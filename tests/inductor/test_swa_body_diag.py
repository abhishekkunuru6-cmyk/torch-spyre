# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bisect the rolled-window body, one construct at a time.

_window_roll_attention adds several constructs at once, so this file builds
one iteration of it up in stages on a single shape. The first stage that fails
names the culprit, which is otherwise a compile error with no line number.

  A  gather + matmul                                (rank 4, one block)
  B  A + the rank-4 band
  C  B + softmax + second matmul
  D  C + online-softmax accumulators
  E  D + the four spyre_hints                       (== one loop iteration)
  F  E over every block, concatenated               (== the whole body)

Stages A-E run block 2 -- the first block whose read start is not 0, so a
dropped window shift shows up here rather than passing by accident.

This file replaces the earlier bisect of the abandoned all-at-once body. That
one was chasing

    RuntimeError: Incompatible host_size and dim_order
    (spyre_tensor_impl.cpp:147 -- a tensor's host_size and dim_order disagree
     in length, i.e. a rank mismatch)

whose prime suspect was that body's rank-5 band view,
scores.view(B, N, Hq, qb, Wb) + band. Rolling has no fold and therefore no
rank-5 anything, so if the stages below pass, that failure was the fold's and
died with it. If A or B still fails, the problem is in the shared flash body
and would have bitten either design.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_swa_body_diag.py -v
"""

import math
import unittest

import torch

from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.swa_window_gather import plan_window_gather
from utils_inductor import cached_randn, compare_with_cpu

BATCH = 1
HEADS = 8
SEQLEN = 256
HEAD_DIM = 64
WINDOW = 64
Q_BLOCK = 64

PLAN = plan_window_gather(SEQLEN, SEQLEN, WINDOW, q_block=Q_BLOCK)
assert PLAN is not None
BUFFER_WIDTH = PLAN.buffer_width
SCALE = 1.0 / math.sqrt(math.sqrt(HEAD_DIM))

# The first block that reads from a nonzero offset.
BLOCK = 2
assert PLAN.read_start(BLOCK) > 0


def _inputs():
    query = cached_randn(
        (BATCH, HEADS, SEQLEN, HEAD_DIM), differentiation=1, dtype=torch.float16
    )
    key = cached_randn(
        (BATCH, HEADS, SEQLEN, HEAD_DIM), differentiation=2, dtype=torch.float16
    )
    value = cached_randn(
        (BATCH, HEADS, SEQLEN, HEAD_DIM), differentiation=3, dtype=torch.float16
    )
    return query, key, value


def _band(block_index: int = BLOCK) -> torch.Tensor:
    """[1, 1, q_block, Wb] additive band for one block."""
    columns = torch.arange(BUFFER_WIDTH)
    start = PLAN.read_start(block_index)
    q_start, _ = PLAN.block_q_range(block_index)
    rows = []
    for offset in range(PLAN.q_block):
        low, high = PLAN.row_window(q_start + offset)
        rows.append((columns + start >= low) & (columns + start < high))
    allowed = torch.stack(rows)
    band = torch.zeros(allowed.shape, dtype=torch.float16)
    band.masked_fill_(~allowed, float("-inf"))
    return band.unsqueeze(0).unsqueeze(0)


def _gather(key: torch.Tensor, value: torch.Tensor, block_index: int = BLOCK):
    """Op on spyre, the equivalent slices on CPU."""
    start = PLAN.read_start(block_index)
    q_start, _ = PLAN.block_q_range(block_index)
    if key.device.type == "spyre":
        return torch.ops.spyre.gather_kv_window(
            key,
            value,
            start,
            BUFFER_WIDTH,
            HEADS,
            Q_BLOCK,
            PLAN.q_kv_offset + q_start,
            WINDOW,
            True,
        )[:2]
    k_win = key[:, :, start : start + BUFFER_WIDTH, :].transpose(-1, -2)
    v_win = value[:, :, start : start + BUFFER_WIDTH, :]
    return k_win, v_win


def _q_rows(query: torch.Tensor, block_index: int = BLOCK) -> torch.Tensor:
    q_start, q_end = PLAN.block_q_range(block_index)
    return query[:, :, q_start:q_end, :]


def _scores(query, key, value, block_index: int = BLOCK):
    k_win, _ = _gather(key, value, block_index)
    return torch.matmul(_q_rows(query, block_index) * SCALE, k_win * SCALE)


def _copy_into(new: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    """spyre.copy_f on device, its value semantics on CPU.

    copy_f is registered for PrivateUse1 only, so calling it on the reference
    side raises NotImplementedError before any comparison happens. It returns
    what it wrote, so on CPU the new value alone is the same answer.
    """
    if new.device.type == "spyre":
        return torch.ops.spyre.copy_f(new, destination)
    return new


def _accumulated_block(q, k, v, band, block_index: int = BLOCK):
    """One block through the full accumulator body, without the hints."""
    k_win, v_win = _gather(k, v, block_index)
    running_max = torch.full(
        (BATCH, HEADS, Q_BLOCK, 64), float("-inf"), device=q.device, dtype=q.dtype
    ).amax(dim=-1)
    denominator = torch.zeros(
        (BATCH, HEADS, Q_BLOCK, 64), device=q.device, dtype=q.dtype
    ).amax(dim=-1)
    output = torch.zeros(
        (BATCH, HEADS, Q_BLOCK, HEAD_DIM), device=q.device, dtype=q.dtype
    )

    scores = torch.matmul(_q_rows(q, block_index) * SCALE, k_win * SCALE) + band
    block_max = torch.amax(scores, dim=-1)
    max_running = torch.maximum(running_max, block_max)
    exp_scores = torch.exp(scores - max_running.unsqueeze(-1))
    correction = torch.exp(running_max - max_running)
    denominator = _copy_into(
        denominator * correction + exp_scores.sum(dim=-1), denominator
    )
    output = _copy_into(
        output * correction.unsqueeze(-1) + torch.matmul(exp_scores, v_win),
        output,
    )
    return _copy_into(output / denominator.unsqueeze(-1), output)


class TestBodyBisect(unittest.TestCase):
    def test_a_gather_and_matmul(self):
        def fn(q, k, v):
            return _scores(q, k, v)

        compare_with_cpu(fn, *_inputs(), run_eager=False)

    def test_b_band(self):
        def fn(q, k, v, band):
            return _scores(q, k, v) + band

        compare_with_cpu(fn, *_inputs(), _band(), run_eager=False)

    def test_c_softmax_and_second_matmul(self):
        def fn(q, k, v, band):
            k_win, v_win = _gather(k, v)
            scores = torch.matmul(_q_rows(q) * SCALE, k_win * SCALE) + band
            block_max = torch.amax(scores, dim=-1)
            exp_scores = torch.exp(scores - block_max.unsqueeze(-1))
            denominator = exp_scores.sum(dim=-1)
            return torch.matmul(exp_scores, v_win) / denominator.unsqueeze(-1)

        compare_with_cpu(fn, *_inputs(), _band(), run_eager=False)

    def test_d_online_softmax_accumulators(self):
        def fn(q, k, v, band):
            return _accumulated_block(q, k, v, band)

        compare_with_cpu(fn, *_inputs(), _band(), run_eager=False)

    def test_e_with_hints(self):
        def fn(q, k, v, band):
            with spyre_hint(tiles={"batch_size": max(1, BATCH // 2)}):
                with spyre_hint(tiles={"num_heads": max(1, HEADS // 4)}):
                    with spyre_hint(tiles={"window_size": max(1, BUFFER_WIDTH // 64)}):
                        with spyre_hint(work_div={"num_heads": 4, "window_size": 8}):
                            return _accumulated_block(q, k, v, band)

        compare_with_cpu(fn, *_inputs(), _band(), run_eager=False)

    def test_f_every_block_concatenated(self):
        # The loop itself: N independent iterations, cat along the sequence.
        # Fails here with E passing => the repetition, not the body.
        #
        # The bands are built OUTSIDE fn and passed in, as in every stage
        # above. Building them inside traces the CPU mask construction --
        # arange, comparisons, masked_fill_ -- into the graph, which crashes
        # make_buffer_reuse on the resulting bool buffers ('FixedLayout' has no
        # attribute 'device_layout'). The real body never does that: its band
        # comes from spyre.window_band_mask, a custom op that is opaque to
        # dynamo. That crash is a genuine backend bug, but it is not on this
        # path and must not be what this stage measures.
        bands = [_band(n) for n in range(PLAN.num_q_blocks)]

        def fn(q, k, v, *block_bands):
            blocks = [
                _accumulated_block(q, k, v, band, n)
                for n, band in enumerate(block_bands)
            ]
            return torch.cat(blocks, dim=2)

        compare_with_cpu(fn, *_inputs(), *bands, run_eager=False)


if __name__ == "__main__":
    unittest.main()
