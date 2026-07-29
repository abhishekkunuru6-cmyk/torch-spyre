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

"""Bisect the increment-5 body failure. Expect some of these to FAIL.

Every gathered-window kernel test died with the same error, independent of
shape:

    RuntimeError: Incompatible host_size and dim_order
    (spyre_tensor_impl.cpp:147 -- a tensor's host_size and dim_order
     disagree in length, i.e. a rank mismatch)

_window_gather_attention adds several constructs at once, so this file builds
the body up one construct at a time on a single shape. The first test that
fails names the culprit.

  A  gather + query fold + matmul                      (rank 4 only)
  B  A + band added through a rank-5 view              <- prime suspect
  C  A + band pre-folded to rank 4                     <- the alternative
  D  C or B + softmax + second matmul
  E  D + online-softmax accumulators
  F  E + the four spyre_hints                          (== the real body)

Reading it:

  A fails                -> the gather/fold/matmul core is wrong
  B fails, C passes      -> the rank-5 view is the problem; emit a rank-4
                            band from the op (costs Hq x band memory)
  B and C pass, D fails  -> softmax shapes, not the band
  E fails, D passes      -> the accumulators
  F fails, E passes      -> a hint names a dim that does not resolve

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
NBLOCKS = PLAN.num_q_blocks
BUFFER_WIDTH = PLAN.buffer_width
FOLDED = NBLOCKS * HEADS
SCALE = 1.0 / math.sqrt(math.sqrt(HEAD_DIM))


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


def _band_blocks() -> torch.Tensor:
    """[N, q_block, buffer_width] additive band, one per Q block."""
    columns = torch.arange(BUFFER_WIDTH)
    blocks = []
    for block_index in range(NBLOCKS):
        start = PLAN.read_start(block_index)
        q_start, _ = PLAN.block_q_range(block_index)
        rows = []
        for offset in range(PLAN.q_block):
            low, high = PLAN.row_window(q_start + offset)
            rows.append((columns + start >= low) & (columns + start < high))
        blocks.append(torch.stack(rows))
    allowed = torch.stack(blocks)
    band = torch.zeros(allowed.shape, dtype=torch.float16)
    band.masked_fill_(~allowed, float("-inf"))
    return band


def _band_rank5() -> torch.Tensor:
    """[1, N, 1, q_block, Wb] -- broadcasts across heads, minimal memory."""
    return _band_blocks().unsqueeze(0).unsqueeze(2)


def _band_rank4() -> torch.Tensor:
    """[1, N*Hq, q_block, Wb] -- materialized per head, block-major."""
    return _band_blocks().repeat_interleave(HEADS, dim=0).unsqueeze(0)


def _fold_query(query: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [query[:, :, n * Q_BLOCK : (n + 1) * Q_BLOCK, :] for n in range(NBLOCKS)],
        dim=1,
    )


def _gather(key: torch.Tensor, value: torch.Tensor):
    """Op on spyre, equivalent slices on CPU."""
    if key.device.type == "spyre":
        return torch.ops.spyre.gather_kv_window(
            key, value, SEQLEN, WINDOW, HEADS, Q_BLOCK, True
        )[:2]
    starts = [PLAN.read_start(n) for n in range(NBLOCKS)]
    k_win = torch.cat(
        [key[:, :, s : s + BUFFER_WIDTH, :].transpose(-1, -2) for s in starts], dim=1
    )
    v_win = torch.cat([value[:, :, s : s + BUFFER_WIDTH, :] for s in starts], dim=1)
    return k_win, v_win


def _scores(query, key, value):
    k_win, _ = _gather(key, value)
    return torch.matmul(_fold_query(query) * SCALE, k_win * SCALE)


class TestBodyBisect(unittest.TestCase):
    def test_a_gather_fold_matmul(self):
        def fn(q, k, v):
            return _scores(q, k, v)

        compare_with_cpu(fn, *_inputs(), run_eager=False)

    def test_b_band_through_rank5_view(self):
        def fn(q, k, v, band):
            scores = _scores(q, k, v)
            return (
                scores.view(BATCH, NBLOCKS, HEADS, Q_BLOCK, BUFFER_WIDTH) + band
            ).view(BATCH, FOLDED, Q_BLOCK, BUFFER_WIDTH)

        compare_with_cpu(fn, *_inputs(), _band_rank5(), run_eager=False)

    def test_c_band_prefolded_rank4(self):
        def fn(q, k, v, band):
            return _scores(q, k, v) + band

        compare_with_cpu(fn, *_inputs(), _band_rank4(), run_eager=False)

    def test_d_softmax_and_second_matmul(self):
        # Uses the rank-4 band so this stage is independent of B's outcome.
        def fn(q, k, v, band):
            k_win, v_win = _gather(k, v)
            scores = torch.matmul(_fold_query(q) * SCALE, k_win * SCALE) + band
            block_max = torch.amax(scores, dim=-1)
            exp_scores = torch.exp(scores - block_max.unsqueeze(-1))
            denominator = exp_scores.sum(dim=-1)
            out = torch.matmul(exp_scores, v_win)
            return out / denominator.unsqueeze(-1)

        compare_with_cpu(fn, *_inputs(), _band_rank4(), run_eager=False)

    def test_e_online_softmax_accumulators(self):
        def fn(q, k, v, band):
            k_win, v_win = _gather(k, v)
            m_reduced = torch.full(
                (BATCH, FOLDED, Q_BLOCK, 64),
                float("-inf"),
                device=q.device,
                dtype=q.dtype,
            )
            running_max = m_reduced.amax(dim=-1)
            denominator = torch.zeros(
                (BATCH, FOLDED, Q_BLOCK, 64), device=q.device, dtype=q.dtype
            ).amax(dim=-1)
            output = torch.zeros(
                (BATCH, FOLDED, Q_BLOCK, HEAD_DIM), device=q.device, dtype=q.dtype
            )

            scores = torch.matmul(_fold_query(q) * SCALE, k_win * SCALE) + band
            block_max = torch.amax(scores, dim=-1)
            max_running = torch.maximum(running_max, block_max)
            exp_scores = torch.exp(scores - max_running.unsqueeze(-1))
            correction = torch.exp(running_max - max_running)
            denominator = torch.ops.spyre.copy_f(
                denominator * correction + exp_scores.sum(dim=-1), denominator
            )
            output = torch.ops.spyre.copy_f(
                output * correction.unsqueeze(-1) + torch.matmul(exp_scores, v_win),
                output,
            )
            return torch.ops.spyre.copy_f(output / denominator.unsqueeze(-1), output)

        compare_with_cpu(fn, *_inputs(), _band_rank4(), run_eager=False)

    def test_f_with_hints(self):
        def fn(q, k, v, band):
            k_win, v_win = _gather(k, v)
            output = torch.zeros(
                (BATCH, FOLDED, Q_BLOCK, HEAD_DIM), device=q.device, dtype=q.dtype
            )
            with spyre_hint(tiles={"batch_size": max(1, BATCH // 2)}):
                with spyre_hint(tiles={"num_heads": max(1, FOLDED // 4)}):
                    with spyre_hint(tiles={"window_size": max(1, BUFFER_WIDTH // 64)}):
                        with spyre_hint(work_div={"num_heads": 4, "window_size": 8}):
                            scores = (
                                torch.matmul(_fold_query(q) * SCALE, k_win * SCALE)
                                + band
                            )
                            block_max = torch.amax(scores, dim=-1)
                            exp_scores = torch.exp(scores - block_max.unsqueeze(-1))
                            denominator = exp_scores.sum(dim=-1)
                            output = torch.ops.spyre.copy_f(
                                torch.matmul(exp_scores, v_win), output
                            )
            return output / denominator.unsqueeze(-1)

        compare_with_cpu(fn, *_inputs(), _band_rank4(), run_eager=False)


if __name__ == "__main__":
    unittest.main()
