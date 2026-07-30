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

_window_roll_attention adds several constructs at once, so this file builds one
iteration of it up in stages. The first stage that fails names the culprit,
which is otherwise a compile error or a wrong number with no line attached.

  A  gather + matmul                                (rank 4, one block)
  B  A + the rank-4 band
  C  B + softmax + second matmul
  D  C + online-softmax accumulators
  E1 D + the four spyre_hints around everything
  E2 D + the hints around the compute only          (== the real body)
  F  the loop over every block, concatenated

Two shapes run the same ladder:

  PREFILL  Lq=Lkv=256, W=64  -> q_block=64, 4 blocks, buffer_width=128. Green
           as of 2026-07-30, A through F.
  DECODE   Lq=1, Lkv=4096, W=64 -> q_block=1, 1 block, buffer_width=64. This
           is the shape test_swa_window_roll_kernel.py fails on, and it fails
           there *identically with the rolled path on and off* -- at Lq=1 both
           paths read the same [4032, 4096) window and do the same math, so
           the bug is in the shared body, not in either window strategy.

           What is unique to it: buffer_width is exactly ONE stick. Every
           passing shape has two or more (prefill W=64 gives 128; decode GQA
           W=128 gives 128). If a stage fails here that passes on prefill,
           a single-stick KV extent is the difference.

Stages A-E run the first block with a nonzero read start, so a dropped window
shift cannot pass by accident. Bands are always built outside the traced
function: building them inside pulls CPU mask construction into the graph and
crashes make_buffer_reuse on the resulting bool buffers, which the real body
never does -- its band comes from spyre.window_band_mask, opaque to dynamo.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_swa_body_diag.py -v
"""

from contextlib import contextmanager
import math
import unittest

import torch

from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.swa_window_gather import plan_window_gather
from utils_inductor import cached_randn, compare_with_cpu

BATCH = 1
HEADS = 8
HEAD_DIM = 64
SCALE = 1.0 / math.sqrt(math.sqrt(HEAD_DIM))


class _BodyLadder:
    """The staged body, shared by every shape. Not a TestCase on its own."""

    seqlen_q = 256
    seqlen_kv = 256
    window = 64
    q_block = 64

    @classmethod
    def setUpClass(cls):
        cls.plan = plan_window_gather(
            cls.seqlen_q, cls.seqlen_kv, cls.window, q_block=cls.q_block
        )
        assert cls.plan is not None, f"shape not supported: {cls.__name__}"
        # The first block that reads from a nonzero offset, so the window shift
        # is exercised rather than trivially zero.
        cls.block = next(
            (n for n in range(cls.plan.num_q_blocks) if cls.plan.read_start(n) > 0),
            0,
        )

    def inputs(self):
        query = cached_randn(
            (BATCH, HEADS, self.seqlen_q, HEAD_DIM),
            differentiation=1,
            dtype=torch.float16,
        )
        key = cached_randn(
            (BATCH, HEADS, self.seqlen_kv, HEAD_DIM),
            differentiation=2,
            dtype=torch.float16,
        )
        value = cached_randn(
            (BATCH, HEADS, self.seqlen_kv, HEAD_DIM),
            differentiation=3,
            dtype=torch.float16,
        )
        return query, key, value

    def band(self, block_index=None):
        """[1, 1, q_block, Wb] additive band for one block."""
        plan = self.plan
        block_index = self.block if block_index is None else block_index
        columns = torch.arange(plan.buffer_width)
        start = plan.read_start(block_index)
        q_start, _ = plan.block_q_range(block_index)
        rows = []
        for offset in range(plan.q_block):
            low, high = plan.row_window(q_start + offset)
            rows.append((columns + start >= low) & (columns + start < high))
        allowed = torch.stack(rows)
        band = torch.zeros(allowed.shape, dtype=torch.float16)
        band.masked_fill_(~allowed, float("-inf"))
        return band.unsqueeze(0).unsqueeze(0)

    def gather(self, key, value, block_index=None):
        """Op on spyre, the equivalent slices on CPU."""
        plan = self.plan
        block_index = self.block if block_index is None else block_index
        start = plan.read_start(block_index)
        q_start, _ = plan.block_q_range(block_index)
        width = plan.buffer_width
        if key.device.type == "spyre":
            return torch.ops.spyre.gather_kv_window(
                key,
                value,
                start,
                width,
                HEADS,
                plan.q_block,
                plan.q_kv_offset + q_start,
                plan.window_size,
                True,
            )[:2]
        k_win = key[:, :, start : start + width, :].transpose(-1, -2)
        v_win = value[:, :, start : start + width, :]
        return k_win, v_win

    def q_rows(self, query, block_index=None):
        block_index = self.block if block_index is None else block_index
        q_start, q_end = self.plan.block_q_range(block_index)
        return query[:, :, q_start:q_end, :]

    def scores(self, query, key, value, block_index=None):
        k_win, _ = self.gather(key, value, block_index)
        return torch.matmul(self.q_rows(query, block_index) * SCALE, k_win * SCALE)

    @contextmanager
    def hints(self):
        """The decomposition's four hints, in its order."""
        width_tiles = max(1, self.plan.buffer_width // 64)
        with spyre_hint(tiles={"batch_size": max(1, BATCH // 2)}):
            with spyre_hint(tiles={"num_heads": max(1, HEADS // 4)}):
                with spyre_hint(tiles={"window_size": width_tiles}):
                    with spyre_hint(work_div={"num_heads": 4, "window_size": 8}):
                        yield

    def accumulators(self, query):
        """M / denominator / output, built as the decomposition builds them."""
        q_block = self.plan.q_block
        running_max = torch.full(
            (BATCH, HEADS, q_block, 64),
            float("-inf"),
            device=query.device,
            dtype=query.dtype,
        ).amax(dim=-1)
        denominator = torch.zeros(
            (BATCH, HEADS, q_block, 64), device=query.device, dtype=query.dtype
        ).amax(dim=-1)
        output = torch.zeros(
            (BATCH, HEADS, q_block, HEAD_DIM), device=query.device, dtype=query.dtype
        )
        return running_max, denominator, output

    def flash_step(self, q_rows, k_win, v_win, band, running_max, denominator, output):
        """The compute the decomposition puts inside the hint nest."""
        scores = torch.matmul(q_rows * SCALE, k_win * SCALE) + band
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

    def accumulated_block(self, q, k, v, band, block_index=None, hint_scope="none"):
        """One block through the accumulator body.

        ``hint_scope`` decides what the hints enclose, which is the whole point
        of stages E and E2:

          "none"     no hints at all
          "compute"  only flash_step -- what _window_roll_attention does today,
                     with the gather and the accumulator init OUTSIDE
          "all"      gather and init inside too
        """
        if hint_scope == "all":
            with self.hints():
                k_win, v_win = self.gather(k, v, block_index)
                return self.flash_step(
                    self.q_rows(q, block_index),
                    k_win,
                    v_win,
                    band,
                    *self.accumulators(q),
                )

        k_win, v_win = self.gather(k, v, block_index)
        pieces = (
            self.q_rows(q, block_index),
            k_win,
            v_win,
            band,
            *self.accumulators(q),
        )
        if hint_scope == "compute":
            with self.hints():
                return self.flash_step(*pieces)
        return self.flash_step(*pieces)

    # ------------------------------------------------------------------ stages

    def test_a_gather_and_matmul(self):
        compare_with_cpu(
            lambda q, k, v: self.scores(q, k, v), *self.inputs(), run_eager=False
        )

    def test_b_band(self):
        def fn(q, k, v, band):
            return self.scores(q, k, v) + band

        compare_with_cpu(fn, *self.inputs(), self.band(), run_eager=False)

    def test_c_softmax_and_second_matmul(self):
        def fn(q, k, v, band):
            k_win, v_win = self.gather(k, v)
            scores = torch.matmul(self.q_rows(q) * SCALE, k_win * SCALE) + band
            block_max = torch.amax(scores, dim=-1)
            exp_scores = torch.exp(scores - block_max.unsqueeze(-1))
            denominator = exp_scores.sum(dim=-1)
            return torch.matmul(exp_scores, v_win) / denominator.unsqueeze(-1)

        compare_with_cpu(fn, *self.inputs(), self.band(), run_eager=False)

    def test_d_online_softmax_accumulators(self):
        def fn(q, k, v, band):
            return self.accumulated_block(q, k, v, band)

        compare_with_cpu(fn, *self.inputs(), self.band(), run_eager=False)

    def test_e1_hints_around_everything(self):
        # Gather and accumulator init inside the hint nest. This is what the
        # ladder measured before, and it passed at both shapes.
        def fn(q, k, v, band):
            return self.accumulated_block(q, k, v, band, hint_scope="all")

        compare_with_cpu(fn, *self.inputs(), self.band(), run_eager=False)

    def test_e2_hints_around_compute_only(self):
        # What _window_roll_attention actually does: the gather and the
        # accumulator init sit OUTSIDE the hints, only flash_step is inside.
        # E1 passing and E2 failing localises the end-to-end decode failure to
        # that placement -- and names the one-line fix.
        def fn(q, k, v, band):
            return self.accumulated_block(q, k, v, band, hint_scope="compute")

        compare_with_cpu(fn, *self.inputs(), self.band(), run_eager=False)

    def test_f_every_block_concatenated(self):
        # The loop itself. Fails here with E passing => the repetition.
        bands = [self.band(n) for n in range(self.plan.num_q_blocks)]

        def fn(q, k, v, *block_bands):
            blocks = [
                self.accumulated_block(q, k, v, band, n)
                for n, band in enumerate(block_bands)
            ]
            return torch.cat(blocks, dim=2)

        compare_with_cpu(fn, *self.inputs(), *bands, run_eager=False)

    def test_g_the_loop_as_the_decomposition_writes_it(self):
        # F plus the real hint placement: the closest this file gets to
        # _window_roll_attention without going through the op itself.
        bands = [self.band(n) for n in range(self.plan.num_q_blocks)]

        def fn(q, k, v, *block_bands):
            blocks = [
                self.accumulated_block(q, k, v, band, n, hint_scope="compute")
                for n, band in enumerate(block_bands)
            ]
            return torch.cat(blocks, dim=2)

        compare_with_cpu(fn, *self.inputs(), *bands, run_eager=False)


def _copy_into(new: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    """spyre.copy_f on device, its value semantics on CPU.

    copy_f is registered for PrivateUse1 only, so calling it on the reference
    side raises NotImplementedError before any comparison happens. It returns
    what it wrote, so on CPU the new value alone is the same answer.
    """
    if new.device.type == "spyre":
        return torch.ops.spyre.copy_f(new, destination)
    return new


class TestPrefillBody(_BodyLadder, unittest.TestCase):
    """Lq=Lkv=256, W=64 -> 4 blocks of 64, buffer_width=128. Green."""


class TestDecodeBody(_BodyLadder, unittest.TestCase):
    """Lq=1, Lkv=4096, W=64 -> one block, buffer_width=64: ONE stick.

    The shape the end-to-end test fails, identically with the rolled path on
    and off. Whichever stage fails here first is where the shared body breaks
    at a single-stick KV extent.
    """

    seqlen_q = 1
    seqlen_kv = 4096
    window = 64
    q_block = 1

    def test_the_window_is_a_single_stick(self):
        # The one structural difference from every shape that passes.
        self.assertEqual(self.plan.buffer_width, 64)
        self.assertEqual(self.plan.num_q_blocks, 1)
        self.assertEqual(self.plan.read_start(0), 4032)


if __name__ == "__main__":
    unittest.main()
