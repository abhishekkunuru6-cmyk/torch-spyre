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

"""Correctness of spyre::gather_kv_window (increment R3).

The op copies **one** Q block's sliding window out of the KV cache and emits
the matching band. K comes out already transposed ([B, Hq, E, buffer_width])
so the caller can matmul it directly.

The block tests below drive every block of a plan and concatenate, which is
also the shape of the loop the decomposition runs: block n reads
plan.read_start(n) and nothing carries between iterations. Where the abandoned
all-at-once gather needed a fold-order contract test -- a mismatch there paired
the wrong window with the wrong head and returned wrong numbers with no error
-- pairing is now positional by iteration and cannot be got out of step.

GQA is tested one block at a time, which is what the body consumes. Cat-ing
several expanded windows together zeroes the leading slots on device; that is
a backend defect, recorded in format/swa_backend_bugs.md, and no code on this
path does it.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_swa_gather_op.py -v
"""

import unittest

import torch

from torch_spyre._inductor.swa_window_gather import (
    check_window_read,
    plan_window_gather,
)
from utils_inductor import cached_randn, compare_with_cpu

BATCH = 1
HEADS = 8
SEQLEN = 256
HEAD_DIM = 64
WINDOW = 64
Q_BLOCK = 64


def _plan(seqlen_q=SEQLEN, seqlen_kv=SEQLEN, window=WINDOW, q_block=Q_BLOCK):
    plan = plan_window_gather(seqlen_q, seqlen_kv, window, q_block=q_block)
    assert plan is not None, f"shape not supported: {seqlen_q}/{seqlen_kv}/{window}"
    return plan


def _expand_kv(tensor: torch.Tensor, expansion: int) -> torch.Tensor:
    if expansion == 1:
        return tensor
    return tensor.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)


def _gather_block(cache: torch.Tensor, plan, block_index: int, num_heads: int, which):
    """Call the op for one block; ``which`` picks k_win (0) or v_win (1)."""
    q_start, _ = plan.block_q_range(block_index)
    return torch.ops.spyre.gather_kv_window(
        cache,
        cache,
        plan.read_start(block_index),
        plan.buffer_width,
        num_heads,
        plan.q_block,
        plan.q_kv_offset + q_start,
        plan.window_size,
        True,
    )[which]


def _reference_window(
    cache: torch.Tensor, plan, block_index: int, num_heads: int, transpose: bool = False
) -> torch.Tensor:
    """One block's window, built independently of the op under test."""
    start = plan.read_start(block_index)
    window = cache[:, :, start : start + plan.buffer_width, :]
    if transpose:
        window = window.transpose(-1, -2)
    return _expand_kv(window, num_heads // cache.size(1))


def _reference_band(plan, block_index: int) -> torch.Tensor:
    """[1, 1, q_block, Wb] additive band, built from the plan's row windows."""
    columns = torch.arange(plan.buffer_width)
    start = plan.read_start(block_index)
    q_start, _ = plan.block_q_range(block_index)
    rows = []
    for offset in range(plan.q_block):
        low, high = plan.row_window(q_start + offset)
        rows.append((columns + start >= low) & (columns + start < high))
    stacked = torch.stack(rows)
    band = torch.zeros(stacked.shape, dtype=torch.float16)
    band.masked_fill_(~stacked, float("-inf"))
    return band.unsqueeze(0).unsqueeze(0)


class TestGatherKVWindow(unittest.TestCase):
    """The gather itself: does it copy the right rows into the buffer?"""

    def _cache(self, differentiation: int, kvheads: int = HEADS) -> torch.Tensor:
        return cached_randn(
            (BATCH, kvheads, SEQLEN, HEAD_DIM),
            differentiation=differentiation,
            dtype=torch.float16,
        )

    def test_key_window_mha(self):
        plan = _plan()

        def fn(k, v):
            blocks = range(plan.num_q_blocks)
            if k.device.type == "spyre":
                windows = [_gather_block(k, plan, n, HEADS, 0) for n in blocks]
            else:
                windows = [
                    _reference_window(k, plan, n, HEADS, transpose=True) for n in blocks
                ]
            return torch.cat(windows, dim=1)

        compare_with_cpu(fn, self._cache(1), self._cache(2), run_eager=False)

    def test_value_window_mha(self):
        plan = _plan()

        def fn(k, v):
            blocks = range(plan.num_q_blocks)
            if k.device.type == "spyre":
                windows = [_gather_block(v, plan, n, HEADS, 1) for n in blocks]
            else:
                windows = [_reference_window(v, plan, n, HEADS) for n in blocks]
            return torch.cat(windows, dim=1)

        compare_with_cpu(fn, self._cache(1), self._cache(2), run_eager=False)

    def test_key_window_gqa(self):
        # 8 query heads from 2 kv heads; the expand happens inside the gather.
        # One block, no cat -- this is exactly what the body consumes.
        plan = _plan()

        def fn(k, v):
            if k.device.type == "spyre":
                return _gather_block(k, plan, 2, HEADS, 0)
            return _reference_window(k, plan, 2, HEADS, transpose=True)

        compare_with_cpu(
            fn, self._cache(1, kvheads=2), self._cache(2, kvheads=2), run_eager=False
        )

    def test_band(self):
        plan = _plan()

        def fn(k, v):
            blocks = range(plan.num_q_blocks)
            if k.device.type == "spyre":
                bands = [_gather_block(k, plan, n, HEADS, 2) for n in blocks]
            else:
                bands = [_reference_band(plan, n) for n in blocks]
            return torch.cat(bands, dim=1)

        compare_with_cpu(fn, self._cache(1), self._cache(2), run_eager=False)

    def test_interior_block_reads_a_shifted_window(self):
        # Block 2 is the first whose read start is not 0, so it is where a
        # dropped shift would show up while blocks 0 and 1 still passed.
        plan = _plan()
        self.assertEqual(plan.read_start(0), 0)
        self.assertGreater(plan.read_start(2), 0)

        def fn(k, v):
            if k.device.type == "spyre":
                return _gather_block(k, plan, 2, HEADS, 0)
            return _reference_window(k, plan, 2, HEADS, transpose=True)

        compare_with_cpu(fn, self._cache(1), self._cache(2), run_eager=False)

    def test_decode_band(self):
        # The decode band, whose all-zeros value is what the body relies on to
        # skip the add entirely -- see block_is_fully_attended and issue 1 in
        # format/swa_backend_bugs.md, where adding it corrupted the output.
        #
        # At decode the band is entirely zeros: buffer_width == window, so the
        # single row can attend to every column. That makes garbage easy to
        # see and makes a wrong -inf impossible to miss.
        plan = _plan(seqlen_q=1, seqlen_kv=4096, q_block=1)
        cache = cached_randn(
            (BATCH, HEADS, 4096, HEAD_DIM), differentiation=3, dtype=torch.float16
        )

        def fn(k, v):
            if k.device.type == "spyre":
                return _gather_block(k, plan, 0, HEADS, 2)
            return _reference_band(plan, 0)

        compare_with_cpu(fn, cache, cache, run_eager=False)

    def test_decode_band_is_all_zeros(self):
        # Guards the test above from passing by agreeing on garbage.
        plan = _plan(seqlen_q=1, seqlen_kv=4096, q_block=1)
        band = _reference_band(plan, 0)
        self.assertEqual(band.shape, (1, 1, 1, 64))
        self.assertTrue(torch.equal(band, torch.zeros_like(band)))

    def test_decode_value_window(self):
        # k_win at decode is covered below; v_win is not, and it is the other
        # operand of the second matmul.
        plan = _plan(seqlen_q=1, seqlen_kv=4096, q_block=1)
        cache = cached_randn(
            (BATCH, HEADS, 4096, HEAD_DIM), differentiation=3, dtype=torch.float16
        )

        def fn(k, v):
            if k.device.type == "spyre":
                return _gather_block(v, plan, 0, HEADS, 1)
            return _reference_window(v, plan, 0, HEADS)

        compare_with_cpu(fn, cache, cache, run_eager=False)

    def test_decode(self):
        # Lq=1 is not a multiple of 64, so decode uses q_block=1 -- one Q block
        # with no intra-block stagger, giving buffer_width == window exactly.
        plan = _plan(seqlen_q=1, seqlen_kv=4096, q_block=1)
        self.assertEqual(plan.buffer_width, WINDOW)
        self.assertEqual(plan.num_q_blocks, 1)
        cache = cached_randn(
            (BATCH, HEADS, 4096, HEAD_DIM), differentiation=3, dtype=torch.float16
        )

        def fn(k, v):
            if k.device.type == "spyre":
                return _gather_block(k, plan, 0, HEADS, 0)
            return _reference_window(k, plan, 0, HEADS, transpose=True)

        compare_with_cpu(fn, cache, cache, run_eager=False)


class TestWindowReadValidation(unittest.TestCase):
    """The op takes its placement as plain ints, so it validates them.

    A caller that computes read_start itself can walk off the end of the cache;
    unchecked that surfaces as a shape mismatch deep in the lowering.
    """

    def _args(self, **overrides):
        args = dict(
            read_start=0,
            buffer_width=128,
            seqlen_kv=256,
            q_block=64,
            window_size=64,
            num_heads=8,
            num_kv_heads=8,
        )
        args.update(overrides)
        return args

    def test_valid_read_passes(self):
        self.assertIsNone(check_window_read(**self._args()))

    def test_last_legal_read_passes(self):
        self.assertIsNone(check_window_read(**self._args(read_start=128)))

    def test_read_past_the_cache_is_rejected(self):
        reason = check_window_read(**self._args(read_start=129))
        self.assertIsNotNone(reason)
        self.assertIn("runs past the cache", reason)

    def test_negative_read_is_rejected(self):
        self.assertIsNotNone(check_window_read(**self._args(read_start=-1)))

    def test_indivisible_gqa_ratio_is_rejected(self):
        reason = check_window_read(**self._args(num_heads=8, num_kv_heads=3))
        self.assertIsNotNone(reason)
        self.assertIn("whole multiple", reason)

    def test_every_plan_read_is_valid(self):
        # The validator must not reject reads the planner itself produces.
        for seqlen_q, seqlen_kv, q_block in ((256, 256, 64), (1, 4096, 1)):
            plan = _plan(seqlen_q=seqlen_q, seqlen_kv=seqlen_kv, q_block=q_block)
            for n in range(plan.num_q_blocks):
                self.assertIsNone(
                    check_window_read(
                        read_start=plan.read_start(n),
                        buffer_width=plan.buffer_width,
                        seqlen_kv=seqlen_kv,
                        q_block=plan.q_block,
                        window_size=plan.window_size,
                        num_heads=HEADS,
                        num_kv_heads=HEADS,
                    ),
                    f"planner produced a read the op rejects (block {n})",
                )


if __name__ == "__main__":
    unittest.main()
