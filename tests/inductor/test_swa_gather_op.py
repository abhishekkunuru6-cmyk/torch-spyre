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

"""Correctness of spyre::gather_kv_window (increment 4).

The op copies each Q block's sliding window out of the KV cache into one
compact [B, N*Hq, buffer_width, E] buffer, folded **block-major** (index
n*Hq + h), and emits the matching band mask.

The headline test here is the **fold-order contract**. If the gather emits
head-major while the caller's query fold is block-major (or vice versa),
every slot pairs the wrong window with the wrong head — the graph still
compiles and still runs, it just returns wrong numbers. That makes it the
most dangerous way to get this subtly wrong, so it is tested directly rather
than trusted.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_swa_gather_op.py -v
"""

import unittest

import torch

from torch_spyre._inductor.swa_window_gather import plan_window_gather
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


def _reference_window(cache: torch.Tensor, plan, num_heads: int) -> torch.Tensor:
    """Block-major gather, built independently of the op under test."""
    expansion = num_heads // cache.size(1)
    blocks = []
    for block_index in range(plan.num_q_blocks):
        start = plan.read_start(block_index)
        window = cache[:, :, start : start + plan.buffer_width, :]
        blocks.append(_expand_kv(window, expansion))
    return torch.cat(blocks, dim=1)


def _reference_band(plan) -> torch.Tensor:
    """[1, N, 1, q_block, Wb] additive band, built from the plan's row windows."""
    columns = torch.arange(plan.buffer_width)
    blocks = []
    for block_index in range(plan.num_q_blocks):
        start = plan.read_start(block_index)
        q_start, _ = plan.block_q_range(block_index)
        rows = []
        for offset in range(plan.q_block):
            low, high = plan.row_window(q_start + offset)
            allowed = (columns + start >= low) & (columns + start < high)
            rows.append(allowed)
        blocks.append(torch.stack(rows))
    stacked = torch.stack(blocks)
    band = torch.zeros(stacked.shape, dtype=torch.float16)
    band.masked_fill_(~stacked, float("-inf"))
    return band.unsqueeze(0).unsqueeze(2)


def _fold_query(query: torch.Tensor, plan) -> torch.Tensor:
    """Block-major query fold — must match the gather's order."""
    blocks = [
        query[:, :, n * plan.q_block : (n + 1) * plan.q_block, :]
        for n in range(plan.num_q_blocks)
    ]
    return torch.cat(blocks, dim=1)


class TestGatherKVWindow(unittest.TestCase):
    """The gather itself: does it copy the right rows into the right slots?"""

    def _cache(self, differentiation: int, kvheads: int = HEADS) -> torch.Tensor:
        return cached_randn(
            (BATCH, kvheads, SEQLEN, HEAD_DIM),
            differentiation=differentiation,
            dtype=torch.float16,
        )

    def test_key_window_mha(self):
        plan = _plan()

        def fn(k, v):
            if k.device.type == "spyre":
                return torch.ops.spyre.gather_kv_window(
                    k, v, SEQLEN, WINDOW, HEADS, Q_BLOCK, True
                )[0]
            return _reference_window(k, plan, HEADS)

        compare_with_cpu(fn, self._cache(1), self._cache(2), run_eager=False)

    def test_value_window_mha(self):
        plan = _plan()

        def fn(k, v):
            if k.device.type == "spyre":
                return torch.ops.spyre.gather_kv_window(
                    k, v, SEQLEN, WINDOW, HEADS, Q_BLOCK, True
                )[1]
            return _reference_window(v, plan, HEADS)

        compare_with_cpu(fn, self._cache(1), self._cache(2), run_eager=False)

    def test_key_window_gqa(self):
        # 8 query heads from 2 kv heads; the expand happens inside the gather.
        plan = _plan()

        def fn(k, v):
            if k.device.type == "spyre":
                return torch.ops.spyre.gather_kv_window(
                    k, v, SEQLEN, WINDOW, HEADS, Q_BLOCK, True
                )[0]
            return _reference_window(k, plan, HEADS)

        compare_with_cpu(
            fn, self._cache(1, kvheads=2), self._cache(2, kvheads=2), run_eager=False
        )

    def test_band(self):
        plan = _plan()

        def fn(k, v):
            if k.device.type == "spyre":
                return torch.ops.spyre.gather_kv_window(
                    k, v, SEQLEN, WINDOW, HEADS, Q_BLOCK, True
                )[2]
            return _reference_band(plan)

        compare_with_cpu(fn, self._cache(1), self._cache(2), run_eager=False)

    def test_decode(self):
        # Lq=1 is not a multiple of 64, so decode uses q_block=1 -- one Q block
        # with no intra-block stagger, giving buffer_width == window exactly.
        plan = _plan(seqlen_q=1, seqlen_kv=4096, q_block=1)
        self.assertEqual(plan.buffer_width, WINDOW)
        cache = cached_randn(
            (BATCH, HEADS, 4096, HEAD_DIM), differentiation=3, dtype=torch.float16
        )

        def fn(k, v):
            if k.device.type == "spyre":
                return torch.ops.spyre.gather_kv_window(
                    k, v, 1, WINDOW, HEADS, 1, True
                )[0]
            return _reference_window(k, plan, HEADS)

        compare_with_cpu(fn, cache, cache, run_eager=False)


class TestFoldOrderContract(unittest.TestCase):
    """The gather's fold order must match the caller's query fold.

    A mismatch pairs the wrong window with the wrong head and returns wrong
    numbers without any error, so it is checked directly.
    """

    def test_scores_pair_block_n_with_window_n(self):
        # scores[.., n*Hq + h, ..] must equal q block n of head h against
        # window n of head h. Built per block on CPU, so a head-major gather
        # (or a head-major query fold) fails here.
        plan = _plan()
        query = cached_randn(
            (BATCH, HEADS, SEQLEN, HEAD_DIM), differentiation=4, dtype=torch.float16
        )
        key = cached_randn(
            (BATCH, HEADS, SEQLEN, HEAD_DIM), differentiation=5, dtype=torch.float16
        )
        value = cached_randn(
            (BATCH, HEADS, SEQLEN, HEAD_DIM), differentiation=6, dtype=torch.float16
        )

        def fn(q, k, v):
            if q.device.type == "spyre":
                k_win, _, _ = torch.ops.spyre.gather_kv_window(
                    k, v, SEQLEN, WINDOW, HEADS, Q_BLOCK, True
                )
                q4 = _fold_query(q, plan)
                return torch.matmul(q4, k_win.transpose(-1, -2))

            per_block = []
            for n in range(plan.num_q_blocks):
                start = plan.read_start(n)
                q_blk = q[:, :, n * plan.q_block : (n + 1) * plan.q_block, :]
                k_blk = k[:, :, start : start + plan.buffer_width, :]
                per_block.append(torch.matmul(q_blk, k_blk.transpose(-1, -2)))
            return torch.cat(per_block, dim=1)

        compare_with_cpu(fn, query, key, value, run_eager=False)

    def test_band_splits_as_a_free_view(self):
        # Block-major is what makes scores.view(B, N, Hq, qb, Wb) valid, which
        # is how the band broadcasts across heads instead of being replicated.
        plan = _plan()
        scores = cached_randn(
            (BATCH, plan.num_q_blocks * HEADS, plan.q_block, plan.buffer_width),
            differentiation=7,
            dtype=torch.float16,
        )
        band = _reference_band(plan)

        def fn(s, b):
            split = s.view(BATCH, plan.num_q_blocks, HEADS, plan.q_block, -1)
            return (split + b).view(BATCH, -1, plan.q_block, plan.buffer_width)

        compare_with_cpu(fn, scores, band, run_eager=False)


if __name__ == "__main__":
    unittest.main()
