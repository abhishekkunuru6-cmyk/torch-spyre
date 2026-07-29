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

"""Rank-5 spike for the gathered-window SWA structure (increment 3).

The planned structure gives the gathered KV buffer a Q-block axis:

    q5    : [B, H, N, q_block, E]
    k_win : [B, H, N, Wb,      E]        window i belongs to Q block i

so N becomes a *batch* dimension of the matmul and Q block i pairs with
window i by ordinary batched-matmul semantics -- no loop, no sliding read.
That is the whole reason the structure works, and it rests on the backend
supporting rank-5 tensors with THREE batch dims. spyre__sdpa_overrideable
only ever works at rank 4, so this is unproven.

This file probes exactly that one variable. No spyre_hint anywhere -- whether
the hints produce real device loops is a separate risk, tracked as increment
6; mixing it in here would confound the answer.

How to read the result:

  - everything passes            -> rank 5 is fine, write the op (increment 4)
  - rank-5 tests fail, rank-4
    folded fallback passes       -> fall back to folding N into the batch axis
                                    ([B*N, H, q_block, E]), at the cost of the
                                    separate Q-block hint
  - both fail                    -> the shapes themselves are wrong, not the rank

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_swa_rank5_spike.py -v
"""

import math
import unittest

import torch

from utils_inductor import cached_randn, compare_with_cpu

# One realistic prefill block decomposition: Lq = 256, window 64 -> Wb = 128,
# N = 4 Q blocks. Small enough to compile fast, structured like the real thing.
BATCH = 1
HEADS = 8
NBLOCKS = 4
Q_BLOCK = 64
WINDOW_BUF = 128
HEAD_DIM = 64

SCALE = 1.0 / math.sqrt(math.sqrt(HEAD_DIM))


def _q5(differentiation: int = 1) -> torch.Tensor:
    return cached_randn(
        (BATCH, HEADS, NBLOCKS, Q_BLOCK, HEAD_DIM),
        differentiation=differentiation,
        dtype=torch.float16,
    )


def _kv5(differentiation: int) -> torch.Tensor:
    return cached_randn(
        (BATCH, HEADS, NBLOCKS, WINDOW_BUF, HEAD_DIM),
        differentiation=differentiation,
        dtype=torch.float16,
    )


def _band(fill: float = float("-inf")) -> torch.Tensor:
    """Additive band over the window: masks the first half of every row.

    ``fill`` is separated out so a rank-5 broadcast failure can be told apart
    from a -inf handling failure.
    """
    band = torch.zeros(1, 1, NBLOCKS, Q_BLOCK, WINDOW_BUF, dtype=torch.float16)
    band[..., : WINDOW_BUF // 2] = fill
    return band


class TestRank5Spike(unittest.TestCase):
    """Does the backend handle rank-5 tensors with three batch dims?"""

    def test_matmul_scores(self):
        # [B,H,N,q,E] @ [B,H,N,E,Wb] -> [B,H,N,q,Wb]; the first attention matmul.
        def fn(q, k):
            return torch.matmul(q, k.transpose(-1, -2))

        compare_with_cpu(fn, _q5(1), _kv5(2), run_eager=False)

    def test_matmul_output(self):
        # [B,H,N,q,Wb] @ [B,H,N,Wb,E] -> [B,H,N,q,E]; the second attention matmul.
        probabilities = cached_randn(
            (BATCH, HEADS, NBLOCKS, Q_BLOCK, WINDOW_BUF),
            differentiation=4,
            dtype=torch.float16,
        )

        def fn(p, v):
            return torch.matmul(p, v)

        compare_with_cpu(fn, probabilities, _kv5(3), run_eager=False)

    def test_add_broadcast_band_finite(self):
        # scores + band, broadcasting [1,1,N,q,Wb] over the leading two dims.
        # Finite fill, so this isolates the broadcast from -inf handling.
        scores = cached_randn(
            (BATCH, HEADS, NBLOCKS, Q_BLOCK, WINDOW_BUF),
            differentiation=5,
            dtype=torch.float16,
        )

        def fn(s, band):
            return s + band

        compare_with_cpu(fn, scores, _band(fill=-100.0), run_eager=False)

    def test_add_broadcast_band_neg_inf(self):
        # Same broadcast, -inf fill -- the value the real band mask uses.
        scores = cached_randn(
            (BATCH, HEADS, NBLOCKS, Q_BLOCK, WINDOW_BUF),
            differentiation=5,
            dtype=torch.float16,
        )

        def fn(s, band):
            return s + band

        compare_with_cpu(fn, scores, _band(), run_eager=False)

    def test_amax_and_broadcast_subtract(self):
        # amax over the window dim, then subtract it back -- softmax's shape moves.
        scores = cached_randn(
            (BATCH, HEADS, NBLOCKS, Q_BLOCK, WINDOW_BUF),
            differentiation=6,
            dtype=torch.float16,
        )

        def fn(s):
            block_max = torch.amax(s, dim=-1)
            return s - block_max.unsqueeze(-1)

        compare_with_cpu(fn, scores, run_eager=False)

    def test_sum_over_window(self):
        # The softmax denominator reduction at rank 5.
        scores = cached_randn(
            (BATCH, HEADS, NBLOCKS, Q_BLOCK, WINDOW_BUF),
            differentiation=7,
            dtype=torch.float16,
        )

        def fn(s):
            return s.sum(dim=-1)

        compare_with_cpu(fn, scores, run_eager=False)

    def test_view_from_rank4(self):
        # query [B,H,Lq,E] -> [B,H,N,q_block,E]; how the Q axis is created.
        query = cached_randn(
            (BATCH, HEADS, NBLOCKS * Q_BLOCK, HEAD_DIM),
            differentiation=8,
            dtype=torch.float16,
        )

        def fn(q):
            reshaped = q.view(BATCH, HEADS, NBLOCKS, Q_BLOCK, HEAD_DIM)
            return reshaped * 2.0

        compare_with_cpu(fn, query, run_eager=False)

    def test_attention_body(self):
        # The whole §4 body at rank 5, hints excluded, softmax built from the
        # same ops SDPA uses rather than torch.softmax.
        def fn(q, k, v, band):
            scores = torch.matmul(q * SCALE, (k * SCALE).transpose(-1, -2))
            scores = scores + band
            block_max = torch.amax(scores, dim=-1)
            exp_scores = torch.exp(scores - block_max.unsqueeze(-1))
            denominator = exp_scores.sum(dim=-1)
            out = torch.matmul(exp_scores, v)
            return out / denominator.unsqueeze(-1)

        compare_with_cpu(fn, _q5(1), _kv5(2), _kv5(3), _band(), run_eager=False)


class TestRank4FoldedFallback(unittest.TestCase):
    """The fallback if rank 5 fails: fold N into the batch axis."""

    def test_attention_body_folded(self):
        # Identical math at [B*N, H, q_block, E] -- rank 4, as SDPA uses.
        folded_batch = BATCH * NBLOCKS
        query = cached_randn(
            (folded_batch, HEADS, Q_BLOCK, HEAD_DIM),
            differentiation=1,
            dtype=torch.float16,
        )
        key = cached_randn(
            (folded_batch, HEADS, WINDOW_BUF, HEAD_DIM),
            differentiation=2,
            dtype=torch.float16,
        )
        value = cached_randn(
            (folded_batch, HEADS, WINDOW_BUF, HEAD_DIM),
            differentiation=3,
            dtype=torch.float16,
        )
        band = torch.zeros(1, 1, Q_BLOCK, WINDOW_BUF, dtype=torch.float16)
        band[..., : WINDOW_BUF // 2] = float("-inf")

        def fn(q, k, v, mask):
            scores = torch.matmul(q * SCALE, (k * SCALE).transpose(-1, -2))
            scores = scores + mask
            block_max = torch.amax(scores, dim=-1)
            exp_scores = torch.exp(scores - block_max.unsqueeze(-1))
            denominator = exp_scores.sum(dim=-1)
            out = torch.matmul(exp_scores, v)
            return out / denominator.unsqueeze(-1)

        compare_with_cpu(fn, query, key, value, band, run_eager=False)


if __name__ == "__main__":
    unittest.main()
