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

"""CPU equivalence for the gathered KV window model (increment 2).

Runs entirely on CPU in float64 — no device, no torch.compile, no HW. The
question here is only whether gathering a compact [B, H, buffer_width, D]
window per Q block computes the same attention as masking the full
[B, H, seqlen_kv, D] cache in place.

Two claims, and the distinction matters:

  1. EXACT — for every query row, the set of KV columns left unmasked inside
     the compact buffer is identical to the set the full-width band mask
     leaves unmasked. Combinatorial, so no tolerance is involved. This is the
     claim that licenses the whole approach.

  2. ROUNDOFF — output values agree to ~1e-14 in float64. This cannot be
     bit-exact: the gathered softmax reduces buffer_width terms where the
     reference reduces seqlen_kv, and though the extra entries are exact
     zeros, sum blocks its reduction differently at different widths, so the
     nonzero terms group differently.

Run:
    python3 -m pytest tests/inductor/test_swa_window_gather_model.py -v
"""

import math

import pytest
import torch

from torch_spyre._inductor.swa_window_gather import (
    WindowGatherPlan,
    plan_window_gather,
)

# (batch, num_heads, num_kvheads, seqlen_q, seqlen_kv, head_dim, window)
SHAPES = [
    (1, 8, 8, 512, 512, 64, 64),  # prefill, MHA
    (1, 8, 8, 512, 512, 64, 128),
    (1, 8, 2, 512, 512, 64, 128),  # prefill, GQA
    (2, 4, 4, 256, 256, 64, 64),  # batch > 1
    (2, 4, 2, 256, 256, 32, 64),  # batch > 1, GQA, narrow head
    (1, 2, 2, 192, 192, 64, 64),
    (1, 8, 8, 1, 4096, 64, 64),  # decode, long cache
    (1, 4, 2, 1, 512, 64, 128),  # decode, GQA
    (2, 4, 2, 100, 256, 64, 64),  # ragged Lq -> unaligned q_kv_offset
]

IDS = [f"b{b}_h{h}kv{kv}_q{lq}_kv{lkv}_d{d}_w{w}" for b, h, kv, lq, lkv, d, w in SHAPES]


def _expand_kv(tensor: torch.Tensor, expansion: int) -> torch.Tensor:
    """GQA broadcast, using the same idiom as the decomposition."""
    if expansion == 1:
        return tensor
    return tensor.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)


def _reference_allowed(
    seqlen_q: int, seqlen_kv: int, window: int, is_causal: bool = True
) -> torch.Tensor:
    """Full-width [seqlen_q, seqlen_kv] boolean band — the definition of SWA."""
    q_kv_offset = seqlen_kv - seqlen_q
    q_coord = torch.arange(seqlen_q) + q_kv_offset
    k_coord = torch.arange(seqlen_kv)
    delta = q_coord.unsqueeze(-1) - k_coord.unsqueeze(0)
    if is_causal:
        return (delta >= 0) & (delta < window)
    return delta.abs() < window


def _block_allowed(plan: WindowGatherPlan, qi: int) -> torch.Tensor:
    """[q_len, buffer_width] boolean band inside Q block qi's compact buffer."""
    q_start, q_end = plan.block_q_range(qi)
    start = plan.read_start(qi)
    cols = torch.arange(start, start + plan.buffer_width)
    rows = []
    for q_index in range(q_start, q_end):
        lo, hi = plan.row_window(q_index)
        rows.append((cols >= lo) & (cols < hi))
    return torch.stack(rows)


def _additive(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Boolean band -> additive 0 / -inf mask."""
    return torch.where(
        allowed,
        torch.zeros((), dtype=dtype),
        torch.full((), float("-inf"), dtype=dtype),
    )


def reference_swa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window: int,
    scale: float,
) -> torch.Tensor:
    """Full attention over the whole cache behind a band mask."""
    expansion = query.size(1) // key.size(1)
    k = _expand_kv(key, expansion)
    v = _expand_kv(value, expansion)
    allowed = _reference_allowed(query.size(2), key.size(2), window)
    scores = torch.matmul(query, k.transpose(-1, -2)) * scale
    scores = scores + _additive(allowed, query.dtype)
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def gathered_swa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: WindowGatherPlan,
    scale: float,
) -> torch.Tensor:
    """Per Q block, attend only against a compact gathered KV window."""
    expansion = query.size(1) // key.size(1)
    out_blocks = []
    for qi in range(plan.num_q_blocks):
        q_start, q_end = plan.block_q_range(qi)
        start = plan.read_start(qi)
        stop = start + plan.buffer_width

        q_blk = query[:, :, q_start:q_end, :]
        k_win = _expand_kv(key[:, :, start:stop, :], expansion)
        v_win = _expand_kv(value[:, :, start:stop, :], expansion)

        band = _additive(_block_allowed(plan, qi), query.dtype)
        scores = torch.matmul(q_blk, k_win.transpose(-1, -2)) * scale
        scores = scores + band
        out_blocks.append(torch.matmul(torch.softmax(scores, dim=-1), v_win))
    return torch.cat(out_blocks, dim=2)


def _make_inputs(batch, heads, kvheads, seqlen_q, seqlen_kv, head_dim, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape_q = (batch, heads, seqlen_q, head_dim)
    shape_kv = (batch, kvheads, seqlen_kv, head_dim)
    query = torch.randn(shape_q, generator=generator, dtype=torch.float64)
    key = torch.randn(shape_kv, generator=generator, dtype=torch.float64)
    value = torch.randn(shape_kv, generator=generator, dtype=torch.float64)
    return query, key, value


class TestColumnSelection:
    """Claim 1 — exact. The compact buffer selects the same (q, k) pairs."""

    @pytest.mark.parametrize(
        "batch,heads,kvheads,seqlen_q,seqlen_kv,head_dim,window", SHAPES, ids=IDS
    )
    def test_unmasked_sets_are_identical(
        self, batch, heads, kvheads, seqlen_q, seqlen_kv, head_dim, window
    ):
        plan = plan_window_gather(seqlen_q, seqlen_kv, window)
        assert plan is not None
        reference = _reference_allowed(seqlen_q, seqlen_kv, window)

        for qi in range(plan.num_q_blocks):
            q_start, q_end = plan.block_q_range(qi)
            start = plan.read_start(qi)
            block = _block_allowed(plan, qi)
            for i, q_index in enumerate(range(q_start, q_end)):
                gathered_cols = (block[i].nonzero().flatten() + start).tolist()
                reference_cols = reference[q_index].nonzero().flatten().tolist()
                assert gathered_cols == reference_cols, (
                    f"block {qi} row {q_index}: gathered {gathered_cols[:8]}... "
                    f"vs reference {reference_cols[:8]}..."
                )

    @pytest.mark.parametrize(
        "batch,heads,kvheads,seqlen_q,seqlen_kv,head_dim,window", SHAPES, ids=IDS
    )
    def test_no_row_is_fully_masked(
        self, batch, heads, kvheads, seqlen_q, seqlen_kv, head_dim, window
    ):
        # A fully masked row would make softmax produce nan.
        plan = plan_window_gather(seqlen_q, seqlen_kv, window)
        assert plan is not None
        for qi in range(plan.num_q_blocks):
            assert _block_allowed(plan, qi).any(dim=-1).all()


class TestNumericAgreement:
    """Claim 2 — roundoff. Same attention output, float64."""

    @pytest.mark.parametrize(
        "batch,heads,kvheads,seqlen_q,seqlen_kv,head_dim,window", SHAPES, ids=IDS
    )
    def test_matches_full_masked_reference(
        self, batch, heads, kvheads, seqlen_q, seqlen_kv, head_dim, window
    ):
        plan = plan_window_gather(seqlen_q, seqlen_kv, window)
        assert plan is not None
        query, key, value = _make_inputs(
            batch, heads, kvheads, seqlen_q, seqlen_kv, head_dim
        )
        scale = 1.0 / math.sqrt(head_dim)

        expected = reference_swa(query, key, value, window, scale)
        actual = gathered_swa(query, key, value, plan, scale)

        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all()
        max_diff = (actual - expected).abs().max().item()
        assert max_diff < 1e-12, f"max abs diff {max_diff:.3e}"

    def test_differs_from_unwindowed_attention(self):
        # Guards against the mask being a no-op: windowing must change the answer.
        batch, heads, seqlen, head_dim, window = 1, 4, 256, 64, 64
        plan = plan_window_gather(seqlen, seqlen, window)
        assert plan is not None
        query, key, value = _make_inputs(batch, heads, heads, seqlen, seqlen, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        windowed = gathered_swa(query, key, value, plan, scale)
        causal = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=scale
        )
        assert (windowed - causal).abs().max().item() > 1e-3


class TestFloat16AgreementWithTheKernelReference:
    """Claim 3 — the kernel test's comparison, on CPU, in its own dtype.

    Every claim above is float64. The device test compares against
    F.scaled_dot_product_attention over the FULL cache behind a band mask, in
    float16, at atol/rtol 0.1 — three differences at once, and one of them is
    a dtype in which a 4096-wide masked softmax and a 64-wide one are no
    longer the same computation.

    This runs that exact comparison with no device involved. A failure here
    exonerates the hardware: it would mean the reference the kernel test
    holds the device to does not agree with the windowed algorithm in float16,
    which is a property of the test, not of the backend.

    The scale is applied as the decomposition applies it -- split across q and
    k as sqrt(sqrt(d)) rather than once after the matmul -- because that split
    is itself a source of float16 difference and the device does it that way.
    """

    ATOL = 0.1
    RTOL = 0.1

    @pytest.mark.parametrize(
        "batch,heads,kvheads,seqlen_q,seqlen_kv,head_dim,window", SHAPES, ids=IDS
    )
    def test_windowed_matches_masked_sdpa_in_float16(
        self, batch, heads, kvheads, seqlen_q, seqlen_kv, head_dim, window
    ):
        plan = plan_window_gather(seqlen_q, seqlen_kv, window)
        assert plan is not None
        query, key, value = _make_inputs(
            batch, heads, kvheads, seqlen_q, seqlen_kv, head_dim
        )
        query, key, value = (t.to(torch.float16) for t in (query, key, value))

        split = 1.0 / math.sqrt(math.sqrt(head_dim))
        windowed = gathered_swa(query * split, key * split, value, plan, 1.0)

        allowed = _reference_allowed(seqlen_q, seqlen_kv, window)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            _additive(allowed, torch.float16),
            enable_gqa=heads != kvheads,
        )

        torch.testing.assert_close(windowed, expected, atol=self.ATOL, rtol=self.RTOL)


class TestWorkReduction:
    """The point of the exercise: fewer KV columns touched per Q block."""

    @pytest.mark.parametrize(
        "batch,heads,kvheads,seqlen_q,seqlen_kv,head_dim,window", SHAPES, ids=IDS
    )
    def test_buffer_is_narrower_than_the_cache(
        self, batch, heads, kvheads, seqlen_q, seqlen_kv, head_dim, window
    ):
        plan = plan_window_gather(seqlen_q, seqlen_kv, window)
        assert plan is not None
        assert plan.buffer_width < seqlen_kv

    def test_decode_touches_only_the_window(self):
        plan = plan_window_gather(1, 4096, 64)
        assert plan is not None
        # 64 of 4096 columns: a 64x reduction in the KV read for one decode step.
        assert plan.buffer_width == 64
