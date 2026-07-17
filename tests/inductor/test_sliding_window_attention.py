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

"""Correctness baseline for Sliding Window Attention (issue #3073) through the
flash-attention SDPA decomposition (PR #2363).

This does not add any new torch-spyre code — it only exercises the existing
generic `attn_bias` path in `spyre__sdpa_overrideable` with a precomputed
sliding-window band mask, the same way hf-adapters' `add_sliding_window_band`
/ `add_causal_sliding_window_band` build masks host-side before compile.

Run:
    python3 -m pytest tests/inductor/test_sliding_window_attention.py -v
"""

import pytest
import unittest
import torch

from utils_inductor import (
    ParameterizedTestMeta,
    cached_randn,
    compare_with_cpu,
)


def sliding_window_causal_mask(
    seqlen_q: int,
    seqlen_kv: int,
    window_size: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Additive causal+window mask, Gemma-3/Gemma-4 style.

    Query row i is aligned to the last seqlen_q positions of the seqlen_kv
    cache (matching hf-adapters' KV-cache coordinate convention) and may
    attend to key j iff 0 <= i - j < window_size.

    seqlen_q == 1 reproduces the real decode shape hf-adapters uses;
    seqlen_q == seqlen_kv reproduces prefill.
    """
    q_pos = torch.arange(seqlen_kv - seqlen_q, seqlen_kv).unsqueeze(-1)
    k_pos = torch.arange(seqlen_kv).unsqueeze(0)
    delta = q_pos - k_pos
    allowed = (delta >= 0) & (delta < window_size)
    mask = torch.zeros(seqlen_q, seqlen_kv, dtype=dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def sliding_window_bidirectional_mask(
    seqlen: int,
    window_size: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Additive symmetric +/-window_size band mask, ModernBERT-style local attention."""
    idx = torch.arange(seqlen)
    off_band = (idx[:, None] - idx[None, :]).abs() > window_size
    mask = torch.zeros(seqlen, seqlen, dtype=dtype)
    mask[off_band] = -torch.inf
    return mask.unsqueeze(0).unsqueeze(0)


class TestSlidingWindowAttention(unittest.TestCase, metaclass=ParameterizedTestMeta):
    torch.manual_seed(0xAFFE)

    PARAMS = {
        ("test_sdpa_sliding_window", "test_sdpa_sliding_window_cpu"): {
            "param_sets": {
                # MHA, prefill, causal+window, window aligned to the 64-elem tile
                # size the flash-attention decomposition uses internally.
                "mha_prefill_causal_w64": (
                    cached_randn(
                        (2, 256, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 32, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 32, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    sliding_window_causal_mask(256, 256, 64),
                    False,
                    False,
                ),
                # GQA, prefill, causal+window — the Gemma-3 shape (32 q heads / 8 kv heads).
                "gqa_prefill_causal_w64": (
                    cached_randn(
                        (2, 256, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 8, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 8, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    sliding_window_causal_mask(256, 256, 64),
                    False,
                    True,
                ),
                # Same, but window size is NOT a multiple of 64 — stresses the
                # decomposition's tile boundary assumptions independent of the mask shape.
                "gqa_prefill_causal_w100": (
                    cached_randn(
                        (2, 256, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 8, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 8, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    sliding_window_causal_mask(256, 256, 100),
                    False,
                    True,
                ),
                # MHA, decode (Lq=1, Lk=full cache) — the real hf-adapters call shape.
                "mha_decode_causal_w64": (
                    cached_randn(
                        (2, 1, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 257, 32, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 257, 32, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    sliding_window_causal_mask(1, 257, 64),
                    False,
                    False,
                ),
                # GQA, decode — Gemma-3 decode shape.
                "gqa_decode_causal_w64": (
                    cached_randn(
                        (2, 1, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 257, 8, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 257, 8, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    sliding_window_causal_mask(1, 257, 64),
                    False,
                    True,
                ),
                # Bidirectional (non-causal) local window — ModernBERT-style.
                "mha_prefill_bidirectional_w64": (
                    cached_randn(
                        (2, 256, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 32, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 32, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    sliding_window_bidirectional_mask(256, 64),
                    False,
                    False,
                ),
                # Isolation case: MHA decode, NO mask at all (plain is_causal=True,
                # attn_mask=None). Control for gqa_decode_no_mask below — if this
                # passes and that one doesn't, the crash is GQA+decode, not masking.
                "mha_decode_no_mask": (
                    cached_randn(
                        (2, 1, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 257, 32, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 257, 32, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    None,
                    True,
                    False,
                ),
                # Isolation case: GQA decode, NO mask at all (plain is_causal=True,
                # attn_mask=None) — tests whether gqa_decode_causal_w64's SIGABRT is
                # caused by GQA+decode alone, independent of any sliding-window mask.
                "gqa_decode_no_mask": (
                    cached_randn(
                        (2, 1, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 257, 8, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 257, 8, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    None,
                    True,
                    True,
                ),
            },
        },
    }

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_sdpa_sliding_window_cpu(self, q, k, v, attn_mask, is_causal, enable_gqa):
        def fn(q, k, v, attn_mask, is_causal, enable_gqa):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask, is_causal=is_causal, enable_gqa=enable_gqa
            )

        compare_with_cpu(fn, q, k, v, attn_mask, is_causal, enable_gqa)
