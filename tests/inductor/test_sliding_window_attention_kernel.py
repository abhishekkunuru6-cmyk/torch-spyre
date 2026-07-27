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

"""Correctness test for the windowed spyre::sliding_window_attention decomposition
(issue #3073), against the SDPA+band-mask reference already used in
tests/inductor/test_sliding_window_attention.py.

GQA+decode was previously excluded here — that combination hit the
pre-existing dxp_standalone crash in issue #3152 (confirmed independent of
sliding-window masking; the no-mask isolation case in the sibling test file
failed identically), unrelated to this decomposition. #3152 was fixed on
main by PR #3184 (spyre__sdpa_overrideable output layout fix); the GQA+decode
case is re-included below to confirm this op is unaffected either way.

Run:
    python3 -m pytest tests/inductor/test_sliding_window_attention_kernel.py -v
"""

import pytest
import unittest
import torch
import torch.nn.functional as F

from utils_inductor import (
    ParameterizedTestMeta,
    cached_randn,
    compare_with_cpu,
)


def _causal_window_mask(
    seqlen_q: int,
    seqlen_kv: int,
    window_size: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reference mask, mirrors sliding_window_block_mask's causal branch."""
    q_pos = torch.arange(seqlen_kv - seqlen_q, seqlen_kv).unsqueeze(-1)
    k_pos = torch.arange(seqlen_kv).unsqueeze(0)
    delta = q_pos - k_pos
    allowed = (delta >= 0) & (delta < window_size)
    mask = torch.zeros(seqlen_q, seqlen_kv, dtype=dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def _bidirectional_window_mask(
    seqlen: int,
    window_size: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reference mask, mirrors sliding_window_block_mask's bidirectional branch."""
    idx = torch.arange(seqlen)
    off_band = (idx[:, None] - idx[None, :]).abs() >= window_size
    mask = torch.zeros(seqlen, seqlen, dtype=dtype)
    mask[off_band] = -torch.inf
    return mask.unsqueeze(0).unsqueeze(0)


class TestSlidingWindowAttentionKernel(
    unittest.TestCase, metaclass=ParameterizedTestMeta
):
    torch.manual_seed(0xAFFE)

    PARAMS = {
        ("test_sliding_window_kernel", "test_sliding_window_kernel_cpu"): {
            "param_sets": {
                # MHA, prefill, causal+window, window aligned to the 64-elem
                # block size the decomposition unrolls with.
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
                    64,
                    True,
                ),
                # GQA, prefill, causal+window — Gemma-3 shape (32 q heads / 8 kv heads).
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
                    64,
                    True,
                ),
                # Window size NOT a multiple of 64 — stresses the block-range
                # rounding independent of the mask shape.
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
                    100,
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
                    64,
                    True,
                ),
                # GQA, decode — Gemma-3 decode shape. #3152 itself is fixed on
                # main (PR #3184), but this op hits a *different* dxp_standalone
                # DDL dimension-mapper crash on this exact shape ("Could not
                # find any suitable dimension mapping", ddl_conversion.cpp:2497
                # — same empty-dimCandidates signature as #3152's diagnostic,
                # different root cause). Not fixable from the Python
                # decomposition side; see expect_fail below.
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
                    64,
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
                    64,
                    False,
                ),
                # MHA, prefill, LONG sequence (2048 vs. 256 elsewhere) — 32
                # blocks instead of 4, each touching a much smaller fraction
                # of the KV range. Added to check correctness before
                # benchmarking at a scale closer to real usage (the toy
                # 256-length shapes above showed the windowed op ~3x SLOWER
                # than the mask-based baseline; this shape tests whether a
                # bigger per-block compute saving can outrun the per-block
                # dispatch overhead that theory blames for that result).
                "mha_prefill_causal_w64_long": (
                    cached_randn(
                        (2, 2048, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 2048, 32, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 2048, 32, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    64,
                    True,
                ),
                # MHA, decode, LONG KV cache (2048 vs. 257 elsewhere) — the
                # realistic inference shape. Always exactly 1 Q-block
                # regardless of cache length, so this isolates the compute-skip
                # win from the per-block overhead concern above entirely.
                "mha_decode_causal_w64_long": (
                    cached_randn(
                        (2, 1, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 2048, 32, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 2048, 32, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    64,
                    True,
                ),
                # Pushing further to trace whether the windowed op's speedup
                # (1.65x at 2048) keeps growing or plateaus with sequence
                # length. Prefill counterparts (mha_prefill_causal_w64_4096/8192)
                # dropped — full O(L^2) SDPA CPU reference at these lengths made
                # the test prohibitively slow; decode is always a single
                # Q-block regardless of cache length, so it stays cheap.
                "mha_decode_causal_w64_4096": (
                    cached_randn(
                        (2, 1, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 4096, 32, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 4096, 32, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    64,
                    True,
                ),
                "mha_decode_causal_w64_8192": (
                    cached_randn(
                        (2, 1, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 8192, 32, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 8192, 32, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    64,
                    True,
                ),
            },
            "expect_fail": ["gqa_decode_causal_w64"],
        },
    }

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_sliding_window_kernel_cpu(self, q, k, v, window_size, is_causal):
        enable_gqa = q.size(1) != k.size(1)

        def fn(q, k, v, window_size, is_causal, enable_gqa):
            if q.device.type == "spyre":
                return torch.ops.spyre.sliding_window_attention(
                    q, k, v, window_size, is_causal
                )
            seqlen_q, seqlen_kv = q.size(2), k.size(2)
            mask = (
                _causal_window_mask(seqlen_q, seqlen_kv, window_size)
                if is_causal
                else _bidirectional_window_mask(seqlen_q, window_size)
            )
            return F.scaled_dot_product_attention(q, k, v, mask, enable_gqa=enable_gqa)

        # spyre::sliding_window_attention has an intentionally empty eager
        # body (matches spyre::exx2 / spyre::layernormscale / the documented
        # dequantize_fp8_with_scale pattern) — the real logic only exists in
        # the register_spyre_decomposition Inductor lowering, so this op
        # only works under torch.compile.
        compare_with_cpu(
            fn, q, k, v, window_size, is_causal, enable_gqa, run_eager=False
        )
