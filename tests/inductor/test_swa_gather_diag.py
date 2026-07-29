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

"""Diagnostic for the two increment-4 failures. Expect some of these to FAIL.

This file is not a correctness suite -- it exists to localize two failures
seen in test_swa_gather_op.py, each with one variable per test:

1. GQA returned ZEROS in the leading slots (MHA passed). The construct unique
   to GQA is cat-ing stride-0 expanded views. Probes C and D compare
   expand-then-cat against cat-then-expand.

2. A compiler crash, not wrong numbers:
       StopIteration in insert_restickify._create_restickify_node
       (cannot find buf5 = k_win [1,32,128,64])
   raised when a matmul consumes the cat result through transpose(-1,-2).
   Probes A, B and E ask whether the trigger is the cat, the transpose, or
   the pair.

How to read it:

  A fails, E passes          -> the cat is the trigger, not the transpose
  A fails, B passes          -> emit K pre-transposed from the gather
  A and E both fail          -> transpose-into-matmul is broken generally,
                                which would be a much larger problem
  C fails, D passes          -> confirms the cat-then-expand fix

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_swa_gather_diag.py -v
"""

import unittest

import torch

from utils_inductor import cached_randn, compare_with_cpu

BATCH = 1
HEADS = 8
KVHEADS = 2
EXPANSION = HEADS // KVHEADS
SEQLEN = 256
HEAD_DIM = 64
Q_BLOCK = 64
BUFFER_WIDTH = 128
# plan_window_gather(256, 256, 64) -- verified in test_swa_window_gather.py
READ_STARTS = [0, 0, 64, 128]
FOLDED = len(READ_STARTS) * HEADS


def _cache(kvheads: int, differentiation: int) -> torch.Tensor:
    return cached_randn(
        (BATCH, kvheads, SEQLEN, HEAD_DIM),
        differentiation=differentiation,
        dtype=torch.float16,
    )


def _query() -> torch.Tensor:
    return cached_randn(
        (BATCH, FOLDED, Q_BLOCK, HEAD_DIM), differentiation=9, dtype=torch.float16
    )


class TestRestickifyTrigger(unittest.TestCase):
    """Failure 2: what makes insert_restickify fail to find the buffer?"""

    def test_a_cat_then_transpose_into_matmul(self):
        # The exact shape that crashed. Expected to FAIL until fixed.
        def fn(q, k):
            windows = torch.cat(
                [k[:, :, s : s + BUFFER_WIDTH, :] for s in READ_STARTS], dim=1
            )
            return torch.matmul(q, windows.transpose(-1, -2))

        compare_with_cpu(fn, _query(), _cache(HEADS, 1), run_eager=False)

    def test_b_cat_of_pretransposed_into_matmul(self):
        # Candidate fix: transpose each slice BEFORE the cat, so no transpose
        # sits between the cat result and the matmul.
        def fn(q, k):
            windows = torch.cat(
                [
                    k[:, :, s : s + BUFFER_WIDTH, :].transpose(-1, -2)
                    for s in READ_STARTS
                ],
                dim=1,
            )
            return torch.matmul(q, windows)

        compare_with_cpu(fn, _query(), _cache(HEADS, 1), run_eager=False)

    def test_e_control_plain_buffer_transpose_into_matmul(self):
        # Control: same transpose-into-matmul on a plain input, no cat. SDPA
        # does exactly this, so a failure here would mean the probe is wrong.
        plain = cached_randn(
            (BATCH, FOLDED, BUFFER_WIDTH, HEAD_DIM),
            differentiation=10,
            dtype=torch.float16,
        )

        def fn(q, k):
            return torch.matmul(q, k.transpose(-1, -2))

        compare_with_cpu(fn, _query(), plain, run_eager=False)


class TestGqaExpandOrder(unittest.TestCase):
    """Failure 1: which side of the cat may the GQA expand sit on?"""

    def test_c_expand_then_cat(self):
        # What increment 4 shipped first; produced zeroed leading slots.
        def fn(k):
            blocks = []
            for s in READ_STARTS:
                window = k[:, :, s : s + BUFFER_WIDTH, :]
                blocks.append(
                    window.unsqueeze(2).expand(-1, -1, EXPANSION, -1, -1).flatten(1, 2)
                )
            return torch.cat(blocks, dim=1)

        compare_with_cpu(fn, _cache(KVHEADS, 2), run_eager=False)

    def test_d_cat_then_expand(self):
        # The applied fix: one expand, over the already-gathered buffer.
        def fn(k):
            windows = torch.cat(
                [k[:, :, s : s + BUFFER_WIDTH, :] for s in READ_STARTS], dim=1
            )
            return windows.unsqueeze(2).expand(-1, -1, EXPANSION, -1, -1).flatten(1, 2)

        compare_with_cpu(fn, _cache(KVHEADS, 2), run_eager=False)

    def test_d_matches_c_ordering_on_cpu(self):
        # Both orders must give the SAME block-major layout, so the fix is a
        # layout-preserving change. Pure CPU -- no device needed.
        key = _cache(KVHEADS, 2)

        expand_then_cat = torch.cat(
            [
                key[:, :, s : s + BUFFER_WIDTH, :]
                .unsqueeze(2)
                .expand(-1, -1, EXPANSION, -1, -1)
                .flatten(1, 2)
                for s in READ_STARTS
            ],
            dim=1,
        )
        windows = torch.cat(
            [key[:, :, s : s + BUFFER_WIDTH, :] for s in READ_STARTS], dim=1
        )
        cat_then_expand = (
            windows.unsqueeze(2).expand(-1, -1, EXPANSION, -1, -1).flatten(1, 2)
        )

        torch.testing.assert_close(cat_then_expand, expand_then_cat)


if __name__ == "__main__":
    unittest.main()
