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

"""End-to-end correctness of the rolled-window SWA path (increment R4).

Runs spyre::sliding_window_attention with config.swa_window_roll ON, so the op
takes _window_roll_attention -- one [B, Hq, Wb, E] KV window buffer per Q
block, a Python loop over the blocks, and the flash body tiled over
window_size instead of max_seqlen_kv. Compared against SDPA with a full band
mask, which is the definition of sliding-window attention.

Note the two things this file does NOT establish:

  - whether the four spyre_hints actually produce device loops. Untiled code
    returns the right answer with one large intermediate, so every test here
    would still pass if the hints silently no-op'd -- which has happened
    before on this op. That is increment R6, and it has to be checked
    structurally.
  - whether the window buffer is actually reused across blocks. The blocks are
    independent, so nothing here forces peak memory to be one buffer rather
    than N; getting that is the entire reason rolling was chosen over
    gathering all windows at once. That is increment R5.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_swa_window_roll_kernel.py -v
"""

import unittest

import torch
import torch._dynamo
import torch.nn.functional as F

from torch_spyre._inductor import config as spyre_config
from utils_inductor import cached_randn, compare_with_cpu

HEAD_DIM = 64


def _band_mask(
    seqlen_q: int, seqlen_kv: int, window_size: int, dtype=torch.float16
) -> torch.Tensor:
    """Full [1, 1, Lq, Lkv] causal sliding-window mask -- the definition."""
    q_pos = torch.arange(seqlen_kv - seqlen_q, seqlen_kv).unsqueeze(-1)
    k_pos = torch.arange(seqlen_kv).unsqueeze(0)
    delta = q_pos - k_pos
    allowed = (delta >= 0) & (delta < window_size)
    mask = torch.zeros(seqlen_q, seqlen_kv, dtype=dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def _attention(q, k, v, window_size):
    """Dispatch: the op on spyre, the masked reference on CPU."""
    if q.device.type == "spyre":
        return torch.ops.spyre.sliding_window_attention(q, k, v, window_size, True)
    mask = _band_mask(q.size(2), k.size(2), window_size)
    return F.scaled_dot_product_attention(
        q, k, v, mask, enable_gqa=q.size(1) != k.size(1)
    )


def _inputs(batch, heads, kvheads, seqlen_q, seqlen_kv):
    query = cached_randn(
        (batch, heads, seqlen_q, HEAD_DIM), differentiation=1, dtype=torch.float16
    )
    key = cached_randn(
        (batch, kvheads, seqlen_kv, HEAD_DIM), differentiation=2, dtype=torch.float16
    )
    value = cached_randn(
        (batch, kvheads, seqlen_kv, HEAD_DIM), differentiation=3, dtype=torch.float16
    )
    return query, key, value


class _RollPathEnabled(unittest.TestCase):
    """Turns the rolled-window path on for the duration of each test."""

    roll_enabled = True

    def setUp(self):
        self._saved = spyre_config.swa_window_roll
        spyre_config.swa_window_roll = self.roll_enabled
        torch._dynamo.reset()

    def tearDown(self):
        spyre_config.swa_window_roll = self._saved
        torch._dynamo.reset()


class TestWindowRollKernel(_RollPathEnabled):
    """Shapes the roll path accepts, against the masked reference."""

    def test_prefill_mha(self):
        # Lq=Lkv=256, W=64 -> q_block=64, 4 blocks, buffer_width=128.
        query, key, value = _inputs(1, 8, 8, 256, 256)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_prefill_mha_wider_window(self):
        # W=128 -> buffer_width=192; the band covers more of each buffer.
        query, key, value = _inputs(1, 8, 8, 256, 256)
        compare_with_cpu(_attention, query, key, value, 128, run_eager=False)

    def test_prefill_gqa(self):
        # 8 query heads from 2 kv heads; the expand happens inside the gather.
        query, key, value = _inputs(1, 8, 2, 256, 256)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_prefill_batch(self):
        query, key, value = _inputs(2, 4, 4, 256, 256)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_decode(self):
        # Lq=1 -> q_block=1, one block, buffer_width == window exactly, and a
        # 64-of-4096 KV read instead of the whole cache.
        query, key, value = _inputs(1, 8, 8, 1, 4096)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_decode_gqa(self):
        query, key, value = _inputs(1, 8, 2, 1, 512)
        compare_with_cpu(_attention, query, key, value, 128, run_eager=False)

    def test_chunked_prefill(self):
        # Lq < Lkv with an aligned offset -- prefill continuing a warm cache.
        query, key, value = _inputs(1, 8, 8, 128, 512)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)


class TestFallbacksStillCorrect(_RollPathEnabled):
    """Shapes the planner rejects must silently keep the unrolled path."""

    def test_window_not_stick_multiple(self):
        # plan_window_gather returns None for window % 64 != 0.
        query, key, value = _inputs(1, 8, 8, 256, 256)
        compare_with_cpu(_attention, query, key, value, 100, run_eager=False)

    def test_query_length_not_stick_multiple(self):
        # 100 % 64 != 0, so choose_q_block returns None and the unrolled loop
        # handles it.
        query, key, value = _inputs(1, 8, 8, 100, 256)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)


class TestDefaultPathUnchanged(_RollPathEnabled):
    """With the flag OFF nothing may change -- this is the regression guard."""

    roll_enabled = False

    def test_prefill_mha(self):
        query, key, value = _inputs(1, 8, 8, 256, 256)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)

    def test_decode(self):
        query, key, value = _inputs(1, 8, 8, 1, 4096)
        compare_with_cpu(_attention, query, key, value, 64, run_eager=False)


if __name__ == "__main__":
    unittest.main()
