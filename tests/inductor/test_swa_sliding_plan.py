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

"""Unit tests for the sliding-window-attention plan and its band mask.

Two things are being pinned down, both without a device:

  1. ``plan_sliding_window`` accepts exactly the shapes whose per-block KV range
     is affine in the block index, and returns ``None`` — never raises — for
     everything else, so unsupported shapes fall back to the unrolled loop.
  2. ``spyre.sliding_window_band_mask`` keeps exactly the positions the op's
     per-block masks keep, and masks every padding column.  That equivalence is
     what licenses reading a constant-width window at a negative origin.

No Spyre device or backend compiler is required.
"""

import unittest

import torch

from torch_spyre._inductor.swa_sliding import (
    STICK,
    build_band_mask_cpu,
    plan_sliding_window,
    unclamped_kv_range,
)

# The shape increment 5a used throughout: 8 Q blocks, a 192-wide window that
# neither divides 512 nor yields 8 tiles.
PREFILL = dict(
    batch_size=1, seqlen_q=512, seqlen_kv=512, window_size=128, is_causal=True
)


def _band_mask(plan, dtype=torch.float32):
    """The mask the spyre::sliding_window_band_mask op wraps, built on CPU."""
    return build_band_mask_cpu(
        plan_seqlen_q(plan),
        plan_seqlen_kv(plan),
        plan.left_pad,
        plan.right_pad,
        plan.q_kv_offset,
        plan.window_size,
        plan.is_causal,
        dtype,
    )


def plan_seqlen_q(plan) -> int:
    return plan.num_q_blocks * plan.q_block


def plan_seqlen_kv(plan) -> int:
    return plan.padded_kv - plan.left_pad - plan.right_pad


class TestPlanAcceptance(unittest.TestCase):
    """Which shapes get the sliding path, and which fall back."""

    def test_prefill_shape_planned(self):
        plan = plan_sliding_window(**PREFILL)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.num_q_blocks, 8)
        self.assertEqual(plan.read_extent, 192)
        self.assertEqual(plan.base_offset, -128)
        self.assertEqual(plan.left_pad, 128)
        self.assertEqual(plan.right_pad, 0)
        self.assertEqual(plan.padded_kv, 640)

    def test_batch_gt_one_falls_back(self):
        """Known-bad above batch 1 (plan section 6.1) — fall back, don't guess."""
        self.assertIsNone(plan_sliding_window(**{**PREFILL, "batch_size": 2}))

    def test_partial_last_q_block_falls_back(self):
        """A short final block would read a different width — not a constant window."""
        self.assertIsNone(plan_sliding_window(**{**PREFILL, "seqlen_q": 512 - 32}))

    def test_unpadded_shape_falls_back(self):
        """Lq=64, Lkv=512: the window sits inside the cache, so no padding.

        Declined because there is then no concatenate op to carry V's
        named_dims hint — see plan_sliding_window.  The unrolled path handles it.
        """
        self.assertIsNone(
            plan_sliding_window(
                batch_size=1,
                seqlen_q=64,
                seqlen_kv=512,
                window_size=128,
                is_causal=True,
            )
        )

    def test_non_causal_needs_right_pad(self):
        """The bidirectional band reaches forward, past the sequence end."""
        plan = plan_sliding_window(**{**PREFILL, "is_causal": False})
        self.assertIsNotNone(plan)
        self.assertGreater(plan.right_pad, 0)
        self.assertGreater(plan.read_extent, 192)  # wider than the causal band

    def test_kv_shorter_than_q_falls_back(self):
        self.assertIsNone(plan_sliding_window(**{**PREFILL, "seqlen_kv": 256}))

    def test_degenerate_sizes_fall_back(self):
        for bad in ({"window_size": 0}, {"seqlen_q": 0}, {"seqlen_kv": 0}):
            with self.subTest(**bad):
                self.assertIsNone(plan_sliding_window(**{**PREFILL, **bad}))


class TestPlanGeometry(unittest.TestCase):
    """The window really is affine, and really stays in bounds."""

    def test_every_block_matches_the_unclamped_range(self):
        plan = plan_sliding_window(**PREFILL)
        for qi in range(plan.num_q_blocks):
            lo, hi = unclamped_kv_range(
                qi,
                plan.q_block,
                plan_seqlen_q(plan),
                plan.q_kv_offset,
                plan.window_size,
                plan.is_causal,
            )
            self.assertEqual(lo, plan.window_lo(qi))
            self.assertEqual(hi - lo, plan.read_extent)

    def test_padded_reads_stay_in_bounds(self):
        for kwargs in (PREFILL, {**PREFILL, "is_causal": False}):
            plan = plan_sliding_window(**kwargs)
            with self.subTest(causal=kwargs["is_causal"]):
                for qi in range(plan.num_q_blocks):
                    lo = plan.padded_window_lo(qi)
                    self.assertGreaterEqual(lo, 0)
                    self.assertLessEqual(lo + plan.read_extent, plan.padded_kv)

    def test_padding_is_stick_aligned(self):
        plan = plan_sliding_window(**PREFILL)
        self.assertEqual(plan.left_pad % STICK, 0)
        self.assertEqual(plan.right_pad % STICK, 0)


class TestBandMask(unittest.TestCase):
    """The band mask keeps the right positions and always kills the padding."""

    def test_shape_and_padding_masked(self):
        plan = plan_sliding_window(**PREFILL)
        mask = _band_mask(plan)
        self.assertEqual(tuple(mask.shape), (1, 1, 512, 640))
        # Every padding column is masked for every query row — the property the
        # constant-width read depends on.
        self.assertTrue(torch.isinf(mask[0, 0, :, : plan.left_pad]).all())

    def test_matches_the_causal_window_rule(self):
        plan = plan_sliding_window(**PREFILL)
        mask = _band_mask(plan)[0, 0]
        seqlen_kv = plan_seqlen_kv(plan)
        for row in (0, 1, 63, 64, 200, 511):
            q_abs = row + plan.q_kv_offset
            for col in range(plan.padded_kv):
                kv_abs = col - plan.left_pad
                expected_keep = (
                    0 <= kv_abs < seqlen_kv and 0 <= q_abs - kv_abs < plan.window_size
                )
                kept = mask[row, col].item() == 0.0
                self.assertEqual(
                    kept,
                    expected_keep,
                    f"row {row} col {col} (kv {kv_abs})",
                )

    def test_each_block_window_covers_its_kept_columns(self):
        """Every position the mask keeps lies inside that block's window.

        If a kept column fell outside the window the block actually reads, the
        constant-width read would silently drop real keys.
        """
        plan = plan_sliding_window(**PREFILL)
        mask = _band_mask(plan)[0, 0]
        for qi in range(plan.num_q_blocks):
            lo = plan.padded_window_lo(qi)
            hi = lo + plan.read_extent
            block = mask[qi * plan.q_block : (qi + 1) * plan.q_block]
            outside = torch.cat([block[:, :lo], block[:, hi:]], dim=1)
            self.assertTrue(
                torch.isinf(outside).all(),
                f"block {qi} keeps a column outside [{lo}, {hi})",
            )

    def test_non_causal_band_is_symmetric(self):
        plan = plan_sliding_window(**{**PREFILL, "is_causal": False})
        mask = _band_mask(plan)[0, 0]
        row = 200
        q_abs = row + plan.q_kv_offset
        for delta in (-plan.window_size, -1, 0, 1, plan.window_size):
            col = q_abs - delta + plan.left_pad
            if 0 <= col < plan.padded_kv:
                kept = mask[row, col].item() == 0.0
                self.assertEqual(kept, abs(delta) < plan.window_size, f"delta {delta}")


class TestDispatchGating(unittest.TestCase):
    """The decomposition takes the sliding path only when it is meant to."""

    def test_flag_defaults_off(self):
        """Opt-in until validated on hardware; the unrolled path works today."""
        from torch_spyre._inductor import config

        self.assertFalse(config.swa_sliding_loop)

    def test_plan_is_the_gate(self):
        """With the flag on, the plan still decides — batch>1 must not slide."""
        self.assertIsNotNone(plan_sliding_window(**PREFILL))
        self.assertIsNone(plan_sliding_window(**{**PREFILL, "batch_size": 4}))

    def test_pad_parts_concatenate_to_the_planned_length(self):
        from torch_spyre._inductor.decompositions import _kv_pad_parts

        plan = plan_sliding_window(**PREFILL)
        kv = torch.zeros(1, 2, plan_seqlen_kv(plan), 8)
        parts = _kv_pad_parts(kv, plan)
        joined = torch.cat(parts, dim=-2)
        # Padding goes on the SEQUENCE axis, not the head axis.
        self.assertEqual(joined.shape[-2], plan.padded_kv)
        self.assertEqual(joined.shape[1], 2)

    def test_pad_parts_keeps_the_real_tensor_at_the_planned_offset(self):
        """The window offsets assume the real K/V starts at left_pad."""
        from torch_spyre._inductor.decompositions import _kv_pad_parts

        plan = plan_sliding_window(**PREFILL)
        kv = torch.ones(1, 2, plan_seqlen_kv(plan), 8)
        joined = torch.cat(_kv_pad_parts(kv, plan), dim=-2)
        self.assertTrue((joined[..., : plan.left_pad, :] == 0).all())
        self.assertTrue((joined[..., plan.left_pad :, :] == 1).all())


if __name__ == "__main__":
    unittest.main()
