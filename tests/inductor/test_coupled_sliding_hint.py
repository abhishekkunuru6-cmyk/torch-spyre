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

"""Unit tests for COUPLED sliding hints — the causal-diagonal cross-dim slide.

A multi-entry ``spyre_hint(sliding={...})`` couples several named dims under
one loop level, each with its own window/stride: sliding-window attention needs
one loop var to partition-slide the Q rows while it overlap-slides the KV
window.  These tests cover the three places that had to learn about coupling:

  1. hint parsing/validation (``_append_sliding_hints``,
     ``_validate_coupled_sliding``)
  2. the mixed output+reduction level gate (``_is_coupled_sliding_level``,
     ``_validate_reduction_tiling``)
  3. full-range reconstruction for a sliding output dim
     (``_compute_full_ranges``)

No Spyre device or backend compiler is required.
"""

import unittest
from unittest.mock import MagicMock

import sympy
from sympy import Integer

import torch_spyre._inductor.propagate_named_dims as pnd
from torch_spyre._inductor.coarse_tile import (
    _compute_full_ranges,
    _is_coupled_sliding_level,
    _validate_reduction_tiling,
)
from torch_spyre._inductor.loop_info import CoarseTileInfo
from torch_spyre._inductor.propagate_named_dims import (
    _append_sliding_hints,
    _validate_coupled_sliding,
)

# Q partition-slides in 64-row blocks; KV overlap-slides a 128-wide window by
# 64.  Both yield 4 tiles, so they can share one loop level.
Q_BLOCK = 64
KV_WINDOW = 128
KV_STRIDE = 64
NUM_TILES = 4
QSEQ = Q_BLOCK * NUM_TILES  # 256
KVSEQ = KV_WINDOW * NUM_TILES  # 512


def _spec(name, window, stride, num_tiles, dim_size):
    """One ``_validate_coupled_sliding`` spec tuple."""
    return (name, window, stride, num_tiles, dim_size)


class TestValidateCoupledSliding(unittest.TestCase):
    """Shape and dim-kind constraints a coupled scope must satisfy."""

    def test_single_dim_scope_is_always_ok(self):
        specs = [_spec("KV", KV_WINDOW, KV_STRIDE, NUM_TILES, KVSEQ)]
        _validate_coupled_sliding(specs, {"KV": sympy.Symbol("c0")}, {"KV"})

    def test_matching_trip_counts_ok(self):
        specs = [
            _spec("QS", Q_BLOCK, Q_BLOCK, NUM_TILES, QSEQ),
            _spec("KV", KV_WINDOW, KV_STRIDE, NUM_TILES, KVSEQ),
        ]
        coords = {"QS": sympy.Symbol("c0"), "KV": sympy.Symbol("r0")}
        _validate_coupled_sliding(specs, coords, {"KV"})

    def test_mismatched_trip_counts_raise(self):
        """One loop level has one trip count — dim_size // window must agree."""
        specs = [
            _spec("QS", Q_BLOCK, Q_BLOCK, 2, 128),
            _spec("KV", KV_WINDOW, KV_STRIDE, NUM_TILES, KVSEQ),
        ]
        coords = {"QS": sympy.Symbol("c0"), "KV": sympy.Symbol("r0")}
        with self.assertRaises(ValueError) as cm:
            _validate_coupled_sliding(specs, coords, {"KV"})
        self.assertIn("share a trip count", str(cm.exception))

    def test_two_output_dims_raise(self):
        """coarse_tile resolves one output-range position per hint_id."""
        specs = [
            _spec("QS", Q_BLOCK, Q_BLOCK, NUM_TILES, QSEQ),
            _spec("KV", KV_WINDOW, KV_STRIDE, NUM_TILES, KVSEQ),
        ]
        coords = {"QS": sympy.Symbol("c0"), "KV": sympy.Symbol("c1")}
        with self.assertRaises(NotImplementedError) as cm:
            _validate_coupled_sliding(specs, coords, set())
        self.assertIn("output dims", str(cm.exception))

    def test_two_reduction_dims_raise(self):
        specs = [
            _spec("QS", Q_BLOCK, Q_BLOCK, NUM_TILES, QSEQ),
            _spec("KV", KV_WINDOW, KV_STRIDE, NUM_TILES, KVSEQ),
        ]
        coords = {"QS": sympy.Symbol("r0"), "KV": sympy.Symbol("r1")}
        with self.assertRaises(NotImplementedError) as cm:
            _validate_coupled_sliding(specs, coords, {"QS", "KV"})
        self.assertIn("reduction dims", str(cm.exception))

    def test_unresolved_dim_does_not_clash(self):
        """A dim this op does not iterate (loop_var None) cannot collide."""
        specs = [
            _spec("QS", Q_BLOCK, Q_BLOCK, NUM_TILES, QSEQ),
            _spec("KV", KV_WINDOW, KV_STRIDE, NUM_TILES, KVSEQ),
        ]
        _validate_coupled_sliding(specs, {"QS": sympy.Symbol("c0")}, set())


class TestAppendSlidingHints(unittest.TestCase):
    """``sliding={...}`` -> DimHints, one per named dim, sharing the hint_id."""

    def setUp(self):
        pnd.reset()
        pnd.declare_tensor_dim("QS", QSEQ)
        pnd.declare_tensor_dim("KV", KVSEQ)

    def tearDown(self):
        pnd.reset()

    def _coupled_hints(self):
        dim_hints = []
        coords = {"QS": sympy.Symbol("c0"), "KV": sympy.Symbol("r0")}
        _append_sliding_hints(
            {
                "QS": {"window": Q_BLOCK, "stride": Q_BLOCK},
                "KV": {"window": KV_WINDOW, "stride": KV_STRIDE},
            },
            hint_id=7,
            coord_for_name=coords,
            reduction_dims={"KV"},
            dim_hints=dim_hints,
        )
        return {h.dim_names[0]: h for h in dim_hints}

    def test_one_hint_per_dim_sharing_hint_id(self):
        """hint_id is the loop level, so coupled dims must share it."""
        hints = self._coupled_hints()
        self.assertEqual(set(hints), {"QS", "KV"})
        self.assertEqual({h.hint_id for h in hints.values()}, {7})

    def test_per_dim_window_and_stride(self):
        """Each dim keeps its OWN extent/stride — Q partitions, KV overlaps."""
        hints = self._coupled_hints()
        self.assertEqual(
            (hints["QS"].read_extent, hints["QS"].slide_stride), (Q_BLOCK, Q_BLOCK)
        )
        self.assertEqual(
            (hints["KV"].read_extent, hints["KV"].slide_stride),
            (KV_WINDOW, KV_STRIDE),
        )

    def test_shared_split_count_and_dim_kinds(self):
        hints = self._coupled_hints()
        self.assertEqual(hints["QS"].split_count, NUM_TILES)
        self.assertEqual(hints["KV"].split_count, NUM_TILES)
        self.assertFalse(hints["QS"].is_reduction)
        self.assertTrue(hints["KV"].is_reduction)

    def test_stride_past_end_raises(self):
        """A gap-read whose last window runs off the end is an OOB read."""
        dim_hints = []
        with self.assertRaises(ValueError) as cm:
            _append_sliding_hints(
                {"KV": {"window": KV_WINDOW, "stride": KV_WINDOW * 2}},
                hint_id=0,
                coord_for_name={"KV": sympy.Symbol("r0")},
                reduction_dims={"KV"},
                dim_hints=dim_hints,
            )
        self.assertIn("past dim_size", str(cm.exception))

    def test_undeclared_dim_raises(self):
        dim_hints = []
        with self.assertRaises(ValueError):
            _append_sliding_hints(
                {"NOPE": {"window": 64, "stride": 64}},
                hint_id=0,
                coord_for_name={},
                reduction_dims=set(),
                dim_hints=dim_hints,
            )


def _reduction_op(loop_info):
    """A minimal Reduction ComputedBuffer stand-in carrying ``loop_info``."""
    from torch._inductor.ir import ComputedBuffer, Reduction

    data = MagicMock(spec=Reduction)
    data.ranges = [Integer(Q_BLOCK)]
    data.reduction_ranges = [Integer(KV_WINDOW)]
    data.reduction_type = "sum"
    op = MagicMock(spec=ComputedBuffer)
    op.data = data
    op.get_name.return_value = "coupled_op"
    op.loop_info = loop_info
    return op


def _coupled_loop_info():
    """One level tiling output dim 0 and reduction dim 0, both sliding."""
    return CoarseTileInfo(
        loop_group_id=(0,),
        loop_count=[Integer(NUM_TILES)],
        loop_tiled_dims=[[0]],
        loop_tiled_reduction_dims=[[0]],
        loop_slide_stride=[Q_BLOCK],
        loop_read_extent=[Q_BLOCK],
        loop_reduction_slide_stride=[KV_STRIDE],
        loop_reduction_read_extent=[KV_WINDOW],
    )


class TestIsCoupledSlidingLevel(unittest.TestCase):
    """A level is coupled only when BOTH kinds carry sliding params."""

    def test_coupled_level_detected(self):
        self.assertTrue(_is_coupled_sliding_level(_coupled_loop_info(), 0))

    def test_reduction_only_slide_is_not_coupled(self):
        info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(NUM_TILES)],
            loop_tiled_dims=[[]],
            loop_tiled_reduction_dims=[[0]],
            loop_reduction_slide_stride=[KV_STRIDE],
            loop_reduction_read_extent=[KV_WINDOW],
        )
        self.assertFalse(_is_coupled_sliding_level(info, 0))

    def test_partition_level_is_not_coupled(self):
        info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(NUM_TILES)],
            loop_tiled_dims=[[0]],
            loop_tiled_reduction_dims=[[0]],
        )
        self.assertFalse(_is_coupled_sliding_level(info, 0))

    def test_no_loop_info(self):
        self.assertFalse(_is_coupled_sliding_level(None, 0))


class TestValidateReductionTilingCoupled(unittest.TestCase):
    """Mixed output+reduction at one level: allowed only when coupled."""

    def test_coupled_mixed_level_allowed(self):
        _validate_reduction_tiling(_reduction_op(_coupled_loop_info()))

    def test_non_sliding_mixed_level_still_raises(self):
        info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(NUM_TILES)],
            loop_tiled_dims=[[0]],
            loop_tiled_reduction_dims=[[0]],
        )
        with self.assertRaises(RuntimeError) as cm:
            _validate_reduction_tiling(_reduction_op(info))
        self.assertIn("mixed output+reduction", str(cm.exception))


class TestComputeFullRangesSlidingOutput(unittest.TestCase):
    """Full-range reconstruction assumes the output dim's tiles partition."""

    def test_partitioning_output_slide_reconstructs_full_dim(self):
        full = _compute_full_ranges(_reduction_op(_coupled_loop_info()))
        self.assertEqual(int(full[0]), QSEQ)

    def test_overlapping_output_slide_raises(self):
        """Overlapping output tiles span less than tile*count; reject, don't mis-size."""
        info = _coupled_loop_info()
        info.loop_slide_stride = [Q_BLOCK // 2]
        with self.assertRaises(NotImplementedError) as cm:
            _compute_full_ranges(_reduction_op(info))
        self.assertIn("OUTPUT dim", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
