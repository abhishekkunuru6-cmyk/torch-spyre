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

"""Coarse-tiling loop metadata attached to ir.Operation objects.

``CoarseTileInfo`` is stamped onto ``ComputedBuffer`` ops by ``coarse_tile()``
and consumed by the scheduler, kernel codegen, and buffer-propagation pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sympy
    from torch._inductor.ir import ComputedBuffer


@dataclass
class CoarseTileInfo:
    """Loop metadata stamped on a ``ComputedBuffer`` by the coarse-tiling pass.

    Attributes
    ----------
    loop_group_id:
        Tuple encoding the nesting path, e.g. ``(0,)`` for an outermost
        group, ``(0, 0)`` for a nested group inside group 0.
    loop_count:
        List of trip counts, one per nesting level from outermost to
        innermost.  ``len(loop_count) == len(loop_group_id)`` always holds.
    loop_tiled_dims:
        List of lists, one sub-list per nesting level.  Each sub-list
        contains the ``data.ranges`` positional indices that are tiled at
        that level.  An empty sub-list means the op is loop-invariant at
        that level.
    loop_tiled_reduction_dims:
        List of lists, one sub-list per nesting level.  Each sub-list
        contains the ``data.reduction_ranges`` positional indices that are
        tiled at that level.  An empty sub-list means no reduction dim is
        tiled at that level.  Parallel to ``loop_tiled_dims``.
    loop_slide_stride:
        Sliding-window tiling only.  List (one entry per nesting level) of the
        per-iteration base advance (block) for that level's tiled OUTPUT dim,
        or ``None`` for ordinary partition tiling.  When set, the loop advances
        a tiled input's base by this stride each iteration while it reads
        ``loop_read_extent`` elements, so consecutive reads OVERLAP when
        ``read_extent > slide_stride``.  Empty list => all levels partition.
    loop_read_extent:
        Parallel to ``loop_slide_stride``: the per-iteration read width
        (window) for a sliding level's output dim, or ``None`` for partition
        tiling.
    loop_reduction_slide_stride:
        As ``loop_slide_stride``, but for the level's tiled REDUCTION dim.
        Kept separate because one level can slide an output dim and a reduction
        dim with DIFFERENT window/stride — the coupled causal-diagonal shape
        (Q partition-slides while KV overlap-slides under the same loop var).
    loop_reduction_read_extent:
        Parallel to ``loop_reduction_slide_stride``: the reduction dim's
        per-iteration read width.
    """

    loop_group_id: tuple[int, ...]
    loop_count: list[sympy.Expr]
    loop_tiled_dims: list[list[int]]
    loop_tiled_reduction_dims: list[list[int]] = field(default_factory=list)
    loop_slide_stride: list[int | None] = field(default_factory=list)
    loop_read_extent: list[int | None] = field(default_factory=list)
    loop_reduction_slide_stride: list[int | None] = field(default_factory=list)
    loop_reduction_read_extent: list[int | None] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Op-metadata helpers
# ---------------------------------------------------------------------------

_SPYRE_METADATA_ATTRS = (
    "dim_hints",
    "work_div_loop_info",
    "loop_info",
    "_restickify_plan",
    "_input_layout_overrides",
    "_emit_set_layout",
)


def copy_op_metadata(src: "ComputedBuffer", dst: "ComputedBuffer") -> None:
    """Copy all Spyre pass metadata from src to dst.

    Call this whenever a pass reconstructs a ComputedBuffer to ensure
    dim_hints, work-division hint metadata, and coarse-tiling attrs are not
    silently dropped.
    """
    for attr in _SPYRE_METADATA_ATTRS:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
