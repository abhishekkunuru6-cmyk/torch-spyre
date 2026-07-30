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

"""Placement arithmetic for sliding-window attention's compact KV buffer.

Sliding-window attention only ever reads a narrow band of the KV cache, so
instead of attending against the whole [B, H, Lkv, D] cache behind a mask, each
Q block gathers just its band into a compact [B, H, buffer_width, D] buffer and
runs the attention body against that.

This module decides *where* that buffer is read from. It is pure integer
arithmetic — no torch, no device — so the window placement can be tested
exhaustively without hardware.

The one invariant everything downstream rests on:

    for every query row, that row's window (intersected with the cache) lies
    entirely inside the buffer its Q block reads.

The arithmetic here is independent of how the windows are *consumed*: it held
for the abandoned all-at-once gather and holds unchanged now that one buffer is
rolled across blocks.

See format/swa_window_roll_plan.md for the design and the increment ladder.
"""

from dataclasses import dataclass

# Elements per 128-byte stick at fp16. Q blocks and KV reads are kept to stick
# boundaries so every gather is a whole number of sticks, matching the block
# size the unrolled decomposition already uses.
STICK = 64


def _floor_stick(value: int) -> int:
    """Round down to a stick boundary, for negative values too."""
    return (value // STICK) * STICK


def _ceil_stick(value: int) -> int:
    """Round up to a stick boundary."""
    return -(-value // STICK) * STICK


@dataclass(frozen=True)
class WindowGatherPlan:
    """Where each Q block reads its compact KV window from.

    All coordinates are absolute positions in the KV cache. Query row ``i``
    sits at cache coordinate ``q_kv_offset + i``, matching hf-adapters'
    convention (add_causal_sliding_window_band in hf_common.py), so prefill
    and decode fall out of the same formula.
    """

    seqlen_q: int
    seqlen_kv: int
    window_size: int
    q_block: int
    num_q_blocks: int
    buffer_width: int
    q_kv_offset: int
    is_causal: bool

    def block_q_range(self, qi: int) -> tuple[int, int]:
        """Half-open query-row range ``[start, end)`` of Q block ``qi``."""
        q_start = qi * self.q_block
        return q_start, min(self.seqlen_q, q_start + self.q_block)

    def row_window(self, q_index: int) -> tuple[int, int]:
        """Half-open cache range query row ``q_index`` may attend to."""
        coord = self.q_kv_offset + q_index
        lo = max(0, coord - self.window_size + 1)
        hi = min(self.seqlen_kv, coord + 1)
        return lo, hi

    def block_is_fully_attended(self, qi: int) -> bool:
        """True when block ``qi``'s band masks nothing.

        Every row of the block can attend every column of its buffer, so the
        additive band is all zeros and adding it is a no-op. Decode is always
        this case: one row, ``buffer_width == window_size``, and a read start
        that lands exactly on the row's own window.

        Worth knowing rather than adding zeros, and worth knowing *in Python*:
        the band is a tensor by the time the body sees it, so a caller cannot
        ask this of the tensor without materializing and reducing it.
        """
        start = self.read_start(qi)
        stop = start + self.buffer_width
        q_start, q_end = self.block_q_range(qi)
        return all(
            self.row_window(q_index)[0] <= start and self.row_window(q_index)[1] >= stop
            for q_index in range(q_start, q_end)
        )

    def read_start(self, qi: int) -> int:
        """Cache offset Q block ``qi``'s buffer is gathered from.

        Stick-aligned, and clamped so the whole ``buffer_width`` span stays
        inside the cache — the ragged first and last blocks shift their read
        rather than reading out of bounds, which keeps the buffer shape
        constant for every block. Columns the shift brings in that the row
        cannot actually attend to are removed by the band mask.
        """
        q_start, _ = self.block_q_range(qi)
        first_coord = self.q_kv_offset + q_start
        window_origin = max(0, _floor_stick(first_coord - self.window_size + 1))
        return min(window_origin, self.seqlen_kv - self.buffer_width)


def check_window_read(
    read_start: int,
    buffer_width: int,
    seqlen_kv: int,
    q_block: int,
    window_size: int,
    num_heads: int,
    num_kv_heads: int,
) -> str | None:
    """Validate one window read, returning the reason it is invalid or None.

    The gather op takes its placement as plain integers, so a caller that
    computes them itself — rather than through ``plan_window_gather`` — can
    hand over a read that runs off the end of the cache or a GQA ratio that
    does not divide. Both would otherwise surface as a shape mismatch deep in
    the lowering, so they are caught at the op boundary instead.

    Returns a string rather than raising so this module stays free of both
    torch and the backend's error classes; the caller raises ``Unsupported``.
    """
    if read_start < 0:
        return f"read_start={read_start} is negative"
    if buffer_width <= 0:
        return f"buffer_width={buffer_width} must be positive"
    if read_start + buffer_width > seqlen_kv:
        return (
            f"window [{read_start}, {read_start + buffer_width}) runs past the "
            f"cache (seqlen_kv={seqlen_kv})"
        )
    if q_block <= 0:
        return f"q_block={q_block} must be positive"
    if window_size <= 0:
        return f"window_size={window_size} must be positive"
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        return (
            f"num_heads={num_heads} is not a whole multiple of "
            f"num_kv_heads={num_kv_heads}"
        )
    return None


def _required_width(
    seqlen_q: int, seqlen_kv: int, window_size: int, q_block: int, q_kv_offset: int
) -> int:
    """Widest span any single Q block must cover, rounded up to a stick.

    Within one Q block the rows' windows are staggered: row ``i`` attends
    ``[c-W+1, c]``, so the block's union spans ``W + q_len - 1`` columns. With
    the read start floored to a stick boundary this comes to ``W + q_block``
    for an aligned prefill, and exactly ``W`` for decode, where a single row
    has no stagger at all.
    """
    widest = 0
    for qi in range(-(-seqlen_q // q_block)):
        q_start = qi * q_block
        q_end = min(seqlen_q, q_start + q_block)
        first_coord = q_kv_offset + q_start
        last_coord = q_kv_offset + q_end - 1
        window_origin = max(0, _floor_stick(first_coord - window_size + 1))
        widest = max(widest, last_coord - window_origin + 1)
    return _ceil_stick(widest)


def choose_q_block(seqlen_q: int) -> int | None:
    """Pick the Q block size for a query length, or None if none fits.

    The block slicing needs ``seqlen_q % q_block == 0`` exactly, so:

      - ``seqlen_q == 1`` (decode) takes ``q_block = 1``. A single row has no
        intra-block stagger, which is what makes ``buffer_width == W`` there
        rather than ``W + q_block``.
      - otherwise a whole number of sticks takes ``q_block = STICK``.
      - anything else falls back. ``q_block = 1`` would technically divide any
        length, but it would mean one gather per query row, which defeats the
        point.
    """
    if seqlen_q == 1:
        return 1
    if seqlen_q > 0 and seqlen_q % STICK == 0:
        return STICK
    return None


def plan_window_gather(
    seqlen_q: int,
    seqlen_kv: int,
    window_size: int,
    is_causal: bool = True,
    q_block: int = STICK,
) -> WindowGatherPlan | None:
    """Plan the compact KV gather, or return None to keep the fallback path.

    Returns None — rather than raising — for any shape this cannot express
    exactly, so an unsupported model shape quietly keeps the unrolled
    decomposition instead of failing to compile.

    Rejected, and why:
      - bidirectional windows, which need a ``2W + q_block`` buffer (deferred)
      - a window that is not a whole number of sticks
      - a cache length that is not a whole number of sticks, whose ragged tail
        would leave the read start unaligned
      - a window that spans the whole cache, where gathering saves nothing and
        the plain masked path is strictly cheaper
      - degenerate or inconsistent lengths
    """
    if not is_causal:
        return None
    if seqlen_q <= 0 or seqlen_kv <= 0 or q_block <= 0:
        return None
    if window_size <= 0 or window_size % STICK != 0:
        return None
    if seqlen_kv % STICK != 0:
        return None

    q_kv_offset = seqlen_kv - seqlen_q
    if q_kv_offset < 0:
        return None

    buffer_width = _required_width(
        seqlen_q, seqlen_kv, window_size, q_block, q_kv_offset
    )
    # Gathering the entire cache is strictly worse than masking it in place.
    if buffer_width >= seqlen_kv:
        return None

    return WindowGatherPlan(
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        window_size=window_size,
        q_block=q_block,
        num_q_blocks=-(-seqlen_q // q_block),
        buffer_width=buffer_width,
        q_kv_offset=q_kv_offset,
        is_causal=is_causal,
    )
