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

"""Shape planning for the sliding-window-attention decomposition.

``spyre_sliding_window_attention`` unrolls a Python loop over Q blocks, slicing
K/V per block.  The sliding-address hint can express that loop as ONE device
``scf.for`` instead — but only for shapes whose per-block KV range is affine in
the block index, which the op's current ``max(0, ...)`` / ``min(seqlen_kv, ...)``
clamps deliberately break.

``plan_sliding_window`` decides whether a given shape qualifies and, if so,
returns the constant window geometry to drive the hint with.  Returning ``None``
means "use the existing unrolled path" — every shape that works today must keep
working, so the answer to anything unsupported is a fallback, never an error.

The equivalence this rests on was verified on hardware in increment 5a: reading
a CONSTANT window at ``base + i*q_block`` over padded K/V and masking the
out-of-sequence positions gives bit-identical results to the op's ragged
clamp-and-shrink loop (max abs difference 0.000000 across 6 shapes).  The
padding is never read as data: every padded position is masked.
"""

from __future__ import annotations

import dataclasses

import torch

STICK = 64


@dataclasses.dataclass(frozen=True)
class SlidingWindowPlan:
    """Constant-window geometry for one sliding-window-attention call.

    Attributes
    ----------
    num_q_blocks:
        Loop trip count — the number of Q blocks, and the only dim that sets it.
    q_block:
        Q rows per iteration, and the KV window's per-iteration advance.
    read_extent:
        KV columns read every iteration.  Constant by construction; that is the
        whole point of the plan.
    base_offset:
        Iteration 0's window start in UNPADDED KV coordinates.  Negative
        whenever the causal window reaches back before the sequence.
    left_pad / right_pad:
        Rows of padding K/V need so every window read stays in bounds.
    padded_kv:
        ``left_pad + seqlen_kv + right_pad`` — the KV named dim's size.
    """

    num_q_blocks: int
    q_block: int
    read_extent: int
    base_offset: int
    left_pad: int
    right_pad: int
    padded_kv: int
    q_kv_offset: int
    window_size: int
    is_causal: bool

    def window_lo(self, qi: int) -> int:
        """Block ``qi``'s window start in unpadded KV coordinates."""
        return self.base_offset + qi * self.q_block

    def padded_window_lo(self, qi: int) -> int:
        """Block ``qi``'s window start as an index into the PADDED tensors."""
        return self.window_lo(qi) + self.left_pad

    def describe(self) -> str:
        return (
            f"{self.num_q_blocks} blocks x {self.q_block} rows, "
            f"W={self.read_extent} S={self.q_block} base={self.base_offset} "
            f"pad={self.left_pad}/{self.right_pad} padded_kv={self.padded_kv}"
        )


def unclamped_kv_range(
    qi: int,
    q_block: int,
    seqlen_q: int,
    q_kv_offset: int,
    window_size: int,
    is_causal: bool,
) -> tuple[int, int]:
    """Block ``qi``'s stick-rounded KV range with the op's clamps REMOVED.

    Mirrors ``spyre_sliding_window_attention``'s kv_start/kv_end arithmetic
    without ``max(0, ...)`` / ``min(seqlen_kv, ...)``.  Those clamps are exactly
    what makes the real ranges ragged; dropping them is what makes the range
    affine in ``qi``, and therefore expressible as a slide.

    Derived rather than hand-written because causal and non-causal bands have
    different widths — the non-causal band reaches ``window_size`` FORWARD too.
    """
    q_lo = qi * q_block
    q_hi = min(seqlen_q, q_lo + q_block)
    r_lo = q_kv_offset + q_lo
    r_hi = q_kv_offset + q_hi - 1
    kv_lo = r_lo - window_size + 1
    kv_hi = r_hi if is_causal else r_hi + window_size - 1
    # Python floor-divides toward -inf, which is the rounding a negative
    # coordinate needs.
    return (kv_lo // STICK) * STICK, ((kv_hi // STICK) + 1) * STICK


def plan_sliding_window(
    batch_size: int,
    seqlen_q: int,
    seqlen_kv: int,
    window_size: int,
    is_causal: bool,
    q_block: int = STICK,
) -> SlidingWindowPlan | None:
    """Return the sliding geometry for this shape, or None to use the unrolled path.

    Unsupported shapes return ``None`` rather than raising: the existing loop
    handles everything, so an unrecognised shape must degrade to it silently.

    Rejected, each for a reason that has been measured rather than assumed:

    * ``batch_size != 1`` — rank-4 attention under the hint returns wrong values
      once batch > 1 and heads >= 4 (plan section 6.1, unexplained; the loop
      structure is right and the addressing is not).  batch == 1 is verified
      across heads 2/4/8, MHA and GQA, seqlen 256 and 512.
    * ``seqlen_q % q_block`` — a partial last Q block makes the block's KV range
      a different width from the others, so the extent stops being constant.
    * a non-affine range — checked directly rather than reasoned about, since
      the whole construction depends on it.
    """
    if batch_size != 1:
        return None
    if q_block <= 0 or seqlen_q <= 0 or seqlen_kv <= 0 or window_size <= 0:
        return None
    if seqlen_q % q_block:
        return None
    if seqlen_kv < seqlen_q:
        return None

    num_q_blocks = seqlen_q // q_block
    q_kv_offset = seqlen_kv - seqlen_q

    def _range(qi: int) -> tuple[int, int]:
        return unclamped_kv_range(
            qi, q_block, seqlen_q, q_kv_offset, window_size, is_causal
        )

    base_offset, first_end = _range(0)
    read_extent = first_end - base_offset

    # The slide model needs start == base + qi*q_block and a constant extent for
    # EVERY block.  Verify instead of trusting the algebra; a shape that does
    # not satisfy it falls back rather than computing something plausible.
    for qi in range(num_q_blocks):
        lo, hi = _range(qi)
        if lo != base_offset + qi * q_block or hi - lo != read_extent:
            return None

    left_pad = max(0, -base_offset)
    last_end = base_offset + (num_q_blocks - 1) * q_block + read_extent
    right_pad = max(0, last_end - seqlen_kv)

    # Padding must not break stick alignment: base_offset is already
    # stick-rounded, so left_pad is a whole number of sticks, but say so.
    if left_pad % STICK or right_pad % STICK:
        return None

    return SlidingWindowPlan(
        num_q_blocks=num_q_blocks,
        q_block=q_block,
        read_extent=read_extent,
        base_offset=base_offset,
        left_pad=left_pad,
        right_pad=right_pad,
        padded_kv=left_pad + seqlen_kv + right_pad,
        q_kv_offset=q_kv_offset,
        window_size=window_size,
        is_causal=is_causal,
    )


def build_band_mask_cpu(
    seqlen_q: int,
    seqlen_kv: int,
    left_pad: int,
    right_pad: int,
    q_kv_offset: int,
    window_size: int,
    is_causal: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the whole sliding-window band over padded K/V, as a CPU tensor.

    Shape ``[1, 1, seqlen_q, left_pad + seqlen_kv + right_pad]``; 0.0 = keep,
    -inf = masked.  Broadcasts over batch and heads.

    Kept separate from the ``spyre::sliding_window_band_mask`` custom op that
    wraps it: the op is registered for the spyre device and cannot be called on
    CPU, so the logic lives here where it can be tested directly instead of
    being mirrored by a second copy in the tests.

    Padding columns are masked unconditionally.  That is the property the whole
    rewrite rests on — it lets the loop read a constant-width window at a
    negative origin without padded rows ever reaching the result.  A padding
    column at KV coordinate < 0 can otherwise satisfy the causal test
    (``0 <= delta < window_size``) and leak uninitialised data into the softmax.
    """
    padded_kv = left_pad + seqlen_kv + right_pad
    # Absolute KV coordinates: padded column c maps to c - left_pad, so leading
    # padding is negative and trailing padding runs past the sequence end.
    # Both fall outside [0, seqlen_kv) and are dropped by `in_sequence`.
    q_idx = torch.arange(seqlen_q, device="cpu") + q_kv_offset
    k_idx = torch.arange(padded_kv, device="cpu") - left_pad
    delta = q_idx.unsqueeze(-1) - k_idx.unsqueeze(0)
    if is_causal:
        in_band = (delta >= 0) & (delta < window_size)
    else:
        in_band = delta.abs() < window_size
    in_sequence = ((k_idx >= 0) & (k_idx < seqlen_kv)).unsqueeze(0)

    mask = torch.zeros(1, 1, seqlen_q, padded_kv, dtype=dtype, device="cpu")
    mask.masked_fill_(~(in_band & in_sequence).unsqueeze(0).unsqueeze(0), float("-inf"))
    return mask
