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

"""SWA integration — Increment 5a: real SWA shapes (SPEC).

Increment 4 ran the whole windowed body under one slide, but at shapes chosen to
satisfy the model rather than shapes the real op produces.  Scoping the rewrite
against `spyre_sliding_window_attention` (decompositions.py:748) turned up two
blockers, both from one root cause.

Blocker A — the trip-count constraint cannot hold
--------------------------------------------------
Increment 2a derived `QSEQ // q_block == KVSEQ // W`: one loop level, one trip
count.  At a real prefill shape (Lq = Lk = 512, window_size = 128, q_block = 64):

    trip count N  = num_q_blocks   = 8       pinned by the Q partition
    window W      = window_size+64 = 192     pinned by the algorithm
    model demands N == KVSEQ // W  == 2

8 != 2, and neither side is tunable.  Pre-padding K/V by window_size does not
rescue it either (640 // 192 == 3).

Blocker B — the window origin is negative
-------------------------------------------
The real causal range is `kv_lo = q_kv_offset + q_start - window_size + 1`,
negative for the first blocks — hence the `max(0, ...)` clamp in the op.  So
`kv_len` is NOT constant; for the shape above it ramps 64, 128, 192, 192, ...
before stabilizing.  The current model assumes a constant extent at base `i*S`
starting from 0 and cannot express that ragged prefix.

Root cause, and what 5b must change
-------------------------------------
`_append_sliding_hints` derives a trip count from `dim_size // window` for EVERY
coupled dim.  That is correct for a dim being PARTITIONED and wrong for a
read-only sliding window over an INPUT: K/V need not claim a trip count at all.
The count should come from the partitioning dim (Q) alone, leaving the sliding
dim to satisfy only `base + S*(N-1) + W <= dim_size`.

Proposed API for 5b — a sliding dim may opt out of setting the trip count, and
may start at a non-zero (possibly negative, once padded) base:

    with spyre_hint(sliding={
        "QS": {"window": 64, "stride": 64},                  # sets N
        "KV": {"window": W, "stride": 64, "base": -(W - 64), # follows N
               "counts_tiles": False},
    }):

Relaxing this DELETES the `QSEQ//q_block == KVSEQ//W` constraint increment 2a
found — it was a symptom of deriving N from the wrong dim, not a law.

How the ragged prefix is handled
----------------------------------
NOT by varying the extent.  Every iteration reads a constant W-wide window at
`base + i*S`; the positions that fall outside the sequence are killed by the
band mask, which the real op already builds for exactly this purpose.  A
negative base needs `window_size` rows of left padding on K/V so the read stays
in bounds — the padded rows are always masked, so their contents never matter.
`padded_reference` below models precisely that, and `masked_windowed_reference`
models the op's current clamp-and-shrink behaviour.  THEY MUST AGREE: that
equality is the core claim of 5a, because it is what licenses replacing a
ragged, data-dependent loop with a constant-shape sliding one.

Run:
    python3 validate_swa_real_shapes.py                 # spec + CPU refs (no HW)
    python3 validate_swa_real_shapes.py --sweep
    SENCORES=1 python3 validate_swa_real_shapes.py --compile   # on the pod (5c)
"""

import argparse
import math
import os
from dataclasses import dataclass

os.environ.setdefault("SENCORES", "1")

import torch  # noqa: E402

STICK = 64
NEG_INF = float("-inf")


@dataclass(frozen=True)
class SwaShape:
    """A real SWA configuration, named the way the decomposition names things."""

    seqlen_q: int
    seqlen_kv: int
    head_dim: int
    window_size: int
    q_block: int = STICK
    is_causal: bool = True

    @property
    def num_q_blocks(self) -> int:
        return -(-self.seqlen_q // self.q_block)  # ceil div, as the op does

    @property
    def q_kv_offset(self) -> int:
        """Absolute KV coordinate of query row 0 (decode and prefill alike)."""
        return self.seqlen_kv - self.seqlen_q

    def unclamped_range(self, qi: int) -> tuple[int, int]:
        """Block ``qi``'s stick-rounded KV range WITHOUT the op's clamps.

        Reproduces decompositions.py's kv_start/kv_end arithmetic but drops the
        ``max(0, ...)`` / ``min(seqlen_kv, ...)``.  Those clamps are exactly what
        makes the real op's ranges ragged; removing them is what makes the range
        affine in ``qi`` — and therefore expressible as a slide.

        Derived rather than hand-written because the causal and non-causal bands
        have different widths (non-causal reaches window_size FORWARD as well).
        """
        q_lo = qi * self.q_block
        q_hi = min(self.seqlen_q, q_lo + self.q_block)
        r_lo = self.q_kv_offset + q_lo
        r_hi = self.q_kv_offset + q_hi - 1
        kv_lo = r_lo - self.window_size + 1
        kv_hi = r_hi if self.is_causal else r_hi + self.window_size - 1
        # Python floor-divides toward -inf, which is what stick-rounding a
        # negative coordinate needs.
        return (kv_lo // STICK) * STICK, ((kv_hi // STICK) + 1) * STICK

    @property
    def read_extent(self) -> int:
        """The CONSTANT window width every iteration reads.

        With seqlen_q a multiple of q_block every block has the same unclamped
        width, so this is a property of the shape, not of the block.
        """
        lo, hi = self.unclamped_range(0)
        return hi - lo

    @property
    def base_offset(self) -> int:
        """Where iteration 0's window starts, in unpadded KV coordinates.

        Negative whenever the window reaches back before the sequence — the
        whole reason left padding is needed.
        """
        return self.unclamped_range(0)[0]

    @property
    def left_pad(self) -> int:
        """Rows of left padding that make the first window read in bounds."""
        return max(0, -self.base_offset)

    @property
    def right_pad(self) -> int:
        """Rows of right padding that make the last window read in bounds.

        Non-zero mainly for the non-causal band, whose forward reach runs past
        the end of the sequence for the final blocks.
        """
        last_end = self.window_lo(self.num_q_blocks - 1) + self.read_extent
        return max(0, last_end - self.seqlen_kv)

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    def window_lo(self, qi: int) -> int:
        """Iteration qi's window start in UNPADDED KV coordinates (may be < 0).

        Affine in qi: the stick rounding is stable because q_lo advances by a
        whole q_block, so ``base_offset + qi * q_block`` is exact.
        """
        return self.base_offset + qi * self.q_block

    def validate(self) -> None:
        if self.seqlen_q % self.q_block:
            raise ValueError(
                f"seqlen_q {self.seqlen_q} must be a multiple of q_block "
                f"{self.q_block} for the partition slide"
            )
        if self.seqlen_kv < self.seqlen_q:
            raise ValueError(f"seqlen_kv {self.seqlen_kv} < seqlen_q {self.seqlen_q}")
        for qi in range(self.num_q_blocks):
            lo, hi = self.unclamped_range(qi)
            if (lo, hi - lo) != (self.window_lo(qi), self.read_extent):
                raise ValueError(
                    f"block {qi} range ({lo}, {hi}) is not affine in qi: "
                    f"expected start {self.window_lo(qi)} extent "
                    f"{self.read_extent} — the slide model does not apply"
                )

    def old_rule_verdict(self) -> tuple[bool, str]:
        """Whether increment 2a's constraint holds, and which half fails if not.

        The rule has two halves — ``window | dim_size`` (so the windows tile the
        dim) and ``N == dim_size // window`` (so both coupled dims agree on the
        trip count).  A shape can satisfy the counts and still fail on
        divisibility, so the two are reported separately rather than collapsed
        into one confusing "N == N -> VIOLATED".
        """
        divides = self.seqlen_kv % self.read_extent == 0
        derived = self.seqlen_kv // self.read_extent
        counts_agree = self.num_q_blocks == derived
        if not divides:
            return False, (
                f"W={self.read_extent} does not divide Lkv={self.seqlen_kv} "
                f"(remainder {self.seqlen_kv % self.read_extent})"
            )
        if not counts_agree:
            return False, f"N={self.num_q_blocks} != Lkv//W={derived}"
        return True, f"N == Lkv//W == {derived}"

    def describe(self) -> str:
        return (
            f"Lq={self.seqlen_q} Lkv={self.seqlen_kv} D={self.head_dim} "
            f"window={self.window_size}{'' if self.is_causal else ' non-causal'} "
            f"-> {self.num_q_blocks} blocks, W={self.read_extent} "
            f"S={self.q_block} base={self.base_offset} "
            f"pad={self.left_pad}/{self.right_pad}"
        )


def pad_kv(t: torch.Tensor, shape: SwaShape) -> torch.Tensor:
    """Left/right-pad a KV tensor so every window read is in bounds.

    The padded rows are always masked out, so their contents never reach the
    result — zeros keep it obvious in a dump.
    """
    if not (shape.left_pad or shape.right_pad):
        return t
    width = t.shape[1]
    parts = []
    if shape.left_pad:
        parts.append(torch.zeros(shape.left_pad, width, dtype=t.dtype))
    parts.append(t)
    if shape.right_pad:
        parts.append(torch.zeros(shape.right_pad, width, dtype=t.dtype))
    return torch.cat(parts, dim=0)


SWEEP_SHAPES: tuple[SwaShape, ...] = (
    # The scoping example from the plan: 8 blocks, W=192, old rule wants 2.
    SwaShape(seqlen_q=512, seqlen_kv=512, head_dim=64, window_size=128),
    SwaShape(seqlen_q=256, seqlen_kv=256, head_dim=64, window_size=128),
    SwaShape(seqlen_q=512, seqlen_kv=512, head_dim=64, window_size=256),
    SwaShape(seqlen_q=128, seqlen_kv=512, head_dim=64, window_size=128),  # decode-ish
    SwaShape(seqlen_q=256, seqlen_kv=256, head_dim=128, window_size=64),
    SwaShape(
        seqlen_q=512, seqlen_kv=512, head_dim=64, window_size=128, is_causal=False
    ),
)

DEFAULT_SHAPE = SWEEP_SHAPES[0]


def band_mask(shape: SwaShape, qi: int) -> torch.Tensor:
    """Additive mask for block ``qi``'s constant W-wide window: 0 keep, -inf drop.

    Kills three things at once, which is why a constant-shape read is safe:
      * left padding (KV coordinate < 0),
      * positions past the end of the sequence,
      * positions outside the causal / sliding band.
    """
    mask = torch.zeros(shape.q_block, shape.read_extent, dtype=torch.float32)
    lo = shape.window_lo(qi)
    for r in range(shape.q_block):
        q_abs = shape.q_kv_offset + qi * shape.q_block + r
        for c in range(shape.read_extent):
            kv_abs = lo + c
            in_sequence = 0 <= kv_abs < shape.seqlen_kv
            if shape.is_causal:
                in_band = q_abs - shape.window_size < kv_abs <= q_abs
            else:
                in_band = abs(q_abs - kv_abs) < shape.window_size
            if not (in_sequence and in_band):
                mask[r, c] = NEG_INF
    return mask


def _softmax_attend(
    q_blk: torch.Tensor, k_win: torch.Tensor, v_win: torch.Tensor, mask, scale: float
) -> torch.Tensor:
    """One block's masked attention; fully-masked rows come back as zeros."""
    s = (q_blk @ k_win.T) * scale
    if mask is not None:
        s = s + mask
    biggest = s.amax(dim=-1, keepdim=True)
    # A row masked everywhere has max -inf; force it to 0 so exp() is finite and
    # the row's denominator is 0 -> output 0, rather than NaN.
    biggest = torch.where(torch.isinf(biggest), torch.zeros_like(biggest), biggest)
    p = torch.exp(s - biggest)
    denom = p.sum(dim=-1, keepdim=True)
    out = p @ v_win
    return torch.where(denom > 0, out / denom.clamp(min=1e-30), torch.zeros_like(out))


def padded_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: SwaShape
) -> torch.Tensor:
    """THE target: constant W-wide window at ``base + i*S`` over padded K/V.

    This is the shape the sliding hint can express — every iteration identical
    in size, the mask handling everything the clamp used to.
    """
    qf = q.to(torch.float32)
    kf = pad_kv(k.to(torch.float32), shape)
    vf = pad_kv(v.to(torch.float32), shape)
    pad = shape.left_pad
    out = torch.zeros(shape.seqlen_q, shape.head_dim, dtype=torch.float32)
    for qi in range(shape.num_q_blocks):
        q_lo = qi * shape.q_block
        lo = shape.window_lo(qi) + pad  # >= 0 by construction
        out[q_lo : q_lo + shape.q_block] = _softmax_attend(
            qf[q_lo : q_lo + shape.q_block],
            kf[lo : lo + shape.read_extent],
            vf[lo : lo + shape.read_extent],
            band_mask(shape, qi),
            shape.scale,
        )
    return out


def masked_windowed_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: SwaShape
) -> torch.Tensor:
    """What the op does TODAY: clamp the range per block, so kv_len varies.

    Mirrors decompositions.py's kv_start/kv_end arithmetic.  Must agree with
    ``padded_reference`` — that equality is what licenses the rewrite.
    """
    qf, kf, vf = (t.to(torch.float32) for t in (q, k, v))
    out = torch.zeros(shape.seqlen_q, shape.head_dim, dtype=torch.float32)
    for qi in range(shape.num_q_blocks):
        q_lo = qi * shape.q_block
        q_hi = min(shape.seqlen_q, q_lo + shape.q_block)
        r_lo = shape.q_kv_offset + q_lo
        r_hi = shape.q_kv_offset + q_hi - 1
        if shape.is_causal:
            kv_lo, kv_hi = r_lo - shape.window_size + 1, r_hi
        else:
            kv_lo, kv_hi = r_lo - shape.window_size + 1, r_hi + shape.window_size - 1
        kv_start = max(0, (kv_lo // STICK) * STICK)
        kv_end = min(shape.seqlen_kv, ((kv_hi // STICK) + 1) * STICK)

        # Same mask rule, over this block's clamped (variable-width) range.
        mask = torch.zeros(q_hi - q_lo, kv_end - kv_start, dtype=torch.float32)
        for r in range(q_hi - q_lo):
            q_abs = r_lo + r
            for c in range(kv_end - kv_start):
                kv_abs = kv_start + c
                if shape.is_causal:
                    in_band = q_abs - shape.window_size < kv_abs <= q_abs
                else:
                    in_band = abs(q_abs - kv_abs) < shape.window_size
                if not in_band:
                    mask[r, c] = NEG_INF

        out[q_lo:q_hi] = _softmax_attend(
            qf[q_lo:q_hi],
            kf[kv_start:kv_end],
            vf[kv_start:kv_end],
            mask,
            shape.scale,
        )
    return out


def unmasked_padded_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: SwaShape
) -> torch.Tensor:
    """DIAGNOSTIC: constant window, NO mask — padding and out-of-band leak in."""
    qf = q.to(torch.float32)
    kf = pad_kv(k.to(torch.float32), shape)
    vf = pad_kv(v.to(torch.float32), shape)
    pad = shape.left_pad
    out = torch.zeros(shape.seqlen_q, shape.head_dim, dtype=torch.float32)
    for qi in range(shape.num_q_blocks):
        q_lo = qi * shape.q_block
        lo = shape.window_lo(qi) + pad
        out[q_lo : q_lo + shape.q_block] = _softmax_attend(
            qf[q_lo : q_lo + shape.q_block],
            kf[lo : lo + shape.read_extent],
            vf[lo : lo + shape.read_extent],
            None,
            shape.scale,
        )
    return out


def full_causal_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: SwaShape
) -> torch.Tensor:
    """DIAGNOSTIC: causal attention with NO sliding window (unbounded history)."""
    qf, kf, vf = (t.to(torch.float32) for t in (q, k, v))
    s = (qf @ kf.T) * shape.scale
    rows = torch.arange(shape.seqlen_q).unsqueeze(1) + shape.q_kv_offset
    cols = torch.arange(shape.seqlen_kv).unsqueeze(0)
    keep = cols <= rows if shape.is_causal else torch.ones_like(s, dtype=torch.bool)
    s = s.masked_fill(~keep, NEG_INF)
    p = torch.exp(s - s.amax(dim=-1, keepdim=True))
    return (p @ vf) / p.sum(dim=-1, keepdim=True)


def _report(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: SwaShape
) -> tuple[torch.Tensor, bool]:
    """Print the constraint check, the target, and every diagnostic's distance."""
    old_rule, why = shape.old_rule_verdict()
    print(
        f"  increment-2a trip-count rule: {why} -> "
        f"{'holds' if old_rule else 'VIOLATED (blocker A)'}"
    )

    ref = padded_reference(q, k, v, shape)
    clamped = masked_windowed_reference(q, k, v, shape)
    equal_err = (ref - clamped).abs().max().item()
    agrees = equal_err < 1e-4 * max(ref.abs().max().item(), 1.0)
    print(
        f"  CORE CLAIM  max|padded_constant_window - clamped_ragged| = "
        f"{equal_err:.6f} -> {'AGREE' if agrees else 'DISAGREE'}"
    )

    print(f"  target  (padded+masked)   peak {ref.abs().max().item():8.3f}")
    for name, alt in (
        ("unmasked padded window", unmasked_padded_reference(q, k, v, shape)),
        ("full causal (no window)", full_causal_reference(q, k, v, shape)),
    ):
        sep = (ref - alt).abs().max().item()
        flag = "  <-- INDISTINGUISHABLE" if sep < 1e-3 else ""
        print(
            f"  vs {name:<26} peak {alt.abs().max().item():8.3f}  "
            f"separation {sep:8.3f}{flag}"
        )
    return ref, agrees


def run_spec(shape: SwaShape) -> bool:
    """Spec-only path: quantify the blocker and prove the two models agree."""
    print("=" * 78)
    print(f"  {shape.describe()}")
    shape.validate()
    torch.manual_seed(0x5EA1)
    q = torch.randn(shape.seqlen_q, shape.head_dim, dtype=torch.float16)
    k = torch.randn(shape.seqlen_kv, shape.head_dim, dtype=torch.float16)
    v = torch.randn(shape.seqlen_kv, shape.head_dim, dtype=torch.float16)
    _ref, agrees = _report(q, k, v, shape)
    return agrees


def run_compile(shape: SwaShape, dump: bool) -> bool:
    """Compile the constant-window form on device (needs 5b's hint extension)."""
    import torch_spyre  # noqa: F401
    import torch_spyre._inductor.propagate_named_dims as pnd
    from torch_spyre._inductor import config, spyre_hint

    from swa_probe_bundle import dump_bundles, snapshot, structural_report

    print("=" * 78)
    print(f"  COMPILE  {shape.describe()}")
    shape.validate()
    pnd.reset()
    torch._dynamo.reset_code_caches()
    torch._inductor.codecache.FxGraphCache.clear()

    torch.manual_seed(0x5EA1)
    q = torch.randn(shape.seqlen_q, shape.head_dim, dtype=torch.float16)
    k = torch.randn(shape.seqlen_kv, shape.head_dim, dtype=torch.float16)
    v = torch.randn(shape.seqlen_kv, shape.head_dim, dtype=torch.float16)

    k_pad = pad_kv(k, shape)
    v_pad = pad_kv(v, shape)
    padded_kv = k_pad.shape[0]
    # The per-block masks stacked along Q: [seqlen_q, read_extent].  A row block
    # is tile-local in KV, so this rides the Q partition slide, not the KV one.
    masks = torch.cat([band_mask(shape, qi) for qi in range(shape.num_q_blocks)]).to(
        torch.float16
    )

    def fn(q, k, v, masks):
        # KV follows Q's trip count instead of setting its own — the 5b
        # extension.  base is folded into the padding here, so the hint itself
        # still starts at 0; a real base= parameter would drop the pre-pad.
        with spyre_hint(
            sliding={
                "QS": {"window": shape.q_block, "stride": shape.q_block},
                "KV": {
                    "window": shape.read_extent,
                    "stride": shape.q_block,
                    "counts_tiles": False,
                },
            }
        ):
            s = torch.matmul(q, k.transpose(0, 1)) * shape.scale + masks
            p = torch.exp(s - torch.amax(s, dim=-1, keepdim=True))
            return torch.matmul(p, v) / torch.sum(p, dim=-1, keepdim=True)

    device = torch.device("spyre")
    q_dev, k_dev, v_dev = q.to(device), k_pad.to(device), v_pad.to(device)
    m_dev = masks.to(device)
    pnd.declare_tensor_dim("QS", shape.seqlen_q)
    pnd.declare_tensor_dim("KV", padded_kv)
    pnd.declare_tensor_dim("D", shape.head_dim)
    pnd.name_tensor_dims(q_dev, ["QS", "D"])
    pnd.name_tensor_dims(k_dev, ["KV", "D"])
    pnd.name_tensor_dims(v_dev, ["KV", "D"])

    before = snapshot()
    try:
        with config.patch({"lx_planning": True, "allow_all_ops_in_lx_planning": True}):
            got = torch.compile(fn)(q_dev, k_dev, v_dev, m_dev).to("cpu", torch.float32)
    except NotImplementedError as e:
        print(f"  UNSUPPORTED (expected until 5b): {e}")
        return False
    except Exception as e:  # noqa: BLE001 — surface compile/codegen failures
        print(f"  COMPILE/RUN FAILED: {type(e).__name__}: {e}")
        return False
    new_dirs = snapshot() - before

    ref = padded_reference(q, k, v, shape)
    tol = max(ref.abs().max().item() * 5e-2, 2e-2)
    max_err = (got - ref).abs().max().item()
    ok = bool(torch.allclose(got, ref, rtol=5e-2, atol=tol))
    print(f"  max|got - padded_ref| = {max_err:.4f} -> {'MATCH' if ok else 'MISMATCH'}")
    reduced = structural_report(new_dirs, shape.num_q_blocks)
    if dump:
        dump_bundles(new_dirs)
    return ok and reduced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqlen-q", type=int, default=DEFAULT_SHAPE.seqlen_q)
    ap.add_argument("--seqlen-kv", type=int, default=DEFAULT_SHAPE.seqlen_kv)
    ap.add_argument("--head-dim", type=int, default=DEFAULT_SHAPE.head_dim)
    ap.add_argument("--window-size", type=int, default=DEFAULT_SHAPE.window_size)
    ap.add_argument("--non-causal", action="store_true")
    ap.add_argument(
        "--sweep", action="store_true", help="run every shape in SWEEP_SHAPES"
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="also compile on device (needs HW; fails until 5b)",
    )
    ap.add_argument(
        "--dump", action="store_true", help="print the full bundle.mlir as well"
    )
    args = ap.parse_args()

    shapes = (
        SWEEP_SHAPES
        if args.sweep
        else (
            SwaShape(
                seqlen_q=args.seqlen_q,
                seqlen_kv=args.seqlen_kv,
                head_dim=args.head_dim,
                window_size=args.window_size,
                is_causal=not args.non_causal,
            ),
        )
    )

    print("SWA increment 5a — real SWA shapes (spec)")
    results = []
    for shape in shapes:
        ok = run_spec(shape)
        if args.compile:
            ok = run_compile(shape, dump=args.dump) and ok
        results.append((shape, ok))

    print("=" * 78)
    print("SUMMARY:")
    violations = 0
    for shape, ok in results:
        old_rule, _why = shape.old_rule_verdict()
        violations += not old_rule
        print(
            f"  Lq={shape.seqlen_q:>4} Lkv={shape.seqlen_kv:>5} "
            f"win={shape.window_size:>4} W={shape.read_extent:>4} "
            f"N={shape.num_q_blocks:>3}  2a-rule "
            f"{'ok  ' if old_rule else 'FAILS'}  {'PASS' if ok else 'FAIL'}"
        )
    n_pass = sum(1 for _, ok in results if ok)
    print(f"  {n_pass}/{len(results)} shapes agree; {violations} violate the 2a rule")
    if not args.compile:
        print()
        print("  Spec only.  PASS here means the constant-window+mask model and")
        print("  the op's clamp-and-shrink model give the SAME answer, so the")
        print("  rewrite is licensed once 5b lifts the trip-count coupling.")


if __name__ == "__main__":
    main()
