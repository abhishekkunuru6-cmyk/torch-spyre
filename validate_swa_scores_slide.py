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

"""SWA integration — Increment 3a: sliding KV as a matmul OUTPUT dim (SPEC).

Increments 1 and 2 slid a dim that the matmul REDUCES.  SWA's first matmul is
the other shape:

    scores = q @ kᵀ        q [QSEQ, D], k [KVSEQ, D]  ->  scores [QSEQ, KVSEQ]

Here the contraction is the head dim D, and the sliding KV dim is an OUTPUT
column of the matmul.  Coupled with the Q-block partition, tile i computes

    scores[64i : 64i+64,  iS : iS+W] = Q[64i : 64i+64] @ K[iS : iS+W]ᵀ

This file is the SPEC increment (3a): it pins down the semantics and the CPU
reference before the mechanism exists (3b).  Nothing here needs the hint to
work yet.

Two findings that change what increment 3 actually is
-----------------------------------------------------
1. **There is no write hazard.**  The plan (and the guard increment 2 left in
   `_compute_full_ranges`) assumed a sliding OUTPUT dim means consecutive tiles
   write overlapping output elements.  They do not: tile i writes rows
   [64i, 64i+64) and tile i+1 writes rows [64i+64, 64i+128).  The Q partition
   separates them, so the 2D regions are DISJOINT even though their column
   ranges overlap.  What the sliding output dim actually breaks is
   `_compute_full_ranges`' span arithmetic — `tile * count` OVER-estimates the
   written span (a bigger buffer, which is safe) rather than double-writing.
   So the increment-2 guard is too strict, not wrong about a hazard.

2. **The real gap is TWO OUTPUT DIMS at one level.**  Both QS and KV are output
   dims of this matmul, so the coupled scope names two dims of the same kind —
   exactly what `_validate_coupled_sliding` rejects today, because
   `_stamp_group` resolves a hint_id to one output-range position per op.
   Increment 2 coupled one output dim + one reduction dim; increment 3 needs
   one level to tile two output-range positions.

Why the numeric check is necessary but NOT sufficient
------------------------------------------------------
In increments 1 and 2 the slide changed the VALUES (overlapping windows were
coverage-weighted into the reduction), so a correct number proved the overlap
happened.  Here the slide changes only WHICH elements are computed, not how:
the target band is literally a slice of the untiled `q @ kᵀ`.  A compiler that
ignored the hint and computed the whole [QSEQ, KVSEQ] product would produce a
band that matches the reference exactly.

So: a MATCH proves the band is correctly ALIGNED (the K read lands on the right
columns and the result is stored at the right offset) — it does NOT prove the
work was reduced.  ``--compile`` therefore ALSO checks the bundle structure
(``swa_probe_bundle.structural_report``): one ``scf.for`` of num_tiles
iterations with the compute inside it, rather than one full-size op.  Both have
to hold for a PASS.  The stuck-KV and disjoint-KV diagnostics below separate
numerically; they are the realistic alignment failure modes.

Shape constraints (unchanged from increment 2)
----------------------------------------------
  * QSEQ % q_block == 0, KVSEQ % W == 0, QSEQ // q_block == KVSEQ // W
  * S*(N-1) + W <= KVSEQ; all dims multiples of 64.

Run:
    python3 validate_swa_scores_slide.py                 # spec + CPU refs (no HW)
    python3 validate_swa_scores_slide.py --sweep
    SENCORES=1 python3 validate_swa_scores_slide.py --compile   # on the pod
"""

import argparse
import os
from dataclasses import dataclass

os.environ.setdefault("SENCORES", "1")

import torch  # noqa: E402

STICK = 64
DEFAULT_Q_BLOCK = 64


@dataclass(frozen=True)
class Shape:
    """One coupled scores-slide configuration.

    ``qseq``/``kvseq``/``d`` are the carrier matmul's dims (D is contracted);
    ``q_block`` is the Q partition width (its own window and stride);
    ``window``/``stride`` are the KV read extent and per-iteration advance.
    """

    qseq: int
    kvseq: int
    d: int
    q_block: int
    window: int
    stride: int

    @property
    def num_tiles(self) -> int:
        return self.qseq // self.q_block

    @property
    def kv_unread_tail(self) -> int:
        """KV columns past the last window's end — never written."""
        return self.kvseq - (self.stride * (self.num_tiles - 1) + self.window)

    def validate(self) -> None:
        if self.qseq % self.q_block:
            raise ValueError(f"q_block {self.q_block} must divide qseq {self.qseq}")
        if self.kvseq % self.window:
            raise ValueError(f"window {self.window} must divide kvseq {self.kvseq}")
        if self.qseq // self.q_block != self.kvseq // self.window:
            raise ValueError(
                f"coupled dims need one trip count: qseq//q_block="
                f"{self.qseq // self.q_block} != kvseq//window="
                f"{self.kvseq // self.window}"
            )
        if self.kv_unread_tail < 0:
            raise ValueError(
                f"last KV window ends at "
                f"{self.stride * (self.num_tiles - 1) + self.window} > kvseq "
                f"{self.kvseq}"
            )
        for name, val in (
            ("qseq", self.qseq),
            ("kvseq", self.kvseq),
            ("d", self.d),
            ("q_block", self.q_block),
            ("window", self.window),
            ("stride", self.stride),
        ):
            if val % STICK:
                raise ValueError(f"{name}={val} must be a multiple of {STICK}")

    def describe(self) -> str:
        return (
            f"QSEQ={self.qseq} KVSEQ={self.kvseq} D={self.d} "
            f"q_block={self.q_block} W={self.window} S={self.stride} -> "
            f"{self.num_tiles} tiles, KV overlap {self.window - self.stride}, "
            f"unwritten KV tail {self.kv_unread_tail}"
        )

    def tile_slices(self, i: int) -> tuple[slice, slice]:
        """The (rows, cols) region of the full scores matrix tile ``i`` owns."""
        q_lo = i * self.q_block
        kv_lo = i * self.stride
        return slice(q_lo, q_lo + self.q_block), slice(kv_lo, kv_lo + self.window)


# QSEQ//q_block == KVSEQ//W is required, so kvseq is always window * num_tiles.
SWEEP_SHAPES: tuple[Shape, ...] = (
    Shape(qseq=128, kvseq=256, d=64, q_block=64, window=128, stride=64),
    Shape(qseq=192, kvseq=384, d=64, q_block=64, window=128, stride=64),
    Shape(qseq=256, kvseq=512, d=64, q_block=64, window=128, stride=64),
    Shape(qseq=128, kvseq=384, d=64, q_block=64, window=192, stride=64),
    Shape(qseq=256, kvseq=1024, d=128, q_block=64, window=256, stride=64),
    Shape(qseq=256, kvseq=512, d=64, q_block=64, window=128, stride=128),  # no overlap
)

DEFAULT_SHAPE = SWEEP_SHAPES[2]


def extract_band(scores: torch.Tensor, shape: Shape) -> torch.Tensor:
    """Gather the [QSEQ, W] band out of a full [QSEQ, KVSEQ] scores matrix.

    Everything outside the band is never written by the tiled loop (the buffer
    comes from an uninitialized allocation), so all comparisons happen in this
    compact frame — never on the full matrix.
    """
    band = torch.zeros(shape.qseq, shape.window, dtype=torch.float32)
    for i in range(shape.num_tiles):
        rows, cols = shape.tile_slices(i)
        band[rows, :] = scores[rows, cols].to(torch.float32)
    return band


def _band_from_kv_window(
    q: torch.Tensor, k: torch.Tensor, shape: Shape, kv_lo_for_tile
) -> torch.Tensor:
    """Band where tile ``i`` scores Q block ``i`` against K[kv_lo : kv_lo+W]."""
    band = torch.zeros(shape.qseq, shape.window, dtype=torch.float32)
    for i in range(shape.num_tiles):
        q_lo = i * shape.q_block
        kv_lo = kv_lo_for_tile(i)
        q_tile = q[q_lo : q_lo + shape.q_block].to(torch.float32)
        k_tile = k[kv_lo : kv_lo + shape.window].to(torch.float32)
        band[q_lo : q_lo + shape.q_block, :] = q_tile @ k_tile.T
    return band


def diagonal_band_reference(
    q: torch.Tensor, k: torch.Tensor, shape: Shape
) -> torch.Tensor:
    """THE target: tile i scores Q rows [64i,64i+64) against K[iS, iS+W)."""
    return _band_from_kv_window(q, k, shape, lambda i: i * shape.stride)


def stuck_kv_band_reference(
    q: torch.Tensor, k: torch.Tensor, shape: Shape
) -> torch.Tensor:
    """DIAGNOSTIC: Q advances but the KV read never does — always K[0, W)."""
    return _band_from_kv_window(q, k, shape, lambda _i: 0)


def disjoint_kv_band_reference(
    q: torch.Tensor, k: torch.Tensor, shape: Shape
) -> torch.Tensor:
    """DIAGNOSTIC: KV advances by its WINDOW, not its stride (partition tiling)."""
    return _band_from_kv_window(q, k, shape, lambda i: i * shape.window)


def _report_refs(q: torch.Tensor, k: torch.Tensor, shape: Shape) -> torch.Tensor:
    """Print the target band and how far each diagnostic sits from it."""
    ref = diagonal_band_reference(q, k, shape)
    print(f"  target  (diagonal band)   peak {ref.abs().max().item():8.3f}")
    for name, alt in (
        ("stuck KV (no slide)", stuck_kv_band_reference(q, k, shape)),
        ("disjoint KV (partition)", disjoint_kv_band_reference(q, k, shape)),
    ):
        sep = (ref - alt).abs().max().item()
        flag = "  <-- INDISTINGUISHABLE" if sep < 1e-3 else ""
        print(
            f"  vs {name:<24} peak {alt.abs().max().item():8.3f}  "
            f"separation {sep:8.3f}{flag}"
        )
    # The untiled product is deliberately NOT a diagnostic: the target band is a
    # slice of it, so it can never be separated numerically (see module doc).
    full = (q.to(torch.float32) @ k.to(torch.float32).T)[:, : shape.kvseq]
    print(
        f"  note: band extracted from the untiled q@kT matches the target "
        f"exactly (sep {(ref - extract_band(full, shape)).abs().max().item():.5f})"
        " — numerics cannot prove work reduction; use --dump."
    )
    return ref


def _self_check(q: torch.Tensor, k: torch.Tensor, shape: Shape) -> bool:
    """Independent element-wise re-derivation of the target band."""
    ref = diagonal_band_reference(q, k, shape)
    naive = torch.zeros(shape.qseq, shape.window, dtype=torch.float32)
    qf, kf = q.to(torch.float32), k.to(torch.float32)
    for r in range(shape.qseq):
        i = r // shape.q_block
        kv_lo = i * shape.stride
        for c in range(shape.window):
            acc = 0.0
            for d in range(shape.d):
                acc += qf[r, d].item() * kf[kv_lo + c, d].item()
            naive[r, c] = acc
    err = (ref - naive).abs().max().item()
    ok = err < 1e-2 * max(ref.abs().max().item(), 1.0)
    print(
        f"  reference self-check: max|vectorized - naive| = {err:.5f} "
        f"-> {'OK' if ok else 'BROKEN'}"
    )
    return ok


def run_spec(shape: Shape, self_check: bool) -> bool:
    """Spec-only path: validate the shape, build the refs, prove separation."""
    print("=" * 78)
    print(f"  {shape.describe()}")
    shape.validate()
    torch.manual_seed(0x5C05)
    q = torch.randn(shape.qseq, shape.d, dtype=torch.float16)
    k = torch.randn(shape.kvseq, shape.d, dtype=torch.float16)
    _report_refs(q, k, shape)
    if not self_check:
        return True
    if shape.qseq * shape.window * shape.d > 2**22:
        print("  reference self-check: skipped (shape too large for naive loop)")
        return True
    return _self_check(q, k, shape)


def run_compile(shape: Shape, dump: bool) -> bool:
    """Compile the coupled scores slide on device and check band + structure."""
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

    torch.manual_seed(0x5C05)
    q = torch.randn(shape.qseq, shape.d, dtype=torch.float16)
    k = torch.randn(shape.kvseq, shape.d, dtype=torch.float16)

    def fn(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # QS partitions the score rows; KV overlap-slides the score COLUMNS.
        # Both are output dims of this matmul (D is the contraction) — the
        # two-output-dims-at-one-level case 3b has to implement.
        with spyre_hint(
            sliding={
                "QS": {"window": shape.q_block, "stride": shape.q_block},
                "KV": {"window": shape.window, "stride": shape.stride},
            }
        ):
            return torch.matmul(q, k.transpose(0, 1))

    device = torch.device("spyre")
    q_dev, k_dev = q.to(device), k.to(device)
    pnd.declare_tensor_dim("QS", shape.qseq)
    pnd.declare_tensor_dim("KV", shape.kvseq)
    pnd.declare_tensor_dim("D", shape.d)
    pnd.name_tensor_dims(q_dev, ["QS", "D"])
    pnd.name_tensor_dims(k_dev, ["KV", "D"])

    before = snapshot()
    try:
        with config.patch({"lx_planning": True, "allow_all_ops_in_lx_planning": True}):
            got = torch.compile(fn)(q_dev, k_dev).to("cpu", torch.float32)
    except NotImplementedError as e:
        print(f"  UNSUPPORTED: {e}")
        return False
    except Exception as e:  # noqa: BLE001 — surface compile/codegen failures
        print(f"  COMPILE/RUN FAILED: {type(e).__name__}: {e}")
        return False
    new_dirs = snapshot() - before

    ref = _report_refs(q, k, shape)
    got_band = extract_band(got, shape)
    tol = max(ref.abs().max().item() * 3e-2, 1e-2)
    max_err = (got_band - ref).abs().max().item()
    ok = bool(torch.allclose(got_band, ref, rtol=3e-2, atol=tol))
    print(
        f"  max|got_band - diagonal_ref| = {max_err:.4f} -> "
        f"{'MATCH' if ok else 'MISMATCH'}"
    )
    if not ok:
        for name, alt in (
            ("stuck KV (slide dropped)", stuck_kv_band_reference(q, k, shape)),
            ("disjoint KV (stride ignored)", disjoint_kv_band_reference(q, k, shape)),
        ):
            if torch.allclose(got_band, alt, rtol=3e-2, atol=tol):
                print(f"  RESULT: band matches the {name} reference.")
                return False
        print("  RESULT: band matches NO reference — inspect the sdsc geometry.")

    # Numbers only prove ALIGNMENT here (the band is a slice of the untiled
    # product), so the loop structure is a separate, required check.
    reduced = structural_report(new_dirs, shape.num_tiles)
    if dump:
        dump_bundles(new_dirs)
    if ok and reduced:
        print("  RESULT: score columns correctly aligned AND the work is tiled.")
    elif ok:
        print("  RESULT: score columns aligned, but the work was NOT reduced.")
    return ok and reduced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qseq", type=int, default=DEFAULT_SHAPE.qseq)
    ap.add_argument("--kvseq", type=int, default=DEFAULT_SHAPE.kvseq)
    ap.add_argument("--d", type=int, default=DEFAULT_SHAPE.d)
    ap.add_argument("--q-block", type=int, default=DEFAULT_Q_BLOCK)
    ap.add_argument("--window", type=int, default=DEFAULT_SHAPE.window)
    ap.add_argument("--stride", type=int, default=DEFAULT_SHAPE.stride)
    ap.add_argument(
        "--sweep", action="store_true", help="run every shape in SWEEP_SHAPES"
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="also compile the coupled scores slide on device (needs HW)",
    )
    ap.add_argument(
        "--dump", action="store_true", help="print the full bundle.mlir as well"
    )
    ap.add_argument(
        "--no-self-check",
        action="store_true",
        help="skip the naive-loop re-derivation of the reference",
    )
    args = ap.parse_args()

    shapes = (
        SWEEP_SHAPES
        if args.sweep
        else (
            Shape(
                qseq=args.qseq,
                kvseq=args.kvseq,
                d=args.d,
                q_block=args.q_block,
                window=args.window,
                stride=args.stride,
            ),
        )
    )

    print("SWA increment 3a — sliding KV as a matmul OUTPUT dim (spec)")
    results = []
    for shape in shapes:
        ok = run_spec(shape, self_check=not args.no_self_check)
        if args.compile:
            ok = run_compile(shape, dump=args.dump) and ok
        results.append((shape, ok))

    print("=" * 78)
    print("SUMMARY:")
    for shape, ok in results:
        print(
            f"  QSEQ={shape.qseq:>4} KVSEQ={shape.kvseq:>5} W={shape.window:>4} "
            f"S={shape.stride:>4}  {'PASS' if ok else 'FAIL'}"
        )
    n_pass = sum(1 for _, ok in results if ok)
    print(f"  {n_pass}/{len(results)} shapes")
    if not args.compile:
        print()
        print("  Spec only — no device run.  The two-output-dim coupling (3b) is")
        print("  NOT implemented yet; --compile is expected to report UNSUPPORTED.")


if __name__ == "__main__":
    main()
