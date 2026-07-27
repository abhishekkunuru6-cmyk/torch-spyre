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

"""SWA integration — Increment 2a: the causal-diagonal cross-dim coupling (SPEC).

Increment 1 (`validate_swa_sliding_matmul.py`) proved a matmul consumes an
OVERLAPPING read on its contraction dim.  That slid ONE dim, and that dim was
also the loop dim.  The real SWA op needs something strictly stronger: one loop
variable `i` must drive TWO different named dims at once —

    Q  : partition-slide, rows [64i, 64i+64)      window == stride  (no overlap)
    KV : overlap-slide,   cols [64i, 64i+W)       window >  stride  (overlap)

That is the causal diagonal.  This file is the SPEC increment (2a): it pins down
the exact semantics and the CPU reference BEFORE the mechanism exists (2b), and
doubles as the RED test for it.  Nothing here needs the hint to work yet.

The carrier program
-------------------
    A [QSEQ, KVSEQ] @ B [KVSEQ, D]  ->  out [QSEQ, D]

with QS = A.dim0 = out.dim0 partition-slid, and KV = A.dim1 = B.dim0 (the
CONTRACTION dim) overlap-slid, both under ONE shared loop var.  Tile i computes

    out[64i : 64i+64, :] = A[64i : 64i+64, iS : iS+W] @ B[iS : iS+W, :]

This is exactly SWA's second matmul `out_blk = exp_scores_blk @ v_blk`, where
Q-block i pairs with KV window i.  It is deliberately the NON-hazardous shape of
the problem: the slid KV dim is reduced, so every output row block is written
exactly once.  (Sliding a dim that is an OUTPUT dim — SWA's `q @ kᵀ`, where
consecutive tiles would write overlapping output columns — is a separate write
hazard, deferred to increment 3.)

The API (implemented by 2b)
---------------------------
Multi-entry `sliding` dict == COUPLED under one loop level, per-dim
window/stride:

    with spyre_hint(sliding={
        "QS": {"window": 64, "stride": 64},    # partition, no overlap
        "KV": {"window": W,  "stride": S},     # overlap when S < W
    }):
        out = a @ b

`hint_id` is what coarse_tile turns into a loop level, so all the DimHints from
one scope share it and stay one level however many dims they name.

Shape constraints implied by the partition+slide model
------------------------------------------------------
  * QSEQ % q_block == 0 and KVSEQ % W == 0   (each dim partitions cleanly)
  * QSEQ // q_block == KVSEQ // W            (ONE loop var => ONE trip count)
    This is new and non-obvious: coupling forces the two dims' sizes to agree
    through their windows.  It is why a real SWA shape must pick W to satisfy
    KVSEQ = W * num_q_blocks.
  * S*(N-1) + W <= KVSEQ                     (last window stays in bounds)
    With overlap (S < W) the last window ends at S*(N-1)+W < W*N = KVSEQ, so a
    tail of (W-S)*(N-1) KV elements is never read.  That is inherent to
    partition+slide (trip count from the partition, base advance from the
    stride), not a bug — the reference below models it exactly.
  * all dims multiples of 64 (stick alignment); S < W for actual overlap.

Run:
    python3 validate_swa_diagonal_coupling.py                 # spec + CPU refs (no HW)
    python3 validate_swa_diagonal_coupling.py --sweep
    SENCORES=1 python3 validate_swa_diagonal_coupling.py --compile   # on the pod
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
    """One coupled-slide configuration.

    ``qseq``/``kvseq``/``d`` are the carrier matmul's dims; ``q_block`` is the
    Q partition width (== its own window and stride); ``window``/``stride`` are
    the KV read extent and per-iteration base advance.
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
        """KV elements past the last window's end — never read (see docstring)."""
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
            f"unread KV tail {self.kv_unread_tail}"
        )


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


def diagonal_reference(a: torch.Tensor, b: torch.Tensor, shape: Shape) -> torch.Tensor:
    """THE target semantics: tile i pairs Q rows [64i,64i+64) with KV [iS, iS+W).

    Each output row block is written exactly once, so this is a plain
    per-tile assignment (not an accumulation like increment 1's reference).
    """
    out = torch.zeros(shape.qseq, shape.d, dtype=torch.float32)
    for i in range(shape.num_tiles):
        q_lo = i * shape.q_block
        kv_lo = i * shape.stride
        a_tile = a[q_lo : q_lo + shape.q_block, kv_lo : kv_lo + shape.window]
        b_tile = b[kv_lo : kv_lo + shape.window, :]
        out[q_lo : q_lo + shape.q_block, :] = a_tile.to(torch.float32) @ b_tile.to(
            torch.float32
        )
    return out


def stuck_kv_reference(a: torch.Tensor, b: torch.Tensor, shape: Shape) -> torch.Tensor:
    """DIAGNOSTIC: Q slides but KV does not — every tile reads KV [0, W).

    Matching this means the coupling attached the loop var to Q only and the KV
    dim never advanced (the slide was dropped on the second dim).
    """
    out = torch.zeros(shape.qseq, shape.d, dtype=torch.float32)
    b_tile = b[: shape.window, :].to(torch.float32)
    for i in range(shape.num_tiles):
        q_lo = i * shape.q_block
        a_tile = a[q_lo : q_lo + shape.q_block, : shape.window].to(torch.float32)
        out[q_lo : q_lo + shape.q_block, :] = a_tile @ b_tile
    return out


def disjoint_kv_reference(
    a: torch.Tensor, b: torch.Tensor, shape: Shape
) -> torch.Tensor:
    """DIAGNOSTIC: KV advances by its WINDOW, not its stride (ordinary partition).

    Matching this means the per-dim ``slide_stride`` was ignored and the KV dim
    degenerated to normal partition tiling.  Indistinguishable from the target
    when stride == window, which is why the sweep includes such a shape.
    """
    disjoint = Shape(
        qseq=shape.qseq,
        kvseq=shape.kvseq,
        d=shape.d,
        q_block=shape.q_block,
        window=shape.window,
        stride=shape.window,
    )
    return diagonal_reference(a, b, disjoint)


def _report_refs(a: torch.Tensor, b: torch.Tensor, shape: Shape) -> torch.Tensor:
    """Print the target reference and how far the diagnostics sit from it."""
    ref = diagonal_reference(a, b, shape)
    stuck = stuck_kv_reference(a, b, shape)
    disjoint = disjoint_kv_reference(a, b, shape)
    full = (a.to(torch.float32) @ b.to(torch.float32))[:, : shape.d]

    print(f"  target  (diagonal)      peak {ref.abs().max().item():8.3f}")
    for name, alt in (
        ("stuck KV (no slide)", stuck),
        ("disjoint KV (partition)", disjoint),
        ("full A@B (no tiling)", full),
    ):
        sep = (ref - alt).abs().max().item()
        flag = "  <-- INDISTINGUISHABLE" if sep < 1e-3 else ""
        print(
            f"  vs {name:<24} peak {alt.abs().max().item():8.3f}  "
            f"separation {sep:8.3f}{flag}"
        )
    return ref


def _self_check(a: torch.Tensor, b: torch.Tensor, shape: Shape) -> bool:
    """Independent re-derivation of the reference, element-wise.

    Guards the reference itself: a wrong reference would silently "validate" a
    wrong mechanism in 2c.
    """
    ref = diagonal_reference(a, b, shape)
    naive = torch.zeros(shape.qseq, shape.d, dtype=torch.float32)
    af, bf = a.to(torch.float32), b.to(torch.float32)
    for r in range(shape.qseq):
        i = r // shape.q_block
        kv_lo = i * shape.stride
        for c in range(shape.d):
            acc = 0.0
            for k in range(kv_lo, kv_lo + shape.window):
                acc += af[r, k].item() * bf[k, c].item()
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
    torch.manual_seed(0xD1A6)
    a = torch.randn(shape.qseq, shape.kvseq, dtype=torch.float16)
    b = torch.randn(shape.kvseq, shape.d, dtype=torch.float16)
    _report_refs(a, b, shape)
    if not self_check:
        return True
    # The triple loop is O(qseq*d*window) Python — only for the small shapes.
    if shape.qseq * shape.d * shape.window > 2**22:
        print("  reference self-check: skipped (shape too large for naive loop)")
        return True
    return _self_check(a, b, shape)


def run_compile(shape: Shape) -> bool:
    """Compile and run the coupled hint on device, checking it against the ref."""
    import torch_spyre  # noqa: F401
    import torch_spyre._inductor.propagate_named_dims as pnd
    from torch_spyre._inductor import config, spyre_hint

    print("=" * 78)
    print(f"  COMPILE  {shape.describe()}")
    shape.validate()
    pnd.reset()
    torch._dynamo.reset_code_caches()
    torch._inductor.codecache.FxGraphCache.clear()

    torch.manual_seed(0xD1A6)
    a = torch.randn(shape.qseq, shape.kvseq, dtype=torch.float16)
    b = torch.randn(shape.kvseq, shape.d, dtype=torch.float16)

    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # ONE loop var drives both: QS partitions (window == stride), KV
        # overlaps (stride < window).  Multi-entry sliding == coupled.
        with spyre_hint(
            sliding={
                "QS": {"window": shape.q_block, "stride": shape.q_block},
                "KV": {"window": shape.window, "stride": shape.stride},
            }
        ):
            return torch.matmul(a, b)

    device = torch.device("spyre")
    a_dev, b_dev = a.to(device), b.to(device)
    pnd.declare_tensor_dim("QS", shape.qseq)
    pnd.declare_tensor_dim("KV", shape.kvseq)
    pnd.declare_tensor_dim("D", shape.d)
    pnd.name_tensor_dims(a_dev, ["QS", "KV"])
    pnd.name_tensor_dims(b_dev, ["KV", "D"])

    try:
        with config.patch({"lx_planning": True, "allow_all_ops_in_lx_planning": True}):
            got = torch.compile(fn)(a_dev, b_dev).to("cpu", torch.float32)
    except NotImplementedError as e:
        print(f"  UNSUPPORTED: {e}")
        return False
    except Exception as e:  # noqa: BLE001 — surface compile/codegen failures
        print(f"  COMPILE/RUN FAILED: {type(e).__name__}: {e}")
        return False

    ref = _report_refs(a, b, shape)
    tol = max(ref.abs().max().item() * 3e-2, 1e-2)
    max_err = (got - ref).abs().max().item()
    ok = bool(torch.allclose(got, ref, rtol=3e-2, atol=tol))
    print(
        f"  max|got - diagonal_ref| = {max_err:.4f} -> {'MATCH' if ok else 'MISMATCH'}"
    )
    if ok:
        print("  RESULT: one loop var DOES couple a partition-slid Q with an")
        print("  overlap-slid KV.  Increment 2 complete; proceed to increment 3.")
    else:
        for name, alt in (
            ("stuck KV (slide dropped on KV)", stuck_kv_reference(a, b, shape)),
            (
                "disjoint KV (per-dim stride ignored)",
                disjoint_kv_reference(a, b, shape),
            ),
        ):
            if torch.allclose(got, alt, rtol=3e-2, atol=tol):
                print(f"  RESULT: output matches the {name} reference.")
                return ok
        print("  RESULT: output matches NO reference — inspect the bundle geometry.")
    return ok


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
        help="also compile and run the coupled hint on device (needs HW)",
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

    print("SWA increment 2a — diagonal cross-dim coupling (spec)")
    results = []
    for shape in shapes:
        ok = run_spec(shape, self_check=not args.no_self_check)
        if args.compile:
            ok = run_compile(shape) and ok
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
        print("  Spec only — no device run.  Rerun with --compile on the pod to")
        print("  check the coupled hint against these references.")


if __name__ == "__main__":
    main()
