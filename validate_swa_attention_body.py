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

"""SWA integration — Increment 4a: the whole attention body under one slide (SPEC).

Increments 2 and 3 each proved one half of SWA's inner body in isolation:

    inc 2   out_blk = exp_scores @ v_win   sliding KV is the REDUCED dim
    inc 3   scores  = q_blk @ k_winT       sliding KV is an OUTPUT dim

Increment 4 puts them in ONE hint scope with the softmax between them, which is
what SWA actually runs per Q-block:

    q_blk = Q[64i : 64i+64]        k_win = K[iS : iS+W]     v_win = V[iS : iS+W]
    s     = (q_blk @ k_winT) * scale
    p     = exp(s - max(s, dim=-1))
    out[64i : 64i+64] = (p @ v_win) / sum(p, dim=-1)

This file is the SPEC increment (4a): semantics and CPU reference before running
anything.  No causal mask inside the window yet — the band mask is increment 5.

What is genuinely new here (neither 2 nor 3 covers it)
------------------------------------------------------
1. **One named dim, both kinds, in one scope.**  KV is an OUTPUT dim of the
   scores matmul and a REDUCTION dim of `p @ v_win`, under the same hint.
   ``is_reduction`` is resolved per op, so the two ops should land in different
   halves of the same level — but nothing has exercised that yet.

2. **Reductions over the slid dim on a TILE-LOCAL tensor.**  `max` and `sum`
   reduce over KV, but they read `s`/`p`, which are already per-tile [64, W]
   intermediates — not the KVSEQ-sized K/V.  Their KV axis must NOT slide: it
   is tile-local scratch with no base advance.  The slide must apply to reads of
   the real K/V tensors and to nothing else.  If a per-tile intermediate picks
   up the affine slide, its reads walk off its own buffer.

3. **The softmax restores the numeric witness increment 3 lost.**  There, the
   band was a slice of the untiled product, so wrong tiling could still produce
   right numbers.  Here the softmax NORMALIZES over the window, so windowed
   attention and full attention differ everywhere — `full attention` below is a
   real diagnostic again, and a MATCH means something on its own.  The bundle
   structure is still checked, since alignment and work reduction remain
   separate claims.

Shape constraints (unchanged from increment 2)
----------------------------------------------
  * QSEQ % q_block == 0, KVSEQ % W == 0, QSEQ // q_block == KVSEQ // W
  * S*(N-1) + W <= KVSEQ; all dims multiples of 64.

Note the unread KV tail: with overlap the last window ends before KVSEQ, so the
tail of K/V never participates.  Increment 5 has to size the real op's KV range
so that tail is not real cache content.

Run:
    python3 validate_swa_attention_body.py                 # spec + CPU refs (no HW)
    python3 validate_swa_attention_body.py --sweep
    SENCORES=1 python3 validate_swa_attention_body.py --compile   # on the pod
"""

import argparse
import math
import os
from dataclasses import dataclass

os.environ.setdefault("SENCORES", "1")

import torch  # noqa: E402

STICK = 64
DEFAULT_Q_BLOCK = 64


@dataclass(frozen=True)
class Shape:
    """One windowed-attention configuration.

    ``d`` is the head dim (contracted by the scores matmul, carried through by
    the output matmul); ``window``/``stride`` are the KV read extent and the
    per-iteration advance.
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
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.d)

    @property
    def kv_unread_tail(self) -> int:
        """KV rows past the last window's end — never attended to."""
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
            f"unattended KV tail {self.kv_unread_tail}"
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


def _windowed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    shape: Shape,
    kv_lo_for_tile,
) -> torch.Tensor:
    """Attention where Q block ``i`` attends only to K/V[kv_lo : kv_lo+W]."""
    out = torch.zeros(shape.qseq, shape.d, dtype=torch.float32)
    qf, kf, vf = (t.to(torch.float32) for t in (q, k, v))
    for i in range(shape.num_tiles):
        q_lo = i * shape.q_block
        kv_lo = kv_lo_for_tile(i)
        q_blk = qf[q_lo : q_lo + shape.q_block]
        k_win = kf[kv_lo : kv_lo + shape.window]
        v_win = vf[kv_lo : kv_lo + shape.window]
        s = (q_blk @ k_win.T) * shape.scale
        p = torch.exp(s - s.amax(dim=-1, keepdim=True))
        out[q_lo : q_lo + shape.q_block] = (p @ v_win) / p.sum(dim=-1, keepdim=True)
    return out


def windowed_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Shape
) -> torch.Tensor:
    """THE target: Q block i attends to the sliding window [iS, iS+W)."""
    return _windowed_attention(q, k, v, shape, lambda i: i * shape.stride)


def stuck_kv_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Shape
) -> torch.Tensor:
    """DIAGNOSTIC: the KV window never advances — every block sees [0, W)."""
    return _windowed_attention(q, k, v, shape, lambda _i: 0)


def disjoint_kv_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Shape
) -> torch.Tensor:
    """DIAGNOSTIC: the window advances by W, not S (ordinary partition tiling)."""
    return _windowed_attention(q, k, v, shape, lambda i: i * shape.window)


def full_attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Shape
) -> torch.Tensor:
    """DIAGNOSTIC: no windowing — softmax over all of KVSEQ.

    Unlike increment 3, this IS numerically distinct from the target: the
    softmax normalizes over whatever set of keys it sees, so attending to more
    keys changes every output element.
    """
    qf, kf, vf = (t.to(torch.float32) for t in (q, k, v))
    s = (qf @ kf.T) * shape.scale
    p = torch.exp(s - s.amax(dim=-1, keepdim=True))
    return (p @ vf) / p.sum(dim=-1, keepdim=True)


def _report_refs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Shape
) -> torch.Tensor:
    """Print the target and how far each diagnostic sits from it."""
    ref = windowed_reference(q, k, v, shape)
    print(f"  target  (windowed attn)   peak {ref.abs().max().item():8.3f}")
    for name, alt in (
        ("stuck KV (no slide)", stuck_kv_reference(q, k, v, shape)),
        ("disjoint KV (partition)", disjoint_kv_reference(q, k, v, shape)),
        ("full attention (no window)", full_attention_reference(q, k, v, shape)),
    ):
        sep = (ref - alt).abs().max().item()
        flag = "  <-- INDISTINGUISHABLE" if sep < 1e-3 else ""
        print(
            f"  vs {name:<26} peak {alt.abs().max().item():8.3f}  "
            f"separation {sep:8.3f}{flag}"
        )
    return ref


def _self_check(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Shape
) -> bool:
    """Re-derive one output row with an independent, unfused softmax."""
    ref = windowed_reference(q, k, v, shape)
    qf, kf, vf = (t.to(torch.float32) for t in (q, k, v))
    worst = 0.0
    for i in range(shape.num_tiles):
        row = i * shape.q_block  # first row of each block
        kv_lo = i * shape.stride
        scores = [
            sum(qf[row, t].item() * kf[kv_lo + c, t].item() for t in range(shape.d))
            * shape.scale
            for c in range(shape.window)
        ]
        biggest = max(scores)
        weights = [math.exp(s - biggest) for s in scores]
        denom = sum(weights)
        for col in range(shape.d):
            acc = sum(w * vf[kv_lo + c, col].item() for c, w in enumerate(weights))
            worst = max(worst, abs(acc / denom - ref[row, col].item()))
    ok = worst < 1e-3 * max(ref.abs().max().item(), 1.0)
    print(
        f"  reference self-check: max|vectorized - naive| = {worst:.6f} "
        f"-> {'OK' if ok else 'BROKEN'}"
    )
    return ok


def run_spec(shape: Shape, self_check: bool) -> bool:
    """Spec-only path: validate the shape, build the refs, prove separation."""
    print("=" * 78)
    print(f"  {shape.describe()}")
    shape.validate()
    torch.manual_seed(0xA77E)
    q = torch.randn(shape.qseq, shape.d, dtype=torch.float16)
    k = torch.randn(shape.kvseq, shape.d, dtype=torch.float16)
    v = torch.randn(shape.kvseq, shape.d, dtype=torch.float16)
    _report_refs(q, k, v, shape)
    if not self_check:
        return True
    return _self_check(q, k, v, shape)


def run_compile(shape: Shape, dump: bool) -> bool:
    """Compile the whole windowed body on device and check values + structure."""
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

    torch.manual_seed(0xA77E)
    q = torch.randn(shape.qseq, shape.d, dtype=torch.float16)
    k = torch.randn(shape.kvseq, shape.d, dtype=torch.float16)
    v = torch.randn(shape.kvseq, shape.d, dtype=torch.float16)

    def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # ONE scope over the whole body.  KV is an OUTPUT dim of the scores
        # matmul and a REDUCTION dim of p @ v; the softmax reduces over it on
        # tile-local intermediates that must not pick up the slide.
        with spyre_hint(
            sliding={
                "QS": {"window": shape.q_block, "stride": shape.q_block},
                "KV": {"window": shape.window, "stride": shape.stride},
            }
        ):
            s = torch.matmul(q, k.transpose(0, 1)) * shape.scale
            p = torch.exp(s - torch.amax(s, dim=-1, keepdim=True))
            return torch.matmul(p, v) / torch.sum(p, dim=-1, keepdim=True)

    device = torch.device("spyre")
    q_dev, k_dev, v_dev = q.to(device), k.to(device), v.to(device)
    pnd.declare_tensor_dim("QS", shape.qseq)
    pnd.declare_tensor_dim("KV", shape.kvseq)
    pnd.declare_tensor_dim("D", shape.d)
    pnd.name_tensor_dims(q_dev, ["QS", "D"])
    pnd.name_tensor_dims(k_dev, ["KV", "D"])
    pnd.name_tensor_dims(v_dev, ["KV", "D"])

    before = snapshot()
    try:
        with config.patch({"lx_planning": True, "allow_all_ops_in_lx_planning": True}):
            got = torch.compile(fn)(q_dev, k_dev, v_dev).to("cpu", torch.float32)
    except NotImplementedError as e:
        print(f"  UNSUPPORTED: {e}")
        return False
    except Exception as e:  # noqa: BLE001 — surface compile/codegen failures
        print(f"  COMPILE/RUN FAILED: {type(e).__name__}: {e}")
        return False
    new_dirs = snapshot() - before

    ref = _report_refs(q, k, v, shape)
    tol = max(ref.abs().max().item() * 5e-2, 2e-2)
    max_err = (got - ref).abs().max().item()
    ok = bool(torch.allclose(got, ref, rtol=5e-2, atol=tol))
    print(
        f"  max|got - windowed_ref| = {max_err:.4f} -> {'MATCH' if ok else 'MISMATCH'}"
    )
    if not ok:
        for name, alt in (
            ("stuck KV (slide dropped)", stuck_kv_reference(q, k, v, shape)),
            ("disjoint KV (stride ignored)", disjoint_kv_reference(q, k, v, shape)),
            (
                "full attention (window ignored)",
                full_attention_reference(q, k, v, shape),
            ),
        ):
            if torch.allclose(got, alt, rtol=5e-2, atol=tol):
                print(f"  RESULT: output matches the {name} reference.")
                return False
        print("  RESULT: output matches NO reference — inspect the body's geometry.")

    reduced = structural_report(new_dirs, shape.num_tiles)
    if dump:
        dump_bundles(new_dirs)
    if ok and reduced:
        print("  RESULT: the whole windowed body runs correctly under ONE slid")
        print("  loop.  Only the causal band mask (increment 5) is left.")
    elif ok:
        print("  RESULT: numerically correct but NOT tiled — the body ran whole.")
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
        help="also compile and run the body on device (needs HW)",
    )
    ap.add_argument(
        "--dump", action="store_true", help="print the full bundle.mlir as well"
    )
    ap.add_argument(
        "--no-self-check",
        action="store_true",
        help="skip the naive re-derivation of the reference",
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

    print("SWA increment 4a — full attention body under one slide (spec)")
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
        print("  Spec only — no device run.  Rerun with --compile on the pod;")
        print("  the body composes increments 2 and 3 in one scope, so a")
        print("  failure there is most likely the composition, not either half.")


if __name__ == "__main__":
    main()
