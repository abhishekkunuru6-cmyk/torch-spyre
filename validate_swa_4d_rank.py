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

"""SWA integration — Increment 5d-pre: 4-D rank and GQA under the slide.

Increments 1-5c all ran at rank 2 (`[Lq, D]`).  The real op is rank 4,
`[B, H, Lq, D]`, so before rewriting the decomposition this isolates the last
mechanism unknowns:

1. **Leading untiled dims.**  B and H are named but NOT slid.  Every op in the
   body must be loop-invariant in them while QS partitions and KV follows —
   the "broadcast at this level" path in coarse_tile, which no probe has
   exercised alongside a slide.

2. **Batch matmul instead of mm.**  At rank 4 the two matmuls lower to
   BATCH_MATMUL_OP, a different codegen path from the rank-2 `mm` every earlier
   increment used.  The per-tensor affine strides have to come out right with a
   batch dim in front.

3. **A broadcast mask.**  The mask is `[1, 1, Lq, padded_kv]`: slid in its last
   two dims, broadcast over the first two.  One tensor that is both, which is
   new.

4. **GQA (optional).**  `key`/`value` carry `num_kvheads < num_heads` and are
   expanded with `unsqueeze/expand/flatten` before the matmul.  The slide then
   has to survive onto a broadcast VIEW of the tensor whose KV dim it names,
   rather than onto the tensor itself.  This is the piece most likely to break;
   `--sweep` includes expansion=2 shapes, and `--no-gqa` skips them.

Both the body's shape and where the GQA expand sits deliberately MIRROR
`decompositions.py`, because the first run of this probe diverged from it in two
places and both diverged into failures that the real op does not have:

  * `amax`/`sum` use no ``keepdim``; the result is unsqueezed at the point of
    use.  With ``keepdim=True`` the reduction output is a rank-4
    ``[B, H, Lq, 1]`` buffer, and when ``B == 1`` that leaves
    ``_resize_device_layout`` two indistinguishable size-1 host dims to place —
    it fails outright.  The real op reduces to rank 3 and never creates that
    buffer.
  * the GQA expand happens OUTSIDE the hint scope, as the real op does.

The window arithmetic is imported from `validate_swa_real_shapes` rather than
re-derived — the causal/non-causal range rounding is fiddly enough that having
one implementation is worth the coupling, and 5a's version is the verified one.

The rank-2 result is the reference for the reference: with B=H=1 the 4-D CPU
model must reproduce 5a's `padded_reference` exactly, which catches a bad 4-D
reference before it can excuse a bad compile.

Run:
    python3 validate_swa_4d_rank.py                 # spec + CPU refs (no HW)
    python3 validate_swa_4d_rank.py --sweep
    SENCORES=1 python3 validate_swa_4d_rank.py --compile   # on the pod
"""

import argparse
import os
from dataclasses import dataclass

os.environ.setdefault("SENCORES", "1")

import torch  # noqa: E402

from validate_swa_real_shapes import (  # noqa: E402
    NEG_INF,
    SwaShape,
    band_mask,
    padded_reference,
)


def pad_kv_nd(t: torch.Tensor, swa: SwaShape) -> torch.Tensor:
    """Pad the SEQUENCE axis (-2) of a K/V tensor of any leading rank.

    5a's ``pad_kv`` pads dim 0, which is the sequence only at rank 2; at rank 4
    that axis is the head dim.  Padded rows are always masked out.
    """
    if not (swa.left_pad or swa.right_pad):
        return t
    lead, width = t.shape[:-2], t.shape[-1]
    parts = []
    if swa.left_pad:
        parts.append(torch.zeros(*lead, swa.left_pad, width, dtype=t.dtype))
    parts.append(t)
    if swa.right_pad:
        parts.append(torch.zeros(*lead, swa.right_pad, width, dtype=t.dtype))
    return torch.cat(parts, dim=-2)


@dataclass(frozen=True)
class Rank4Shape:
    """A rank-4 SWA configuration: ``[batch, heads, seqlen, head_dim]``."""

    batch: int
    heads: int
    kv_heads: int
    swa: SwaShape

    @property
    def expansion(self) -> int:
        return self.heads // self.kv_heads

    @property
    def is_gqa(self) -> bool:
        return self.expansion != 1

    def validate(self) -> None:
        self.swa.validate()
        if self.heads % self.kv_heads:
            raise ValueError(
                f"heads {self.heads} must be a multiple of kv_heads {self.kv_heads}"
            )

    def describe(self) -> str:
        gqa = f" GQA x{self.expansion}" if self.is_gqa else ""
        return f"B={self.batch} H={self.heads} KVH={self.kv_heads}{gqa} | {self.swa.describe()}"


def _swa(seqlen: int = 256, window: int = 128, head_dim: int = 64) -> SwaShape:
    return SwaShape(
        seqlen_q=seqlen,
        seqlen_kv=seqlen,
        head_dim=head_dim,
        window_size=window,
    )


# Shape sizes are load-bearing here, not arbitrary.  Both failures the first HW
# runs found were dim-identification AMBIGUITIES rather than slide problems, and
# an ambiguity only exists when two dims share a size — so the sweep deliberately
# includes colliding and non-colliding variants of each.
SWEEP_SHAPES: tuple[Rank4Shape, ...] = (
    # B == H == 1: two size-1 leading dims, which _resize_device_layout cannot
    # tell apart.  Degenerate (real models have many heads) but kept as the
    # documented lower bound of what works.
    Rank4Shape(batch=1, heads=1, kv_heads=1, swa=_swa()),
    Rank4Shape(batch=1, heads=2, kv_heads=2, swa=_swa()),
    Rank4Shape(batch=2, heads=2, kv_heads=2, swa=_swa()),
    Rank4Shape(batch=1, heads=4, kv_heads=2, swa=_swa()),  # GQA x2, B breaks the tie
    # B == kv_heads == expansion == 2: every dim the same size.  This is the one
    # that produced a wrong answer with the right loop structure.
    Rank4Shape(batch=2, heads=4, kv_heads=2, swa=_swa()),  # GQA x2, all sizes 2
    # Same GQA, but B != expansion.  If the collision hypothesis is right these
    # pass while the one above fails; if they also fail, GQA is broken for
    # batch > 1 generally and the size collision is a red herring.
    Rank4Shape(batch=2, heads=8, kv_heads=2, swa=_swa()),  # GQA x4, B=2 != 4
    Rank4Shape(batch=4, heads=4, kv_heads=2, swa=_swa()),  # GQA x2, B=4 != 2
    Rank4Shape(batch=1, heads=2, kv_heads=2, swa=_swa(seqlen=512, window=256)),
)

DEFAULT_SHAPE = SWEEP_SHAPES[1]


def build_mask(swa: SwaShape, padded_kv: int) -> torch.Tensor:
    """The full ``[1, 1, Lq, padded_kv]`` additive mask, broadcast over B and H.

    Full width, not the compact ``[Lq, W]`` band: the untiled carrier computes
    ``q @ kT`` at full width, so that is the only shape the add typechecks at.
    Everything outside the band is -inf and never read.
    """
    mask = torch.full((swa.seqlen_q, padded_kv), NEG_INF, dtype=torch.float32)
    for qi in range(swa.num_q_blocks):
        q_lo = qi * swa.q_block
        kv_lo = swa.window_lo(qi) + swa.left_pad
        mask[q_lo : q_lo + swa.q_block, kv_lo : kv_lo + swa.read_extent] = band_mask(
            swa, qi
        )
    return mask.unsqueeze(0).unsqueeze(0)


def _attend_4d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    shape: Rank4Shape,
    kv_lo_for_tile,
) -> torch.Tensor:
    """Rank-4 windowed attention; head ``h`` reads kv head ``h // expansion``."""
    swa = shape.swa
    qf = q.to(torch.float32)
    kf = pad_kv_nd(k.to(torch.float32), swa)
    vf = pad_kv_nd(v.to(torch.float32), swa)
    out = torch.zeros(
        shape.batch, shape.heads, swa.seqlen_q, swa.head_dim, dtype=torch.float32
    )
    for b in range(shape.batch):
        for h in range(shape.heads):
            kvh = h // shape.expansion
            for qi in range(swa.num_q_blocks):
                q_lo = qi * swa.q_block
                lo = kv_lo_for_tile(qi)
                s = (
                    qf[b, h, q_lo : q_lo + swa.q_block]
                    @ kf[b, kvh, lo : lo + swa.read_extent].T
                ) * swa.scale + band_mask(swa, qi)
                biggest = s.amax(dim=-1, keepdim=True)
                biggest = torch.where(
                    torch.isinf(biggest), torch.zeros_like(biggest), biggest
                )
                p = torch.exp(s - biggest)
                denom = p.sum(dim=-1, keepdim=True)
                blk = p @ vf[b, kvh, lo : lo + swa.read_extent]
                out[b, h, q_lo : q_lo + swa.q_block] = torch.where(
                    denom > 0, blk / denom.clamp(min=1e-30), torch.zeros_like(blk)
                )
    return out


def windowed_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Rank4Shape
) -> torch.Tensor:
    """THE target: every (b, h) runs the verified rank-2 body on its own slice."""
    swa = shape.swa
    return _attend_4d(q, k, v, shape, lambda qi: swa.window_lo(qi) + swa.left_pad)


def stuck_kv_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Rank4Shape
) -> torch.Tensor:
    """DIAGNOSTIC: the KV window never advances — every block reads window 0."""
    swa = shape.swa
    return _attend_4d(q, k, v, shape, lambda _qi: swa.window_lo(0) + swa.left_pad)


def head_collapse_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Rank4Shape
) -> torch.Tensor:
    """DIAGNOSTIC: every query head reads KV head 0.

    The GQA failure that matters: an expansion that broadcasts the wrong kv head
    still produces plausible numbers, so it needs its own reference.  Identical
    to the target when kv_heads == 1, and flagged as such.
    """
    swa = shape.swa
    collapsed = Rank4Shape(
        batch=shape.batch, heads=shape.heads, kv_heads=shape.heads, swa=swa
    )
    k0 = k[:, :1].expand(-1, shape.heads, -1, -1).contiguous()
    v0 = v[:, :1].expand(-1, shape.heads, -1, -1).contiguous()
    return _attend_4d(q, k0, v0, collapsed, lambda qi: swa.window_lo(qi) + swa.left_pad)


def _rank2_agreement(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Rank4Shape
) -> float:
    """Largest gap between the 4-D model and 5a's verified rank-2 one.

    Checks the reference itself: run per (b, h) slice, the rank-4 body IS the
    rank-2 body, so any difference is a bug in this file rather than in the
    compiler.
    """
    swa = shape.swa
    worst = 0.0
    ref = windowed_reference(q, k, v, shape)
    for b in range(shape.batch):
        for h in range(shape.heads):
            kvh = h // shape.expansion
            flat = padded_reference(q[b, h], k[b, kvh], v[b, kvh], swa)
            worst = max(worst, (ref[b, h] - flat).abs().max().item())
    return worst


def _report(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Rank4Shape
) -> torch.Tensor:
    ref = windowed_reference(q, k, v, shape)
    print(f"  target  (rank-4 windowed) peak {ref.abs().max().item():8.3f}")
    diagnostics = [("stuck KV (no slide)", stuck_kv_reference(q, k, v, shape))]
    if shape.kv_heads > 1:
        diagnostics.append(
            ("all heads read KV head 0", head_collapse_reference(q, k, v, shape))
        )
    for name, alt in diagnostics:
        sep = (ref - alt).abs().max().item()
        flag = "  <-- INDISTINGUISHABLE" if sep < 1e-3 else ""
        print(
            f"  vs {name:<26} peak {alt.abs().max().item():8.3f}  "
            f"separation {sep:8.3f}{flag}"
        )
    return ref


def _inputs(shape: Rank4Shape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0x4D4D)
    swa = shape.swa
    q = torch.randn(
        shape.batch, shape.heads, swa.seqlen_q, swa.head_dim, dtype=torch.float16
    )
    k = torch.randn(
        shape.batch, shape.kv_heads, swa.seqlen_kv, swa.head_dim, dtype=torch.float16
    )
    v = torch.randn(
        shape.batch, shape.kv_heads, swa.seqlen_kv, swa.head_dim, dtype=torch.float16
    )
    return q, k, v


def run_spec(shape: Rank4Shape) -> bool:
    """Spec-only: check the 4-D reference against the verified rank-2 one."""
    print("=" * 78)
    print(f"  {shape.describe()}")
    shape.validate()
    q, k, v = _inputs(shape)
    _report(q, k, v, shape)
    gap = _rank2_agreement(q, k, v, shape)
    ok = gap < 1e-4
    print(
        f"  reference check vs rank-2 padded_reference: max gap {gap:.6f} "
        f"-> {'OK' if ok else 'BROKEN'}"
    )
    return ok


def run_compile(shape: Rank4Shape, dump: bool, gqa_naming: str = "explicit") -> bool:
    """Compile the rank-4 body on device and check values + loop structure.

    ``gqa_naming`` selects how the GQA-expanded k/v get their named dims:
    ``"explicit"`` annotates the materialized result, ``"implicit"`` leaves it
    to propagation (the configuration that produced ~1e9 errors at batch > 1).
    """
    import torch_spyre  # noqa: F401
    import torch_spyre._inductor.propagate_named_dims as pnd
    from torch_spyre._inductor import config, spyre_hint

    from swa_probe_bundle import dump_bundles, snapshot, structural_report

    swa = shape.swa
    print("=" * 78)
    print(f"  COMPILE  {shape.describe()}")
    shape.validate()
    pnd.reset()
    torch._dynamo.reset_code_caches()
    torch._inductor.codecache.FxGraphCache.clear()

    q, k, v = _inputs(shape)
    k_pad = pad_kv_nd(k, swa)
    v_pad = pad_kv_nd(v, swa)
    padded_kv = k_pad.shape[-2]
    mask = build_mask(swa, padded_kv).to(torch.float16)
    expansion = shape.expansion

    def _gqa_expand(t):
        """Broadcast each kv head to its query heads, naming the result.

        The expand rewrites the head dim from kv_heads to kv_heads*expansion.
        Name propagation carries the SOURCE tensor's name list onto the derived
        buffer and matches names to layout extents by product prefix
        (_consume_names), so "HKV" (size kv_heads) no longer matches the new
        head extent and the whole list shifts by an axis — that is the
        `_untracked_N` warning, and it lands "KV" on the wrong dim, which the
        slide then strides through.  Naming the materialized result explicitly
        removes the guess.  ``.contiguous()`` inside the scope forces a rank-4
        buffer for the hint to name, following the flash decomposition's
        pattern in test_coarse_tile_e2e.py.
        """
        t = t.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        if gqa_naming == "explicit":
            with spyre_hint(named_dims=["B", "H", "KV", "D"]):
                t = t.contiguous()
        return t

    def fn(q, k, v, mask):
        # GQA expansion happens OUTSIDE the sliding scope, matching the real
        # decomposition (decompositions.py builds k_blk/v_blk before entering
        # its spyre_hint blocks).  Inside the scope the expanded tensors are
        # just ordinary inputs.
        if expansion != 1:
            k = _gqa_expand(k)
            v = _gqa_expand(v)
        with spyre_hint(
            sliding={
                "QS": {"window": swa.q_block, "stride": swa.q_block},
                "KV": {
                    "window": swa.read_extent,
                    "stride": swa.q_block,
                    "counts_tiles": False,
                },
            }
        ):
            s = torch.matmul(q, k.transpose(-1, -2)) * swa.scale + mask
            # amax/sum WITHOUT keepdim, unsqueezed at the point of use — the
            # real decomposition's formulation.  keepdim=True materializes a
            # rank-4 [B, H, Lq, 1] reduction output, and when B == 1 that gives
            # _resize_device_layout two indistinguishable size-1 host dims to
            # place, which it cannot do (see issue #3116's stick-dim recovery).
            m = torch.amax(s, dim=-1)
            p = torch.exp(s - m.unsqueeze(-1))
            denom = torch.sum(p, dim=-1)
            return torch.matmul(p, v) / denom.unsqueeze(-1)

    device = torch.device("spyre")
    q_dev, k_dev, v_dev = q.to(device), k_pad.to(device), v_pad.to(device)
    m_dev = mask.to(device)
    head_name = "H" if not shape.is_gqa else "HKV"
    pnd.declare_tensor_dim("B", shape.batch)
    pnd.declare_tensor_dim("H", shape.heads)
    pnd.declare_tensor_dim("HKV", shape.kv_heads)
    pnd.declare_tensor_dim("ONE", 1)
    pnd.declare_tensor_dim("QS", swa.seqlen_q)
    pnd.declare_tensor_dim("KV", padded_kv)
    pnd.declare_tensor_dim("D", swa.head_dim)
    pnd.name_tensor_dims(q_dev, ["B", "H", "QS", "D"])
    pnd.name_tensor_dims(k_dev, ["B", head_name, "KV", "D"])
    pnd.name_tensor_dims(v_dev, ["B", head_name, "KV", "D"])
    # Broadcast over B and H, slid in QS and KV — one tensor that is both.
    pnd.name_tensor_dims(m_dev, ["ONE", "ONE", "QS", "KV"])

    before = snapshot()
    try:
        with config.patch({"lx_planning": True, "allow_all_ops_in_lx_planning": True}):
            got = torch.compile(fn)(q_dev, k_dev, v_dev, m_dev).to("cpu", torch.float32)
    except NotImplementedError as e:
        print(f"  UNSUPPORTED: {e}")
        return False
    except Exception as e:  # noqa: BLE001 — surface compile/codegen failures
        print(f"  COMPILE/RUN FAILED: {type(e).__name__}: {e}")
        return False
    new_dirs = snapshot() - before

    ref = windowed_reference(q, k, v, shape)
    tol = max(ref.abs().max().item() * 5e-2, 2e-2)
    max_err = (got - ref).abs().max().item()
    ok = bool(torch.allclose(got, ref, rtol=5e-2, atol=tol))
    print(f"  max|got - rank4_ref| = {max_err:.4f} -> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        alts = [("stuck KV (slide dropped)", stuck_kv_reference(q, k, v, shape))]
        if shape.kv_heads > 1:
            alts.append(
                ("all heads read KV head 0", head_collapse_reference(q, k, v, shape))
            )
        for name, alt in alts:
            if torch.allclose(got, alt, rtol=5e-2, atol=tol):
                print(f"  RESULT: output matches the {name} reference.")
                return False
        print("  RESULT: output matches NO reference — inspect the batch geometry.")

    reduced = structural_report(new_dirs, swa.num_q_blocks)
    if dump:
        dump_bundles(new_dirs)
    return ok and reduced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=DEFAULT_SHAPE.batch)
    ap.add_argument("--heads", type=int, default=DEFAULT_SHAPE.heads)
    ap.add_argument("--kv-heads", type=int, default=DEFAULT_SHAPE.kv_heads)
    ap.add_argument("--seqlen", type=int, default=DEFAULT_SHAPE.swa.seqlen_q)
    ap.add_argument("--window-size", type=int, default=DEFAULT_SHAPE.swa.window_size)
    ap.add_argument("--head-dim", type=int, default=DEFAULT_SHAPE.swa.head_dim)
    ap.add_argument(
        "--sweep", action="store_true", help="run every shape in SWEEP_SHAPES"
    )
    ap.add_argument("--no-gqa", action="store_true", help="skip the GQA shapes")
    ap.add_argument(
        "--compile", action="store_true", help="also compile on device (needs HW)"
    )
    ap.add_argument(
        "--dump", action="store_true", help="print the full bundle.mlir as well"
    )
    ap.add_argument(
        "--gqa-naming",
        choices=("explicit", "implicit"),
        default="explicit",
        help="name the GQA-expanded k/v explicitly (default) or leave it to "
        "propagation ('implicit' reproduces the ~1e9 batch>1 failures)",
    )
    args = ap.parse_args()

    shapes = (
        SWEEP_SHAPES
        if args.sweep
        else (
            Rank4Shape(
                batch=args.batch,
                heads=args.heads,
                kv_heads=args.kv_heads,
                swa=_swa(
                    seqlen=args.seqlen,
                    window=args.window_size,
                    head_dim=args.head_dim,
                ),
            ),
        )
    )
    if args.no_gqa:
        shapes = tuple(s for s in shapes if not s.is_gqa)

    print("SWA increment 5d-pre — 4-D rank and GQA under the slide")
    results = []
    for shape in shapes:
        ok = run_spec(shape)
        if args.compile:
            ok = run_compile(shape, dump=args.dump, gqa_naming=args.gqa_naming) and ok
        results.append((shape, ok))

    print("=" * 78)
    print("SUMMARY:")
    for shape, ok in results:
        print(
            f"  B={shape.batch} H={shape.heads} KVH={shape.kv_heads} "
            f"Lq={shape.swa.seqlen_q:>4} win={shape.swa.window_size:>4}  "
            f"{'PASS' if ok else 'FAIL'}"
        )
    n_pass = sum(1 for _, ok in results if ok)
    print(f"  {n_pass}/{len(results)} shapes")
    print()
    print("  Sizes are load-bearing: a dim-identification ambiguity exists only")
    print("  when two dims share a size.  Compare B=2/KVH=2/exp=2 (all the same)")
    print("  against B=2/exp=4 and B=4/exp=2 to tell a size collision apart from")
    print("  GQA simply being broken for batch > 1.")
    if not args.compile:
        print()
        print("  Spec only — the 4-D CPU model agrees with 5a's verified rank-2")
        print("  one.  Rerun with --compile on the pod for the real unknowns:")
        print("  batch matmul, broadcast mask, and the GQA expand view.")


if __name__ == "__main__":
    main()
