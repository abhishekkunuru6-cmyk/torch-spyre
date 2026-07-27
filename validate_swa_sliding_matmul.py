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

"""SWA integration — Increment 1: sliding read on a MATMUL's reduced dim.

The validated `validate_sliding_prototype.py` proved the sliding hint works for
a plain reduction (`x.sum(dim=0)`) where the slid dim IS the reduction dim.  SWA
is different: the sliding KV window feeds two MATMULs.  Before rewriting the real
decomposition we isolate the decisive unknown here:

    Can a matmul consume an OVERLAPPING (sliding) read on its CONTRACTION dim?

This mirrors SWA's second matmul `exp_scores @ v_blk`, which reduces over the
sliding `kv_len`.  We compute `K @ V` reducing over the KV-sequence dim, with a
sliding hint on that dim, and compare to the coverage-weighted matmul reference

    ref = sum_i  K[:, i*S : i*S+W] @ V[i*S : i*S+W, :]

the exact matmul analog of the reduction prototype's coverage-weighted sum.
BOTH operands slide in lockstep over the shared KV dim (unlike the reduction,
where only one tensor slid) — that lockstep is itself part of what this probes.

Interpreting the result:
  * MATCH                -> a matmul DOES consume a sliding reduced-dim read;
                            the hard half of SWA (exp@V) is de-risked.
  * matches PLAIN K@V    -> the slide did not take effect (no overlap);
                            the hint did not attach to the matmul contraction.
  * neither / crash      -> the sliding mechanism does not compose with matmul
                            reduction codegen; needs a different attach point.

Run:
    SENCORES=1 python3 validate_swa_sliding_matmul.py
    SENCORES=1 python3 validate_swa_sliding_matmul.py --dump    # bundle.mlir
"""

import argparse
import os

os.environ.setdefault("SENCORES", "1")

import regex as re  # noqa: E402, F401
import torch  # noqa: E402
import torch_spyre  # noqa: E402, F401
from torch._inductor.runtime.runtime_utils import cache_dir  # noqa: E402

DEVICE = torch.device("spyre")
BUNDLE_ROOT = os.path.join(cache_dir(), "inductor-spyre")

# Q rows, KV sequence (the slid/reduced dim), D head-dim.  KVSEQ is the sliding
# dim: window W, stride S, num_tiles = KVSEQ // W.  Defaults mirror SWA sticks
# (64) with a 2-tile overlap.
Q = 64
KVSEQ = 256
D = 64
WINDOW = 128
STRIDE = 64


def _snapshot() -> set[str]:
    return set(os.listdir(BUNDLE_ROOT)) if os.path.isdir(BUNDLE_ROOT) else set()


def _dump_bundles(new_dirs: set[str]) -> None:
    for name in sorted(new_dirs):
        mlir = os.path.join(BUNDLE_ROOT, name, "bundle.mlir")
        if not os.path.isfile(mlir):
            continue
        with open(mlir) as f:
            text = f.read()
        if "scf.for" not in text and "matmul" not in text.lower():
            continue
        print(f"    --- {os.path.basename(name)}/bundle.mlir ---")
        for ln in text.splitlines():
            print(f"      {ln}")


def _cpu_overlap_matmul(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Coverage-weighted matmul: sum of the per-window K_window @ V_window.

    PARTITION+SLIDE: num_tiles = KVSEQ // WINDOW windows, each sliding by STRIDE.
    With KVSEQ=256, W=128, S=64 -> 2 windows over KV rows 0:128 and 64:192.
    """
    num_tiles = KVSEQ // WINDOW
    acc = torch.zeros(Q, D, dtype=torch.float32)
    for i in range(num_tiles):
        lo = STRIDE * i
        k_win = k[:, lo : lo + WINDOW].to(torch.float32)
        v_win = v[lo : lo + WINDOW, :].to(torch.float32)
        acc += k_win @ v_win
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", action="store_true", help="dump loop-bearing bundle.mlir")
    args = ap.parse_args()

    assert KVSEQ % WINDOW == 0, f"WINDOW {WINDOW} must divide KVSEQ {KVSEQ}"
    torch.manual_seed(0xAFFE)
    print(f"SENCORES={os.environ.get('SENCORES')}")
    print(
        f"Q={Q} KVSEQ={KVSEQ} D={D} W={WINDOW} S={STRIDE} -> "
        f"{KVSEQ // WINDOW} tiles (partition+slide), overlap {WINDOW - STRIDE}"
    )
    print("=" * 78)

    from torch_spyre._inductor import config, spyre_hint
    import torch_spyre._inductor.propagate_named_dims as _pnd

    _pnd.reset()
    torch._dynamo.reset_code_caches()
    torch._inductor.codecache.FxGraphCache.clear()

    k = torch.randn(Q, KVSEQ, dtype=torch.float16)
    v = torch.randn(KVSEQ, D, dtype=torch.float16)

    def fn(k, v):
        # Slide over the shared KV dim (K's dim 1, V's dim 0) — the matmul
        # contraction.  Both operands slide in lockstep.
        with spyre_hint(sliding={"KV": {"window": WINDOW, "stride": STRIDE}}):
            return torch.matmul(k, v)

    k_dev = k.to(DEVICE)
    v_dev = v.to(DEVICE)
    _pnd.declare_tensor_dim("Q", Q)
    _pnd.declare_tensor_dim("KV", KVSEQ)
    _pnd.declare_tensor_dim("D", D)
    _pnd.name_tensor_dims(k_dev, ["Q", "KV"])
    _pnd.name_tensor_dims(v_dev, ["KV", "D"])

    before = _snapshot()
    try:
        with config.patch({"lx_planning": True, "allow_all_ops_in_lx_planning": True}):
            got = torch.compile(fn)(k_dev, v_dev).to("cpu", torch.float32)
    except Exception as e:  # noqa: BLE001 — surface compile/codegen failures
        print(f"COMPILE/RUN FAILED: {type(e).__name__}: {e}")
        if args.dump:
            _dump_bundles(_snapshot() - before)
        raise

    if args.dump:
        _dump_bundles(_snapshot() - before)

    ref = _cpu_overlap_matmul(k, v)  # coverage-weighted (overlap) matmul
    plain = k.to(torch.float32) @ v.to(torch.float32)  # NON-overlap full K@V
    max_err = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, rtol=3e-2, atol=ref.abs().max().item() * 3e-2)
    is_plain = torch.allclose(
        got, plain, rtol=3e-2, atol=plain.abs().max().item() * 3e-2
    )

    print(
        f"  overlap ref peak: {ref.abs().max().item():.3f}  "
        f"plain K@V peak: {plain.abs().max().item():.3f}  "
        f"got peak: {got.abs().max().item():.3f}"
    )
    print(
        f"  max|got - overlap_ref| = {max_err:.4f} -> {'MATCH' if ok else 'MISMATCH'}"
    )
    print()
    if ok:
        print("  RESULT: a matmul CONSUMES a sliding reduced-dim read correctly.")
        print("  The exp@V half of SWA is de-risked; proceed to increment 2.")
    elif is_plain:
        print("  RESULT: output matches the PLAIN full K@V, not the overlap ref.")
        print("  => the slide did NOT take effect on the matmul contraction dim.")
        print("  The sliding hint likely did not attach to the matmul's reduced")
        print("  dim (check propagate_named_dims gating for matmul reductions).")
    else:
        print("  RESULT: output matches NEITHER reference. The sliding mechanism")
        print("  does not compose with matmul reduction codegen as-is — inspect")
        print("  the bundle (--dump) and the sdsc geometry.")


if __name__ == "__main__":
    main()
