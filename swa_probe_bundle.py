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

"""Bundle inspection shared by the SWA sliding-window probes.

Some probes cannot prove what they need from numbers alone.  When a slide
changes only WHICH elements are computed and not their values (`q @ kT`'s score
band is a slice of the untiled product), a compile that ignored the hint
entirely still produces the right numbers.  The evidence for work reduction is
structural and lives in ``bundle.mlir``: one ``scf.for`` with the expected trip
count, and ONE ``sdsc_execute`` inside it rather than N unrolled ones.

``structural_report`` extracts exactly that, so a probe can assert it instead of
asking a human to eyeball a dump.
"""

import os
from dataclasses import dataclass, field

from torch._inductor.runtime.runtime_utils import cache_dir

BUNDLE_ROOT = os.path.join(cache_dir(), "inductor-spyre")


def snapshot() -> set[str]:
    """Names of the bundle directories that exist right now."""
    return set(os.listdir(BUNDLE_ROOT)) if os.path.isdir(BUNDLE_ROOT) else set()


@dataclass(frozen=True)
class BundleStructure:
    """What ``bundle.mlir`` says about the emitted loop nest.

    ``executes_in_loop`` vs ``executes_outside`` is the decisive pair: the
    unhinted compile emits one op outside any loop, the tiled compile emits one
    op inside a loop that runs ``trip_counts[0]`` times.
    """

    path: str
    trip_counts: list[int] = field(default_factory=list)
    executes_in_loop: int = 0
    executes_outside: int = 0
    affine_maps: list[str] = field(default_factory=list)

    @property
    def looped(self) -> bool:
        return bool(self.trip_counts) and self.executes_in_loop > 0

    def describe(self) -> str:
        return (
            f"scf.for trip counts {self.trip_counts or '[]'}; "
            f"sdsc_execute {self.executes_in_loop} inside / "
            f"{self.executes_outside} outside; "
            f"{len(self.affine_maps)} affine map(s)"
        )


def _parse_structure(path: str) -> BundleStructure:
    with open(path) as f:
        text = f.read()

    bounds: dict[str, int] = {}
    affine_maps: list[str] = []
    depth = 0
    in_loop = 0
    outside = 0
    order: list[int] = []

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#map_"):
            affine_maps.append(line)
            continue
        if line.startswith("%loop_bound_"):
            name, _, rest = line.partition(" = ")
            # "%loop_bound_0 = arith.constant 4 : index"
            parts = rest.split()
            if len(parts) >= 2 and parts[0] == "arith.constant":
                bounds[name] = int(parts[1])
            continue
        if line.startswith("scf.for"):
            depth += 1
            for name in bounds:
                if name in line:
                    order.append(bounds[name])
            continue
        if line == "}" and depth > 0:
            depth -= 1
            continue
        if line.startswith("sdscbundle.sdsc_execute"):
            if depth > 0:
                in_loop += 1
            else:
                outside += 1

    return BundleStructure(
        path=path,
        trip_counts=order,
        executes_in_loop=in_loop,
        executes_outside=outside,
        affine_maps=affine_maps,
    )


def structural_report(new_dirs: set[str], expect_trip_count: int) -> bool:
    """Print what each new bundle's MLIR says and check it against expectations.

    Returns True when some bundle shows a loop whose outermost trip count is
    ``expect_trip_count`` carrying its work inside — i.e. the hint produced a
    real loop instead of one full-size op.
    """
    reports = []
    for name in sorted(new_dirs):
        mlir = os.path.join(BUNDLE_ROOT, name, "bundle.mlir")
        if os.path.isfile(mlir):
            reports.append(_parse_structure(mlir))

    if not reports:
        print("  structure: no bundle.mlir found — cannot check work reduction")
        return False

    ok = False
    for r in reports:
        matched = r.looped and r.trip_counts[0] == expect_trip_count
        ok = ok or matched
        print(
            f"  structure [{os.path.basename(os.path.dirname(r.path))}]: {r.describe()}"
        )
        for m in r.affine_maps:
            print(f"    {m}")

    if ok:
        print(
            f"  -> WORK REDUCED: one loop of {expect_trip_count} iterations "
            "carries the compute; the hint produced a real scf.for."
        )
    else:
        print(
            f"  -> NOT REDUCED: no loop with trip count {expect_trip_count} "
            "wrapping the compute.  The numbers may still be correct — the "
            "compiler can compute the full result and match the reference — "
            "but the tiling did not take effect."
        )
    return ok


def dump_bundles(new_dirs: set[str]) -> None:
    """Print the full MLIR of every new loop-bearing bundle."""
    for name in sorted(new_dirs):
        mlir = os.path.join(BUNDLE_ROOT, name, "bundle.mlir")
        if not os.path.isfile(mlir):
            continue
        with open(mlir) as f:
            text = f.read()
        if "scf.for" not in text:
            continue
        print(f"    --- {name}/bundle.mlir ---")
        for ln in text.splitlines():
            print(f"      {ln}")
