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

"""Read the loop structure back out of a generated ``bundle.mlir``.

A numeric pass proves nothing about tiling: untiled code returns exactly the
right answer, just with one large intermediate. The only way to know whether a
spyre_hint produced a device loop is to look at what was emitted, which is what
this module is for.

It parses the two constructs ``bundle.py`` writes:

    %loop_bound_0 = arith.constant 4 : index
    scf.for %i_0 = %c0 to %loop_bound_0 step %c1 {
        sdscbundle.sdsc_execute (...) {sdsc_filename="...", "symbol_ids"=[...]}
    }

Pure text in, dataclasses out -- no torch, no device -- so the parser itself
can be tested on a literal string.
"""

from dataclasses import dataclass, field
import os
import regex as re

_LOOP_BOUND = re.compile(r"%loop_bound_(\d+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*index")
_LOOP_OPEN = re.compile(r"scf\.for\s+(%\S+)\s*=\s*%\S+\s+to\s+%loop_bound_(\d+)\b")
_EXECUTE = re.compile(r"sdscbundle\.sdsc_execute")
_CLOSE = re.compile(r"^\s*\}\s*$")


@dataclass(frozen=True)
class Loop:
    """One ``scf.for`` in the emitted bundle."""

    bound_index: int
    trip_count: int
    depth: int
    variable: str


@dataclass(frozen=True)
class Execute:
    """One ``sdsc_execute``, and the loops it sits inside."""

    depth: int
    enclosing_trip_counts: tuple[int, ...]

    @property
    def repetitions(self) -> int:
        """How many times this execute runs once the loops are counted."""
        total = 1
        for count in self.enclosing_trip_counts:
            total *= count
        return total


@dataclass(frozen=True)
class BundleStructure:
    """What one ``bundle.mlir`` says about its own loop nest."""

    path: str
    loops: list[Loop] = field(default_factory=list)
    executes: list[Execute] = field(default_factory=list)

    @property
    def executes_inside_loops(self) -> list[Execute]:
        return [e for e in self.executes if e.depth > 0]

    @property
    def executes_at_top_level(self) -> list[Execute]:
        return [e for e in self.executes if e.depth == 0]

    @property
    def max_depth(self) -> int:
        return max((loop.depth + 1 for loop in self.loops), default=0)

    @property
    def trip_counts(self) -> list[int]:
        return [loop.trip_count for loop in self.loops]

    def has_loop_with_trip_count(self, count: int) -> bool:
        return any(loop.trip_count == count for loop in self.loops)

    def report(self) -> str:
        """One human-readable block per bundle, for pasting into a thread."""
        lines = [f"{os.path.basename(os.path.dirname(self.path))}:"]
        if not self.loops:
            lines.append("  NO scf.for -- nothing was tiled into a device loop")
        else:
            for loop in self.loops:
                indent = "  " + "  " * loop.depth
                lines.append(
                    f"{indent}scf.for {loop.variable} trip_count={loop.trip_count}"
                )
        inside = self.executes_inside_loops
        outside = self.executes_at_top_level
        lines.append(f"  sdsc_execute: {len(inside)} inside loops, {len(outside)} flat")
        if inside:
            total = sum(e.repetitions for e in inside)
            lines.append(f"  execute invocations once loops unroll: {total}")
        return "\n".join(lines)


def parse_bundle_mlir(text: str, path: str = "<string>") -> BundleStructure:
    """Parse the loop nest out of ``bundle.mlir`` text.

    Depth is tracked from the ``scf.for ... {`` lines and the bare ``}`` lines
    that close them; every other brace ``bundle.py`` writes is on a line that
    also carries content, so a bare close brace is unambiguous.
    """
    bounds: dict[int, int] = {}
    for match in _LOOP_BOUND.finditer(text):
        bounds[int(match.group(1))] = int(match.group(2))

    loops: list[Loop] = []
    executes: list[Execute] = []
    open_loops: list[Loop] = []

    for line in text.splitlines():
        opened = _LOOP_OPEN.search(line)
        if opened is not None:
            bound_index = int(opened.group(2))
            loop = Loop(
                bound_index=bound_index,
                trip_count=bounds.get(bound_index, 0),
                depth=len(open_loops),
                variable=opened.group(1),
            )
            loops.append(loop)
            open_loops.append(loop)
            continue
        if _EXECUTE.search(line) is not None:
            executes.append(
                Execute(
                    depth=len(open_loops),
                    enclosing_trip_counts=tuple(x.trip_count for x in open_loops),
                )
            )
            continue
        if _CLOSE.match(line) and open_loops:
            open_loops.pop()

    return BundleStructure(path=path, loops=loops, executes=executes)


def read_bundle(path: str) -> BundleStructure:
    """Parse one ``bundle.mlir`` from disk."""
    with open(path) as handle:
        return parse_bundle_mlir(handle.read(), path=path)


def find_bundles_since(roots, timestamp: float) -> list[BundleStructure]:
    """Parse every bundle.mlir under ``roots`` written at or after ``timestamp``.

    Selection is by mtime across several roots rather than by set-difference on
    one, because the root itself moves: disabling inductor's caches sends
    cache_dir() to a fresh temporary directory, so a root sampled before the
    compile is not where the bundles land. Pass both the before and after
    roots and let the timestamp decide.

    Tolerant of a directory with no bundle.mlir: dxp_standalone runs *after*
    generate_bundle, so a failed backend compile still leaves a parseable
    bundle -- which is exactly when its structure is most worth reading.
    """
    structures = []
    seen: set[str] = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            candidate = os.path.join(root, name, "bundle.mlir")
            if candidate in seen or not os.path.isfile(candidate):
                continue
            if os.path.getmtime(candidate) >= timestamp:
                seen.add(candidate)
                structures.append(read_bundle(candidate))
    return structures
