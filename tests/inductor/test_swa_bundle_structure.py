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

"""Increment R6: does the window hint actually produce a device loop?

Every numeric test of sliding-window attention can pass with nothing tiled at
all -- untiled code returns the right answer, just with one large intermediate.
This file looks at the emitted bundle.mlir instead of the numbers, and it runs
the SAME check against both paths so they can be compared directly:

  - the unrolled fallback (SPYRE_SWA_WINDOW_ROLL off), which slices a
    variable-width KV range per Q block
  - the rolled window (on), which reads a constant buffer_width per block

The claim under test is that spyre_hint(tiles={"window_size": Wb // 64})
becomes an scf.for over the window. Recorded from earlier sessions: compiling
the unrolled decomposition emitted ZERO scf.for at seqlen 2048, ~40 flat
sdsc_execute groups. If that is still true for both paths, then windowed
tiling is not working for EITHER design, and that finding matters more than
any further body variant.

The report is printed whether or not the assertions hold -- read the output,
do not just read pass/fail.

Run on hardware:
    SENCORES=1 python3 -m pytest tests/inductor/test_swa_bundle_structure.py -v -s
"""

import os
import time
import unittest

import torch
import torch._dynamo

from bundle_structure import find_bundles_since, parse_bundle_mlir
from inductor_cache_isolation import isolated_inductor_cache
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor import decompositions
from torch_spyre._inductor.swa_window_gather import plan_window_gather
from utils_inductor import cached_randn

BATCH = 1
HEADS = 8
SEQLEN = 256
HEAD_DIM = 64
WINDOW = 64

SAMPLE_MLIR = """\
module {
\tfunc.func @sdsc_bundle() {
\t\t%c0 = arith.constant 0 : index
\t\t%c1 = arith.constant 1 : index
\t\t%loop_bound_0 = arith.constant 4 : index
\t\t%loop_bound_1 = arith.constant 2 : index
\t\tsdscbundle.sdsc_execute (%sym_1) {sdsc_filename="a.json", "symbol_ids"=[-1]}
\t\tscf.for %i_0 = %c0 to %loop_bound_0 step %c1 {
\t\t\tscf.for %i_1 = %c0 to %loop_bound_1 step %c1 {
\t\t\t\tsdscbundle.sdsc_execute (%addr_0) {sdsc_filename="b.json", "symbol_ids"=[-2]}
\t\t\t}
\t\t}
\t\treturn
\t}
}
"""


def _inputs(device):
    query = cached_randn(
        (BATCH, HEADS, SEQLEN, HEAD_DIM), differentiation=1, dtype=torch.float16
    )
    key = cached_randn(
        (BATCH, HEADS, SEQLEN, HEAD_DIM), differentiation=2, dtype=torch.float16
    )
    value = cached_randn(
        (BATCH, HEADS, SEQLEN, HEAD_DIM), differentiation=3, dtype=torch.float16
    )
    return tuple(t.to(device) for t in (query, key, value))


def _compile_swa(roll_enabled: bool):
    """Compile the SWA op on one path; return (bundles, roll_branch_entries).

    Both paths emitting identical structure is a result that could equally
    mean "the flag changed nothing in the graph", so plan_window_gather is
    counted while tracing: it is called only under `if config.swa_window_roll`,
    so a nonzero count is proof the rolled branch was actually taken. Without
    that, a null result here cannot be told apart from a flag that never
    reached the decomposition.

    The compile also gets a private Inductor cache. The flag is read during
    lowering, after the FX graph cache key is computed, so both paths share a
    key and the second compile would otherwise replay the first's artifact --
    which is what made the first run's two paths report identical structure.

    A backend compile failure is caught, not raised: generate_bundle writes
    bundle.mlir before dxp_standalone runs, so a failed compile still leaves
    structure worth reading.
    """
    started = time.time()

    entries: list = []
    original_plan = decompositions.plan_window_gather

    def counting_plan(*args, **kwargs):
        plan = original_plan(*args, **kwargs)
        entries.append(plan)
        return plan

    saved_flag = spyre_config.swa_window_roll
    spyre_config.swa_window_roll = roll_enabled
    decompositions.plan_window_gather = counting_plan
    torch._dynamo.reset()
    try:
        with isolated_inductor_cache() as cache_root:
            query, key, value = _inputs("spyre")

            def fn(q, k, v):
                return torch.ops.spyre.sliding_window_attention(q, k, v, WINDOW, True)

            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            try:
                compiled(query, key, value)
            except Exception as exc:  # noqa: BLE001 -- structure still parseable
                print(
                    f"\n[compile raised, reading structure anyway] "
                    f"{type(exc).__name__}: {exc}"
                )
            root = os.path.join(cache_root, "inductor-spyre")
            structures = find_bundles_since({root}, started)
    finally:
        spyre_config.swa_window_roll = saved_flag
        decompositions.plan_window_gather = original_plan
        torch._dynamo.reset()

    return structures, len(entries)


def _print_report(label: str, structures, roll_entries: int) -> None:
    plan = plan_window_gather(SEQLEN, SEQLEN, WINDOW)
    expected = None if plan is None else max(1, plan.buffer_width // 64)
    print(f"\n=== {label} ===")
    print(
        f"bundles emitted: {len(structures)}  (expected window trip count: {expected})"
    )
    print(f"rolled branch entered: {roll_entries} time(s) during tracing")
    for structure in structures:
        print(structure.report())
    loops = sum(len(s.loops) for s in structures)
    flat = sum(len(s.executes_at_top_level) for s in structures)
    inside = sum(len(s.executes_inside_loops) for s in structures)
    print(f"TOTAL: {loops} scf.for, {inside} execute inside loops, {flat} flat")


class TestBundleParser(unittest.TestCase):
    """The parser itself, on a literal bundle -- no device needed."""

    def setUp(self):
        self.structure = parse_bundle_mlir(SAMPLE_MLIR)

    def test_finds_both_loops_with_trip_counts(self):
        self.assertEqual(self.structure.trip_counts, [4, 2])

    def test_tracks_nesting_depth(self):
        self.assertEqual([loop.depth for loop in self.structure.loops], [0, 1])
        self.assertEqual(self.structure.max_depth, 2)

    def test_separates_flat_executes_from_looped_ones(self):
        self.assertEqual(len(self.structure.executes_at_top_level), 1)
        self.assertEqual(len(self.structure.executes_inside_loops), 1)

    def test_counts_repetitions_through_the_nest(self):
        self.assertEqual(self.structure.executes_inside_loops[0].repetitions, 8)

    def test_a_bundle_with_no_loops_reports_so(self):
        flat = parse_bundle_mlir(
            '\tsdscbundle.sdsc_execute (%sym_1) {sdsc_filename="a.json"}\n'
        )
        self.assertEqual(flat.loops, [])
        self.assertIn("NO scf.for", flat.report())


class TestWindowTilingIsReal(unittest.TestCase):
    """[HW] Compile both paths and compare their emitted loop structure."""

    def test_the_flag_actually_changes_what_is_traced(self):
        # Guards the two tests below: a null result there means nothing if the
        # flag never reached the decomposition.
        _, rolled_entries = _compile_swa(roll_enabled=True)
        _, fallback_entries = _compile_swa(roll_enabled=False)
        self.assertGreater(
            rolled_entries, 0, "the rolled branch was never entered with the flag ON"
        )
        self.assertEqual(
            fallback_entries, 0, "the rolled branch was entered with the flag OFF"
        )

    def test_rolled_path_emits_a_window_loop(self):
        structures, entries = _compile_swa(roll_enabled=True)
        _print_report("rolled window (SPYRE_SWA_WINDOW_ROLL=1)", structures, entries)

        self.assertTrue(structures, "no bundle.mlir was emitted at all")
        plan = plan_window_gather(SEQLEN, SEQLEN, WINDOW)
        assert plan is not None
        expected = max(1, plan.buffer_width // 64)
        self.assertTrue(
            any(s.has_loop_with_trip_count(expected) for s in structures),
            f"no scf.for with trip count {expected} (the window hint) -- the "
            "hint produced no device loop, so nothing is tiled over the window",
        )

    def test_unrolled_fallback_for_comparison(self):
        # Not an assertion about the fallback being right: this is the baseline
        # the rolled path has to beat, printed so the two can be compared.
        structures, entries = _compile_swa(roll_enabled=False)
        _print_report("unrolled fallback (flag off)", structures, entries)
        self.assertTrue(structures, "no bundle.mlir was emitted at all")


if __name__ == "__main__":
    unittest.main()
