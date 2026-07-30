# Backend defects found while implementing sliding-window attention

Six issues found on branch `swa-window-roll` (2026-07-30) that are **not**
sliding-window attention problems. SWA is how they were reached; each one is
reproducible from constructs any other op could use. Recorded here so the SWA
branch stays about SWA and these can be filed and fixed on their own branch.

Ordered by how much they cost to work around.

---

## 1. Adding an all-zeros additive mask corrupts a fused attention body

**Severity: wrong numbers, silently.**

A flash-attention body over one 64-column KV window. The additive band is all
zeros — every column is attended, so the add is arithmetically a no-op. Adding
it anyway disagreed with a CPU reference on **199 of 512 output elements
(38.9%)**. Not adding it made the identical body correct.

Two independent SWA decompositions — the rolled window and the older unrolled
loop, which share no code — failed **bit-identically**, so it is not specific
to either.

| | |
|---|---|
| shape | `Lq=1`, one 64-column window, `[B,H,D] = [1,8,64]`, fp16 |
| band | `[1, 1, 1, 64]`, all zeros, produced inside the graph |
| symptom | ~39% of outputs outside `atol=0.1`; max abs diff 0.345 |
| worked around by | `5f826e7`, `cfbb6fe` — skip the add when the band provably masks nothing |

**What makes it interesting.** Every component is correct in isolation: the
gathered K, the gathered V, the band tensor itself, the algorithm in float64
and float16, and a staged rebuild of the body reproducing every construct
including hint placement and the block loop. **Only the composition is wrong.**

The one property that separates passing from failing is the band's *role*:

| role | result |
|---|---|
| graph **input** | passes |
| graph **output** | passes |
| fused **intermediate** | **fails** |

So the suspicion is how a fused constant participates in the add, not the
addition. Adding zeros is a strange thing to do deliberately, but a mask that
happens to be empty at some shape is not, and the failure is silent.

**Reproducer:** drafted, eight variants separating where the zeros come from
(argument / `torch.zeros` in-graph / `spyre.window_band_mask`), what they look
like (broadcast `[1,1,1,W]` vs materialised `[B,H,1,W]`), and what surrounds the
add (accumulators, hints, `Lq=1` vs a full stick). Parked at
`test_zero_band_add_repro.py` in this session's scratchpad, to land on the
bug-fix branch. It has **not been run** — the failing subset is the bug report.

---

## 2. Lowering-time config is not in the Inductor cache key

**Severity: tests silently compare a path against itself.**

`config.swa_window_roll` is read during *lowering*, after Inductor computes the
FX graph cache key. Two compiles of the same graph under different flag values
therefore share a key, and the second replays the first's artifact.

This cost two hardware runs and produced a confident wrong conclusion: two
paths reporting identical structure looked like a finding rather than a cache
hit. Nothing here is SWA-specific — **any** config consulted at lowering time
has it, and any test toggling one within a process is exposed.

Worked around by `tests/inductor/inductor_cache_isolation.py`: a private
`TORCHINDUCTOR_CACHE_DIR` per compile.

Note `torch._inductor.config.force_disable_caches` is *not* a usable
workaround — it redirects `cache_dir()` to a fresh temporary directory, so
emitted bundles land somewhere the caller did not sample.

---

## 3. `make_buffer_reuse` assumes every layout is tiled

**Severity: compile crash.**

`torch_spyre/_inductor/wrapper.py:156`

```text
AttributeError: 'FixedLayout' object has no attribute 'device_layout'
```

Reached with `old` a `FixedTiledLayout` `torch.bool` buffer of size `[8192]`
and `new` a plain `FixedLayout` `torch.bool` `[64, 128]`. Triggered by tracing
CPU-side mask construction (`arange`, comparisons, `masked_fill_`) into the
graph, which produces bool buffers that never get a device layout.

`make_buffer_reuse` reaches for `new.get_layout().device_layout` unconditionally.

---

## 4. `cat` of expanded (stride-0) buffers zeroes the leading slots

**Severity: wrong numbers, silently.**

GQA only — MHA has no expand. Slicing a KV cache, GQA-expanding each slice with
`unsqueeze(2).expand(...).flatten(1, 2)`, then `cat`-ing the results, returns
**zeros** in the leading slots on device while CPU is correct.

| | |
|---|---|
| symptom | 15006 / 262144 elements wrong (5.7%), leading rows exactly 0.0 |
| isolated by | `tests/inductor/test_swa_gather_op.py::test_gqa_windows_survive_a_cat` |
| control | the same expanded window **not** cat-ed passes (`test_key_window_gqa`) |

Cat-ing the raw slices first and expanding the result once avoids it. The
current SWA path does not cat gathered windows at all, so this is dormant
there — but the test is kept as a live probe.

---

## 5. `insert_restickify` `StopIteration` on a `cat` consumed through a view

**Severity: compile crash.**

A matmul consuming a `cat` result through `transpose(-1, -2)` crashes
`insert_restickify` — `_create_restickify_node` cannot resolve the `cat`
buffer's FX node when the consumer reaches it through a view.

Probes in `tests/inductor/test_swa_gather_diag.py` separate the pieces: a plain
buffer through a transpose is fine, a `cat` straight into a matmul is fine, the
combination is not.

Worked around by transposing each slice before the `cat`.

---

## 6. The tiling hint produces no device loop

**Severity: performance only — silent, and invisible to every numeric test.**

`spyre_hint(tiles={"window_size": ...})` emits **zero** `scf.for` in
`bundle.mlir`, on the rolled path and the unrolled fallback alike: 6 bundles
each, 96 flat `sdsc_execute`, none inside a loop. So nothing is tiled over the
window on either design, and every numeric pass to date was untiled code
getting the right answer with one large intermediate.

Not the unroller: `UNROLL_LOOPS` and `codegen/unroll.py` were removed in #3235,
so `scf.for` is the only path and flat output means no `LoopSpec` was created.
Not dynamic trip counts either — every count here is a compile-time int
(`Wb//64 = 2`, `N = 4`).

First place to look: `assign_dim_hints` gates on **named dims**, and
`propagate_named_dims` has an `_untracked_<size>` fallback that only warns.

**Confirmed 2026-07-30, after the fix for issue 2.** Re-run with an isolated
Inductor cache per compile *and* with the roll branch's entry counted during
tracing (1 with the flag on, 0 with it off), so the two paths are known to
have traced differently and neither replayed the other's artifact. Both still
emit 6 bundles, 96 flat `sdsc_execute`, **0 `scf.for`**, with the same
distribution (23, 23, 23, 16, 7, 4). The earlier identical totals were not a
cache artifact after all; the two designs genuinely compile to the same
structure, and neither tiles.

Reproducer: `tests/inductor/test_swa_bundle_structure.py`, parser in
`bundle_structure.py`.

**Knock-on effect worth mentioning in the report.** With no window loop, each
block computes its whole `q_block x buffer_width` score matrix in one pass, so
the online-softmax accumulators both SWA bodies carry are dead weight: `M` is
`-inf` throughout, `correction` is `exp(-inf) == 0`, and the running terms drop
out every time. They are correct, and they are what makes the body legal *if*
the hint ever starts producing a loop, but today they cost work and buy
nothing.
