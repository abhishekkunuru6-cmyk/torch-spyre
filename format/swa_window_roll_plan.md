# SWA via a rolled KV window — plan

**Supersedes** `format/swa_window_gather_plan.md`. Same layer-A goal (get the
attention math right), same placement arithmetic, **different consumption**: one
window buffer reused across Q blocks instead of every block's window
materialized at once.

## 0. What changed, and why

The open question raised on the thread was whether a `[B, H, W, E]` KV buffer
is enough for prefill. It is not, and for the reason increment 1 already
derived: a tile of `T` query rows has staggered windows, so `W + T - 1` keys are
live at once and the buffer must be `[B, H, W + T, E]`. That part of the design
was already correct and is unchanged.

The consequence is the part that was open. Two ways to serve `N = Lq/T` blocks:

| | buffer memory | kernel launches |
|---|---|---|
| **roll** — one buffer, advanced per block | `W + T` | `N`, sequential |
| **store all at once** — the previous plan | `N × (W + T)` | 1 |

**Antoni's call:**

> "the whole point of this is to save the memory, and usually to not pay the
> price of rolling the buffer you'd use paged attention + a mask so you write to
> the next page and just ignore the first few tokens that have been overwritten
> in the previous iterations. But again, that's why I said yesterday to focus on
> getting the math for the attention itself correct and assume the query, key,
> and value come to you already correctly computed, and then we build from there
> to add all these complexities"

> "in any case, **prefer rolling for now**, knowing there's a better way if you
> page + mask"

So: **roll**. Store-all-at-once is dropped — it spent the exact resource the
feature exists to save (the old plan's own risk 3 put prefill at 65x the cache
at Gemma sizes). Paged + mask is the named successor, deferred to layer B.

## 1. What survives, and what dies

**Survives unchanged — this is most of the work already done:**

| | where |
|---|---|
| placement arithmetic: `read_start(n)`, `buffer_width = W + T` (`W` at decode) | `torch_spyre/_inductor/swa_window_gather.py`, inc 1 |
| the coverage invariant + CPU equivalence, 38/38 green | `tests/inductor/test_swa_window_gather_model.py`, inc 2 |
| the op boundary: a named op with a *decomposition*, not a kernel (route A) | `customops.py` / `decompositions.py` |
| flash attention's hint nest with `buffer_width` in place of `max_seqlen_kv` | §3 |
| **slice before expanding, never after** (`lower_pad_sequence: pad_extent=-129`) | inc 4 |
| the band mask op | `spyre::window_band_mask` |

**Dies with the fold — all of it complexity that only all-at-once needed:**

- **The block-major fold** (`n*Hq + h`) and the query fold / output unfold cats.
  There is no `N` axis in any tensor now, so there is nothing to fold.
- **The fold-order contract** and its test. Was the previous plan's risk 2 — the
  one that returns wrong numbers without erroring. It is now unreachable by
  construction.
- **The rank-5 band view** `scores.view(B, N, Hq, qb, Wb) + band`. The band is
  rank 4 and broadcasts over batch and heads. This is worth noting twice: it is
  the prime suspect for the increment-5 blocker at `a561a30` (*Incompatible
  host_size and dim_order*, a rank assertion in `spyre_tensor_impl.cpp`), so
  rolling likely removes the current failure as a side effect rather than
  needing it fixed.
- **The `cat` in the gather**, and with it the two device constraints the diag
  probes cornered (`cat`→`transpose`→matmul = `StopIteration` in
  `insert_restickify`; expand→`cat` = zeroed GQA slots). One block reads **one**
  contiguous slice. Keep the diag file: the restickify bug is a real backend
  limitation worth filing separately, it is just no longer on our path.
- **The prefill memory risk.** Buffer is `W + T` for every `Lq`.

**New, and it is the whole cost:** `N` sequential attention bodies instead of
one. Traced, that is a Python `for` over Q blocks in the decomposition.

## 2. The one thing that could make this pointless

The iterations are **independent** — block `n` reads the cache at a compile-time
offset and shares nothing with block `n+1`. Nothing in the graph forces the
buffers to be sequentially reused, so "rolling" is not something the source
expresses; it is something the memory planner has to *grant*. If every
iteration's window buffer stays live, peak memory is `N × (W + T)` and we have
paid the sequential cost for the all-at-once memory profile — strictly the worst
cell of the table.

This is not hypothetical. `SpyrePythonWrapperCodegen.codegen_free_buffer`
(`torch_spyre/_inductor/wrapper.py:137`, from #2872) **suppresses the free for
any pool buffer**, so freeing is entirely the allocator's liveness analysis, not
the wrapper's.

Two routes, in order:

- **A — earn it.** The buffers are dead the moment their block's output is
  written, so a reuse-aware scratchpad allocator should already collapse them.
  Measure before assuming (inc R5).
- **B — force it.** Allocate one `[B, Hq, Wb, E]` buffer outside the loop and
  `spyre.copy_f` each block's slice into it, the same in-place idiom the
  accumulators use. This creates a true write-after-read chain: one physical
  buffer, and the sequentiality Antoni said we would pay for becomes explicit
  rather than hoped for.

**Route B is the honest implementation of "rolling."** Route A is worth one
measurement first because it is free if it holds.

## 3. The structure

Per Q block, an ordinary rank-4 flash body. Everything is `[B, Hq, ·, ·]`.

```python
outputs = []
for n in range(plan.num_q_blocks):
    k_win, v_win, band = torch.ops.spyre.gather_kv_window(
        key, value, plan.read_start(n), plan.buffer_width, num_heads, ...)
    #   k_win : [B, Hq, E, Wb]   pre-transposed (see §4)
    #   v_win : [B, Hq, Wb, E]
    #   band  : [1, 1, T, Wb]    broadcasts over batch and heads

    q_blk = query[:, :, n * T : (n + 1) * T, :]
    # M / denominator / output accumulators over [B, Hq, T, ...], as SDPA

    with spyre_hint(tiles={"batch_size": max(1, B // 2)}):
      with spyre_hint(tiles={"num_heads": max(1, Hq // 4)}):
        with spyre_hint(tiles={"window_size": max(1, Wb // 64)}):   # was max_seqlen_kv
          with spyre_hint(work_div={"num_heads": 4, "window_size": 8}):
              scores = torch.matmul(q_blk * scale, k_win * scale) + band

              block_max   = torch.amax(scores, dim=-1)
              max_running = torch.maximum(M, block_max)
              exp_scores  = torch.exp(scores - max_running.unsqueeze(-1))
              correction  = torch.exp(M - max_running)

              denominator = copy_f(denominator * correction
                                   + exp_scores.sum(dim=-1), denominator)
              output      = copy_f(output * correction.unsqueeze(-1)
                                   + torch.matmul(exp_scores, v_win), output)
              M           = copy_f(max_running, M)

    outputs.append(copy_f(output / denominator.unsqueeze(-1), output))

return torch.cat(outputs, dim=2)      # concat along the sequence
```

Note the accumulators are **per block**, initialized inside the loop. A block's
entire window is in one buffer, so there is no cross-block softmax state — the
online-softmax machinery is here only because the `window_size` hint tiles
`buffer_width` into `Wb/64` chunks, and each chunk is a partial softmax. That is
the same reason SDPA carries it.

**Decode falls out for free.** `Lq = 1` ⇒ `N = 1` ⇒ one iteration, `T = 1`,
`Wb = W`. Identical code, no special case, and the loop costs nothing. Decode is
also where the win is largest (64x fewer KV rows read at `Lkv=4096, W=64`), so
it stays the first target.

### `T` is now a real dial

Rolling makes the block size a tradeoff the plan can tune, which all-at-once
could not afford:

| `T` | buffer | launches at `Lq=8192` |
|---:|---:|---:|
| 64 | `W + 64` | 128 |
| 256 | `W + 256` | 32 |
| 512 | `W + 512` | 16 |

Bigger `T` buys back the sequential cost for a small additive memory increase —
additive, not multiplicative, which is exactly what all-at-once could not do.
Start at `T = 64` (validated, stick-aligned), sweep in layer B. Note the graph
is unrolled at trace time, so `N` also sets **graph size and compile time**; at
`Lq = 8192, T = 64` that is 128 copies of the body, which is its own argument
for a larger `T`.

## 4. The op

```python
spyre::gather_kv_window(key, value, read_start, buffer_width,
                        num_heads, q_block, seqlen_q, is_causal)
    -> (k_win, v_win, band)
```

Changed from the gather version: it takes **`read_start` directly** — one
compile-time int, one block — instead of deriving all `N` starts internally.
The op no longer knows about blocking at all, which is what makes it reusable if
the consumption strategy changes again (e.g. to paging).

Body is now one slice per tensor, no `cat`:

```python
k_win = key[:, :, s : s + Wb, :].transpose(-1, -2)
v_win = value[:, :, s : s + Wb, :]
if expansion != 1:
    k_win = k_win.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
    v_win = v_win.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
```

**Keep K pre-transposed** (`b100e54`): transposing per slice before the consumer
is the shape SDPA's matmul wants, and it costs nothing to keep. The
`insert_restickify` `StopIteration` that forced it was `cat`-specific and no
longer applies, but transposing here is still the right layout, so leave it and
drop the *reason* from the comment rather than the code.

**Keep slice-before-expand.** The `cat`-before-expand constraint is gone with
the `cat`; the full-length-cache constraint is not, and is the one that matters.

## 4a. What the first hardware run said (2026-07-30)

Three runs, three results. The third one outranks the rest of this plan.

**The body is healthy through stage C.** `test_swa_body_diag.py` A (gather +
matmul), B (+ the rank-4 band) and C (+ softmax + second matmul) all **pass**.
The *Incompatible host_size and dim_order* rank assertion that blocked the
all-at-once body is **gone**, which retires it as the fold's problem — it died
with the rank-5 band view, exactly as §1 predicted. D/E/F failed on a test bug,
not the device: `spyre.copy_f` is registered for PrivateUse1 only and the
helper called it on the CPU reference side too. Fixed with `_copy_into`.

**GQA needs one more probe.** `test_key_window_gqa` failed with zeroed leading
rows (5.7% of elements, `actual == 0.0` where the reference is not) — the
signature of the abandoned design's expand→`cat` bug. But the *test* cats the
expanded windows; the body never does, it feeds each straight to a matmul. Split
into a single-block assertion (what the body actually does) and a `cat` probe,
so the next run says whether the limitation is the `cat` or the expand.

**Neither path produces a device loop.** `bundle.mlir` at `Lq=Lkv=256, W=64`:

| | bundles | `scf.for` | `sdsc_execute` |
|---|---:|---:|---:|
| rolled (flag ON) | 6 | **0** | 96, all flat |
| unrolled fallback (flag OFF) | 6 | **0** | 96, all flat |

Risk 2 is now a measured fact rather than a worry, and it lands on **both**
designs identically. The `window_size` hint is not becoming a loop for either,
so nothing is tiled over the window in either — every numeric pass to date was
untiled code getting the right answer with one big intermediate.

*The totals being identical is itself suspicious* — it is also what you would
see if the flag never reached the decomposition. `test_the_flag_actually_
changes_what_is_traced` now counts `plan_window_gather` calls during tracing
(called only under `if config.swa_window_roll`) and inductor's caches are
disabled during the compile, so the next run can tell a real null apart from a
flag that did nothing. **Read that test before believing the table above.**

**What this does to the plan.** If the null holds, the rolled path's remaining
claim over the unrolled fallback is memory alone (R5), and §5 risk 1 says even
that is not expressed by the graph. Two designs now agree that windowed tiling
is not reaching the device, which is a backend-level finding and worth more
than a third body variant. Take it to Antoni before spending on R7 onward.

## 4b. Second run: the body is green, the null is not yet measured

**The flag is honored.** `plan_window_gather` was entered once with the flag on
and zero times with it off. So the two paths genuinely trace differently, and
the identical first-run totals were *not* a no-op flag — but see below, they may
still have been one path's cached artifact reported twice.

**One iteration of the body is fully green.** A, B, C, **D** (accumulators) and
**E** (all four hints) pass. That is §3's structure, complete, on device. Only F
(every block, concatenated) failed, on `AttributeError: 'FixedLayout' object has
no attribute 'device_layout'` in `make_buffer_reuse` (`wrapper.py:156`) over a
`torch.bool` buffer — the test built its bands *inside* the traced function, so
CPU mask construction entered the graph. The real body's band comes from
`spyre.window_band_mask`, a custom op opaque to dynamo, so this is off our path.
Bands are now passed in like every other stage. **The crash is still a real
backend bug worth filing** — `make_buffer_reuse` assumes every layout is a
`FixedTiledLayout`.

**GQA is answered.** Single-block GQA **passes**; only the `cat` probe fails,
with the zeroed-slot signature. So the limitation is `cat` of expanded
(stride-0) buffers, *not* the expand inside the gather — and the body never cats
gathered windows. GQA is fine for this design. Also worth filing.

**The R6 null is not yet measured.** The second run emitted **zero** bundles on
both paths: disabling inductor's caches moves `cache_dir()` to a fresh temporary
directory, and the root was sampled before the flag was set. Now found by mtime
across both roots. Note what forced the cache flag in the first place: the
config is read during *lowering*, after the FX cache key is computed, so **the
two paths share a cache key** and one can replay the other's artifact. That is
the likeliest reading of the first run's identical totals.

**The unroller is not the explanation.** `UNROLL_LOOPS` and
`codegen/unroll.py` were deleted in #3235, already on this branch, so `scf.for`
is the only path — flat `sdsc_execute` means no `LoopSpec` was created, not a
loop that got flattened. Backend loop support requiring compile-time trip
counts is also not the blocker: every count here (`Wb//64 = 2`, `N = 4`) is a
compile-time int.

## 4c. R5 and R6 measured (2026-07-30)

**R5: the buffer is reused. The design's one claim holds.** Sweeping the window
at a fixed `Lq=512` (8 blocks), so nothing but the window varies:

| `W` | `buffer_width` | `pool_size` | kernels |
|---:|---:|---:|---:|
| 64 | 128 | 1048576 | 10 |
| 128 | 192 | 1310720 | 10 |
| 192 | 256 | 1441792 | 10 |
| 256 | 320 | 1441792 | 10 |

**2048 bytes of planned scratch per row of `buffer_width`** — exactly one
window pair (`2 x B x H x D x 2` at fp16). Eight live windows would be 16384,
and would have taken the pool from 1 MiB to over 4 MiB; it went to 1.375 MiB.
Peak scratch tracks **one** window while eight blocks run, which is the whole
reason rolling was chosen over materialising all `N`.

Read the slope as an order-of-magnitude result, not a precise one: the pool is
clearly quantised (the last step is flat, the per-step deltas are 4096, 2048,
0), and `pool_size` is planned scratch rather than measured peak HBM. The
discrimination was 8x, far past that noise.

**The sequential cost is smaller than the plan assumed.** 10 kernel calls for
8 blocks, constant in `W` — Inductor fuses across the unrolled block loop, so
"one kernel launch per tile" over-counts. §3's worry about `N` launches at
large `Lq` should be re-measured before it drives a `T` sweep.

**R6: still no device loop, and now properly measured.** Isolated caches, and
the roll branch's entry counted during tracing (1 with the flag on, 0 with it
off), so the two paths are known to have traced differently. Both emit 6
bundles, 96 flat `sdsc_execute`, **0 `scf.for`**. The identical totals were not
a cache artifact: the two designs genuinely compile to the same structure, and
neither tiles over the window.

**So the accumulators are currently dead weight.** With no window loop each
block is a single pass: `M` is `-inf`, `correction` is `exp(-inf) == 0`, and
the running terms drop out. They stay because they are what makes the body
legal if the hint ever starts producing a loop, but today they are cost with no
benefit. Detail in `format/swa_backend_bugs.md` issue 6.

## 5. Risks, in order

1. **The rolling may not roll** (§2). The whole justification for choosing this
   over all-at-once is memory, and the graph does not express it — the allocator
   grants it. Measured by inc R5; forced by route B if the measurement is bad.
   *A numeric pass says nothing about this.*
2. **The hints may be no-ops.** Unchanged from the previous plan and still
   first-class: compiling `spyre_sliding_window_attention` previously emitted
   **zero** device loops (~40 flat `sdsc_execute` groups at seqlen 2048, no
   `scf.for`). `assign_dim_hints` gates on named dims and
   `propagate_named_dims`'s `_untracked_<size>` fallback only warns. Untiled
   code returns the right answer with one big intermediate, so **verify
   structurally** (inc R6).
3. **Graph size at large `N`.** 128 unrolled bodies at `Lq=8192, T=64`. Mitigated
   by `T`, bounded by measurement, not a correctness risk.
4. **`Lq % T`** — block slicing needs exact divisibility; otherwise pad Q or fall
   back. Unchanged.

Gone: the fold-order risk and the prefill-memory risk, both structural
casualties of dropping the fold.

## 6. Increment ladder

Layer A is still the job: attention math correctness, Q/K/V assumed correct.
`[HW]` = the user runs it on the pod.

Carried over, no rework:

- [x] **Inc 1 — placement arithmetic.** `c49f04e`. `read_start`/`buffer_width`
      are consumption-agnostic and hold verbatim.
- [x] **Inc 2 — CPU equivalence.** `b89d41f`, 38/38 green. Claims are per-block,
      so they carry; add a rolled-loop model alongside the folded one (R2).
- [x] **Inc 3 — rank spike.** `6c1ddd5`. Recorded fact: **rank-5 matmul is
      unsupported**. Now moot — rolling is rank 4 everywhere by construction.

New from here:

- [ ] **Inc R1 — close out the old blocker.** `[HW]` Run the `a561a30` bisect
      (`test_swa_body_diag.py`) **once** and record which construct produces
      *Incompatible host_size and dim_order*. If it is the rank-5 band view,
      rolling has already fixed it; if it is something in the shared flash body,
      we need to know before rebuilding on top of it. One run, no follow-up work
      either way.
- [ ] **Inc R2 — rolled CPU model.** Extend the inc-2 model to a per-block loop
      and assert the concatenated output matches the full masked reference.
      Cheap, no HW, and it locks the loop's block↔window pairing before device
      work — pairing is now positional-by-iteration rather than by fold index.
- [ ] **Inc R3 — the op, single-window form.** §4 signature, fake, decomposition.
      Rewrites `8f02074`/`3a8cbd2`/`b100e54` down to roughly a third.
- [ ] **Inc R4 — the body.** §3 behind `config.swa_window_roll`
      (`SPYRE_SWA_WINDOW_ROLL=1`, default OFF). `[HW]` **decode first** (`N=1`,
      one iteration, the case the previous structure never got a clean answer
      for), then prefill at `N=2`, then general `N`.
- [ ] **Inc R5 — does it roll?** `[HW]` The increment that decides whether this
      approach delivered anything. Measure peak live KV-window bytes across `N`
      (allocation plan / scratchpad report, `LX_PLANNING=1`). Expect flat in
      `N`; if it grows linearly, take route B (§2) and re-measure. Note
      `codegen_free_buffer` suppresses pool-buffer frees, so a linear result is
      a live possibility, not a paranoid one.
- [ ] **Inc R6 — verify the tiling is real.** `[HW]` Parse `bundle.mlir` for
      `scf.for` trip counts and `sdsc_execute` counts inside vs outside the
      window loop. Risk 2 means a green numeric run proves nothing here.
- [ ] **Inc R7 — shape sweep.** `[HW]` `B>1`, GQA and MHA, decode and prefill.
      Layer A is done when R7 is green **and** R5 is flat **and** R6 shows real
      loops.

Layer B — the cost, which is what "we build from there" means:

- [ ] **Inc R8 — benchmark** `[HW]` against the mask-based SDPA baseline. The
      claim to test is that saved work beats `N` launches plus the copy.
- [ ] **Inc R9 — sweep `T`** (§3). Trades launches against `W + T` and graph size.
- [ ] **Inc R10 — paged + mask.** Antoni's named better way: write into the next
      page and mask off the overwritten prefix, removing the serialization
      rolling pays for. Only after R8/R9 say what it is worth.

Layer C — projections and RoPE. Still not ours.

## 7. Deferred

- **Bidirectional** (`is_causal=False`) — needs `Wb = 2W + T`.
- **`Lq % T != 0`** — pad or fall back.
- **Multi-core (`SENCORES > 1`)** — `work_div` carries over unchanged; no claim
  until R8 measures it.
- **A real gather kernel** (the old route B) — superseded by paging as the thing
  worth building if the copy proves expensive.
