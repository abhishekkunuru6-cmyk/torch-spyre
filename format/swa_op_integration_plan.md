# SWA op integration — plan & next tasks (resume here)

Self-contained hand-off for wiring the validated **sliding-address hint** into the
**real** sliding-window-attention op. A fresh session should be able to continue
from this file alone.

Branch: **`swa-sliding-address`** (all sliding work). The sibling branch
`swa-windowed-decomposition` is the for-loop approach ONLY (reset to `e22622b`,
shared with others — do NOT put sliding work there).

---

## 1. What is actually done vs. NOT done (read this first)

**DONE — the sliding *mechanism*, validated in isolation on HW:**

- `spyre_hint(sliding={"A": {"window": W, "stride": S}})` produces a correct
  overlapping windowed read. Full frontend path:
  `propagate_hints` → `propagate_named_dims` → `coarse_tile` → `op_spec` →
  `spyre_kernel` → `compute_ops`/`superdsc` (affine-stride scaling + kept backGap).
- `validate_sliding_prototype.py` — sliding reduction (`x.sum(dim=0)`):
  `--sweep` = 8/8 shapes (1–4 col-sticks, 2/3/4 windows, W∈{128,256}); SENCORES
  1/2/4 all MATCH.
- `validate_swa_sliding_matmul.py` — sliding read on a MATMUL contraction dim
  (SWA's `exp@V` analog): MATCH (max err 0.21).

- `validate_swa_diagonal_coupling.py` — COUPLED diagonal slide (Q partition-slides
  while KV overlap-slides under one loop level): MATCH (max err 0.13), plus 19
  unit tests in `tests/inductor/test_coupled_sliding_hint.py` and the 202-test
  coarse-tiling suite green.

**IN PROGRESS — the op is now rewritten but NOT yet correct.**
`spyre_sliding_window_attention` (`torch_spyre/_inductor/decompositions.py`)
takes the sliding path via `_sliding_window_attention_looped` when
`config.swa_sliding_loop` is set AND `plan_sliding_window` returns a plan;
otherwise it falls back to the original Python-unrolled
`for qi in range(num_q_blocks)` loop. As of this session the sliding path
**compiles and executes** on HW for a power-of-two `padded_kv` but produces
**44.2% wrong elements**. That aggregate has since been localized: **only head
0 is wrong** — 22 of 24 (block, head) pairs match CPU to fp16 precision, so the
sliding machinery itself is working. See 5h for the head table, the prime
suspect and the one-line experiment to run first. The flag defaults OFF, so the
default path is untouched.

Status at a glance:

| | state |
|---|---|
| sliding hint mechanism (inc 1–5c) | DONE, HW-verified |
| op rewritten to one loop | DONE (behind flag) |
| op-group contiguity (5e) | SOLVED — value dependency, not ordering edge |
| `padded_kv` factorization (5g) | root-caused, **fix not written** |
| numerics (5h) | **current blocker — narrowed to HEAD 0 ONLY** (22/24 block-head pairs are correct; suspect the 5e hack's 1→8 head broadcast) |
| inner KV sweep (5f) | not started (scalability) |
| multi-core work division | **not started** (all runs `SENCORES=1`) |

**The diagonal coupling that blocked this is now BUILT (increment 2).** A
multi-entry `spyre_hint(sliding={...})` couples dims under one loop level with
per-dim window/stride:

```python
with spyre_hint(sliding={"QS": {"window": 64, "stride": 64},    # partition
                         "KV": {"window": W,  "stride": S}}):   # overlap
```

Coupling is **sharing a `hint_id`**, not sharing a sympy loop var: `hint_id` is
what coarse_tile turns into a loop level, so one scope stays one level however
many dims it names, while each op still resolves its own per-dim loop var.
Codegen needed no change — `OpSpec.sliding_symbols` is already per-symbol and
`bundle.py` already emits multi-term affine maps, so a tensor indexed by both
dims gets `base + i*qs_stride + i*kv_slide`. What changed was the middle:
`CoarseTileInfo.loop_reduction_{slide_stride,read_extent}`, `_stamp_group`
keying params by dim kind, `create_op_spec` assigning per symbol by kind, and
`_validate_reduction_tiling` allowing a mixed output+reduction level when it is
coupled.

The one semantic call: a coupled level SKIPS `_propagate_tiled_reduction_op`.
Its reduction completes *inside* iteration `i` (block `i` pairs with window
`i`), so the fill+combine accumulator would sum every block into one slot — a
wrong answer, not a crash.

Two constraints the partition+slide model forces: one level has one trip count,
so `QSEQ // q_block == KVSEQ // W`; and with overlap the last window ends short,
leaving a `(W−S)(N−1)` KV tail that is **never read**. The first of these turns
out to be unsatisfiable at real SWA shapes and has to be relaxed — see
increment 5 below.

---

## 2. The real op (what we are replacing)

`decompositions.py:681` `spyre_sliding_window_attention(query, key, value,
window_size, is_causal, scale)`:

- `q_block_size = kv_block_size = 64`; `num_q_blocks = ceil(max_seqlen_q/64)`.
- Python `for qi`: `q_start=qi*64`; causal window
  `kv_lo = q_kv_offset+q_start−window_size+1`, rounded to 64 →
  `kv_start` advances **+64 per qi**; `kv_len ≈ window_size+64`.
- Per block: slice `key[:,:,kv_start:kv_end,:]` / `value[...]`, GQA-broadcast,
  `band_mask = spyre.sliding_window_block_mask(...)` (built on CPU per-qi),
  then inner online-softmax sweep: `scores=q@kᵀ` (reduces head_dim), softmax
  over `kv_len`, `out=exp@v` (reduces `kv_len`). Accumulators M/denominator/out.
- `return torch.cat(out_blocks, dim=2)`.

The docstring (lines 696–707) explicitly says hint tiling can't express the
index-dependent window slice — that is exactly the gap our sliding hint closes,
ONCE it supports the diagonal coupling.

---

## 2b. The rewrite as it stands (read before changing it)

New file `torch_spyre/_inductor/swa_sliding.py`: frozen `SlidingWindowPlan`
(`num_q_blocks, q_block, read_extent, base_offset, left_pad, right_pad,
padded_kv, q_kv_offset, window_size, is_causal`), `unclamped_kv_range`,
`plan_sliding_window`, `build_band_mask_cpu`. New custom op
`spyre::sliding_window_band_mask` builds the full band over padded K/V.

```python
k_pad = _prepare_kv(key, plan, expansion)      # zero blocks built OUTSIDE
v_pad = _prepare_kv(value, plan, expansion)    # any hint scope

with spyre_hint(named_dims=["ONE", "ONE", "QS", "KV"]):
    band = torch.ops.spyre.sliding_window_band_mask(...)

diag_col = plan.q_kv_offset + plan.left_pad
zero_from_band = band[:, :, :1, diag_col : diag_col + 1]   # exactly 0.0
v_probe = v_pad[:1, :1, :1, :1]

with spyre_hint(named_dims=["B", "H", "QS", "D"]):
    q_scaled = query * scaling_factor
with spyre_hint(named_dims=["B", "H", "KV", "D"]):
    k_scaled = k_pad * scaling_factor + zero_from_band * v_probe   # see 5e

with spyre_hint(sliding={
        "QS": {"window": plan.q_block,     "stride": plan.q_block},
        "KV": {"window": plan.read_extent, "stride": plan.q_block,
               "counts_tiles": False}}):
    scores = torch.matmul(q_scaled, k_scaled.transpose(-1, -2))
    scores = scores + band
    block_max = torch.amax(scores, dim=-1)
    exp_scores = torch.exp(scores - block_max.unsqueeze(-1))
    denominator = exp_scores.sum(dim=-1)
    out = torch.matmul(exp_scores, v_pad)
    return out / denominator.unsqueeze(-1)
```

**Two invariants that are easy to break by accident:**

1. **One op per `named_dims` scope.** `_named_dims` uses `setdefault`, so the
   *first* op in a scope wins the name. Wrapping several differently-shaped ops
   in one scope caused
   `ValueError: spyre_hint(sliding=...) on 'KV': need 0 < window (128) <=
   dim_size (64)` — a `[1,8,64,128]` zero-pad block registered `KV=64` before
   the `cat` could register 320. Hence `_kv_pad_parts` builds zeros *outside*
   the scopes.
2. **`amax` without `keepdim`**, unsqueezed at point of use (5d item 3).

## 3. Next tasks (increment ladder)

Discipline that worked for the prototype: **isolate each unknown in a
`validate_*.py`, run on HW (user runs; no HW in sandbox), THEN integrate.**

- [x] **Inc 1 — matmul consumes sliding reduced-dim read.**
  `validate_swa_sliding_matmul.py` (commit 55df22c). MATCH.

- [x] **Inc 2 — diagonal cross-dim coupling (the core extension).**
  2a spec + CPU reference (`ff9f105`), 2b mechanism + unit tests (`6a28810`),
  2c HW MATCH. Carrier: `A[QSEQ,KVSEQ] @ B[KVSEQ,D]`, tile `i` computing
  `out[64i:64i+64,:] = A[64i:64i+64, iS:iS+W] @ B[iS:iS+W, :]` — SWA's
  `out_blk = exp_scores_blk @ v_blk`. Deliberately the non-hazardous shape:
  the slid KV dim is REDUCED, so each output row block is written once.

- [x] **Inc 3 — non-reduced matmul slide** (`4e4d2ed` spec, `d601cd2` impl).
  `scores = q@kᵀ`, KV an OUTPUT col. Needed TWO OUTPUT DIMS per level, so
  sliding params became per-(level, dim). No write hazard after all: the Q
  partition keeps the written regions disjoint. HW: band MATCH + structure
  shows 2 executes inside a loop of 4, 0 outside, two-term affine map.

- [x] **Inc 4 — full inner attention body under the slide** (`ea2a82a`).
  `s=(q@kᵀ)*scale; p=exp(s-max s); out=(p@v)/sum p` in one scope. HW: err
  0.0032; 9 executes inside one loop of 4, 0 outside, one map `8192*d0` (the
  slid stride — a partition would be 16384). Per-tile intermediates did not
  pick up the slide.

  **Verification note that applies to every future increment:** numbers alone
  cannot prove work reduction when the slide changes only WHICH elements are
  computed (inc 3's band is a slice of the untiled product — separation
  0.00000). Use `swa_probe_bundle.structural_report`, which checks the
  `scf.for` trip count and counts `sdsc_execute` inside vs outside the loop.

- [ ] **Inc 5 — real SWA shapes, then the rewrite.**

  **Increment 5 is NOT just the rewrite the earlier plan assumed.** Scoping it
  against `decompositions.py:748` turned up two blockers, both from the same
  root cause.

  *Blocker A — the trip-count constraint cannot hold at real shapes.* Increment
  2a derived `QSEQ // q_block == KVSEQ // W`. Take prefill `Lq = Lk = 512`,
  `window_size = 128`, `q_block = 64`:

  | quantity | value | pinned by |
  |---|---|---|
  | trip count N | `num_q_blocks = 8` | the Q partition |
  | window W | `window_size + 64 = 192` | the algorithm |
  | model demands | `N == 512 // 192 == 2` | `dim_size // window` |

  8 ≠ 2, and neither side is tunable. Pre-padding K/V by `window_size` does not
  help either (`640 // 192 == 3`).

  *Blocker B — the window origin is negative.* The real range is
  `kv_lo = q_kv_offset + q_start − window_size + 1`, negative for the first
  blocks, which is why the op clamps with `max(0, …)`. So `kv_len` is not
  constant — for the case above it ramps `64, 128, 192, 192, …`. The current
  model assumes a constant extent at base `i·S` starting from 0 and cannot
  express that ragged prefix.

  *Root cause.* `_append_sliding_hints` derives a trip count from
  `dim_size // window` for EVERY coupled dim. That is right for a dim being
  partitioned and wrong for a read-only sliding window over an input: K/V need
  not claim a trip count at all. The count should come from the partitioning
  dim (Q) alone, leaving the sliding dim to satisfy only
  `S·(N−1) + W ≤ dim_size`. Relaxing this DELETES the
  `QSEQ//q_block == KVSEQ//W` constraint — it was a symptom, not a law.

  Sub-increments:

  - [x] **5a.** Spec (`890c4c7`, `5152fb6`). 6/6 shapes: the constant-window +
    mask model and the op's clamp-and-shrink model agree at **exactly
    0.000000** — not a tolerance, the same computation. 6/6 violate the 2a rule
    (some on the count, some on `W ∤ Lkv`).
  - [x] **5b.** Implemented (`7d1ac39`): `counts_tiles: False` lets a dim follow
    another's trip count, skipping divisibility and bounded only by
    `S·(N−1) + W ≤ dim_size`. **No `base` parameter** — every expressible case
    reduces to base 0 (`base > 0` is a slice; `base < 0` needs padding, after
    which it is 0), so blocker B is handled by padding.
  - [x] **5c.** HW: real shape Lq=Lkv=512, window=128 → err 0.0086, trip count
    **[8] from Q's partition**, 10 executes inside one loop, 0 outside, and a
    two-term affine map (the mask riding both slides).
  - [x] **5d-pre.** Rank-4 + GQA probe (`validate_swa_4d_rank.py`). Cleared
    leading untiled dims, `BATCH_MATMUL_OP`, and the broadcast-and-slid mask.
    Found the batch>1 defect recorded in section 6 — **`batch == 1` works for
    every shape tested** (H=2/4/8, MHA and GQA, Lq 256 and 512).
  - [ ] **5d — the rewrite.** Rewrite `spyre_sliding_window_attention`, scoped
    to `batch == 1`, falling back to the existing unrolled loop otherwise.
    Known work, in rough risk order:

    1. **The padding decision (open).** The negative window origin needs
       `left_pad = -base_offset` rows on K/V. Padding a production KV cache
       per call is a real copy; peeling the ragged prefix instead means
       unrolling `left_pad // 64` blocks — 2 for `window_size=128` but **64 for
       Gemma's 4096**, so peeling does not obviously win. Measure before
       choosing.
    2. **band_mask.** Currently CPU per-qi. Must become a full
       `[Lq, padded_kv]` tensor whose KV axis rides the same slide (5a's probe
       shows the compact `[Lq, W]` form does not typecheck against the untiled
       carrier). **Name its dims** — 5c passed with them unnamed only via the
       `_untracked_` fallback coincidentally aligning.
    3. **Softmax shape.** Use `amax(dim=-1)` WITHOUT keepdim and unsqueeze at
       the point of use, as the op already does. `keepdim=True` materializes a
       rank-4 `[B,H,Lq,1]` buffer that `_resize_device_layout` cannot place
       when a second leading dim is also 1.
    4. **GQA broadcast**, kept as-is — it is not implicated (see section 6).
    5. Validate against `tests/inductor/test_sliding_window_attention.py` and
       the additive-mask SDPA path.

  - [x] **5e — op-group contiguity (SOLVED).** `coarse_tile` requires
    every op sharing a hint scope to be **contiguous** in `graph.operations`;
    anything unhinted landing in the middle splits the scope into two groups
    over the same `hint_id` and `validate_coarse_tile_groups` rejects it.
    Hit **twice**, by two different ops, for the same underlying reason —
    Inductor schedules an op with no forced-early consumer wherever it likes,
    and both times it chose "just before my first use", i.e. inside the loop
    body:

    - **V's pad chain** (fixed, `683d795`). K feeds the scores matmul (first op
      in the scope) so it schedules early; V is not needed until the *second*
      matmul at the end. Fixed by `cat([key, value], dim=1)` **before**
      padding, so one shared buffer must be ready for K's use and carries V
      along — slicing back out afterward is a view, not a deferrable chain.
    - **The band mask** (OPEN). `op11: FallbackKernel` / `op12: MultiOutput`
      land between the matmul (`op10`) and `scores + band` (`op13`). Cannot use
      the same trick: the mask is built from plan *constants*, has **no tensor
      input at all**, so there is nothing to merge it into. It was already
      earliest-possible in Python source order and still landed late —
      confirming source order does not drive schedule order.

    **What did NOT work.** A `schedule_anchor=k_scaled` argument (an ignored
    tensor input, added purely to create an un-optimizable dependency edge)
    and a separate `spyre::order_after` custom op. Both failed, and the reason
    generalizes: an ordering edge makes the band a *sibling* of the matmul, not
    a **predecessor of the op that consumes it**. `order_after` also broke
    worse — being a `FallbackKernel`, `propagate_named_dims` cannot traverse it,
    so QS/KV names stopped there, `_is_coupled_sliding_level` went false, the
    reduction-accumulator path fired, and spliced buffers broke topological
    order (`KeyError: 'op13'` in `Scheduler.compute_ancestors`).

    **What worked — a value dependency, not an ordering hint.** Force the band
    to be a genuine input to an op that is *already* early in the scope:

    ```python
    zero_from_band = band[:, :, :1, diag_col : diag_col + 1]   # exactly 0.0
    v_probe = v_pad[:1, :1, :1, :1]
    k_scaled = k_pad * scaling_factor + zero_from_band * v_probe
    ```

    `k_scaled` feeds the *first* matmul, so the band must be ready before the
    scope opens. `diag_col = q_kv_offset + left_pad` is on the band's diagonal,
    where the value is exactly `0.0` (verified across causal/non-causal and
    several window sizes), so the term is numerically inert. `v_probe` drags
    V's pad chain in by the same mechanism, which is why the earlier
    `cat([key, value], dim=1)` merge (`683d795`) could be **reverted**.

    Note the slice must be **indexed**, not reduced: `band.amax()` over a
    rank-4 band reduces QS and KV simultaneously and raises `expected exactly
    1 reduction variable, got {d1, d0}`.

  - [x] **5g — `padded_kv` factorization (ROOT-CAUSED, fix not yet written).**
    With contiguity solved, the two `seqlen 256` shapes still died at
    `Unsupported coordinate expression 2*(Mod(c1, 20))/5`. This was
    misdiagnosed twice (first as the doubled head axis from the K/V merge, then
    as a naming gap) before a `pow2` test shape settled it:

    | shape | `padded_kv` | factors | result |
    |---|---:|---|---|
    | `mha_prefill_causal_w64_b1` | 320 | 2⁶ × **5** | coordinate assertion |
    | `mha_prefill_causal_w64_b1_pow2` | 256 | 2⁸ | **compiles + runs** |

    `views.py` carries a `# TODO: handle non-unit fractions` for exactly this
    (issue #1353): a `padded_kv` with an awkward factor produces a fractional
    coordinate expression the backend rejects. **Fix:** round `padded_kv` up to
    a friendly factorization inside `plan_sliding_window`. The extra columns
    are masked by the band anyway, so this costs read bandwidth, not
    correctness. Not a design flaw in the sliding approach — a shape
    constraint.

  - [ ] **5h — CURRENT BLOCKER: wrong numerics on the shape that runs.**
    `mha_prefill_causal_w64_b1_pow2` executes end-to-end but reports
    `Mismatched elements: 86975 / 196608 (44.2%)`, greatest abs diff 3.4296875
    at `(0, 0, 188, 52)`.

    `diagnose_swa_sliding_blocks.py` breaks that aggregate down **per Q block**
    (B=1, H=8, Lq=Lkv=192, D=128, window=64 → 3 blocks, windows `[0, 64, 128]`,
    `read_extent=128`). First run, max abs err per block:

    | Q block | rows | vs `correct` |
    |---:|---|---:|
    | 0 | 0–63 | 2.5693 |
    | 1 | 64–127 | 0.8586 |
    | 2 | 128–191 | **0.0097** ← correct to fp16 precision |

    **The last block is right and the error grows the further back you go.**
    That is the decisive clue: "stuck at block 0" (the assumption driving every
    earlier hypothesis) predicts the exact opposite gradient, so it is ruled
    out. The read appears pinned at or near its *final* position for every
    iteration.

    The first run's `stuck`/`shifted` columns showed `nan` — an artifact of the
    *diagnostic*, not the device: those references give some rows a window with
    no valid column, so `softmax` over all `-inf` produced `nan`. Fixed
    (`f9d1ecc`); fully-masked rows now resolve to 0. The reference set was also
    replaced to probe the newly-indicated end of the range: `last` (read never
    moved off its final position), `reversed` (slide runs backwards), `desync`
    (data and band slide at different rates), plus `correct`/`unmasked`.
    **Re-run is pending — the discriminating result is not in yet.**

    **RESOLVED TO A SINGLE HEAD (2026-07-28) — start here next session.**
    The per-head profile is decisive:

    | block | h0 | h1 | h2 | h3 | h4 | h5 | h6 | h7 |
    |---:|---:|---:|---:|---:|---:|---:|---:|---:|
    | 0 | **2.569** | 0.008 | 0.005 | 0.004 | 0.009 | 0.007 | 0.009 | 0.008 |
    | 1 | **0.859** | 0.004 | 0.005 | 0.004 | 0.005 | 0.004 | 0.004 | 0.006 |
    | 2 | 0.004 | 0.004 | 0.004 | 0.007 | 0.006 | 0.010 | 0.005 | 0.005 |

    **22 of 24 (block, head) pairs are correct to fp16 precision (≤0.010).
    Only head 0 is wrong, and only in blocks 0 and 1.** The sliding read, the
    windowing, the band mask and the softmax are therefore all WORKING — the
    44.2% aggregate was one head in eight dragging the average. Every
    hypothesis chased before this (stuck-at-zero, pinned-at-last, reversed,
    desync, unmasked, padding) is dead; the block-level and row-level tables
    only ever showed head 0's error because they max over heads.

    Corroborating detail from the row profile: within block 0 the error does
    **not** track the padding-column count (row 0 has 63 padding columns and
    err 0.0020; row 3 has 60 and err 0.8238; row 63 has 0 and err 0.3724). So
    padding handling is fine too.

    *Prime suspect — the 5e contiguity hack, on the H-axis broadcast.* It is
    the only construct in the rewrite that singles out index 0 of a leading
    axis:

    ```python
    zero_from_band = band[:, :, :1, diag_col : diag_col + 1]   # [1, 1, 1, 1]
    v_probe = v_pad[:1, :1, :1, :1]                            # [1, 1, 1, 1]
    k_scaled = k_pad * scaling_factor + zero_from_band * v_probe
    ```

    `band` is `[1, 1, Lq, padded_kv]`, so its H extent is **1** while `k_pad`
    has `H = 8`. The add therefore REQUIRES a 1→8 broadcast on the head axis.
    If that broadcast is not materialized and the term lands on head 0 alone,
    head 0's K is perturbed while heads 1–7 get clean `k_pad * scaling_factor`
    — exactly the observed signature. (`zero_from_band` is exactly 0.0 in
    *value*, so this is about where the term is APPLIED, not what it is.)

    *Discriminating experiment, one line, run this first:* delete
    `+ zero_from_band * v_probe` from `k_scaled` and re-run
    `diagnose_swa_sliding_blocks.py`.

    - head 0 becomes correct → confirmed; the hack is the bug. (The contiguity
      error from 5e will very likely return, since that term is what orders
      the band — that is expected, and is the thing to re-solve.)
    - head 0 still wrong → the hack is exonerated and the fault is elsewhere;
      go after head 0's K/V layout directly.

    *Candidate fixes if confirmed*, in order of preference: (a) expand the
    zero term to the full head extent explicitly
    (`zero_from_band.expand(-1, num_heads, -1, -1)`) so no implicit 1→8
    broadcast is needed; (b) source the probe from a tensor that already has
    `H = num_heads` so the broadcast never arises; (c) find an ordering
    construct that does not require indexing a leading axis at 0 at all.
    Whichever is chosen, re-check that the band still lands **before** the
    scope — the whole point of the term.

    *Also worth re-testing once head 0 is fixed:* `batch > 1` with `heads >= 4`
    (section 6.1). A leading-axis broadcast defect of this shape is a plausible
    common cause, and 6.1 has been open and unexplained for a while.

    Secondary observation, now much less likely to be the cause: the run warns
    `buf17/buf8: loop var d1 has no named dim mapping -- using _untracked_192`
    and `_untracked_256`, likewise `buf21/buf11`. Those are the QS and KV
    extents reaching body tensors **without their names**. This was the leading
    suspect before the head profile; it is demoted because a dropped sliding
    hint would corrupt all eight heads equally, not one. Still worth cleaning
    up — 5d item 2 already flagged that 5c passed with unnamed mask dims only
    because the `_untracked_` fallback happened to align.

    *Method note for the next session.* Four hypotheses were wrong before the
    head profile landed, and every one of them was reached by reasoning about
    aggregate or block-level numbers. The measurement that actually resolved
    it — split the error along an axis where every slice runs the **identical**
    program — cost one small function. Prefer that shape of experiment over
    another round of candidate references.

  - [ ] **5f — inner KV sweep (SCALABILITY GAP, not correctness).** The single
    loop currently processes the **entire** `read_extent` in one shot: one
    `[64, D] x [D, read_extent]` matmul and one softmax over the full width.
    That reproduces the unrolled path's **outer** block-skip but **drops its
    inner online-softmax sweep** (the old `spyre_hint(tiles={"kv_window":
    ceil(kv_len/64)})` plus the `M` / `denominator` / `out_blk` accumulators).

    `read_extent = 64 - floor((1 - window_size)/64) * 64`, so the per-iteration
    scores intermediate grows with the window:

    | window | read_extent | scores tile (B=1, H=8, fp16) |
    |---:|---:|---:|
    | 64 | 128 | ~131 KB |
    | 128 | 192 | ~197 KB |
    | **4096** (Gemma-3 default) | **4160** | **~4.3 MB** |

    Correct at the small windows tested; **will not scale to production window
    sizes** — this is exactly the "large-intermediate problem flash attention's
    tiling exists to avoid" that the *unrolled* decomposition's own docstring
    cites as the reason its inner sweep exists. Fixing it means nesting a
    second tiling level with online-softmax accumulators *inside* the sliding
    loop. Sequenced after 5e: no point nesting an inner loop inside an outer
    loop that does not yet compile.

---

## 3b. Why the single loop is worth it (measured, not assumed)

`benchmark_sliding_window_attention.py`, windowed decomposition vs. the
mask-based baseline:

| case | speedup |
|---|---:|
| prefill 256 | 0.34x (slower) |
| prefill 2048 | 1.84x |
| prefill 4096 | 3.81x |
| decode 4096 → 131072 | 3.24x → 5.80x → 16.17x → 32.86x → 65.65x → **129.72x** |

The shape of that curve is what matters. Per-Q-block cost is essentially
**constant** (1.44 / 1.27 / 1.30 ms at 256 / 2048 / 4096), and every `Lq=1`
decode costs ~1.8 ms regardless of cache length. So:

> runtime ≈ `num_q_blocks` × fixed per-kernel overhead

The Python-unrolled loop emits `num_q_blocks` **separate device kernels**, so it
pays that overhead N times. Collapsing to one `scf.for` attacks exactly the
dominant term — which is why the 256 case is a *loss* (overhead dominates the
tiny win) and the gap widens monotonically with sequence length.

## 3c. Open gaps not yet on the increment ladder

- **Work-division hints are absent from the sliding path.** The rewrite only
  ever solved *block selection* (which KV window a Q block reads). It never
  emits `tiles=` / `work_div=` for batch or heads, i.e. **multi-core work
  division was never implemented**. This is invisible in every result recorded
  in this document because **every run so far used `SENCORES=1`**. Any
  benchmark or correctness claim at `SENCORES>1` is currently unsupported.
- **Section 6.1 (batch>1, heads≥4 wrong results) remains OPEN and unexplained.**
  The rewrite is scoped to `batch == 1` to avoid it, not because it is fixed.
- **Issue draft for #3248 is written but NOT filed** —
  `format/issue_sdpa_unaligned_kv.md`.

---

## 4. How to run (user, on the pod — no HW in the Claude sandbox)

```bash
git checkout swa-sliding-address
python -m pytest tests/inductor/test_coupled_sliding_hint.py -q     # no HW
python -m pytest tests/inductor/test_coarse_tiling.py -q            # no HW
SENCORES=1 python validate_sliding_prototype.py --sweep --no-dump   # reduction
SENCORES=1 python validate_swa_sliding_matmul.py                    # inc 1
SENCORES=1 python validate_swa_diagonal_coupling.py --compile       # inc 2
SENCORES=1 python validate_swa_scores_slide.py --compile            # inc 3
SENCORES=1 python validate_swa_attention_body.py --compile          # inc 4
SENCORES=1 python validate_swa_real_shapes.py --compile             # inc 5

# The sliding rewrite is OFF by default; opt in per run.
SENCORES=1 SPYRE_SWA_SLIDING_LOOP=1 \
  python -m pytest tests/inductor/test_sliding_window_attention_kernel.py -k b1 -q
SENCORES=1 python diagnose_swa_sliding_blocks.py                    # inc 5h
python -m pytest tests/inductor/test_swa_sliding_plan.py -q          # no HW
```

`config.swa_sliding_loop` (`SPYRE_SWA_SLIDING_LOOP=1`) gates the whole
rewrite; with it unset the op runs the original unrolled loop, so nothing on
this branch can regress the default path. `plan_sliding_window` returns `None`
— i.e. falls back — for `batch_size != 1`, `seqlen_q % q_block`, a non-affine
KV range, or `left_pad == 0 and right_pad == 0`.

`--compile` on the increment 3+ probes also checks the bundle structure
(`swa_probe_bundle.py`): a numeric MATCH alone does not prove work reduction.
Add `--dump` to print the whole `bundle.mlir`.

Constraints of the current hint: `W | dim_size`, stick dims multiple of 64,
`S < W` for overlap, `S <= W` always (a gap-read past the end now raises).
Coupled scopes additionally need equal trip counts across the coupled dims and
allow at most one output dim + one reduction dim per op. `SWA-DEBUG` prints were stripped (commit 6dca4c6) — re-add
locally while debugging a new increment, strip before any PR.

## 5. Related

Memory: `project_swa_op_integration.md`, `project_swa_sliding_address_probe.md`.
Backend was pre-cleared to execute `stride < extent` overlap (probes
`probe_sliding_address.py`, `probe_windowed_numeric.py`).

---

## 6. Known issues — NOT caused by the sliding work, tracked separately

Both were found by `validate_swa_4d_rank.py` / the SDPA test and are unrelated
to each other. Neither blocks increment 5d, which is scoped to `batch == 1`.

### 6.1 Wrong results at `batch > 1` with `heads >= 4` (OPEN, unexplained)

Under the sliding hint, rank-4 attention returns wrong values — not a crash —
once **both** `batch > 1` and `heads >= 4`. The loop structure is correct
throughout (`scf.for` with the right trip count, all executes inside it), so
the tiling is right and only the addressing is wrong; error magnitudes of
`~1e9` say it is striding through unwritten memory.

| B | H | GQA? | B×H | result |
|---|---|---|---|---|
| 1 | 2 / 4 / 8 | either | 2–8 | PASS |
| 2 | 2 | MHA | 4 | PASS |
| 2 | 4 | **MHA** | 8 | **FAIL** |
| 2 | 4 | GQA | 8 | FAIL |
| 1 | 8 | MHA | 8 | PASS |
| 2 | 8 | GQA | 16 | FAIL |
| 4 | 4 | GQA | 16 | FAIL |

**Ruled out, each by a control:**

- **GQA** — `B=2 H=4 KVH=4` is pure MHA with no expand in the graph and fails
  identically to the GQA shape at the same B and H.
- **B×H (the BMM batch count)** — `B=1 H=8` and `B=2 H=4` are both B×H=8 and
  land on opposite sides.
- **A size collision between dims** — `B=2/KVH=2/exp=2` (all sizes equal) and
  `B=2/exp=4`, `B=4/exp=2` (all distinct) fail alike.
- **Name propagation through the GQA expand** — untested, not refuted: the
  `spyre_hint(named_dims=...)` attempt folded away and changed nothing (errors
  identical to the digit). Superseded anyway, since MHA fails too.

**Next step:** stop hypothesising and diff `bundle.mlir` across the minimal
pair, which differs in one parameter:

```bash
SENCORES=1 python validate_swa_4d_rank.py --batch 2 --heads 2 --kv-heads 2 --compile --dump
SENCORES=1 python validate_swa_4d_rank.py --batch 2 --heads 4 --kv-heads 4 --compile --dump
```

### 6.2 `batch == 1 and heads == 1` layout failure (degenerate, low priority)

`_resize_device_layout: cannot uniquely identify the stick host dim` on
`old_host_size=[1, 1, 256]` — two size-1 leading dims it cannot tell apart.
Only reachable when B and H are *both* 1; real models have many heads. Related
to issue #3116's stick-dim recovery.

### 6.3 GQA + decode SIGABRT (PRE-EXISTING, not ours)

`dxp_standalone` aborts with `DtException: Could not find any suitable
dimension mapping` (`ddl_conversion.cpp:2497`) on GQA decode (`Lq=1`).
Reproducible from `test_sliding_window_attention.py::gqa_decode_no_mask` with
`attn_mask=None`, `is_causal=True` and **no sliding hint anywhere** — the SDPA
path, not ours. It gates GQA-decode SWA independently of this work and needs a
backend fix. Worth filing on its own.

Note that test file is not on this branch: added in `e22622b`, removed in
`584a392`. It is the control that told us which GQA bug was whose, so it
probably should not stay deleted.

### 6.4 Main-side regression: #3248 breaks SWA decode at non-stick kv lengths

`test_sliding_window_kernel_mha_decode_causal_w64` (B=2, Lq=1, **Lkv=257**,
32 heads, window 64) fails at **90.9% mismatched elements** on any branch
carrying current main. It passes on `swa-windowed-decomposition` only because
that branch still sits on the older `e4cd21e`.

**Bisected to `2365a59` — "fix(sdpa): mask exp padding lanes so unpadded kv
seqlen doesn't produce inf" (#3248).** Not ours: reproduced with
`propagate_named_dims.py`, `coarse_tile.py`, `spyre_kernel.py` and
`loop_info.py` all reverted to main's versions, i.e. with none of the sliding
work in the build. `SPYRE_SWA_SLIDING_LOOP` is off by default, so increment 5d
is not involved either.

*Mechanism.* `_get_coordinate_mask` used to mask a padded dim only when
`arg.scales[dim] == -2` (the reduced stick dim). #3248 changed that to
`== -2 or mask_pointwise`, so for `exp` EVERY padded dim is masked with `-inf`,
unconditionally by op name. SWA's `exp` falls outside what that was tested
against, in two ways the commit's own comments call out:

> Multi-dim masking is UNTESTED. SDPA only pads the stick dim … treat
> multi-padded-dim pointwise ops as unverified.
>
> Masking is unconditional by op-name, not gated on whether the output actually
> feeds a contraction … TODO(consumer-gating).

SWA's `exp_scores` feeds BOTH a reduction (`.sum(dim=-1)`) and a contraction
(`matmul(exp_scores, v_blk)`), and it runs inside the `tiles={"kv_window": …}`
coarse-tiled loop, so the mask extent `iteration_space[dim] - padding` is
computed against a tiled iteration space rather than SDPA's untiled one. At
Lkv=257 the decode block's kv_len is 65 (kv_start=192, kv_end=257) — 63 padding
lanes in a 2-stick buffer.

*Note this is the same ragged tail the ceil-div fix (`413eb3c`) addresses*, and
the failure reproduces the pre-fix 90.9% signature exactly, so #3248 re-breaks
the computation that fix protects, by a different route.

The fix belongs in `superdsc.py` (the bandage is explicitly temporary — see
#3290 for the principled replacement), not in the SWA decomposition. Repro:

```bash
python3 -m pytest tests/inductor/test_sliding_window_attention_kernel.py \
    -k "mha_decode_causal_w64 and not 4096 and not 8192 and not long" -q
```

The stick-aligned decode cases (`_4096`, `_8192`, `_long`) all pass — only the
non-stick-aligned 257 fails. Increment 5d's sliding path does NOT cover this
shape either way: `plan_sliding_window` rejects `Lq=1` (not a whole Q block),
so it falls back to the unrolled path and is neither helped nor hurt.
