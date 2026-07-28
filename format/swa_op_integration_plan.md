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

**NOT done — the actual SWA op is UNCHANGED.** `spyre_sliding_window_attention`
(`torch_spyre/_inductor/decompositions.py:681`) still runs the original
Python-unrolled `for qi in range(num_q_blocks)` loop (N separate op-groups). The
sliding hint is NOT wired into it. "Tests pass" = the *building block* works;
the *payoff* (one sliding `scf.for` instead of the unroll) is not implemented.

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

  - [ ] **5e — CURRENT BLOCKER: op-group contiguity.** `coarse_tile` requires
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

    Current attempt (`10ab9f8`, **UNVERIFIED — the run that was meant to test
    it aborted on a failed `git pull` and silently tested the old code**):
    pass `schedule_anchor=k_scaled`, a tensor the op ignores for values,
    purely to create a dependency edge a custom op cannot optimize away.
    Risk: this only guarantees *no earlier than* `k_scaled`, not *as early as*
    — and every observation so far says the scheduler defers toward first use.
    If the split persists at the same point, the fallback is building the mask
    as ordinary Pointwise ops (`arange`/compare/`where`) **inside** the sliding
    scope, which carries its own open question: `arange` has no tensor inputs
    and `dim_hint` propagation flows through tensor data flow, so it is not
    obvious `coarse_tile` would infer QS/KV tiling for it without explicit
    naming.

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
```

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
