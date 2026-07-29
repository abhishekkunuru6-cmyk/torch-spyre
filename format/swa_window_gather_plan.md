# SWA via a compact gathered KV window — plan

**Branch:** `swa-window-gather` = `origin/main` (8adbf38) + three cherry-picks
that bring the SWA op and its tests, and *nothing else*:

| commit | what |
|---|---|
| `57c04aa` (`-x e22622b`) | `spyre::sliding_window_attention`, `spyre::sliding_window_block_mask`, the unrolled decomposition, both test files |
| `befd67e` (`-x 652e6bc`) | ceil-div fix for the `kv_window` tile count |
| `5d3af53` (`-x 49329de`) | drop the slow 4096/8192 prefill cases from the kernel test |

No sliding-hint frontend changes come along — this is a clean-frontend restart.

## 0. Steer from the DevSync (Antoni)

> "Assume it's been computed correctly — query, key and value have been computed
> correctly — and **get the math running for the sliding window attention**."

> "Let's build it up like an onion. Let's start with just the attention itself —
> inputs query, key, value, and then you get the attention output."

And on copying each Q index's window into a separate buffer: *"Do not worry
about that. We'll have to worry about that later."*

So **layer A = attention math correctness; layer B = the cost of the copy;
layer C = projections and RoPE, not ours yet.**

## 1. The idea

Attend against a **compact gathered KV window** instead of the full cache
behind a mask, in a body **structurally identical to the existing flash
attention** (`spyre__sdpa_overrideable`, `decompositions.py:527`) — the same
nested hints and the same online-softmax accumulators — with **`window_size` in
place of `max_seqlen_kv`**.

**There is no Python `for` loop in the attention body.** The unrolled
decomposition (`spyre_sliding_window_attention`) stays in the tree only as a
fallback and a numeric reference.

**Everything is rank 4.** Increment 3 proved rank-5 matmul is unsupported
(`lowering.py` `spyre_bmm` accepts only `(3,3)`, `(4,4)`, `(3,2)`), so the
Q-block axis is folded into the head axis rather than kept separate.

## 2. Decisions taken

| question | decision |
|---|---|
| structure | flash attention's hint nest, `window_size` replacing `max_seqlen_kv` |
| Python loop | **removed** from the body |
| rank | **4 throughout** — Q blocks folded into heads |
| fold order | **block-major**, index `n*Hq + h` (see §4) |
| buffer | `[B, N*Hq, Wb, E]`; decode (`N=1`) is exactly `[B, Hq, W, E]` |
| who builds it | custom op **`spyre::gather_kv_window`** |
| window position | compile-time ints from `plan_window_gather` |
| op also does | GQA expand; K and V in one call; emits the band mask |
| scope | prefill **and** decode in one structure |

## 3. How a Q block finds its window

Worked example: `Lq = Lkv = 256`, `W = 64`, `q_block = 64`, so `N = 4`,
`Wb = W + q_block = 128`, `q_kv_offset = 0`.

**Step 1 — derive (Python, trace time, pure integers).**

```text
read_start(n) = clamp( floor64(q_kv_offset + q_block*n - W + 1), 0, Lkv - Wb )
```

| block `n` | q rows | rows' true windows | `read_start` | buffer covers |
|---:|---|---|---:|---|
| 0 | 0–63 | `[0,0] … [0,63]` | 0 | `[0, 128)` |
| 1 | 64–127 | `[1,64] … [64,127]` | 0 | `[0, 128)` |
| 2 | 128–191 | `[65,128] … [128,191]` | 64 | `[64, 192)` |
| 3 | 192–255 | `[129,192] … [192,255]` | 128 | `[128, 256)` |

Implemented as `plan_window_gather(...).read_start(n)` (increment 1); the
coverage invariant is verified exhaustively for every row (increment 2).

**Step 2 — gather.** Window `n` is *physically copied* into slot `n`. After
that there is nothing left to derive: the association is the buffer's layout.

**Step 3 — pair.** The query is folded with the same indexing, so a rank-4
batched matmul pairs equal batch indices, i.e. block `n` with window `n`. This
is what the copy buys — the earlier sliding approach needed a *read address* to
advance per iteration (new machinery in `coarse_tile`/`superdsc`); gathering
de-overlaps the data so plain positional matching suffices.

**Step 4 — mask.** The band removes what the buffer over-covers: the *stagger*
inside a block (row 128 sees `[65,128]`, row 191 sees `[128,191]`) and the
*clamp overhang* at the ragged ends.

### Width and out-of-range

Rows within a block have staggered windows spanning `W + q_block - 1` columns,
so **`Wb = W + q_block`** for prefill and **`Wb = W`** for decode (one row, no
stagger). Early blocks want a negative start and the last wants to run past
`Lkv`; rather than zero-filling, the whole window **shifts**
(`clamp(win_start, 0, Lkv - Wb)`), keeping one contiguous stick-aligned read
and a constant buffer shape.

## 4. Why block-major, and why it is not arbitrary

The folded axis is `n*Hq + h` (block outer, head inner). This is forced by
three things at once:

1. **Query fold and output unfold are `cat`s of slices at rank 4** — no
   permute, no copy beyond the gather:

   ```python
   q4  = cat([query[:, :, n*qb:(n+1)*qb, :] for n in range(N)], dim=1)
   out = cat([out4[:, n*Hq:(n+1)*Hq, :, :] for n in range(N)], dim=2)
   ```

2. **The band stays minimal.** `scores.view(B, N, Hq, qb, Wb)` is a *free*
   split under block-major, so the band is `[1, N, 1, qb, Wb]` — `Lq × Wb`
   elements, the same as any windowed mask. Head-major (`h*N + n`) cannot
   broadcast a per-block band across a flat axis, forcing `N*Hq*qb*Wb`
   (~1e9 elements at Gemma sizes).

3. **The rank-5 broadcast add it needs is proven.** Increment 3 showed rank-5
   pointwise/broadcast/`amax`/`sum`/`view` all work; only *matmul* does not.

**The contract:** the gather and the query fold must use the *same* order. Get
them out of step and every slot pairs the wrong window with the wrong head —
which still runs and returns wrong numbers. This is the single most likely way
to be subtly wrong, so increment 4 tests it explicitly.

## 5. The structure

```python
k4, v4, band = torch.ops.spyre.gather_kv_window(key, value, ...)
#   k4, v4 : [B, N*Hq, Wb, E]          band : [1, N, 1, q_block, Wb]

q4 = torch.cat([query[:, :, n*qb:(n+1)*qb, :] for n in range(N)], dim=1)
# M / denominator / output accumulators over [B, N*Hq, q_block, ...], as SDPA

with spyre_hint(tiles={"batch_size": max(1, B // 2)}):
  with spyre_hint(tiles={"num_heads": max(1, (N * Hq) // 4)}):
    with spyre_hint(tiles={"window_size": max(1, Wb // 64)}):   # was max_seqlen_kv
      with spyre_hint(work_div={"num_heads": 4, "window_size": 8}):
          scores = torch.matmul(q4 * scale, (k4 * scale).transpose(-1, -2))
          scores = (scores.view(B, N, Hq, qb, Wb) + band).view(B, N * Hq, qb, Wb)

          block_max   = torch.amax(scores, dim=-1)
          max_running = torch.maximum(M, block_max)
          exp_scores  = torch.exp(scores - max_running.unsqueeze(-1))
          correction  = torch.exp(M - max_running)

          denominator = copy_f(denominator * correction
                               + exp_scores.sum(dim=-1), denominator)
          output      = copy_f(output * correction.unsqueeze(-1)
                               + torch.matmul(exp_scores, v4), output)
          M           = copy_f(max_running, M)

output = copy_f(output / denominator.unsqueeze(-1), output)
return torch.cat([output[:, n*Hq:(n+1)*Hq] for n in range(N)], dim=2)
```

The `num_heads` axis is now `N*Hq` rather than `Hq` — every `(block, head)`
pair is an independent attention problem, so this gives `work_div` strictly
more parallelism than the unfolded form would.

## 6. The op

```python
spyre::gather_kv_window(key, value, seqlen_q, window_size,
                        num_heads, q_block, is_causal)
    -> (k4, v4, band)
```

**Route A: a named op with a *decomposition*, not a new kernel.** It expands to
per-block slices, the GQA expand, and a `cat` — plus a CPU-built band. No new
lowering and no C++; Antoni explicitly deferred an optimized gather. Route B (a
real gather kernel) is costed only if layer B measurements demand it.

**Slice before expanding, never after.** The unrolled op carries a comment
earned the hard way: expanding the full-length K/V and slicing afterwards
feeds the stick-padding pass a consumer needing a small window out of a tensor
it still thinks is full-length, which it cannot reconcile
(`lower_pad_sequence: pad_extent=-129 ... original_size_dim=257`).

## 7. Risks, in order

1. **The hints may be no-ops.** Compiling `spyre_sliding_window_attention`
   previously emitted **zero** device loops — ~40 flat `sdsc_execute` groups at
   seqlen 2048, no `scf.for`. `assign_dim_hints` gates on **named dims**, and
   `propagate_named_dims` raises `Named dim 'X' used in name_tensor_dims but
   not declared` with an `_untracked_<size>` fallback that only warns. A numeric
   pass will **not** catch this — untiled code returns the right answer with one
   big intermediate. Verify **structurally**.
2. **Fold-order mismatch** (§4) — wrong numbers, not a crash.
3. **Prefill memory.** The buffer holds `N * Wb` KV rows against the cache's
   `Lkv`:

   | case | cache rows | buffer rows | ratio |
   |---|---:|---:|---:|
   | decode `Lkv=4096, W=64` | 4096 | 64 | **0.016x** |
   | prefill `Lq=Lkv=512, W=64` | 512 | 1024 | 2x |
   | prefill `Lq=Lkv=8192, W=4096` | 8192 | 532k | **65x** |

   Intrinsic to copying a window per block. **Worth raising with Antoni before
   investing in large-window prefill.**
4. **`Lq % q_block`** — the block slicing needs exact divisibility; otherwise
   pad Q or fall back.

## 8. Increment ladder

Layer A is the current job. `[HW]` = the user runs it on the pod.

- [x] **Inc 1 — placement arithmetic.** `swa_window_gather.py`. `c49f04e`.
- [x] **Inc 2 — CPU equivalence.** 38/38 green. `b89d41f`.
- [x] **Inc 3 — rank spike.** `6c1ddd5`. **Rank-5 matmul UNSUPPORTED**; rank-5
      pointwise/`amax`/`sum`/`view` all fine; rank-4 body passes. Decided
      against extending `spyre_bmm` — that is shared backend code on every
      BMM's lowering path, and folding to rank 4 costs nothing.
- [ ] **Inc 4 — the op.** `spyre::gather_kv_window` + fake + decomposition,
      plus an explicit **fold-order** test (risk 2).
- [ ] **Inc 5 — the body.** The §5 structure behind `config.swa_window_gather`
      (`SPYRE_SWA_WINDOW_GATHER=1`, default OFF). `[HW]` decode first (`N=1`),
      then prefill.
- [ ] **Inc 6 — verify the tiling is real.** `[HW]` Parse `bundle.mlir` for
      `scf.for` trip counts and `sdsc_execute` counts inside vs outside. Risk 1
      means a green numeric run proves nothing about tiling.
- [ ] **Inc 7 — shape sweep.** `[HW]` `B>1`, GQA, decode, prefill. Layer A is
      done when this is green **and** inc 6 shows real loops.

Layer B — the copy ("worry about that later"):

- [ ] **Inc 8 — benchmark** `[HW]` vs the mask-based SDPA baseline.
- [ ] **Inc 9 — materialize vs fuse.** `[HW]` The work reduction is banked
      either way; this is the copy's cost against the layout it buys.
- [ ] **Inc 10 — route B**, only if 8 and 9 demand it.

Layer C — projections and RoPE. Not ours yet.

## 9. Deferred

- **Bidirectional** (`is_causal=False`) — needs `Wb = 2W + q_block`.
- **`Lq % q_block != 0`** — pad or fall back.
- **Multi-core (`SENCORES > 1`)** — `work_div` carries over unchanged, no claim
  until inc 8 measures it.
