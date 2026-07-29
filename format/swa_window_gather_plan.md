# SWA via a compact gathered KV window — plan

**Branch:** `swa-window-gather` = `origin/main` (8adbf38) + three cherry-picks
that bring the SWA op and its tests, and *nothing else*:

| commit | what |
|---|---|
| `57c04aa` (`-x e22622b`) | `spyre::sliding_window_attention`, `spyre::sliding_window_block_mask`, the unrolled decomposition, both test files |
| `befd67e` (`-x 652e6bc`) | ceil-div fix for the `kv_window` tile count |
| `5d3af53` (`-x 49329de`) | drop the slow 4096/8192 prefill cases from the kernel test |

No sliding-hint frontend changes (`coarse_tile`, `superdsc`,
`propagate_named_dims` …) come along — this is a clean-frontend restart.

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

**There is no Python `for` loop.** The unrolled per-Q-block decomposition
(`spyre_sliding_window_attention`) is not the model for this work; it stays in
the tree only as a fallback and a numeric reference.

## 2. Decisions taken

| question | decision |
|---|---|
| structure | flash attention's hint nest verbatim, `window_size` replacing `max_seqlen_kv` |
| Python loop | **removed** |
| Q blocking | expressed as a **`spyre_hint`**, not a loop |
| buffer | gathered window `[B, Hq, N, Wb, E]` — decode (`N=1`) is exactly `[B, Hq, W, E]` |
| who builds it | new custom op **`spyre::gather_kv_window`** |
| window position | compile-time ints, resolved once per call by `plan_window_gather` |
| op also does | GQA expand to `Hq`; K and V in one call; emits the band mask |
| scope | prefill **and** decode in one structure |

## 3. Why the buffer needs a Q-block axis

A single shared `[B, H, W, E]` window is correct only when **every** query row
has the same window — that is, `Lq == 1` (decode). At `Lq > 1` the rows'
windows differ, so one shared buffer gives the wrong answer.

The fix is to make the Q block an **axis of the buffer** rather than an
iteration of a loop:

```text
q5     : [B, H, N, q_block, E]        N = num_q_blocks, Lq = N * q_block
k_win  : [B, H, N, Wb,      E]        window i belongs to Q block i
```

`N` is then a **batch dimension of the matmul**, so `torch.matmul` pairs Q
block `i` with window `i` automatically. That is the diagonal Q↔KV pairing SWA
needs, obtained from ordinary batched-matmul semantics — **no coupling
machinery, no sliding read, no loop.** This is precisely what the copy buys:
gathering de-overlaps the data, so a plain partition does the job that
previously required a sliding hint.

At `N == 1` this degenerates to exactly `[B, H, W, E]`.

### Width within a block

Rows inside one Q block have staggered windows: the row at coordinate `c`
attends `[c-W+1, c]`, so a block of `q_block` rows spans `W + q_block - 1`
columns. With the read start floored to a stick boundary:

> **`Wb = W + q_block`** for prefill, and **`Wb = W`** for decode (one row, no
> stagger).

Both fall out of `plan_window_gather` (increment 1), which derives the width
from the per-block maximum rather than assuming alignment.

### Out-of-range windows

Early blocks want a negative start; the last block wants to run past `Lkv`.
Rather than zero-filling or clamping per row, **shift the whole window**:

```python
read_start = min(max(win_start, 0), Lkv - Wb)     # stick-aligned
```

Every column is then real, contiguous data and the buffer shape is constant for
every block; the band mask removes the columns the shift brings in.

## 4. The structure (this is the deliverable)

Flash attention's body, unchanged except for what it reads:

```python
k_win, v_win, band = torch.ops.spyre.gather_kv_window(key, value, ...)
#   k_win, v_win : [B, Hq, N, Wb, E]
#   band         : [1, 1, N, q_block, Wb]     0.0 keep / -inf masked

q5 = query.view(B, Hq, N, q_block, E)
output = torch.zeros_like(q5)
# M / denominator accumulators over [B, Hq, N, q_block], exactly as SDPA

with spyre_hint(tiles={"batch_size": max(1, B // 2)}):
  with spyre_hint(tiles={"num_heads": max(1, Hq // 4)}):
    with spyre_hint(tiles={"num_q_blocks": N}):                  # the Q hint
      with spyre_hint(tiles={"window_size": max(1, Wb // 64)}):  # was max_seqlen_kv
        with spyre_hint(work_div={"num_heads": 4,
                                  "num_q_blocks": 8,
                                  "window_size": 8}):
            scaled_keys = k_win * scaling_factor
            scores = torch.matmul(q5 * scaling_factor,
                                  scaled_keys.transpose(-1, -2))  # [B,H,N,q_block,Wb]
            scores = scores + band

            block_max   = torch.amax(scores, dim=-1)
            max_running = torch.maximum(M, block_max)
            exp_scores  = torch.exp(scores - max_running.unsqueeze(-1))
            correction  = torch.exp(M - max_running)

            denominator = copy_f(denominator * correction
                                 + exp_scores.sum(dim=-1), denominator)
            output      = copy_f(output * correction.unsqueeze(-1)
                                 + torch.matmul(exp_scores, v_win), output)
            M           = copy_f(max_running, M)

output = copy_f(output / denominator.unsqueeze(-1), output)
return output.view(B, Hq, Lq, E)
```

Line for line the same as `spyre__sdpa_overrideable` (lines 604–660), with four
substitutions: `key`/`value` → the gathered window; `causal_mask` → `band`;
`max_seqlen_q` → the `(N, q_block)` pair; **`max_seqlen_kv` → `window_size`**.

Hint count matches SDPA: four `tiles` dims plus `work_div`.

## 5. The op

```python
@torch.library.custom_op("spyre::gather_kv_window", mutates_args=(), ...)
def gather_kv_window(
    key: torch.Tensor,      # [B, Hkv, Lkv, E]
    value: torch.Tensor,    # [B, Hkv, Lkv, E]
    seqlen_q: int, window_size: int, num_heads: int,
    q_block: int, is_causal: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    # k_win, v_win : [B, Hq, N, Wb, E];  band : [1, 1, N, q_block, Wb]
```

**Route A (v1): a named op with a *decomposition*, not a new kernel.** It
expands to per-block slices stacked along `N`, plus the GQA expand and the
CPU-built band. No new lowering and no C++ — and Antoni explicitly deferred an
optimized gather. Route B (a real gather kernel) is costed only if layer B
measurements demand it.

## 6. Risks, in order

1. **Rank 5.** SDPA works at rank 4; `[B, H, N, Wb, E]` is rank 5, and a
   persistent rank-5 matmul (three batch dims) is untested here. *Fallback if
   the backend balks:* fold `N` into the batch axis → `[B*N, H, q_block, E]`
   and `[B*N, H, Wb, E]`, rank 4, at the cost of the separate Q-block hint.
   **Settle this first — it gates the whole structure.**
2. **Prefill memory.** The buffer holds `N * Wb` KV rows against the cache's
   `Lkv`. Decode is a large *saving*; prefill is a *blowup* growing with the
   window:

   | case | cache rows | buffer rows | ratio |
   |---|---:|---:|---:|
   | decode `Lkv=4096, W=64` | 4096 | 64 | **0.016x** |
   | prefill `Lq=Lkv=512, W=64` | 512 | 8 × 128 = 1024 | 2x |
   | prefill `Lq=Lkv=8192, W=4096` | 8192 | 128 × 4160 = 532k | **65x** |

   This is the intrinsic cost of copying a window per block, and it is the
   strongest argument for letting Inductor keep the gather a *view* where it
   can (inc 9). Flag it before anyone runs large-window prefill.
3. **The hints may be no-ops.** Compiling `spyre_sliding_window_attention`
   previously emitted **zero** device loops — ~40 flat `sdsc_execute` groups at
   seqlen 2048, no `scf.for`. `assign_dim_hints` gates on **named dims**, and
   the decomposition never declared `kv_window` / `max_seqlen_q`; it also wanted
   `allow_all_ops_in_lx_planning=True`. A numeric pass will **not** catch this —
   untiled code returns the right answer with one big intermediate. Tiling must
   be verified **structurally**.
4. **`Lq % q_block`.** The reshape to `[B,H,N,q_block,E]` needs exact
   divisibility; otherwise pad Q or fall back.

## 7. Increment ladder

Layer A is the current job. `[HW]` = the user runs it on the pod.

- [x] **Inc 1 — placement arithmetic.** `swa_window_gather.py`:
      `WindowGatherPlan` + `plan_window_gather`, torch-free. Commit `c49f04e`.
- [x] **Inc 2 — CPU equivalence.** 38/38 green on the pod. Exact claim: the
      unmasked column set inside the compact buffer is identical to the
      full-width band mask's. Roundoff claim: float64 agreement `<1e-12`.
      Commit `b89d41f`.
- [ ] **Inc 3 — rank-5 spike.** `[HW]` Smallest possible probe: does a
      `[B,H,N,q,E] @ [B,H,N,E,Wb]` batched matmul compile and run? Risk 1 gates
      everything else, so settle it before writing the op.
- [ ] **Inc 4 — the op.** `spyre::gather_kv_window` + fake + decomposition
      (route A), returning the stacked buffer and the band. Compile-only, then
      `[HW]` one shape.
- [ ] **Inc 5 — the body.** The §4 structure behind `config.swa_window_gather`
      (`SPYRE_SWA_WINDOW_GATHER=1`, default OFF). `[HW]` decode first (`N=1`,
      the simplest case), then prefill.
- [ ] **Inc 6 — verify the tiling is real.** `[HW]` Parse `bundle.mlir` for
      `scf.for` trip counts and count `sdsc_execute` inside vs outside the loop.
      Risk 3 means a green numeric run proves nothing about tiling.
- [ ] **Inc 7 — shape sweep.** `[HW]` `B>1`, GQA, decode, prefill. Layer A is
      done when this is green **and** inc 6 shows real loops.

Layer B — the copy ("worry about that later"):

- [ ] **Inc 8 — benchmark.** `[HW]` vs the mask-based SDPA baseline.
- [ ] **Inc 9 — materialize vs fuse.** `[HW]` The work reduction is banked
      either way (the matmul reads `Wb` columns regardless); this is about the
      copy's cost against the layout it buys. Risk 2 makes it matter more for
      prefill than earlier framing suggested.
- [ ] **Inc 10 — route B**, only if 8 and 9 demand it.

Layer C — projections and RoPE. Not ours yet.

## 8. Deferred

- **Bidirectional** (`is_causal=False`) — needs `Wb = 2W + q_block`.
- **`Lq % q_block != 0`** — pad or fall back.
- **Multi-core (`SENCORES > 1`)** — `work_div` carries over from SDPA
  unchanged, but no claim is made until inc 8 measures it.
