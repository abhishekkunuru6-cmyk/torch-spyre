# SWA via a compact gathered KV window — plan

**Branch:** `swa-window-gather` = `origin/main` (8adbf38) + three cherry-picks
that bring the SWA op and its tests, and *nothing else*:

| commit | what |
|---|---|
| `57c04aa` (`-x e22622b`) | `spyre::sliding_window_attention`, `spyre::sliding_window_block_mask`, the unrolled decomposition, both test files |
| `befd67e` (`-x 652e6bc`) | ceil-div fix for the `kv_window` tile count |
| `5d3af53` (`-x 49329de`) | drop the slow 4096/8192 prefill cases from the kernel test |

No sliding-hint frontend changes (`coarse_tile`, `superdsc`, `propagate_named_dims`
…) come along — this branch is a clean-frontend restart. The unrolled
per-Q-block loop stays in the tree as the **fallback and the reference**.

## 0. Steer from the DevSync (Antoni)

Two directions from the review, and they set the whole shape of the ladder:

> "Assume it's been computed correctly — query, key and value have been computed
> correctly — and **get the math running for the sliding window attention**. And
> then we'll worry about how do we optimize [...] for the projections and
> everything, which is the next step."

> "Let's build it up like an onion. Let's start with just the attention itself —
> inputs query, key, value, and then you get the attention output — and from
> there we can build the onion."

And directly on the question of copying each Q index's window into a separate
buffer:

> "Do not worry about that. We'll have to worry about that later."

So: **correctness of the attention math is layer A; the efficiency of getting
the window into the buffer is layer B; projections and RoPE are layer C and not
ours yet.** §7 is ordered accordingly. This is also the second reason route A
(§5) is right — an optimized gather kernel is explicitly deferred, not merely
expensive.

## 1. The idea

Instead of attending against the full `[B, Hkv, Lkv, D]` KV cache with a mask,
**materialize only the in-window KV into a compact buffer** and run a
flash-attention body against that buffer. Per Q block:

```
gather  ->  k_win, v_win : [B, Hq, Wb, D]      (Wb << Lkv)
flash   ->  out_blk      : [B, Hq, q_len, D]
```

The work per Q block stops scaling with `Lkv` and starts scaling with `Wb`.

## 2. Decisions taken (from the design discussion)

| question | decision |
|---|---|
| scope | prefill **and** decode from the start |
| buffer width | `Wb = W + q_block` (see §3 — width exactly `W` is arithmetically impossible for a 64-row Q block) |
| `W` | **exactly `window_size`**, assumed a multiple of 64; fall back otherwise |
| who builds it | a new custom op **`spyre::gather_kv_window`** |
| window position | **compile-time ints** — `q_start` etc. are Python ints, op traced once per Q block |
| op also does | GQA expand to `Hq`; K and V in **one call**; **emits the band mask** |
| op does *not* do | zero-fill out-of-range (see §4 — handled by shifting the read instead) |

## 3. The width arithmetic (why `Wb = W + 64`, not `W`)

Let `q_kv_offset = Lkv - Lq`, `a = q_kv_offset + q_start` = absolute KV
coordinate of the block's first row. Causal row at coordinate `c` attends
`[c-W+1, c]`. Over the 64 rows of one Q block the union is

```
[a - W + 1,  a + q_len - 1]        width = W + q_len - 1  <=  W + 63
```

so a width-`W` buffer cannot hold one Q block's window; only `q_block == 1`
would make `Wb == W`. With stick alignment folded in — `win_start` floor-aligned
to 64 so the DMA starts on a stick boundary:

```
win_start = align64_floor(a - W + 1)
width     = (a + 63) - win_start + 1
```

When `W % 64 == 0` **and** `q_kv_offset % 64 == 0`, `a - W + 1` sits exactly 1
above the boundary `a - W`, so `win_start = a - W` and `width = W + 64` exactly.
Hence:

> **`Wb = W + 64`**, given `W % 64 == 0` and `q_kv_offset % 64 == 0`.

Drop either alignment precondition and floor-aligning can lose up to 63 more
columns, forcing `Wb = W + 128`. v1 **falls back** rather than widening.

*Decode note:* at `Lq = 1`, `q_len = 1`, so only `W` of the `W+64` columns can
ever be valid. A `q_block = 1` special case would give `Wb = W`. Deferred —
listed in §8.

## 4. Out-of-range windows: shift the read, don't clamp per row

Early blocks want `win_start < 0`; the last block wants `win_start + Wb > Lkv`.
Since the op emits the band mask itself, it knows exactly which columns are
invalid, so **no zero-fill is needed**. Rather than clamping row-by-row (which
duplicates edge rows and breaks the contiguous DMA), clamp the *whole window*:

```python
read_start = min(max(win_start, 0), Lkv - Wb)     # still 64-aligned
```

Every column in the buffer is then real, contiguous, stick-aligned data; the
band mask is built against `read_start` and zeroes whatever the row's true
window does not cover. This keeps one clean DMA per block and a constant
`[B, Hq, Wb, D]` shape for every block including the ragged ends.

Requires `Lkv >= Wb`. When the whole cache is narrower than one window,
sliding-window attention *is* full attention — fall back.

**Preconditions for the gather path** (else return `None` and take the unrolled
fallback): `W % 64 == 0`, `q_kv_offset % 64 == 0`, `Lkv % 64 == 0`,
`Lkv >= Wb`, `is_causal` (bidirectional needs `Wb = 2W + 64`, see §8).

## 5. The op

```python
@torch.library.custom_op("spyre::gather_kv_window", mutates_args=(), ...)
def gather_kv_window(
    key: torch.Tensor,      # [B, Hkv, Lkv, D]
    value: torch.Tensor,    # [B, Hkv, Lkv, D]
    q_start: int, q_end: int, q_kv_offset: int,
    window_size: int, num_heads: int, is_causal: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    # k_win [B, Hq, Wb, D], v_win [B, Hq, Wb, D], band [1, 1, q_len, Wb]
```

**Implementation route A (v1, recommended): a named op with a *decomposition*,
not a new kernel.** Register the op for graph identity, then
`register_spyre_decomposition` expands it to `slice -> expand -> flatten` for
K and V plus the existing CPU-built band mask. No new lowering, no C++, and
Inductor lowers the slices through paths that already work. The op is a
*boundary*, which is what makes it swappable later.

**Route B (later, only if measured hot):** a real gather kernel with its own
lowering. Costed only after route A gives a number to beat.

Two things to watch, both layer B (measure later, do not guess now):

- **Materialize vs. fuse.** A slice is a view, and Inductor may fuse it into the
  matmul rather than building a compact buffer. **This does not cost the work
  reduction** — `q @ k_win^T` reads `Wb` columns whether `k_win` is a view or a
  real buffer, so the out-of-window matmuls are skipped either way. What
  materialization changes is an *extra copy*, which only pays for itself if it
  buys better DMA/layout (contiguous, stick-aligned operand). Forcing it uses
  the `torch.ops.spyre.copy_f`-into-zeros pattern the op already uses for its
  accumulators. Purely an optimization question — §7 layer B.
- **GQA expand inside the gather** means the copy is `Hq`-wide rather than
  `Hkv`-wide — more bytes copied, but one op. If GQA regresses, moving the
  expand back out is a one-line change.

## 6. The flash body

Structurally the existing inner online-softmax sweep, with one change that
matters: `kv_len` becomes the **constant** `Wb` for every block, instead of a
per-block ragged length.

```python
with spyre_hint(tiles={"kv_window": Wb // 64}):     # constant across blocks
    scores = matmul(q_blk * s, (k_win * s).transpose(-1, -2)) + band
    # running max / denominator / out_blk accumulators, unchanged
```

Constant shape across blocks means every block emits an **identical** kernel —
which is also the precondition for later collapsing the Python unroll into one
device loop, if that is ever wanted.

## 7. Increment ladder

Ordered by §0's onion. **Layer A is the whole job until it is done** — the math,
`(query, key, value) -> attention output`, nothing else. No HW in the sandbox;
the user runs anything marked **[HW]** on the pod.

### Layer A — get the math running (the current job)

- [ ] **Inc 0 — baseline.** [HW] Run both test files on this branch *before any
      change*, so we know what passes. Expect one known failure:
      `mha_decode_causal_w64` at `Lkv=257` is broken on current main by #3248
      (a main-side regression, not ours).
- [ ] **Inc 1 — `swa_window_gather.py` planning module.** Frozen
      `WindowGatherPlan` + `plan_window_gather()` returning `None` for every
      unsupported shape in §4. Pure arithmetic. Unit tests first, no HW.
- [ ] **Inc 2 — CPU equivalence.** Prove the gathered-buffer model computes the
      same attention as the full masked reference, across a shape sweep. No HW.

  ```
Two claims, and it matters which is which:
  
  1. **Exact, and this is the real content.** For every query row, the set
     of KV columns left unmasked inside the compact buffer is *identical*
     to the set the full-width band mask leaves unmasked. This is a
     combinatorial claim about the window arithmetic, so it is exactly
     testable — no tolerance.
  2. **Roundoff-level, in float64.** Output values agree to ~1e-14. This
     cannot be bit-exact and should not be asserted as such: the gathered
     softmax reduces `buffer_width` terms where the reference reduces
     `seqlen_kv`, and although the extra entries are exact zeros, `sum`
     blocks its reduction differently at different widths, so the nonzero
   terms group differently.
  
  If claim 1 fails, the window arithmetic is wrong and nothing downstream
  matters.

  ```
- [ ] **Inc 3 — the op.** `spyre::gather_kv_window` + fake + decomposition
      (route A). Compile-only first, then [HW] a single prefill shape.
- [ ] **Inc 4 — wire it in.** `spyre_sliding_window_attention` takes the gather
      path when `config.swa_window_gather` is set and the plan is not `None`.
    [HW] the `b1` prefill cases.
- [ ] **Inc 5 — shape sweep.** [HW] `B>1`, GQA, decode. Band the failures rather
    than chasing the first one. **Layer A is done when this sweep is green.**

### Layer B — the copy, once the math is right ("worry about that later")

- [ ] **Inc 6 — benchmark.** [HW] vs the unrolled path and vs the mask-based
      SDPA baseline, at prefill 256/2048/4096 and decode 4096→131072. First
      evidence that skipping out-of-window matmuls actually pays.
- [ ] **Inc 7 — materialize vs fuse.** [HW] Measure whether forcing the compact
      buffer to materialize beats letting Inductor fuse the slice away. Note
      (§5) the work reduction is already banked either way — this is about the
      cost of the extra copy against the layout it buys.
- [ ] **Inc 8 — route B, only if inc 6/7 say so.** A real gather kernel with its
      own lowering.

### Layer C — not ours yet

Projections and RoPE, per §0. Out of scope for this branch.

**Flag:** `config.swa_window_gather` / `SPYRE_SWA_WINDOW_GATHER=1`, **default
OFF**. With it unset the op runs the existing unrolled loop, so nothing on this
branch can regress the default path.

## 8. Deferred, deliberately

- **Bidirectional (`is_causal=False`)** — union is `[c-W+1, c+W-1]`, so
  `Wb = 2W + 64`. Mechanically the same, just wider; falls back for now.
- **`q_block = 1` for decode**, which would give `Wb = W` exactly (§3).
- **Unaligned `q_kv_offset` / `Lkv`**, which need `Wb = W + 128` (§3).
- **Collapsing the unroll to one device loop.** The constant per-block shape
  from §6 is the precondition; this plan does not depend on it.
- **Multi-core (`SENCORES > 1`).** The existing `work_div` hints carry over
  untouched, but no claim is made until inc 6 measures it.
