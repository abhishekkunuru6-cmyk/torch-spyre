# Sliding-window attention

In sliding-window attention each query row attends only to the last
`window_size` key positions. The `[seqlen_q, seqlen_kv]` score matrix is
therefore a narrow diagonal band, and everything outside it is masked to
`-inf` before the softmax.

The obvious implementation — compute every score, then mask — does the full
quadratic work and throws most of it away. At a 4096-token cache with a
64-token window, 98% of the scores are discarded. This decomposition does not
compute them.

## The idea

A query row at cache coordinate `c` can only reach keys in `[c - W + 1, c]`.
So instead of attending against the whole cache behind a mask, copy just that
range out of the cache and attend against the copy.

Rows are processed in blocks of `q_block` rows at a time, and rows within a
block have *staggered* windows — the first row of a 64-row block reaches
`[c-W+1, c]`, the last reaches `[c-W+64, c+63]`. Their union spans `W + 64`
columns, so a block's buffer is `W + q_block` rows wide, not `W`.

```text
cache:   ........................................................
block 0:      [<--- buffer_width --->]
block 1:              [<--- buffer_width --->]
block 2:                      [<--- buffer_width --->]
```

Two consequences make this cheap:

- **The buffer does not grow with the cache.** `buffer_width` depends on
  `window_size` and `q_block` only. A decode step against a 4096-row cache
  reads 64 rows.
- **Every block's buffer is the same shape**, so one allocation serves all of
  them in turn. The peak scratch is one window, not one per block.

Decode falls out of the same arithmetic with no special case: `seqlen_q == 1`
gives one block of one row, no stagger, and `buffer_width == window_size`.

## Code flow

`spyre::sliding_window_attention(query, key, value, window_size, is_causal,
scale)` is the entry point. Its decomposition runs during Inductor lowering,
so every quantity below is a compile-time Python integer, not a tensor.

```text
spyre_sliding_window_attention                     decompositions.py
│
├── choose_q_block(seqlen_q)                       swa_window_gather.py
│      1 for decode, 64 otherwise, or None if seqlen_q divides into neither
│
├── plan_window_gather(...) -> WindowGatherPlan | None
│      buffer_width, num_q_blocks, and read_start(n) for every block
│
├── if the plan is None:                           ── the fallback ──
│      log why, then _masked_sliding_window_attention:
│      a full band mask + spyre__sdpa_overrideable. Correct for any shape,
│      quadratic, and only reached by shapes the placement cannot express.
│
└── _window_roll_attention(query, key, value, plan, scaling_factor, num_heads)
       │
       for each Q block n (a Python loop, unrolled at trace time):
       │
       ├── spyre::gather_kv_window(key, value, read_start(n), buffer_width, …)
       │      one slice of K (transposed) and V, GQA-expanded, plus the band
       │      for this block
       │
       ├── query[:, :, n*q_block : (n+1)*q_block, :]
       │
       └── under four spyre_hints, a flash-attention step:
              scores = q @ k_win  (+ band, unless the band masks nothing)
              online-softmax accumulators, then scores @ v_win
       │
       └── torch.cat(per-block outputs, dim=2)
```

### Where each block reads from

`WindowGatherPlan.read_start(n)` is the only non-obvious arithmetic:

```python
window_origin = max(0, floor_to_stick(first_coord - window_size + 1))
read_start    = min(window_origin, seqlen_kv - buffer_width)
```

The `floor_to_stick` keeps every read on a 128-byte stick boundary. The two
clamps handle the ragged ends: the first blocks would want to start before the
cache, and the last would want to run past it. Rather than reading out of
bounds or changing the buffer's shape, **the window shifts** and stays inside
the cache. The columns the shift brings in are ones those rows cannot attend
to — the band mask removes them.

That is what keeps `buffer_width` constant for every block, which is in turn
what lets one buffer serve them all.

### The band

`spyre::window_band_mask` builds one block's additive mask,
`[1, 1, q_block, buffer_width]`, broadcasting over batch and heads. It removes
what the buffer over-covers: the stagger between rows inside the block, and
the overhang from a shifted read.

When a block's rows can attend *every* column of their buffer, the band is all
zeros and the add is skipped (`WindowGatherPlan.block_is_fully_attended`).
Decode is always that case. This is not only saved work — see the note at the
end.

### The attention itself

Structurally `spyre__sdpa_overrideable`'s body, with one substitution that is
the whole point: the tiled KV extent is `buffer_width` rather than
`max_seqlen_kv`. The online-softmax accumulators are carried because the
`window_size` hint may tile the buffer into several chunks, each a partial
softmax.

## Supported shapes

The windowed path requires, and `rejection_reason` reports which of these
failed:

| constraint | why |
|---|---|
| causal windows | bidirectional needs a `2W + q_block` buffer; not implemented |
| `window_size % 64 == 0` | the window is read in whole sticks |
| `seqlen_kv % 64 == 0` | a ragged tail leaves the read start unaligned |
| `seqlen_q == 1` or `seqlen_q % 64 == 0` | rows are processed in whole blocks |
| `buffer_width < seqlen_kv` | otherwise the window covers the cache and there is nothing to save |

Anything else is served correctly by the masked fallback, with a warning
naming the constraint. The fallback builds a `seqlen_q x seqlen_kv` mask, so a
long prefill landing there is expensive — which is what the warning is for.

## A note for anyone touching the band

Skipping the all-zeros band is not merely an optimisation. Adding a band that
masks nothing, as a fused intermediate, was found to corrupt roughly 39% of
the output at `seqlen_q=1, seqlen_kv=4096, window_size=64` — on two
independent decompositions, in the same way. Every component was correct in
isolation; only the composition was wrong. If you reintroduce an unconditional
add here, check decode numerically against a masked reference.
