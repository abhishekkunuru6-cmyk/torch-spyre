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

"""Localize WHERE the sliding SWA path goes numerically wrong.

The sliding path compiles and runs (padded_kv must be a friendly power of two;
320 = 2^6*5 still trips issue #1353) but disagrees with CPU on ~44% of
elements.  An aggregate number cannot say whether the loop reads the wrong KV
window, drops the mask, or mis-addresses one Q block -- so this breaks the
error down per Q block and scores each iteration against several "what if it
went wrong THIS way" references.

The first run showed the error shrinking monotonically with block index
(2.57 / 0.86 / 0.0097), i.e. the LAST block is right and earlier ones are
progressively wrong.  That rules out "stuck at block 0" and points at the read
sitting at or near its final position for every iteration, so the references
here are chosen to separate those:

    correct   : block i reads window i                     (the target)
    last      : every block reads the LAST window          (base stuck at end)
    reversed  : block i reads window N-1-i                 (slide runs backwards)
    unmasked  : correct window, band mask never applied    (mask dropped)
    desync    : reads the last window, masks as if at its own window

Fully-masked rows are forced to 0 rather than left as softmax(-inf) = nan, so
every reference stays comparable even where a window holds no valid column.

    SENCORES=1 python3 diagnose_swa_sliding_blocks.py
"""

import os

os.environ.setdefault("SENCORES", "1")
os.environ.setdefault("SPYRE_SWA_SLIDING_LOOP", "1")

import torch  # noqa: E402
import torch_spyre  # noqa: E402, F401

from torch_spyre._inductor.swa_sliding import plan_sliding_window  # noqa: E402

# padded_kv = 256 = 2^8, the shape that compiles.
B, H, LQ, LKV, D, WINDOW = 1, 8, 192, 192, 128, 64
Q_BLOCK = 64


def _attend(q_blk, k_pad, v_pad, plan, q_lo, kv_lo, mask_lo, apply_mask=True):
    """One Q block against the padded-K/V window starting at kv_lo.

    q_lo is the block's absolute first query row.  mask_lo is the padded column
    the BAND believes the window starts at -- normally equal to kv_lo, but
    passing them apart models a mask that failed to slide with the data.
    """
    kv = slice(kv_lo, kv_lo + plan.read_extent)
    s = q_blk @ k_pad[:, :, kv].mT / (D**0.5)

    if apply_mask:
        rows = (q_lo + torch.arange(q_blk.shape[-2])).view(-1, 1) + plan.q_kv_offset
        cols = (mask_lo + torch.arange(plan.read_extent)).view(1, -1) - plan.left_pad
        delta = rows - cols
        keep = (delta >= 0) & (delta < WINDOW) & (cols >= 0) & (cols < LKV)
        s = s.masked_fill(~keep, float("-inf"))

    # Softmax that tolerates an all -inf row instead of producing nan.
    biggest = s.amax(dim=-1, keepdim=True)
    biggest = torch.where(torch.isinf(biggest), torch.zeros_like(biggest), biggest)
    weights = torch.exp(s - biggest)
    denom = weights.sum(dim=-1, keepdim=True)
    out = weights @ v_pad[:, :, kv]
    return torch.where(denom > 0, out / denom.clamp(min=1e-30), torch.zeros_like(out))


def build_ref(q, k, v, plan, window_for_block, apply_mask=True, mask_follows=True):
    """Full output assembled from a per-block choice of which window to read."""
    qf = q.float()
    pad = torch.zeros(B, H, plan.left_pad, D)
    k_pad = torch.cat([pad, k.float()], dim=-2)
    v_pad = torch.cat([pad, v.float()], dim=-2)

    out = torch.zeros(B, H, LQ, D)
    for blk in range(plan.num_q_blocks):
        q_lo = blk * Q_BLOCK
        kv_lo = window_for_block(plan, blk)
        mask_lo = kv_lo if mask_follows else plan.padded_window_lo(blk)
        out[:, :, q_lo : q_lo + Q_BLOCK] = _attend(
            qf[:, :, q_lo : q_lo + Q_BLOCK],
            k_pad,
            v_pad,
            plan,
            q_lo,
            kv_lo,
            mask_lo,
            apply_mask,
        )
    return out


def main() -> None:
    plan = plan_sliding_window(
        batch_size=B, seqlen_q=LQ, seqlen_kv=LKV, window_size=WINDOW, is_causal=True
    )
    print(f"plan: {plan.describe()}")
    print(
        f"SENCORES={os.environ['SENCORES']} "
        f"SPYRE_SWA_SLIDING_LOOP={os.environ['SPYRE_SWA_SLIDING_LOOP']}"
    )
    windows = [plan.padded_window_lo(i) for i in range(plan.num_q_blocks)]
    print(f"padded window starts per block: {windows}  read_extent={plan.read_extent}")

    torch.manual_seed(0)
    q = torch.randn(B, H, LQ, D, dtype=torch.float16)
    k = torch.randn(B, H, LKV, D, dtype=torch.float16)
    v = torch.randn(B, H, LKV, D, dtype=torch.float16)

    def fn(q, k, v):
        return torch.ops.spyre.sliding_window_attention(q, k, v, WINDOW, True)

    got = torch.compile(fn)(q.to("spyre"), k.to("spyre"), v.to("spyre")).cpu().float()

    last = plan.num_q_blocks - 1
    refs = {
        "correct": build_ref(q, k, v, plan, lambda p, i: p.padded_window_lo(i)),
        "last": build_ref(q, k, v, plan, lambda p, i: p.padded_window_lo(last)),
        "reversed": build_ref(q, k, v, plan, lambda p, i: p.padded_window_lo(last - i)),
        "unmasked": build_ref(
            q, k, v, plan, lambda p, i: p.padded_window_lo(i), apply_mask=False
        ),
        "desync": build_ref(
            q, k, v, plan, lambda p, i: p.padded_window_lo(last), mask_follows=False
        ),
    }

    header = f"{'Q block':>8}  {'rows':>10}  " + "  ".join(f"{n:>10}" for n in refs)
    print()
    print(header)
    print("-" * len(header))
    for blk in range(plan.num_q_blocks):
        lo = blk * Q_BLOCK
        sl = slice(lo, lo + Q_BLOCK)
        errs = {
            n: (got[:, :, sl] - r[:, :, sl]).abs().max().item() for n, r in refs.items()
        }
        best = min(errs, key=errs.get)
        cells = "  ".join(
            f"{errs[n]:>10.4f}" + ("*" if n == best else " ") for n in refs
        )
        print(f"{blk:>8}  {f'{lo}-{lo + Q_BLOCK - 1}':>10}  {cells}")

    print()
    print("  * = closest reference for that block.  All blocks closest to 'correct'")
    print("  with small errors means only precision is left.  'last' means the read")
    print("  never moved off its final position; 'reversed' that the slide runs the")
    print("  wrong way; 'unmasked' that the band is not reaching the scores;")
    print("  'desync' that data and mask slide at different rates.")


if __name__ == "__main__":
    main()
