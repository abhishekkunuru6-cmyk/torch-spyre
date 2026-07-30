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

"""Increment R5: does the rolled window buffer actually get reused?

This is the claim the whole design rests on. Rolling one W+T buffer across Q
blocks was chosen over materialising all N windows at once *because it saves
memory* -- and it costs N sequential kernel launches to do so. But the blocks
are independent: block n reads a compile-time offset and shares nothing with
block n+1, so nothing in the graph forces the buffers to be reused. Reuse is
granted by the memory planner, not expressed by the source. If it is not
granted, the design paid the sequential cost for the memory profile it was
chosen to avoid, which is strictly the worst of both.

**Sweeping the window, not the query length.** The obvious experiment -- grow
Lq and watch memory -- cannot answer this: query and output grow linearly with
Lq under either hypothesis, and so does N x (W + T). Growing W at a FIXED Lq
separates them cleanly, because nothing else in the graph depends on W:

    reused      pool grows by ~one window per unit of buffer_width
    not reused  pool grows by ~N windows per unit of buffer_width

At Lq=512, T=64 that is an eightfold difference in slope, which no amount of
allocator noise can blur.

The measurement is V.graph.pool_size, the scratch the graph plans for, read
where Inductor computes it rather than inferred from a debug dump.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_swa_buffer_reuse.py -v -s
"""

import unittest

import torch
import torch._dynamo

from captured_wrapper import capture_generated_graphs
from inductor_cache_isolation import isolated_inductor_cache
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.swa_window_gather import plan_window_gather
from utils_inductor import cached_randn

BATCH = 1
HEADS = 8
HEAD_DIM = 64
SEQLEN = 512  # fixed, so only the window varies
Q_BLOCK = 64
WINDOWS = (64, 128, 192, 256)

# One row of one window buffer, in bytes: K and V, fp16.
BYTES_PER_WINDOW_ROW = 2 * BATCH * HEADS * HEAD_DIM * 2


def _compile_and_measure(window: int) -> tuple[int, int]:
    """Compile SWA at one window size; return (pool_size, gather call count)."""
    query = cached_randn(
        (BATCH, HEADS, SEQLEN, HEAD_DIM), differentiation=1, dtype=torch.float16
    )
    key = cached_randn(
        (BATCH, HEADS, SEQLEN, HEAD_DIM), differentiation=2, dtype=torch.float16
    )
    value = cached_randn(
        (BATCH, HEADS, SEQLEN, HEAD_DIM), differentiation=3, dtype=torch.float16
    )

    saved = spyre_config.swa_window_roll
    spyre_config.swa_window_roll = True
    torch._dynamo.reset()
    try:
        with isolated_inductor_cache():
            with capture_generated_graphs() as graphs:

                def fn(q, k, v):
                    return torch.ops.spyre.sliding_window_attention(
                        q, k, v, window, True
                    )

                compiled = torch.compile(fn, backend="inductor", fullgraph=True)
                compiled(query.to("spyre"), key.to("spyre"), value.to("spyre"))
    finally:
        spyre_config.swa_window_roll = saved
        torch._dynamo.reset()

    assert graphs, "no graph was generated -- the capture hook did not fire"
    pool = max(g.pool_size for g in graphs)
    kernels = sum(g.count(".run(") for g in graphs)
    return pool, kernels


class TestWindowBufferIsReused(unittest.TestCase):
    """Grow the window at a fixed query length and watch the planned pool."""

    def test_pool_grows_with_one_window_not_with_all_of_them(self):
        plans = {w: plan_window_gather(SEQLEN, SEQLEN, w) for w in WINDOWS}
        for window, plan in plans.items():
            self.assertIsNotNone(plan, f"window {window} unsupported at Lq={SEQLEN}")

        measurements = {}
        for window in WINDOWS:
            pool, kernels = _compile_and_measure(window)
            measurements[window] = (plans[window].buffer_width, pool, kernels)

        blocks = plans[WINDOWS[0]].num_q_blocks
        print(f"\n=== R5: pool size vs window (Lq={SEQLEN}, {blocks} blocks) ===")
        print(f"{'W':>5} {'buffer_width':>13} {'pool_size':>12} {'kernels':>9}")
        for window, (width, pool, kernels) in measurements.items():
            print(f"{window:>5} {width:>13} {pool:>12} {kernels:>9}")

        first, last = WINDOWS[0], WINDOWS[-1]
        width_delta = measurements[last][0] - measurements[first][0]
        pool_delta = measurements[last][1] - measurements[first][1]
        slope = pool_delta / width_delta
        reused_slope = BYTES_PER_WINDOW_ROW
        print(
            f"slope: {slope:.0f} bytes per window row "
            f"(one buffer ~= {reused_slope}, {blocks} buffers ~= "
            f"{reused_slope * blocks})"
        )

        # Halfway between the two hypotheses, on a log scale it is not close.
        self.assertLess(
            slope,
            reused_slope * blocks / 2,
            f"pool grows by {slope:.0f} bytes per window row, closer to "
            f"{blocks} live window buffers ({reused_slope * blocks}) than to one "
            f"({reused_slope}) -- the window buffer is NOT being reused across "
            f"blocks, so rolling is paying sequential launches for the "
            f"all-at-once memory profile",
        )

    def test_kernel_count_grows_with_the_block_count(self):
        # The cost side of the same trade: rolling is N sequential launches by
        # construction. Recorded so the memory win above is read against it.
        pool_small, kernels_small = _compile_and_measure(WINDOWS[0])
        plan = plan_window_gather(SEQLEN, SEQLEN, WINDOWS[0])
        assert plan is not None
        print(
            f"\n{plan.num_q_blocks} blocks -> {kernels_small} kernel calls, "
            f"pool {pool_small}"
        )
        self.assertGreater(kernels_small, 0)


if __name__ == "__main__":
    unittest.main()
