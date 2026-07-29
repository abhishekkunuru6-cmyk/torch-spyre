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

"""Unit tests for the sliding-window KV gather plan (increment 1).

Pure integer arithmetic — no device, no torch.compile, no HW. The plan decides
where each Q block's compact [B, H, buffer_width, D] KV window is read from;
these tests pin the width formula and, more importantly, the coverage
invariant every downstream step depends on:

    every query row's true window, intersected with the cache, lies inside
    the buffer its Q block reads.

Run:
    python3 -m pytest tests/inductor/test_swa_window_gather.py -v
"""

import pytest

from torch_spyre._inductor.swa_window_gather import (
    WindowGatherPlan,
    choose_q_block,
    plan_window_gather,
)

STICK = 64


# Shapes the gather path is expected to accept: (seqlen_q, seqlen_kv, window).
SUPPORTED_SHAPES = [
    (512, 512, 64),  # prefill, square
    (512, 512, 128),
    (512, 512, 256),
    (256, 256, 64),
    (128, 512, 64),  # chunked prefill, aligned offset
    (1, 4096, 64),  # decode
    (1, 4096, 256),
    (1, 128, 64),
    (192, 192, 64),
    (100, 256, 64),  # ragged Lq -> unaligned q_kv_offset
]


def _reference_width(seqlen_q: int, seqlen_kv: int, window: int, q_block: int) -> int:
    """Widest span any Q block needs, derived from first principles."""
    q_kv_offset = seqlen_kv - seqlen_q
    num_q_blocks = -(-seqlen_q // q_block)
    widest = 0
    for qi in range(num_q_blocks):
        q_start = qi * q_block
        q_end = min(seqlen_q, q_start + q_block)
        first_abs = q_kv_offset + q_start
        last_abs = q_kv_offset + q_end - 1
        win_start = max(0, ((first_abs - window + 1) // STICK) * STICK)
        widest = max(widest, last_abs - win_start + 1)
    return -(-widest // STICK) * STICK


class TestRejection:
    """plan_window_gather returns None for anything it cannot express exactly."""

    def test_bidirectional_is_deferred(self):
        assert plan_window_gather(512, 512, 64, is_causal=False) is None

    @pytest.mark.parametrize("window", [1, 63, 65, 100, 127])
    def test_window_not_stick_multiple(self, window):
        assert plan_window_gather(512, 512, window) is None

    @pytest.mark.parametrize("window", [0, -64])
    def test_non_positive_window(self, window):
        assert plan_window_gather(512, 512, window) is None

    @pytest.mark.parametrize("seqlen_kv", [257, 300, 511])
    def test_cache_not_stick_multiple(self, seqlen_kv):
        assert plan_window_gather(64, seqlen_kv, 64) is None

    @pytest.mark.parametrize(
        "seqlen_q,seqlen_kv,window", [(256, 256, 256), (64, 64, 64), (128, 128, 128)]
    )
    def test_window_spanning_the_whole_cache(self, seqlen_q, seqlen_kv, window):
        # Gathering the entire cache saves nothing; the masked path is cheaper.
        assert plan_window_gather(seqlen_q, seqlen_kv, window) is None

    def test_query_longer_than_cache(self):
        assert plan_window_gather(512, 256, 64) is None

    @pytest.mark.parametrize("seqlen_q", [0, -1])
    def test_non_positive_query_length(self, seqlen_q):
        assert plan_window_gather(seqlen_q, 512, 64) is None


class TestBufferWidth:
    """The compact buffer's width, which is the whole premise of the design."""

    def test_prefill_is_window_plus_one_block(self):
        plan = plan_window_gather(512, 512, 64)
        assert plan is not None
        assert plan.buffer_width == 64 + 64

    def test_prefill_wider_window(self):
        plan = plan_window_gather(512, 512, 128)
        assert plan is not None
        assert plan.buffer_width == 128 + 64

    def test_decode_is_exactly_the_window(self):
        # Lq == 1 has no intra-block stagger, so the buffer is [B, H, W, E].
        plan = plan_window_gather(1, 4096, 256)
        assert plan is not None
        assert plan.buffer_width == 256

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SUPPORTED_SHAPES)
    def test_width_matches_first_principles(self, seqlen_q, seqlen_kv, window):
        plan = plan_window_gather(seqlen_q, seqlen_kv, window)
        assert plan is not None
        assert plan.buffer_width == _reference_width(seqlen_q, seqlen_kv, window, 64)

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SUPPORTED_SHAPES)
    def test_width_is_stick_aligned_and_fits(self, seqlen_q, seqlen_kv, window):
        plan = plan_window_gather(seqlen_q, seqlen_kv, window)
        assert plan is not None
        assert plan.buffer_width % STICK == 0
        assert plan.buffer_width <= seqlen_kv


class TestReadStart:
    """Where each block's DMA begins."""

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SUPPORTED_SHAPES)
    def test_stick_aligned_and_in_bounds(self, seqlen_q, seqlen_kv, window):
        plan = plan_window_gather(seqlen_q, seqlen_kv, window)
        assert plan is not None
        for qi in range(plan.num_q_blocks):
            start = plan.read_start(qi)
            assert start % STICK == 0
            assert 0 <= start <= seqlen_kv - plan.buffer_width

    def test_advances_one_block_per_step_in_the_steady_state(self):
        # Away from both clamps the window slides exactly q_block per Q block.
        plan = plan_window_gather(512, 512, 64)
        assert plan is not None
        starts = [plan.read_start(qi) for qi in range(plan.num_q_blocks)]
        assert starts == [0, 0, 64, 128, 192, 256, 320, 384]

    def test_decode_reads_the_final_window(self):
        plan = plan_window_gather(1, 4096, 64)
        assert plan is not None
        assert plan.read_start(0) == 4096 - 64

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SUPPORTED_SHAPES)
    def test_never_moves_backwards(self, seqlen_q, seqlen_kv, window):
        plan = plan_window_gather(seqlen_q, seqlen_kv, window)
        assert plan is not None
        starts = [plan.read_start(qi) for qi in range(plan.num_q_blocks)]
        assert starts == sorted(starts)


class TestQBlocks:
    """Q-side partitioning, which the KV read is keyed off."""

    @pytest.mark.parametrize(
        "seqlen_q,expected",
        [(512, 8), (64, 1), (1, 1), (100, 2), (192, 3), (256, 4)],
    )
    def test_block_count(self, seqlen_q, expected):
        plan = plan_window_gather(seqlen_q, 4096, 64)
        assert plan is not None
        assert plan.num_q_blocks == expected

    def test_ranges_tile_the_query_exactly(self):
        plan = plan_window_gather(100, 256, 64)
        assert plan is not None
        ranges = [plan.block_q_range(qi) for qi in range(plan.num_q_blocks)]
        assert ranges == [(0, 64), (64, 100)]


class TestCoverage:
    """The invariant everything downstream rests on."""

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SUPPORTED_SHAPES)
    def test_every_row_window_lies_inside_its_buffer(self, seqlen_q, seqlen_kv, window):
        plan = plan_window_gather(seqlen_q, seqlen_kv, window)
        assert plan is not None
        for qi in range(plan.num_q_blocks):
            start = plan.read_start(qi)
            stop = start + plan.buffer_width
            q_start, q_end = plan.block_q_range(qi)
            for q_index in range(q_start, q_end):
                lo, hi = plan.row_window(q_index)
                assert lo >= start, (qi, q_index, lo, start)
                assert hi <= stop, (qi, q_index, hi, stop)

    @pytest.mark.parametrize("seqlen_q,seqlen_kv,window", SUPPORTED_SHAPES)
    def test_row_windows_are_clamped_to_the_cache(self, seqlen_q, seqlen_kv, window):
        plan = plan_window_gather(seqlen_q, seqlen_kv, window)
        assert plan is not None
        for q_index in range(seqlen_q):
            lo, hi = plan.row_window(q_index)
            assert 0 <= lo < hi <= seqlen_kv
            assert hi - lo <= window

    def test_row_window_is_the_causal_band(self):
        plan = plan_window_gather(512, 512, 128)
        assert plan is not None
        assert plan.row_window(0) == (0, 1)
        assert plan.row_window(127) == (0, 128)
        assert plan.row_window(128) == (1, 129)
        assert plan.row_window(511) == (384, 512)

    def test_decode_row_window_is_the_cache_tail(self):
        plan = plan_window_gather(1, 4096, 64)
        assert plan is not None
        assert plan.row_window(0) == (4032, 4096)


class TestChooseQBlock:
    """Q block size must divide the query length exactly."""

    def test_decode_takes_a_single_row(self):
        # Lq=1 is not a multiple of 64; one row also has no intra-block stagger.
        assert choose_q_block(1) == 1

    @pytest.mark.parametrize("seqlen_q", [64, 128, 256, 512, 4096])
    def test_stick_multiples_take_a_full_block(self, seqlen_q):
        assert choose_q_block(seqlen_q) == STICK

    @pytest.mark.parametrize("seqlen_q", [2, 63, 65, 100, 257])
    def test_anything_else_falls_back(self, seqlen_q):
        # q_block=1 would divide these, but that is one gather per query row.
        assert choose_q_block(seqlen_q) is None

    @pytest.mark.parametrize("seqlen_q", [0, -1])
    def test_degenerate_lengths_fall_back(self, seqlen_q):
        assert choose_q_block(seqlen_q) is None

    def test_decode_buffer_is_exactly_the_window(self):
        # The pairing that matters: q_block=1 -> no stagger -> Wb == W.
        plan = plan_window_gather(1, 4096, 64, q_block=choose_q_block(1))
        assert plan is not None
        assert plan.buffer_width == 64


class TestPlanIsImmutable:
    def test_frozen(self):
        plan = plan_window_gather(512, 512, 64)
        assert isinstance(plan, WindowGatherPlan)
        with pytest.raises(Exception):
            plan.buffer_width = 999  # type: ignore[misc]
