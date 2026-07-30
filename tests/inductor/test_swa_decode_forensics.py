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

"""What is the device actually computing at the failing decode shape?

Everything that could be checked by reconstruction has been, and all of it
passes: the windowed algorithm is exact in float64 and agrees with the kernel
test's own float16 masked-SDPA reference; the staged body reproduces on device
at this shape through every construct including the real hint placement and
the block loop. Yet `spyre::sliding_window_attention` is wrong here, on both
the rolled path and the untouched fallback, identically.

So this file stops rebuilding the body and instead asks what the wrong answer
IS. Rather than one reference, it scores the device output against a set of
candidate computations, per head:

    correct       the window at read_start = 4032, which is the definition
    offset_zero   the window read from 0 -- the read offset dropped entirely
    off_by_stick  the window read one stick early, 3968
    unwindowed    plain causal attention over the whole 4096-row cache
    first_row     every query row attending only to cache row 0

A device output that matches one of the wrong candidates names the bug
outright. One that matches none of them, but is finite and plausibly scaled,
says the arithmetic is being corrupted rather than misdirected -- a different
investigation. Per head, because 199 of 512 mismatched elements is close to
three of eight heads, and an earlier investigation on a different SWA design
narrowed its error to a single head.

This asserts nothing about the diagnosis: it prints a table and asserts only
that the correct candidate wins, so the failure message carries the evidence.

Run:
    SENCORES=1 python3 -m pytest tests/inductor/test_swa_decode_forensics.py -v -s
"""

import unittest

import torch
import torch._dynamo
import torch.nn.functional as F

from torch_spyre._inductor import config as spyre_config
from utils_inductor import cached_randn

BATCH = 1
HEADS = 8
HEAD_DIM = 64
SEQLEN_KV = 4096
WINDOW = 64
READ_START = 4032  # plan_window_gather(1, 4096, 64, q_block=1).read_start(0)


def _inputs():
    query = cached_randn(
        (BATCH, HEADS, 1, HEAD_DIM), differentiation=1, dtype=torch.float16
    )
    key = cached_randn(
        (BATCH, HEADS, SEQLEN_KV, HEAD_DIM), differentiation=2, dtype=torch.float16
    )
    value = cached_randn(
        (BATCH, HEADS, SEQLEN_KV, HEAD_DIM), differentiation=3, dtype=torch.float16
    )
    return query, key, value


def _attend_to_slice(query, key, value, start, width):
    """Plain attention over one contiguous KV slice, nothing masked."""
    k_win = key[:, :, start : start + width, :]
    v_win = value[:, :, start : start + width, :]
    return F.scaled_dot_product_attention(query, k_win, v_win)


def _candidates(query, key, value):
    """Named hypotheses for what the device might be computing."""
    return {
        "correct": _attend_to_slice(query, key, value, READ_START, WINDOW),
        "offset_zero": _attend_to_slice(query, key, value, 0, WINDOW),
        "off_by_stick": _attend_to_slice(query, key, value, READ_START - 64, WINDOW),
        "unwindowed": F.scaled_dot_product_attention(query, key, value),
        "first_row": _attend_to_slice(query, key, value, 0, 1),
    }


def _run_on_device(query, key, value, roll_enabled: bool) -> torch.Tensor:
    saved = spyre_config.swa_window_roll
    spyre_config.swa_window_roll = roll_enabled
    torch._dynamo.reset()
    try:

        def fn(q, k, v):
            return torch.ops.spyre.sliding_window_attention(q, k, v, WINDOW, True)

        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        result = compiled(query.to("spyre"), key.to("spyre"), value.to("spyre"))
        return result.cpu().float()
    finally:
        spyre_config.swa_window_roll = saved
        torch._dynamo.reset()


def _score(actual: torch.Tensor, candidates: dict) -> dict:
    """Max abs difference per head against every candidate."""
    return {
        name: [
            (actual[0, h] - reference[0, h].float()).abs().max().item()
            for h in range(HEADS)
        ]
        for name, reference in candidates.items()
    }


def _report(label: str, scores: dict) -> str:
    names = list(scores)
    header = f"{'head':>5} " + " ".join(f"{n:>13}" for n in names)
    lines = [f"\n=== {label} ===", header]
    for h in range(HEADS):
        row = " ".join(f"{scores[n][h]:>13.5f}" for n in names)
        best = min(names, key=lambda n: scores[n][h])
        lines.append(f"{h:>5} {row}   <- {best}")
    worst = {n: max(scores[n]) for n in names}
    lines.append(
        "worst over heads: " + ", ".join(f"{n}={v:.5f}" for n, v in worst.items())
    )
    return "\n".join(lines)


class TestWhatDecodeComputes(unittest.TestCase):
    """Score the device's decode output against candidate computations."""

    def _check(self, roll_enabled: bool, label: str):
        query, key, value = _inputs()
        actual = _run_on_device(query, key, value, roll_enabled)
        scores = _score(actual, _candidates(query, key, value))
        print(_report(label, scores))

        self.assertTrue(torch.isfinite(torch.tensor(scores["correct"])).all())
        for name, per_head in scores.items():
            if name == "correct":
                continue
            self.assertLess(
                max(scores["correct"]),
                max(per_head),
                f"device output is closer to '{name}' than to the correct "
                f"window -- see the table above",
            )

    def test_rolled_path(self):
        self._check(roll_enabled=True, label="rolled (SPYRE_SWA_WINDOW_ROLL=1)")

    def test_unrolled_fallback(self):
        self._check(roll_enabled=False, label="unrolled fallback")


if __name__ == "__main__":
    unittest.main()
