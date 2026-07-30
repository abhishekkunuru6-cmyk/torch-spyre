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

"""Capture what Inductor generated, and how much memory it planned for.

``SpyrePythonWrapperCodegen.generate`` is the one place where both the wrapper
source and ``V.graph.pool_size`` are in hand, so wrapping it is enough to see
how much scratch a graph asked for without running it or reading a debug dump.

pool_size is the measurement that matters for buffer reuse: if a body allocates
one window buffer and reuses it across blocks, the pool grows with the window;
if every block's buffer stays live, it grows with the window times the number of
blocks. Nothing in the source distinguishes those -- only the plan does.
"""

from contextlib import contextmanager
from dataclasses import dataclass

from torch._inductor.virtualized import V

from torch_spyre._inductor.wrapper import SpyrePythonWrapperCodegen


@dataclass(frozen=True)
class GeneratedGraph:
    """One compiled graph's planned memory and generated wrapper."""

    pool_size: int
    source: str

    def count(self, needle: str) -> int:
        return self.source.count(needle)


@contextmanager
def capture_generated_graphs():
    """Yield a list that fills with a GeneratedGraph per graph compiled."""
    captured: list[GeneratedGraph] = []
    original = SpyrePythonWrapperCodegen.generate

    def recording_generate(self, is_inference):
        result = original(self, is_inference)
        wrapper_value, _ = result
        captured.append(
            GeneratedGraph(
                pool_size=int(getattr(V.graph, "pool_size", 0) or 0),
                source=str(wrapper_value.value),
            )
        )
        return result

    SpyrePythonWrapperCodegen.generate = recording_generate
    try:
        yield captured
    finally:
        SpyrePythonWrapperCodegen.generate = original
