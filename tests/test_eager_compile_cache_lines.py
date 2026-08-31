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

# Owner(s): ["module: dynamo"]

"""Regression tests for the dynamo cache budget of standalone-compiled eager ops.

Compiled eager kernels used to share one dynamo cache line, and so one
recompile budget: ``torch.compile`` on an ``OpOverload`` routes through
``wrap_inline``, whose ``inner`` is a single code object. Exhausting the budget
made dynamo run an op eagerly, which re-dispatched into the Spyre kernel that
called it and recursed (hf-adapters#402).

CPU-only; no Spyre device needed.
"""

import unittest

import torch
import torch._dynamo

# Importing torch_spyre applies the dynamo config changes under test.
import torch_spyre  # noqa: F401
from torch_spyre.ops.eager import _guard_reentry, _op_frame


class TestDynamoCacheLimits(unittest.TestCase):
    def test_accumulated_limit_is_raised_with_the_per_line_limit(self):
        # Checked first, so a default 256 caps the cache whatever the other says.
        config = torch._dynamo.config
        self.assertGreaterEqual(
            config.accumulated_recompile_limit, config.recompile_limit
        )
        self.assertEqual(config.accumulated_recompile_limit, 1024)


class TestPerOpCacheLine(unittest.TestCase):
    def test_each_op_gets_its_own_code_object(self):
        add = _op_frame(torch.ops.aten.add.Tensor)
        mul = _op_frame(torch.ops.aten.mul.Tensor)
        self.assertIsNot(add.__code__, mul.__code__)

    def test_frame_forwards_args_and_kwargs_to_the_op(self):
        add = _op_frame(torch.ops.aten.add.Tensor)
        x, y = torch.ones(3), torch.full((3,), 2.0)
        torch.testing.assert_close(add(x, y, alpha=2), x + 2 * y)

    def test_one_op_exhausting_its_budget_leaves_other_ops_compilable(self):
        graphs = []

        def counting_backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        limit = 4
        with torch._dynamo.config.patch(
            recompile_limit=1024, accumulated_recompile_limit=limit
        ):
            compiled_add = torch.compile(
                _op_frame(torch.ops.aten.add.Tensor),
                backend=counting_backend,
                dynamic=False,
            )
            compiled_mul = torch.compile(
                _op_frame(torch.ops.aten.mul.Tensor),
                backend=counting_backend,
                dynamic=False,
            )
            # One graph per shape, so this spends add's budget.
            for n in range(1, limit + 1):
                compiled_add(torch.randn(n), torch.randn(n))
            spent_on_add = len(graphs)
            for n in range(1, limit + 1):
                compiled_mul(torch.randn(n), torch.randn(n))
            spent_on_mul = len(graphs) - spent_on_add

        self.assertEqual(spent_on_add, limit)
        # Sharing one cache line, as before the fix, this would be 0.
        self.assertEqual(spent_on_mul, limit)


class TestReentryGuard(unittest.TestCase):
    def test_reentering_the_same_op_raises_instead_of_recursing(self):
        op = torch.ops.aten.add.Tensor
        with self.assertRaisesRegex(RuntimeError, "re-entered itself"):
            with _guard_reentry(op):
                with _guard_reentry(op):
                    pass

    def test_a_different_op_may_nest(self):
        with _guard_reentry(torch.ops.aten.add.Tensor):
            with _guard_reentry(torch.ops.aten.mul.Tensor):
                pass

    def test_the_op_is_released_after_an_exception(self):
        op = torch.ops.aten.add.Tensor
        with self.assertRaises(ValueError):
            with _guard_reentry(op):
                raise ValueError
        with _guard_reentry(op):  # must not still be marked in flight
            pass


# NB: plain unittest, not run_tests() -- that seeds the RNG, which initializes
# the Spyre device and turns this CPU-only test into a device test.
# See tests/test_nested_compile_region_guard.py.
if __name__ == "__main__":
    unittest.main()
