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

"""Diagnostic: confirm dataFormat_ in SDSC JSON for fp32 comparison ops.

Goal
----
Verify the exact ``dataFormat_`` values that torch-spyre writes into the SDSC
JSON for a float32 comparison op (e.g. ``greaterequal``).

The layout-propagation pass assigns ``IEEE_FP32`` (32 elems/stick) to both
input tensors AND the bool output.  The hardware SFP comparison unit, however,
always writes its result in ``SEN169_FP16`` format (64 elems/stick), regardless
of the operand precision.  If the JSON emits ``IEEE_FP32`` for the output
tensor, the DMA engine interprets the 64 fp16 result values per stick as 32
fp32 values → every other element is read from the wrong location → ~50 %
mismatch.

What this test does
-------------------
Builds ``OpSpec`` objects directly (no ``torch.compile``, no hardware, no
backend compiler) and calls ``compile_op_spec`` — the same code path used by
the production pipeline — then checks ``labeledDs_[i]["dataFormat_"]`` in the
resulting SDSC JSON.

JSON structure (from compute_ops.py generate_sdsc):
  { "0_greaterequal": {            # top-level key = idx_opfunc
      "dscs_": [
        { "greaterequal": {        # inner key = opfunc name
            "labeledDs_": [
              {"ldsIdx_": 0, "dataFormat_": "...", ...},  # input 0
              {"ldsIdx_": 1, "dataFormat_": "...", ...},  # input 1
              {"ldsIdx_": 2, "dataFormat_": "...", ...},  # output (bool)
            ],
            "computeOp_": [
              {"attributes_": {"dataFormat_": "..."}, ...}
            ],
        }}
      ]
    }
  }

Run
---
    python tests/inductor/test_fp32_cmp_sdsc_format.py

No pytest, no hardware, no backend compiler required.
"""

import unittest

from sympy import Integer, Mod, Symbol
import torch_spyre  # noqa: F401 – registers the spyre backend

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.op_spec import OpSpec, TensorArg

# ---------------------------------------------------------------------------
# Helpers to build a minimal 1-D pointwise OpSpec and parse the SDSC JSON
# ---------------------------------------------------------------------------

_HBM_BASE = 0x400000000
_N = 128  # number of elements


def _make_fp32_cmp_op_spec(op_name: str) -> OpSpec:
    """Build a minimal 2-input, 1-output pointwise OpSpec for an fp32 comparison.

    Inputs: IEEE_FP32, 128 elems  (32 elems/stick → device_size = [4, 32])
    Output: the layout-propagation pass assigns device_dtype = IEEE_FP32 too
            (this is the current behaviour we are diagnosing).

    Stick layout for IEEE_FP32: 32 elems/stick, so 128 elems → 4 sticks.
    device_size = [num_sticks, elems_per_stick] = [4, 32]
    device_coordinates = [x // 32, x % 32]  (x is the iteration symbol)
    """
    x = Symbol("x")
    elems_per_stick = 32  # IEEE_FP32
    num_sticks = _N // elems_per_stick  # 4

    def _arg(is_input: bool, arg_index: int, hbm_offset: int) -> TensorArg:
        return TensorArg(
            is_input=is_input,
            arg_index=arg_index,
            device_dtype=DataFormats.IEEE_FP32,
            device_size=[num_sticks, elems_per_stick],
            device_coordinates=[x // elems_per_stick, Mod(x, elems_per_stick)],
            allocation={"hbm": _HBM_BASE + hbm_offset},
        )

    return OpSpec(
        op=op_name,
        is_reduction=False,
        iteration_space={x: (Integer(_N), 1)},
        args=[
            _arg(True, 0, 0x000000),  # input A
            _arg(True, 1, 0x010000),  # input B
            _arg(False, 2, 0x100000),  # output (bool, stored as IEEE_FP32)
        ],
        op_info={},
    )


def _make_fp16_cmp_op_spec(op_name: str) -> OpSpec:
    """Same as above but with fp16 inputs/output (baseline — known-good path)."""
    x = Symbol("x")
    elems_per_stick = 64  # SEN169_FP16
    num_sticks = _N // elems_per_stick  # 2

    def _arg(is_input: bool, arg_index: int, hbm_offset: int) -> TensorArg:
        return TensorArg(
            is_input=is_input,
            arg_index=arg_index,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[num_sticks, elems_per_stick],
            device_coordinates=[x // elems_per_stick, Mod(x, elems_per_stick)],
            allocation={"hbm": _HBM_BASE + hbm_offset},
        )

    return OpSpec(
        op=op_name,
        is_reduction=False,
        iteration_space={x: (Integer(_N), 1)},
        args=[
            _arg(True, 0, 0x000000),
            _arg(True, 1, 0x010000),
            _arg(False, 2, 0x100000),
        ],
        op_info={},
    )


def _labeled_ds_formats(sdsc_json: dict) -> list[tuple[int, str]]:
    """Return [(ldsIdx_, dataFormat_)] for every tensor in the SDSC JSON."""
    result = []
    for top_val in sdsc_json.values():
        for dsc_wrapper in top_val.get("dscs_", []):
            for inner in dsc_wrapper.values():  # unwrap opfunc key
                for entry in inner.get("labeledDs_", []):
                    result.append((entry["ldsIdx_"], entry["dataFormat_"]))
    return result


def _compute_op_format(sdsc_json: dict) -> str | None:
    """Return the dataFormat_ from computeOp_[0].attributes_."""
    for top_val in sdsc_json.values():
        for dsc_wrapper in top_val.get("dscs_", []):
            for inner in dsc_wrapper.values():
                for op in inner.get("computeOp_", []):
                    return op.get("attributes_", {}).get("dataFormat_")
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFp32CmpSdscFormat(unittest.TestCase):
    """Verify dataFormat_ values in SDSC JSON for fp32 and fp16 comparison ops."""

    def _compile_and_get_formats(self, op_spec: OpSpec) -> list[tuple[int, str]]:
        sdsc_json, _, _, _ = compile_op_spec(idx=0, op_spec=op_spec, symbols=[])
        return _labeled_ds_formats(sdsc_json), _compute_op_format(sdsc_json)

    # ------------------------------------------------------------------
    # fp16 baseline: all tensors (inputs + bool output) → SEN169_FP16
    # ------------------------------------------------------------------

    def test_fp16_ge_all_formats_are_SEN169_FP16(self):
        """fp16 comparison: every tensor dataFormat_ must be SEN169_FP16."""
        op_spec = _make_fp16_cmp_op_spec("greaterequal")
        formats, op_fmt = self._compile_and_get_formats(op_spec)
        print("\n--- fp16 greaterequal tensor formats ---")
        for idx, fmt in formats:
            label = "OUTPUT" if idx == len(formats) - 1 else f"INPUT[{idx}]"
            print(f"  Tensor{idx} ({label}): dataFormat_={fmt!r}")
        print(f"  computeOp_ dataFormat_: {op_fmt!r}")
        for idx, fmt in formats:
            self.assertEqual(
                fmt, "SEN169_FP16", f"Tensor{idx} expected SEN169_FP16, got {fmt!r}"
            )

    # ------------------------------------------------------------------
    # fp32 comparisons: the hypothesis says the output will show IEEE_FP32
    # but the hardware writes SEN169_FP16 sticks → ~50 % element mismatch.
    #
    # These tests ASSERT the CORRECT behaviour (output must be SEN169_FP16).
    # They will FAIL today, confirming the root cause.
    # Once the fix is applied they will PASS.
    # ------------------------------------------------------------------

    def _assert_fp32_cmp_output_is_fp16(self, op_name: str):
        op_spec = _make_fp32_cmp_op_spec(op_name)
        formats, op_fmt = self._compile_and_get_formats(op_spec)
        print(f"\n--- fp32 {op_name} tensor formats ---")
        for idx, fmt in formats:
            label = "OUTPUT" if idx == len(formats) - 1 else f"INPUT[{idx}]"
            print(f"  Tensor{idx} ({label}): dataFormat_={fmt!r}")
        print(f"  computeOp_ dataFormat_: {op_fmt!r}")
        # Inputs must remain IEEE_FP32
        for idx, fmt in formats[:-1]:
            self.assertEqual(
                fmt,
                "IEEE_FP32",
                f"fp32 {op_name} INPUT[{idx}] expected IEEE_FP32, got {fmt!r}",
            )
        # Output (bool) must be SEN169_FP16: the hardware always writes fp16 sticks
        out_idx, out_fmt = formats[-1]
        self.assertEqual(
            out_fmt,
            "SEN169_FP16",
            f"fp32 {op_name} OUTPUT Tensor{out_idx}: got {out_fmt!r}, "
            f"want 'SEN169_FP16'. "
            f"Root cause: hardware SFP writes 64 fp16 values/stick for comparison "
            f"results regardless of operand precision; telling DMA to read "
            f"IEEE_FP32 (32 values/stick) causes ~50% element mismatch.",
        )

    def test_fp32_greaterequal_output_format(self):
        self._assert_fp32_cmp_output_is_fp16("greaterequal")

    def test_fp32_greaterthan_output_format(self):
        self._assert_fp32_cmp_output_is_fp16("greaterthan")

    def test_fp32_lesserequal_output_format(self):
        self._assert_fp32_cmp_output_is_fp16("lesserequal")

    def test_fp32_lesserthan_output_format(self):
        self._assert_fp32_cmp_output_is_fp16("lesserthan")

    def test_fp32_equal_output_format(self):
        self._assert_fp32_cmp_output_is_fp16("equal")

    def test_fp32_notequal_output_format(self):
        self._assert_fp32_cmp_output_is_fp16("notequal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
