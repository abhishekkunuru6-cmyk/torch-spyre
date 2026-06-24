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

"""
Dtype conversion operator table for torch-spyre.

This module provides a centralized table for dtype conversion operators,
mapping PyTorch dtype pairs to Spyre hardware operators.
"""

from typing import Mapping, Optional

import torch

from torch_spyre._C import DataFormats
from torch_spyre._inductor.constants import (
    IDENTITY_OP,
    DL16TOFP32_OP,
    FP32TODL16_OP,
    FP8TODL16_OP,
)

# Spyre has no native bool: a bool tensor reuses whichever physical format
# produced it (e.g. fp16 vs fp32 comparison results). Maps that format to
# the logical dtype that natively uses it.
_BOOL_EQUIVALENT_DTYPES: Mapping[DataFormats, torch.dtype] = {
    DataFormats.SEN169_FP16: torch.float16,
    DataFormats.IEEE_FP32: torch.float32,
}


def bool_equivalent_dtype(device_dtype: DataFormats) -> Optional[torch.dtype]:
    """Logical dtype matching a bool's physical format, or None if unsupported."""
    return _BOOL_EQUIVALENT_DTYPES.get(device_dtype)


class DtypeOpTable:
    _IDENTITY_DTYPES = [
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.bool),
        (torch.bfloat16, torch.bool),
        (torch.float32, torch.bool),
    ]

    _FP16_TO_FP32_DTYPES = [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ]

    _FP32_TO_FP16_DTYPES = [
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
    ]

    _FP8_TO_FP16_DTYPES = [
        (torch.float8_e4m3fn, torch.float16),
    ]

    _TYPECAST_OPS_TABLE = {
        **{pair: IDENTITY_OP for pair in _IDENTITY_DTYPES},
        **{pair: DL16TOFP32_OP for pair in _FP16_TO_FP32_DTYPES},
        **{pair: FP32TODL16_OP for pair in _FP32_TO_FP16_DTYPES},
        **{pair: FP8TODL16_OP for pair in _FP8_TO_FP16_DTYPES},
    }

    _TYPECAST_OP_NAMES = set(_TYPECAST_OPS_TABLE.values())
    _TYPECAST_OP_DTYPES = set(_TYPECAST_OPS_TABLE.keys())

    @classmethod
    def get_operator(
        cls, src_dtype: torch.dtype, dst_dtype: torch.dtype
    ) -> Optional[str]:
        return cls._TYPECAST_OPS_TABLE.get((src_dtype, dst_dtype))

    @classmethod
    def get_bool_src_operator(
        cls, device_dtype: DataFormats, dst_dtype: torch.dtype
    ) -> Optional[str]:
        """Resolve the conversion op for a torch.bool source tensor.

        Treats the bool as its physically-equivalent dtype (from
        `device_dtype`) and resolves the op for that pair.
        """
        equivalent_src_dtype = bool_equivalent_dtype(device_dtype)
        if equivalent_src_dtype is None:
            return None
        if equivalent_src_dtype == dst_dtype:
            return IDENTITY_OP
        return cls.get_operator(equivalent_src_dtype, dst_dtype)

    @classmethod
    def is_supported(cls, src_dtype: torch.dtype, dst_dtype: torch.dtype) -> bool:
        """Whether Spyre can natively perform this dtype conversion.

        For torch.bool sources, checks whether *some* physical format
        could convert to dst_dtype (see get_bool_src_operator).
        """
        if src_dtype == torch.bool:
            return any(
                cls.get_bool_src_operator(fmt, dst_dtype) is not None
                for fmt in _BOOL_EQUIVALENT_DTYPES
            )
        return cls.get_operator(src_dtype, dst_dtype) is not None

    @classmethod
    def get_table(
        cls,
    ) -> Mapping[tuple[torch.dtype, torch.dtype], str]:
        return cls._TYPECAST_OPS_TABLE

    @classmethod
    def get_dtype_pairs(cls) -> set[tuple[torch.dtype, torch.dtype]]:
        return cls._TYPECAST_OP_DTYPES

    @classmethod
    def is_dtype_op(cls, op: str) -> bool:
        return op in cls._TYPECAST_OP_NAMES
