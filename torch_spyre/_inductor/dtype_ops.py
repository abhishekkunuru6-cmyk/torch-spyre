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
)

# Spyre has no native bool representation: a torch.bool tensor physically
# reuses whichever hardware format produced it (e.g. comparing two float16
# tensors yields a 16-bit-wide SEN169_FP16 bool, comparing two float32
# tensors yields a 32-bit-wide IEEE_FP32 bool). This maps each physical
# format a bool can take to the logical dtype that natively uses that same
# format, so a bool's conversions can be resolved by treating it as if it
# were that equivalent dtype (see DtypeOpTable.get_bool_src_operator).
_BOOL_EQUIVALENT_DTYPES: Mapping[DataFormats, torch.dtype] = {
    DataFormats.SEN169_FP16: torch.float16,
    DataFormats.IEEE_FP32: torch.float32,
}


def bool_equivalent_dtype(device_dtype: DataFormats) -> Optional[torch.dtype]:
    """Logical dtype that natively uses the same physical format as a bool.

    Returns None if `device_dtype` has no known bool-compatible equivalent,
    i.e. a bool with that physical format isn't supported.
    """
    return _BOOL_EQUIVALENT_DTYPES.get(device_dtype)


class DtypeOpTable:
    _IDENTITY_DTYPES = [
        # float16 -> bool: the bool result reuses float16's physical format
        # (a bool inherits its producing operand's physical width -- see
        # bool_equivalent_dtype), so this is a same-width reinterpret.
        # NOTE: the reverse, bool -> float16, is *not* always identity --
        # a bool can also be 32-bit (e.g. produced by comparing float32
        # tensors), so it is resolved dynamically from the source's actual
        # physical format instead -- see get_bool_src_operator.
        (torch.float16, torch.bool),
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
    ]

    _FP16_TO_FP32_DTYPES = [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ]

    _FP32_TO_FP16_DTYPES = [
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
    ]

    _TYPECAST_OPS_TABLE = {
        **{pair: IDENTITY_OP for pair in _IDENTITY_DTYPES},
        **{pair: DL16TOFP32_OP for pair in _FP16_TO_FP32_DTYPES},
        **{pair: FP32TODL16_OP for pair in _FP32_TO_FP16_DTYPES},
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

        A bool's logical dtype alone doesn't determine its physical format
        (see bool_equivalent_dtype), so bool sources can't be looked up in
        the static table above by logical dtype pair. Instead, treat the
        bool as if it were the dtype that's physically equivalent to its
        actual, already-propagated `device_dtype`, and resolve the op for
        that pair (an identity reinterpret if the formats already match).
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

        For torch.bool sources the concrete op depends on the tensor's
        physical format, which isn't known until layout propagation runs
        (see get_bool_src_operator); this only checks whether *some*
        physical format of a bool source could convert to dst_dtype.
        """
        if src_dtype == torch.bool:
            return dst_dtype in _BOOL_EQUIVALENT_DTYPES.values()
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
