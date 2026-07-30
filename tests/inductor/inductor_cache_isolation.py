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

"""Give a compile its own cold Inductor cache.

Any torch-spyre config read during *lowering* — `swa_window_roll`, and it is
not the only one — is consulted after Inductor has already computed the FX
graph cache key. Two compiles of the same graph under different flag values
therefore share a key, and the second replays the first's artifact. A test that
toggles such a flag within one process silently compares a path against itself.

`torch._inductor.config.force_disable_caches` is not a usable fix here: it
redirects `cache_dir()` to a fresh temporary directory, so the emitted bundles
land somewhere the caller did not sample. Setting TORCHINDUCTOR_CACHE_DIR gives
both properties at once — a cold cache, and a known location.
"""

from contextlib import contextmanager
import os
import tempfile


@contextmanager
def isolated_inductor_cache():
    """Point Inductor at a private cache directory, yielded to the caller.

    The directory is deliberately not removed on exit: the bundles inside are
    frequently the reason the test ran at all.
    """
    previous = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    directory = tempfile.mkdtemp(prefix="spyre_isolated_cache_")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = directory
    try:
        yield directory
    finally:
        if previous is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = previous
