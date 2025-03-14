#  Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

from typing import Dict
from ..tensor import Tensor, from_buffer
from ..dtype import Dtype
import struct, pathlib, os, hashlib, urllib.request, json, ctypes
from tqdm import tqdm

def _fetch(url: str, invalidate_cache: bool = False) -> pathlib.Path:
    cache_dir = pathlib.Path.home() / '.cache' / 'dsc' / 'blob'
    if invalidate_cache and cache_dir.exists():
        os.removedirs(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / hashlib.md5(url.encode('utf-8')).hexdigest()
    if not fp.exists():
        with urllib.request.urlopen(url, timeout=10) as r:
            assert r.status == 200
            pbar = tqdm(total=r.length, unit='B', unit_scale=True, desc=url)
            with open(fp, mode="w+b") as f:
                while chunk := r.read(8192):
                    pbar.update(f.write(chunk))
    return fp


def safe_load(url: str, invalidate_cache: bool = False) -> Dict[str, Tensor]:
    fp = _fetch(url, invalidate_cache)
    b = fp.read_bytes()
    n = struct.unpack_from('<Q', b)[0]
    assert 0 < n < (len(b) - 8)

    header = json.loads(b[8:8+n].decode('utf-8'))
    res = {}
    for key, val in header.items():
        if key == '__metadata__':
            continue

        dtype = Dtype.from_string(val['dtype'])
        shape = val['shape']
        offset_start = val['data_offsets'][0]
        offset_stop = val['data_offsets'][1]

        data_ptr = ctypes.cast(ctypes.c_char_p(b[8+n+offset_start:8+n+offset_stop]), ctypes.c_void_p)

        res[key] = from_buffer(shape, dtype, data_ptr)

    return res