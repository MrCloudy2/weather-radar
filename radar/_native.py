"""Optional native (Rust) reflectivity CMAX kernel, loaded via ctypes.

If `libcmax.so` isn't present (e.g. local dev without the build step), HAVE_NATIVE
is False and callers fall back to the pure-NumPy path — same result, ~2x slower.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_F32 = ctypes.POINTER(ctypes.c_float)
_I32 = ctypes.POINTER(ctypes.c_int32)
_F64 = ctypes.POINTER(ctypes.c_double)

_lib = None
for _p in (Path(__file__).with_name("libcmax.so"), Path("/app/radar/libcmax.so")):
    if _p.exists():
        try:
            _l = ctypes.CDLL(str(_p))
            _l.cmax_sweep.restype = None
            _l.cmax_sweep.argtypes = [
                ctypes.c_int32, _I32, _F32, _F32, ctypes.c_int32, _F64, _F32,
            ]
            _lib = _l
            break
        except OSError:
            pass

HAVE_NATIVE = _lib is not None


def cmax_sweep(ray_flat, dist_flat, polar, edges, best) -> None:
    """Accumulate one sweep's NaN-aware column-max into `best` (native kernel).

    ray_flat  int32   [npix]      ray index per grid pixel
    dist_flat float32 [npix]      ground distance per pixel
    polar     float32 [nrays,nbins] sweep data, NaN for invalid gates (C-contig)
    edges     float64 [nbins+1]   ground-range bin edges (monotonic)
    best      float32 [npix]      running max, NaN-initialised (updated in place)
    """
    npix = ray_flat.size
    nbins = polar.shape[1]
    _lib.cmax_sweep(
        npix,
        ray_flat.ctypes.data_as(_I32),
        dist_flat.ctypes.data_as(_F32),
        polar.ctypes.data_as(_F32),
        nbins,
        edges.ctypes.data_as(_F64),
        best.ctypes.data_as(_F32),
    )
