"""Optional native (Rust) kernels, loaded via ctypes.

If the shared lib is missing, or a given kernel returns NOT_IMPLEMENTED (-1),
callers fall back to the pure-NumPy path — identical result, slower.

Each wrapper returns True if the native kernel ran (status 0), False otherwise.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_F32 = ctypes.POINTER(ctypes.c_float)
_I32 = ctypes.POINTER(ctypes.c_int32)
_F64 = ctypes.POINTER(ctypes.c_double)
_c32 = ctypes.c_int32
_cf = ctypes.c_float

_SIGS = {
    "cmax_sweep":     [_c32, _I32, _F32, _F32, _c32, _F64, _F32],
    "sample_sweep":   [_c32, _I32, _F32, _F32, _c32, _F64, _F32],
    "echo_top_sweep": [_c32, _I32, _F32, _F32, _c32, _F64, _F32, _cf, _F32],
    "vil_column":     [_c32, _c32, _F32, _F32, _F32],
    "nearest_mosaic": [_c32, _c32, _F32, _F32, _F32],
    "dealias_rays":   [_c32, _c32, _F32, _cf],
    "motion_ssd":     [_c32, _c32, _F32, _F32, _c32, _c32, _c32, _cf,
                       _I32, _I32, _F32, _F32, _c32],
    "dual_doppler":   [_c32, _F32, _F32, _F32, _F32, _cf, _F32, _F32, _F32],
}

_lib = None
for _name in ("libradar_native.so", "libcmax.so"):
    for _d in (Path(__file__).with_name(_name), Path("/app/radar") / _name):
        if _d.exists():
            try:
                _l = ctypes.CDLL(str(_d))
                for _fn, _args in _SIGS.items():
                    if hasattr(_l, _fn):
                        getattr(_l, _fn).restype = _c32
                        getattr(_l, _fn).argtypes = _args
                _lib = _l
                break
            except OSError:
                pass
    if _lib is not None:
        break

HAVE_NATIVE = _lib is not None


def _p(a, t):
    return a.ctypes.data_as(t)


# --- reflectivity (implemented) -------------------------------------------
def cmax_sweep(ray_flat, dist_flat, polar, edges, best) -> bool:
    if _lib is None:
        return False
    return _lib.cmax_sweep(
        ray_flat.size, _p(ray_flat, _I32), _p(dist_flat, _F32),
        _p(polar, _F32), polar.shape[1], _p(edges, _F64), _p(best, _F32)) == 0


def sample_sweep(ray_flat, dist_flat, polar, edges, out) -> bool:
    if _lib is None:
        return False
    return _lib.sample_sweep(
        ray_flat.size, _p(ray_flat, _I32), _p(dist_flat, _F32),
        _p(polar, _F32), polar.shape[1], _p(edges, _F64), _p(out, _F32)) == 0


# --- column products (stubs until written) --------------------------------
def echo_top_sweep(ray_flat, dist_flat, polar, edges, heights_bin, thr, et) -> bool:
    if _lib is None:
        return False
    return _lib.echo_top_sweep(
        ray_flat.size, _p(ray_flat, _I32), _p(dist_flat, _F32),
        _p(polar, _F32), polar.shape[1], _p(edges, _F64),
        _p(heights_bin, _F32), float(thr), _p(et, _F32)) == 0


def vil_column(dbz_stack, height_stack, vil) -> bool:
    if _lib is None:
        return False
    nsweeps, npix = dbz_stack.shape
    return _lib.vil_column(npix, nsweeps, _p(dbz_stack, _F32),
                           _p(height_stack, _F32), _p(vil, _F32)) == 0


# --- velocity (stubs) ------------------------------------------------------
def nearest_mosaic(values, dists, out) -> bool:
    if _lib is None:
        return False
    nradar, npix = values.shape
    return _lib.nearest_mosaic(npix, nradar, _p(values, _F32),
                               _p(dists, _F32), _p(out, _F32)) == 0


def dealias_rays(vel, nyquist) -> bool:
    if _lib is None:
        return False
    nrays, nbins = vel.shape
    return _lib.dealias_rays(nrays, nbins, _p(vel, _F32), float(nyquist)) == 0


# --- analysis (stubs) ------------------------------------------------------
def dual_doppler(v1, v2, az1, az2, min_cross_deg, u, v, speed) -> bool:
    if _lib is None:
        return False
    return _lib.dual_doppler(
        v1.size, _p(v1, _F32), _p(v2, _F32), _p(az1, _F32), _p(az2, _F32),
        float(min_cross_deg), _p(u, _F32), _p(v, _F32), _p(speed, _F32)) == 0


IMPLEMENTED = {"cmax_sweep", "sample_sweep"}   # the rest return NOT_IMPLEMENTED
