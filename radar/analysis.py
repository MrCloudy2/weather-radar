"""Temporal + multi-radar analysis products.

  * accumulate()   - integrate rain rate over time -> accumulation (mm)
  * motion_field() - block cross-correlation between two frames -> storm motion
  * advect()       - shift a field along a motion vector -> simple nowcast
  * dual_doppler() - combine two radars' radial velocities -> horizontal wind

These operate on already-gridded fields (shared Web Mercator grid), so the
geometry is uniform and the maths stays simple.
"""

from __future__ import annotations

import math

import numpy as np

from . import _native, geo


# Bearing field from a fixed site to every grid pixel is constant, cache it
# (mirrors render._GEOM_CACHE; sites x grids is a handful of entries).
_AZ_CACHE: dict = {}


def _bearings_f32(site: tuple[float, float], grid) -> np.ndarray:
    key = (round(site[0], 4), round(site[1], 4),
           grid.west, grid.south, grid.east, grid.north, grid.width)
    hit = _AZ_CACHE.get(key)
    if hit is None:
        lon, lat = grid.lonlat()
        _, az = geo.haversine_bearing(site[0], site[1], lat, lon)
        hit = np.ascontiguousarray(az, np.float32)
        if len(_AZ_CACHE) > 8:
            _AZ_CACHE.clear()
        _AZ_CACHE[key] = hit
    return hit


def _mpp(grid) -> tuple[float, float]:
    """Metres per pixel (east, north) at the grid's central latitude."""
    latc = math.radians((grid.north + grid.south) / 2)
    mpp_x = (grid.east - grid.west) / grid.width * 111_320 * math.cos(latc)
    mpp_y = (grid.north - grid.south) / grid.height * 111_320
    return mpp_x, mpp_y


# --- rainfall accumulation --------------------------------------------------
def accumulate(rate_grids: list[np.ma.MaskedArray], minutes_per_frame: float
               ) -> np.ma.MaskedArray:
    """Sum rain-rate (mm/h) grids over time into accumulation (mm)."""
    dt_h = minutes_per_frame / 60.0
    acc = None
    for g in rate_grids:
        contrib = np.ma.filled(g, 0.0) * dt_h
        acc = contrib if acc is None else acc + contrib
    if acc is None:
        return None
    valid = acc > 0.1
    return np.ma.array(np.where(valid, acc, 0.0), mask=~valid)


# --- storm motion + nowcast -------------------------------------------------
def motion_field(prev: np.ma.MaskedArray, curr: np.ma.MaskedArray, grid,
                 dt_seconds: float, block: int = 64, search: int = 16,
                 step: int = 2, min_dbz: float = 12.0) -> list[dict]:
    """Block cross-correlation (min SSD) motion vectors between two dBZ grids.

    Returns one vector per echo block: geographic centre + wind-style u/v (m/s)
    and speed/heading of the echo motion.
    """
    a = np.ascontiguousarray(np.ma.filled(prev, 0.0), np.float32)
    b = np.ascontiguousarray(np.ma.filled(curr, 0.0), np.float32)
    H, W = a.shape
    lon, lat = grid.lonlat()
    mpp_x, mpp_y = _mpp(grid)
    out: list[dict] = []

    if _native.HAVE_NATIVE and "motion_ssd" in _native.IMPLEMENTED:
        cap = ((H // block) + 1) * ((W // block) + 1)
        cx = np.empty(cap, np.int32)
        cy = np.empty(cap, np.int32)
        dxs = np.empty(cap, np.float32)
        dys = np.empty(cap, np.float32)
        cnt = _native.motion_ssd(a, b, block, search, step, min_dbz,
                                 cx, cy, dxs, dys)
        if cnt >= 0:
            for i in range(cnt):
                u = -float(dxs[i]) * mpp_x / dt_seconds
                v = float(dys[i]) * mpp_y / dt_seconds
                speed = math.hypot(u, v)
                heading = (math.degrees(math.atan2(u, v)) + 360) % 360
                out.append({
                    "lat": round(float(lat[cy[i], cx[i]]), 4),
                    "lon": round(float(lon[cy[i], cx[i]]), 4),
                    "u": round(u, 2), "v": round(v, 2),
                    "speed": round(speed, 1), "heading": round(heading),
                })
            return out
    for by in range(0, H - block, block):
        for bx in range(0, W - block, block):
            patch = b[by:by + block, bx:bx + block]
            if patch.max() < min_dbz:
                continue
            best = (1e30, 0, 0)
            for dy in range(-search, search + 1, step):
                for dx in range(-search, search + 1, step):
                    y0, x0 = by + dy, bx + dx
                    if y0 < 0 or x0 < 0 or y0 + block > H or x0 + block > W:
                        continue
                    ref = a[y0:y0 + block, x0:x0 + block]
                    ssd = float(np.mean((patch - ref) ** 2))
                    if ssd < best[0]:
                        best = (ssd, dx, dy)
            _, dx, dy = best
            if dx == 0 and dy == 0:
                continue
            cy, cx = by + block // 2, bx + block // 2
            # echo now at (bx,by) matched prev at (bx+dx,by+dy) => moved (-dx,-dy)px
            u = -dx * mpp_x / dt_seconds                 # east m/s
            v = dy * mpp_y / dt_seconds                  # north m/s (row+ = south)
            speed = math.hypot(u, v)
            heading = (math.degrees(math.atan2(u, v)) + 360) % 360  # toward, 0=N
            out.append({
                "lat": round(float(lat[cy, cx]), 4),
                "lon": round(float(lon[cy, cx]), 4),
                "u": round(u, 2), "v": round(v, 2),
                "speed": round(speed, 1), "heading": round(heading),
            })
    return out


def mean_motion(vectors: list[dict]) -> tuple[float, float]:
    """Robust (median) mean motion (u, v) m/s from a vector list."""
    if not vectors:
        return 0.0, 0.0
    u = float(np.median([v["u"] for v in vectors]))
    v = float(np.median([v["v"] for v in vectors]))
    return u, v


def advect(field: np.ma.MaskedArray, u_ms: float, v_ms: float,
           lead_seconds: float, grid) -> np.ma.MaskedArray:
    """Shift a field by a uniform motion for a lead time (simple nowcast)."""
    mpp_x, mpp_y = _mpp(grid)
    sx = int(round(u_ms * lead_seconds / mpp_x))    # east -> +cols
    sy = int(round(-v_ms * lead_seconds / mpp_y))   # north -> -rows
    data = np.ma.filled(field, np.nan)
    shifted = np.roll(data, (sy, sx), axis=(0, 1))
    # Blank the wrapped-in edges.
    if sy > 0:
        shifted[:sy, :] = np.nan
    elif sy < 0:
        shifted[sy:, :] = np.nan
    if sx > 0:
        shifted[:, :sx] = np.nan
    elif sx < 0:
        shifted[:, sx:] = np.nan
    return np.ma.array(np.nan_to_num(shifted), mask=np.isnan(shifted))


# --- dual-Doppler horizontal wind ------------------------------------------
def dual_doppler(v1: np.ma.MaskedArray, v2: np.ma.MaskedArray,
                 site1: tuple[float, float], site2: tuple[float, float],
                 grid, min_cross_deg: float = 20.0):
    """Solve horizontal wind (u,v) from two radars' radial velocities.

    Vr = u*sin(az) + v*cos(az) for each radar; 2x2 solve per pixel. Valid only
    where the beams cross at a healthy angle (else the inversion is unstable).
    Returns (speed [m/s] masked, u masked, v masked).
    """
    if _native.HAVE_NATIVE and "dual_doppler" in _native.IMPLEMENTED:
        shape = v1.shape
        f1 = np.ascontiguousarray(np.ma.filled(v1, np.nan), np.float32).ravel()
        f2 = np.ascontiguousarray(np.ma.filled(v2, np.nan), np.float32).ravel()
        z1 = _bearings_f32(site1, grid).ravel()
        z2 = _bearings_f32(site2, grid).ravel()
        u = np.empty(f1.size, np.float32)
        v = np.empty(f1.size, np.float32)
        sp = np.empty(f1.size, np.float32)
        if _native.dual_doppler(f1, f2, z1, z2, min_cross_deg, u, v, sp):
            def mk(arr):
                arr = arr.reshape(shape).astype(np.float64)
                return np.ma.array(np.nan_to_num(arr), mask=np.isnan(arr))
            return mk(sp), mk(u), mk(v)

    lon, lat = grid.lonlat()
    _, az1 = geo.haversine_bearing(site1[0], site1[1], lat, lon)
    _, az2 = geo.haversine_bearing(site2[0], site2[1], lat, lon)
    a1, b1 = np.sin(np.radians(az1)), np.cos(np.radians(az1))
    a2, b2 = np.sin(np.radians(az2)), np.cos(np.radians(az2))
    det = a1 * b2 - a2 * b1
    d1 = np.ma.filled(v1, np.nan)
    d2 = np.ma.filled(v2, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (d1 * b2 - d2 * b1) / det
        v = (a1 * d2 - a2 * d1) / det
    speed = np.hypot(u, v)
    cross = np.abs(np.sin(np.radians(az1 - az2)))          # ~|det|
    bad = (np.ma.getmaskarray(v1) | np.ma.getmaskarray(v2)
           | np.isnan(d1) | np.isnan(d2)
           | (cross < math.sin(math.radians(min_cross_deg))))
    mk = lambda arr: np.ma.array(np.nan_to_num(arr), mask=bad)
    return mk(speed), mk(u), mk(v)
