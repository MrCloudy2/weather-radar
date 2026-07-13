"""Rasterise polar sweeps onto the shared Web Mercator grid and colour them.

Reverse-mapping: for every grid pixel we compute ground range + bearing from a
radar, look up the owning (ray, bin) in the polar array, and sample it. Because
every panel/composite renders on the same EPSG:3857 grid, overlays register
pixel-perfectly under Leaflet and multi-panel hover lines up.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import _native, colormaps, geo, odim, process


@dataclass
class RenderResult:
    png: bytes
    bounds: list[list[float]]   # Leaflet [[south, west], [north, east]]
    width: int
    height: int


def _lookup_indices(
    geom: odim.SweepGeometry, dist: np.ndarray, bearing: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each pixel's (ground dist, bearing) to polar (ray_idx, bin_idx, in_range)."""
    ground = geo.beam_ground_range(geom.slant_ranges, geom.elangle_rad)
    edges = np.empty(ground.size + 1)
    edges[1:-1] = (ground[:-1] + ground[1:]) / 2.0
    edges[0] = ground[0] - (ground[1] - ground[0]) / 2.0
    edges[-1] = ground[-1] + (ground[-1] - ground[-2]) / 2.0
    bin_idx = np.searchsorted(edges, dist) - 1
    in_range = (bin_idx >= 0) & (bin_idx < ground.size)
    bin_idx = np.clip(bin_idx, 0, ground.size - 1)

    az = geom.az_mid
    n = az.size
    hi = np.searchsorted(az, bearing) % n
    lo = (hi - 1) % n

    def ang_gap(a, b):
        d = np.abs(a - b) % 360.0
        return np.minimum(d, 360.0 - d)

    pick_lo = ang_gap(bearing, az[lo]) <= ang_gap(bearing, az[hi])
    # int32 indices (grid < 2.1e9, rays<360, bins<1000) halve index memory.
    ray_idx = np.where(pick_lo, lo, hi).astype(np.int32)
    return ray_idx, bin_idx.astype(np.int32), in_range


def _polar_lookup(
    data: np.ma.MaskedArray,
    geom: odim.SweepGeometry,
    dist: np.ndarray,
    bearing: np.ndarray,
) -> np.ma.MaskedArray:
    """Sample the polar sweep at each pixel's (ground dist, bearing)."""
    ray_idx, bin_idx, in_range = _lookup_indices(geom, dist, bearing)
    sampled = data[ray_idx, bin_idx]
    mask = np.ma.getmaskarray(sampled) | ~in_range
    return np.ma.array(np.ma.getdata(sampled), mask=mask)


def _sample_f32(data, geom, dist, bearing) -> np.ndarray:
    """Sample a sweep onto the grid as float32 with NaN for invalid gates.

    NaN sentinel lets the CMAX loop use np.fmax (NaN-aware max) with no mask
    bookkeeping and half the memory of masked float64 arrays.
    """
    ray_idx, bin_idx, in_range = _lookup_indices(geom, dist, bearing)
    out = np.ma.getdata(data)[ray_idx, bin_idx].astype(np.float32)
    invalid = np.ma.getmaskarray(data)[ray_idx, bin_idx]
    invalid |= ~in_range
    out[invalid] = np.nan
    return out


# Per-radar (site + grid) distance and bearing->ray mapping are constant across
# frames (only elevation-dependent bin lookup changes per sweep), so cache them.
_GEOM_CACHE: dict = {}


def _radar_grid_geom(vol, grid, sample_geom):
    """Cached (dist_f32, ray_idx int32) from a radar site to every grid pixel."""
    sig = (round(vol.site_lat, 4), round(vol.site_lon, 4),
           grid.west, grid.south, grid.east, grid.north, grid.width)
    hit = _GEOM_CACHE.get(sig)
    if hit is not None:
        return hit
    lon, lat = grid.lonlat()
    dist, bearing = geo.haversine_bearing(vol.site_lat, vol.site_lon, lat, lon)
    az = sample_geom.az_mid           # rays ~identical across sweeps of a radar
    n = az.size
    hi = np.searchsorted(az, bearing) % n
    lo = (hi - 1) % n

    def ang_gap(a, b):
        d = np.abs(a - b) % 360.0
        return np.minimum(d, 360.0 - d)

    ray_idx = np.where(ang_gap(bearing, az[lo]) <= ang_gap(bearing, az[hi]),
                       lo, hi).astype(np.int32)
    res = (dist.astype(np.float32), ray_idx)
    if len(_GEOM_CACHE) > 16:
        _GEOM_CACHE.clear()
    _GEOM_CACHE[sig] = res
    return res


def _bin_edges(geom) -> np.ndarray:
    """Ground-range bin edges [nbins+1] for one sweep (shared by numpy + native)."""
    ground = geo.beam_ground_range(geom.slant_ranges, geom.elangle_rad)
    edges = np.empty(ground.size + 1)
    edges[1:-1] = (ground[:-1] + ground[1:]) / 2.0
    edges[0] = ground[0] - (ground[1] - ground[0]) / 2.0
    edges[-1] = ground[-1] + (ground[-1] - ground[-2]) / 2.0
    return edges


def _bin_lookup(geom, dist_f):
    """Ground-range bin index + in-range mask for one sweep (per elevation)."""
    edges = _bin_edges(geom)
    nbins = geom.slant_ranges.size
    bin_idx = np.searchsorted(edges, dist_f).astype(np.int32) - 1
    in_range = (bin_idx >= 0) & (bin_idx < nbins)
    np.clip(bin_idx, 0, nbins - 1, out=bin_idx)
    return bin_idx, in_range


def _sample_cached(data, ray_idx, bin_idx, in_range) -> np.ndarray:
    """Flat-index sample -> float32 grid with NaN for invalid gates."""
    nbins = data.shape[1]
    flat = ray_idx * np.int32(nbins) + bin_idx     # < 360*1000, fits int32
    raw = np.ma.getdata(data).ravel()
    out = raw.take(flat).astype(np.float32)
    invalid = np.ma.getmaskarray(data).ravel().take(flat)
    invalid |= ~in_range
    out[invalid] = np.nan
    return out


def prepare_sweep(
    path: str | Path,
    quantity: str,
    sweep_index: int = 0,
    dealias: bool = True,
    gate_clutter: bool = True,
) -> np.ma.MaskedArray:
    """Read a moment sweep and apply per-moment processing.

    Handles file moments (DBZH/TH/VRADH/RHOHV/ZDR) and derived ones
    (RATE from reflectivity, AZSHEAR from velocity).
    """
    # AZSHEAR is derived from velocity; it isn't a stored moment.
    if quantity == "AZSHEAR":
        vel = prepare_sweep(path, "VRADH", sweep_index, dealias, gate_clutter)
        geom = odim.sweep_geometry(path, sweep_index)
        return process.azimuthal_shear(vel, geom.slant_ranges, vel.shape[0])

    # RATE is derived from reflectivity via Z-R.
    src = "DBZH" if quantity == "RATE" else quantity
    data = odim.read_sweep(path, src, sweep_index)
    if quantity in ("DBZH", "TH", "RATE"):
        if src == "DBZH":
            data = process.floor(data, 5.0)      # drop <5 dBZ clutter haze
        if gate_clutter:
            data = process.despeckle(data, min_valid=3)
        if quantity == "RATE":
            data = process.dbz_to_rate(data)
    elif quantity == "VRADH":
        if gate_clutter and "DBZH" in odim.summarize(path).sweeps[sweep_index].moments:
            refl = odim.read_sweep(path, "DBZH", sweep_index)
            data = process.gate_by_reflectivity(data, refl)
            data = process.despeckle(data, min_valid=3)
        if dealias:
            geom = odim.sweep_geometry(path, sweep_index)
            data = process.dealias(data, geom.nyquist)
    elif quantity in ("RHOHV", "ZDR"):
        # Polarimetric moments (Croatian radars only). Only meaningful in
        # precipitation, so gate to gates with real reflectivity -- this keeps
        # tornado-debris / hail signatures (they have echo) but drops clear-air
        # and clutter noise that otherwise dominates the field.
        if gate_clutter and "DBZH" in odim.summarize(path).sweeps[sweep_index].moments:
            refl = process.floor(odim.read_sweep(path, "DBZH", sweep_index), 5.0)
            data = process.gate_by_reflectivity(data, refl)
            data = process.despeckle(data, min_valid=3)
    return data


def sample_to_grid(
    path: str | Path,
    quantity: str,
    grid: geo.Grid,
    sweep_index: int = 0,
    dealias: bool = True,
    gate_clutter: bool = True,
) -> tuple[np.ma.MaskedArray, np.ndarray]:
    """Sample one radar's sweep onto the shared grid.

    Returns (values [H x W] masked physical array, dist_m [H x W] ground range
    from this radar to each pixel). dist_m feeds the nearest-radar velocity
    mosaic; masked where out of range.
    """
    vol = odim.summarize(path)
    geom = odim.sweep_geometry(path, sweep_index)
    data = prepare_sweep(path, quantity, sweep_index, dealias, gate_clutter)
    lon, lat = grid.lonlat()
    dist, bearing = geo.haversine_bearing(vol.site_lat, vol.site_lon, lat, lon)
    values = _polar_lookup(data, geom, dist, bearing)
    return values, dist


def sample_cmax_to_grid(
    path: str | Path,
    quantity: str,
    grid: geo.Grid,
    dealias: bool = True,
    gate_clutter: bool = True,
) -> tuple[np.ma.MaskedArray, np.ndarray]:
    """Column-max: max over ALL elevation tilts at each grid pixel.

    This is how national mosaics (e.g. ARSO SIRAD) are built. In mountainous
    terrain the low beam is blocked/overshoots, so combining tilts recovers echo
    the base sweep misses. Reflectivity only -- averaging/maxing radial velocity
    across tilts is not physical.
    """
    vol = odim.summarize(path)
    geom0 = odim.sweep_geometry(path, vol.sweeps[0].index)
    # Distance + ray mapping are constant per radar across frames (cached); only
    # the per-elevation bin lookup changes per sweep.
    dist_f, ray_idx = _radar_grid_geom(vol, grid, geom0)
    shape = (grid.height, grid.width)

    def _sweeps():
        for s in vol.sweeps:
            try:
                data = prepare_sweep(path, quantity, s.index, dealias, gate_clutter)
                geom = odim.sweep_geometry(path, s.index)
            except (KeyError, IndexError):
                continue
            yield data, geom

    if _native.HAVE_NATIVE:
        # Fused native kernel: bin-search + sample + NaN-max in one pass/sweep.
        ray_flat = np.ascontiguousarray(ray_idx.ravel(), np.int32)
        dist_flat = np.ascontiguousarray(dist_f.ravel(), np.float32)
        best = np.full(ray_flat.size, np.nan, np.float32)
        used = False
        for data, geom in _sweeps():
            polar = np.ascontiguousarray(np.ma.filled(data, np.nan), np.float32)
            _native.cmax_sweep(ray_flat, dist_flat, polar, _bin_edges(geom), best)
            used = True
        if not used:
            return np.ma.array(np.zeros(shape, np.float32), mask=np.ones(shape, bool)), dist_f
        best = best.reshape(shape)
        return np.ma.array(np.nan_to_num(best), mask=np.isnan(best)), dist_f

    # Pure-NumPy fallback (identical result).
    best = None
    for data, geom in _sweeps():
        bin_idx, in_range = _bin_lookup(geom, dist_f)
        vals = _sample_cached(data, ray_idx, bin_idx, in_range)
        best = vals if best is None else np.fmax(best, vals, out=best)
    if best is None:
        return np.ma.array(np.zeros(shape, np.float32), mask=np.ones(shape, bool)), dist_f
    return np.ma.array(np.nan_to_num(best), mask=np.isnan(best)), dist_f


def render_cmax(
    path: str | Path, quantity: str, grid: geo.Grid, dealias: bool = True
) -> RenderResult:
    """Render a single radar's column-max onto the shared grid."""
    values, _ = sample_cmax_to_grid(path, quantity, grid, dealias)
    return render_grid(values, quantity, grid)


ECHO_TOP_DBZ = 18.0          # reflectivity defining the echo top
VIL_CAP_DBZ = 56.0           # cap to keep hail from inflating VIL


def _reflectivity_layers(path, grid, dist, bearing, site_height_m):
    """Per-tilt (reflectivity dBZ, beam-height m) grids for column products."""
    vol = odim.summarize(path)
    layers = []
    for s in vol.sweeps:
        data = prepare_sweep(path, "DBZH", s.index)
        geom = odim.sweep_geometry(path, s.index)
        ray_idx, bin_idx, in_range = _lookup_indices(geom, dist, bearing)
        vals = data[ray_idx, bin_idx]
        vmask = np.ma.getmaskarray(vals) | ~in_range
        r = geom.slant_ranges[bin_idx]
        el = geom.elangle_rad
        height = (
            np.sqrt(r * r + geo.A_E**2 + 2 * r * geo.A_E * np.sin(el))
            - geo.A_E + site_height_m
        )
        layers.append((np.ma.getdata(vals), vmask, height))
    return layers


def sample_column_to_grid(
    path: str | Path, product: str, grid: geo.Grid, dealias: bool = True
) -> tuple[np.ma.MaskedArray, np.ndarray]:
    """Compute a vertical-column product (ET or VIL) on the shared grid."""
    vol = odim.summarize(path)
    lon, lat = grid.lonlat()
    dist, bearing = geo.haversine_bearing(vol.site_lat, vol.site_lon, lat, lon)
    layers = _reflectivity_layers(path, grid, dist, bearing, vol.site_height_m)
    shape = (grid.height, grid.width)

    if product == "ET":
        et = np.full(shape, -np.inf)
        for vals, vmask, height in layers:
            echo = ~vmask & (vals >= ECHO_TOP_DBZ)
            et = np.where(echo, np.maximum(et, height), et)
        valid = np.isfinite(et) & (et > -np.inf)
        return np.ma.array(np.where(valid, et / 1000.0, 0.0), mask=~valid), dist

    # VIL: integrate liquid-water content M(Z) over height (trapezoid on the
    # per-pixel vertical profile of tilts).
    n = len(layers)
    Z = np.stack([np.where(m, np.nan, np.minimum(v, VIL_CAP_DBZ))
                  for v, m, _ in layers])          # [n,H,W] dBZ, nan where empty
    H = np.stack([h for _, _, h in layers])        # [n,H,W] metres
    order = np.argsort(H, axis=0)                   # ascending height per pixel
    H = np.take_along_axis(H, order, axis=0)
    Z = np.take_along_axis(Z, order, axis=0)
    zlin = np.power(10.0, Z / 10.0)
    M = 3.44e-6 * np.power(zlin, 4.0 / 7.0)         # kg/m^3, nan where empty
    M = np.nan_to_num(M, nan=0.0)
    dh = np.diff(H, axis=0)                          # [n-1,H,W]
    vil = np.sum((M[1:] + M[:-1]) / 2.0 * dh, axis=0)
    valid = vil > 0.05
    return np.ma.array(np.where(valid, vil, 0.0), mask=~valid), dist


def render_column(
    path: str | Path, product: str, grid: geo.Grid, dealias: bool = True
) -> RenderResult:
    values, _ = sample_column_to_grid(path, product, grid, dealias)
    return render_grid(values, product, grid)


def render_grid(
    values: np.ma.MaskedArray, quantity: str, grid: geo.Grid
) -> RenderResult:
    """Colour a gridded physical field into a PNG + Leaflet bounds."""
    rgba = colormaps.apply(values, quantity)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return RenderResult(
        png=buf.getvalue(),
        bounds=grid.leaflet_bounds,
        width=grid.width,
        height=grid.height,
    )


def render_sweep(
    path: str | Path,
    quantity: str,
    grid: geo.Grid,
    sweep_index: int = 0,
    dealias: bool = True,
    gate_clutter: bool = True,
) -> RenderResult:
    """Render a single radar's moment onto the shared grid."""
    values, _ = sample_to_grid(
        path, quantity, grid, sweep_index, dealias, gate_clutter
    )
    return render_grid(values, quantity, grid)


def sample_point_column(
    path: str | Path, product: str, lat: float, lon: float, dealias: bool = True
) -> dict:
    """ET/VIL value at one geographic point (hover tool)."""
    vol = odim.summarize(path)
    latA, lonA = np.array([lat]), np.array([lon])
    dist, bearing = geo.haversine_bearing(vol.site_lat, vol.site_lon, latA, lonA)
    profile = []  # (dbz, height_m)
    for s in vol.sweeps:
        data = prepare_sweep(path, "DBZH", s.index, dealias)
        geom = odim.sweep_geometry(path, s.index)
        ray, bin_i, inr = _lookup_indices(geom, dist, bearing)
        if not inr[0]:
            continue
        v = data[ray[0], bin_i[0]]
        r = geom.slant_ranges[bin_i[0]]
        h = (np.sqrt(r * r + geo.A_E**2 + 2 * r * geo.A_E * np.sin(geom.elangle_rad))
             - geo.A_E + vol.site_height_m)
        profile.append((None if np.ma.is_masked(v) else float(v), float(h)))
    if not profile:
        return {"value": None, "in_range": False, "range_km": round(float(dist[0]) / 1000, 1)}
    units = colormaps.UNITS.get(product, "")
    if product == "ET":
        tops = [h for v, h in profile if v is not None and v >= ECHO_TOP_DBZ]
        val = round(max(tops) / 1000.0, 1) if tops else None
    else:  # VIL
        pts = sorted((h, v) for v, h in profile if v is not None)
        vil = 0.0
        for (h0, z0), (h1, z1) in zip(pts, pts[1:]):
            m0 = 3.44e-6 * (10 ** (min(z0, VIL_CAP_DBZ) / 10)) ** (4 / 7)
            m1 = 3.44e-6 * (10 ** (min(z1, VIL_CAP_DBZ) / 10)) ** (4 / 7)
            vil += (m0 + m1) / 2 * (h1 - h0)
        val = round(vil, 1) if vil > 0.05 else None
    return {"value": val, "units": units, "in_range": True,
            "range_km": round(float(dist[0]) / 1000, 1)}


def sample_point(
    path: str | Path,
    quantity: str,
    lat: float,
    lon: float,
    sweep_index: int = 0,
    dealias: bool = True,
    gate_clutter: bool = True,
) -> dict:
    """Return the moment value + geometry at one geographic point (hover tool)."""
    vol = odim.summarize(path)
    geom = odim.sweep_geometry(path, sweep_index)
    data = prepare_sweep(path, quantity, sweep_index, dealias, gate_clutter)

    dist, bearing = geo.haversine_bearing(
        vol.site_lat, vol.site_lon, np.array([lat]), np.array([lon])
    )
    dist = float(dist[0])
    bearing = float(bearing[0])

    ground = geo.beam_ground_range(geom.slant_ranges, geom.elangle_rad)
    if dist > ground[-1]:
        return {"value": None, "in_range": False, "range_km": round(dist / 1000.0, 1)}
    bin_idx = int(np.argmin(np.abs(ground - dist)))
    gap = np.abs(((bearing - geom.az_mid) + 180) % 360 - 180)
    ray_idx = int(np.argmin(gap))

    val = data[ray_idx, bin_idx]
    r = geom.slant_ranges[bin_idx]
    height_m = (
        np.sqrt(r * r + geo.A_E**2 + 2 * r * geo.A_E * np.sin(geom.elangle_rad))
        - geo.A_E
        + vol.site_height_m
    )
    return {
        "value": None if np.ma.is_masked(val) else round(float(val), 2),
        "units": colormaps.UNITS.get(quantity, ""),
        "in_range": True,
        "range_km": round(dist / 1000.0, 1),
        "azimuth_deg": round(bearing, 1),
        "beam_height_m": round(float(height_m), 0),
        "elevation_deg": round(float(np.degrees(geom.elangle_rad)), 1),
    }


def sample_point_cmax(
    path: str | Path, quantity: str, lat: float, lon: float,
    dealias: bool = True,
) -> dict:
    """Column-max point value: the strongest tilt at this location (with its
    beam height/elevation), matching the CMAX display."""
    vol = odim.summarize(path)
    best = None
    for s in vol.sweeps:
        r = sample_point(path, quantity, lat, lon, s.index, dealias)
        if not r.get("in_range"):
            continue
        if best is None:
            best = r                      # first in-range tilt (may have no echo)
        elif r.get("value") is not None and (
            best.get("value") is None or r["value"] > best["value"]
        ):
            best = r                      # a stronger (or first real) echo
    return best or {"value": None, "in_range": False, "range_km": None}
