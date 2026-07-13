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

from . import colormaps, geo, odim, process


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
    ray_idx = np.where(pick_lo, lo, hi)
    return ray_idx, bin_idx, in_range


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
    # Distance + bearing from this radar to every grid pixel are the SAME for all
    # its tilts, so compute them once (was the dominant cost per sweep).
    lon, lat = grid.lonlat()
    dist, bearing = geo.haversine_bearing(vol.site_lat, vol.site_lon, lat, lon)
    best = None          # running max over valid gates (masked -> -inf)
    any_valid = None     # union of valid pixels across tilts
    for s in vol.sweeps:
        try:
            data = prepare_sweep(path, quantity, s.index, dealias, gate_clutter)
            geom = odim.sweep_geometry(path, s.index)
        except (KeyError, IndexError):
            continue
        vals = _polar_lookup(data, geom, dist, bearing)
        m = np.ma.getmaskarray(vals)
        d = np.where(m, -np.inf, np.ma.getdata(vals))
        best = d if best is None else np.maximum(best, d)
        any_valid = ~m if any_valid is None else (any_valid | ~m)
    if best is None:
        shape = (grid.height, grid.width)
        return np.ma.array(np.zeros(shape), mask=np.ones(shape, bool)), dist
    return np.ma.array(np.where(any_valid, best, 0.0), mask=~any_valid), dist


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
