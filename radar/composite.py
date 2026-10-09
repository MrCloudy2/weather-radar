"""Multi-radar compositing onto the shared grid.

Reflectivity (DBZH/TH): max-value mosaic -- each pixel takes the strongest echo
across radars (standard national-mosaic method; robust to per-radar beam
blockage / cone-of-silence).

Velocity (VRADH): nearest-radar mosaic -- radial velocity is radar-relative and
cannot be averaged across sites, so each pixel takes the value from its nearest
radar that has a valid echo. Seams at equidistant boundaries are expected and
honest; each patch is physically valid for its own radar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _native, colormaps, geo, opera, process, render

COLUMN_PRODUCTS = {"ET", "VIL"}                        # vertical, all-tilt
CMAX_MOMENTS = {"DBZH", "TH", "RATE"}                  # support the CMAX toggle
MAX_MOMENTS = {"DBZH", "TH", "RATE", "ET", "VIL"}      # combine by max across radars


def _dbz_as(dbz: np.ndarray, quantity: str) -> np.ndarray:
    """A dBZ field expressed as `quantity` (dBZ as-is, or Marshall-Palmer rate).

    Applies the same <5 dBZ clutter floor as render.prepare_sweep.
    """
    with np.errstate(invalid="ignore"):
        dbz = np.where(dbz >= 5.0, dbz, np.float32(np.nan)).astype(np.float32)
        if quantity != "RATE":
            return dbz
        z = np.power(10.0, np.minimum(dbz, process.HAIL_CAP_DBZ) / 10.0)
        return np.power(z / process.ZR_A, 1.0 / process.ZR_B).astype(np.float32)


def build_composite(
    paths: list[Path],
    quantity: str,
    grid: geo.Grid,
    sweep_index: int = 0,
    dealias: bool = True,
    cmax: bool = False,
    fill_dbz: np.ndarray | None = None,
) -> np.ma.MaskedArray:
    """Combine several radars into one gridded field.

    - Column products (ET/VIL) and cmax reflectivity use all tilts per radar.
    - MAX_MOMENTS combine by max across radars; others by nearest-radar mosaic.
    - `fill_dbz` (float32 dBZ on `grid`, NaN = none; e.g. the OPERA composite)
      gap-fills pixels no radar covers, for reflectivity-family products only.
      Never modified in place (it is cached by the caller).
    """
    is_column = quantity in COLUMN_PRODUCTS
    use_cmax = cmax and quantity in CMAX_MOMENTS
    shape = (grid.height, grid.width)

    def _sample_one(p):
        if is_column:
            return render.sample_column_to_grid(p, quantity, grid, dealias)
        if use_cmax:
            return render.sample_cmax_to_grid(p, quantity, grid, dealias)
        return render.sample_to_grid(p, quantity, grid, sweep_index, dealias)

    # Max mosaic (reflectivity / column): stream a NaN-aware float32 running max
    # so we never hold all radars' grids at once.
    if quantity in MAX_MOMENTS:
        best = None
        for p in paths:
            try:
                v, _ = _sample_one(p)
            except (KeyError, IndexError):
                continue
            arr = np.ma.filled(v, np.nan).astype(np.float32, copy=False)
            best = arr if best is None else np.fmax(best, arr, out=best)
        if fill_dbz is not None and quantity in CMAX_MOMENTS:
            # Gap-fill only: OPERA runs 5-10 dBZ hotter than our processed
            # radars where both see echo, so it must not override them.
            extra = _dbz_as(fill_dbz, quantity)
            if best is None:
                best = extra
            else:
                gap = np.isnan(best)
                best[gap] = extra[gap]
        if best is None:
            return np.ma.array(np.zeros(shape, np.float32), mask=np.ones(shape, bool))
        return np.ma.array(np.nan_to_num(best), mask=np.isnan(best))

    # Nearest-radar mosaic for velocity (and any non-max moment): needs all.
    samples = []
    for p in paths:
        try:
            samples.append(_sample_one(p))
        except (KeyError, IndexError):
            continue
    if not samples:
        return np.ma.array(np.zeros(shape), mask=np.ones(shape, bool))
    if _native.HAVE_NATIVE and "nearest_mosaic" in _native.IMPLEMENTED:
        vals = np.ascontiguousarray(np.stack(
            [np.ma.filled(v, np.nan).astype(np.float32).ravel()
             for v, _ in samples]))
        ds = np.ascontiguousarray(np.stack(
            [np.asarray(d, np.float32).ravel() for _, d in samples]))
        out = np.empty(vals.shape[1], np.float32)
        if _native.nearest_mosaic(vals, ds, out):
            out = out.reshape(shape).astype(np.float64)
            bad = np.isnan(out)
            return np.ma.array(np.nan_to_num(out), mask=bad)
    values = [v for v, _ in samples]
    dists = np.stack([d for _, d in samples])          # [n x H x W]
    masks = np.stack([np.ma.getmaskarray(v) for v in values])
    data = np.stack([np.ma.getdata(v) for v in values])
    dists = np.where(masks, np.inf, dists)             # ignore masked pixels
    pick = np.argmin(dists, axis=0)                    # nearest valid radar
    hh, ww = np.indices(pick.shape)
    out = data[pick, hh, ww]
    out_mask = np.min(dists, axis=0) == np.inf         # no valid radar here
    return np.ma.array(out, mask=out_mask)


def composite_value(
    paths_nodes: list[tuple[Path, str]],
    quantity: str,
    lat: float,
    lon: float,
    sweep_index: int = 0,
    dealias: bool = True,
    cmax: bool = False,
    opera_path: Path | None = None,
) -> dict:
    """Point value for a composite (max for reflectivity, nearest for velocity).

    With `opera_path`, the OPERA composite answers for CMAX reflectivity
    products where no radar has a value, matching build_composite's gap-fill.
    """
    is_column = quantity in COLUMN_PRODUCTS
    use_cmax = cmax and quantity in CMAX_MOMENTS
    results = []
    for path, node in paths_nodes:
        try:
            if is_column:
                s = render.sample_point_column(path, quantity, lat, lon, dealias)
            elif use_cmax:
                s = render.sample_point_cmax(path, quantity, lat, lon, dealias)
            else:
                s = render.sample_point(path, quantity, lat, lon, sweep_index, dealias)
        except (KeyError, IndexError):
            continue
        s["node"] = node
        results.append(s)
    if (opera_path is not None and use_cmax
            and not any(r.get("value") is not None for r in results)):
        dbz = opera.point_dbz(opera_path, lat, lon)
        if dbz is not None and dbz >= 5.0:
            val = float(_dbz_as(np.array([dbz], np.float32), quantity)[0])
            results.append({"value": round(val, 1), "in_range": True,
                            "range_km": None, "node": "opera",
                            "units": colormaps.UNITS.get(quantity, "")})
    if not results:
        return {"value": None, "in_range": False, "source": None}
    valid = [r for r in results if r.get("value") is not None]
    if not valid:
        # Still report geometry from the nearest in-range radar if any.
        in_range = [r for r in results if r.get("in_range")]
        base = min(in_range, key=lambda r: r["range_km"]) if in_range else results[0]
        return {**base, "value": None, "source": None}
    if quantity in MAX_MOMENTS:
        best = max(valid, key=lambda r: r["value"])
    else:
        best = min(valid, key=lambda r: r["range_km"])  # nearest radar
    return {**best, "source": best["node"]}
