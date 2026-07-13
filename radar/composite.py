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

from . import geo, render

COLUMN_PRODUCTS = {"ET", "VIL"}                        # vertical, all-tilt
CMAX_MOMENTS = {"DBZH", "TH", "RATE"}                  # support the CMAX toggle
MAX_MOMENTS = {"DBZH", "TH", "RATE", "ET", "VIL"}      # combine by max across radars


def build_composite(
    paths: list[Path],
    quantity: str,
    grid: geo.Grid,
    sweep_index: int = 0,
    dealias: bool = True,
    cmax: bool = False,
) -> np.ma.MaskedArray:
    """Combine several radars into one gridded field.

    - Column products (ET/VIL) and cmax reflectivity use all tilts per radar.
    - MAX_MOMENTS combine by max across radars; others by nearest-radar mosaic.
    """
    is_column = quantity in COLUMN_PRODUCTS
    use_cmax = cmax and quantity in CMAX_MOMENTS
    samples = []
    for p in paths:
        try:
            if is_column:
                samples.append(render.sample_column_to_grid(p, quantity, grid, dealias))
            elif use_cmax:
                samples.append(render.sample_cmax_to_grid(p, quantity, grid, dealias))
            else:
                samples.append(
                    render.sample_to_grid(p, quantity, grid, sweep_index, dealias)
                )
        except (KeyError, IndexError):
            continue  # this radar lacks the moment/sweep; skip it
    if not samples:
        shape = (grid.height, grid.width)
        return np.ma.array(np.zeros(shape), mask=np.ones(shape, bool))

    values = [v for v, _ in samples]
    if quantity in MAX_MOMENTS:
        stack = np.ma.stack(values)          # [n_radar x H x W]
        return stack.max(axis=0)             # masked where all radars masked

    # Nearest-radar mosaic for velocity (and any non-max moment).
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
) -> dict:
    """Point value for a composite (max for reflectivity, nearest for velocity)."""
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
