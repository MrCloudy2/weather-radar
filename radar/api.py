"""FastAPI backend for the Slovenia radar dashboard.

Subpath-aware (root_path=/radar) so it drops behind Nginx cleanly. Serves radar
sources, a master timeline, single-radar + composite Web Mercator tiles, and
point values for the hover inspector. Static frontend (Phase 3) mounts at /.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response

from . import colormaps, composite, config, render as render_mod, store as store_mod

ROOT_PATH = os.environ.get("RADAR_ROOT_PATH", "")
# Products available on every radar (derived or in every file).
BASE_MOMENTS = ["DBZH", "RATE", "ET", "VIL", "VRADH", "AZSHEAR", "TH"]
# Polarimetric products only the Croatian radars carry.
POLARIMETRIC = ["RHOHV", "ZDR"]
MOMENTS = BASE_MOMENTS + POLARIMETRIC


def _moments_for(node_id: str) -> list[str]:
    country = config.NODES[node_id].country
    return BASE_MOMENTS + POLARIMETRIC if country != "SI" else BASE_MOMENTS


# Analysis products (temporal / multi-radar), not stored moments.
SCALAR_ANALYSIS = {"ACCUM", "NOWCAST", "WIND"}   # rendered as PNG fields
_stop = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(store_mod.worker_loop(_stop))
    try:
        yield
    finally:
        _stop.set()
        await task


app = FastAPI(title="Slovenia Radar", root_path=ROOT_PATH, lifespan=lifespan)
store = store_mod.store


def _nodes_param(node: str | None, nodes: str | None) -> list[str]:
    """Resolve a single `node` or csv `nodes` into a validated list."""
    if nodes:
        ids = [n for n in nodes.split(",") if n]
    elif node:
        ids = [node]
    else:
        raise HTTPException(400, "provide ?node= or ?nodes=")
    for n in ids:
        if n not in config.NODES:
            raise HTTPException(404, f"unknown node {n}")
    return ids


@app.get("/api/sources")
def sources():
    out = []
    for n in config.NODES.values():
        latest = store.latest(n.id) if n.id in config.ACTIVE_NODES else None
        out.append({
            "id": n.id,
            "name": n.name,
            "country": n.country,
            "lat": n.lat,
            "lon": n.lon,
            "active": n.id in config.ACTIVE_NODES,
            "in_default_composite": n.composite,
            "latest": latest.time.isoformat() if latest else None,
            "frames": len(store.frames(n.id)) if n.id in config.ACTIVE_NODES else 0,
            "moments": _moments_for(n.id),
        })
    return {
        "sources": out,
        "moments": MOMENTS,
        "default_composite": [n.id for n in config.NODES.values() if n.composite],
        "grid": {"bounds": config.SLOVENIA_GRID.leaflet_bounds},
    }


@app.get("/api/timeline")
def timeline(node: str = Query("silis")):
    """Master timeline (frame keys) driven by a reference node (Slovenian)."""
    if node not in config.ACTIVE_NODES:
        raise HTTPException(404, "unknown/inactive node")
    frs = store.frames(node)
    return {
        "node": node,
        "frames": [
            {"key": f.time.strftime("%Y%m%dT%H%M%S"), "iso": f.time.isoformat()}
            for f in frs
        ],
        "sweeps": [
            {"index": i, "elevation": e}
            for i, e in _sweep_list(frs[-1].path) if frs
        ] if frs else [],
    }


@app.get("/api/archive")
def archive():
    """Full-day timeline of the persisted flagship (composite CMAX reflectivity).

    Extends beyond the 90-min raw window using the on-disk 24h archive.
    """
    from datetime import datetime, timezone
    keys = store.archive_keys()
    return {
        "quantity": config.ARCHIVE_QUANTITY,
        "nodes": config.ARCHIVE_NODES,
        "hours": config.ARCHIVE_HOURS,
        "frames": [
            {"key": k, "iso": datetime.strptime(k, "%Y%m%dT%H%M%S")
                .replace(tzinfo=timezone.utc).isoformat()}
            for k in keys
        ],
    }


def _sweep_list(path):
    from . import odim
    vol = odim.summarize(path)
    return [(s.index, s.elangle) for s in vol.sweeps]


@app.get("/api/legend")
def legend(quantity: str = Query(...)):
    return colormaps.legend(quantity)


@app.get("/api/tile.png")
def tile(
    quantity: str = Query(...),
    time: str = Query(..., description="reference frame key YYYYMMDDThhmmss"),
    node: str | None = Query(None),
    nodes: str | None = Query(None, description="csv -> composite"),
    sweep: int = Query(0, ge=0),
    cmax: bool = Query(False, description="column-max over all tilts (reflectivity)"),
    dealias: bool = Query(True),
    window: int = Query(60, description="accumulation window (min)"),
    lead: int = Query(30, description="nowcast lead time (min)"),
):
    ids = _nodes_param(node, nodes)
    if quantity in SCALAR_ANALYSIS:
        params = {"window": window} if quantity == "ACCUM" else \
                 {"lead": lead} if quantity == "NOWCAST" else {}
        res = store.render_product(quantity, ids, time, **params)
    elif len(ids) == 1:
        res = store.render_single(ids[0], time, quantity, sweep, dealias, cmax)
    else:
        res = store.render_composite(ids, time, quantity, sweep, dealias, cmax)
    if res is None:
        raise HTTPException(404, "frame not available")
    s, w = res.bounds[0]
    n, e = res.bounds[1]
    return Response(
        content=res.png,
        media_type="image/png",
        headers={
            "X-Radar-Bounds": f"{s},{w},{n},{e}",
            "Cache-Control": "public, max-age=300",
        },
    )


@app.get("/api/value")
def value(
    quantity: str = Query(...),
    time: str = Query(...),
    lat: float = Query(...),
    lon: float = Query(...),
    node: str | None = Query(None),
    nodes: str | None = Query(None),
    sweep: int = Query(0, ge=0),
    cmax: bool = Query(False),
    dealias: bool = Query(True),
    window: int = Query(60),
    lead: int = Query(30),
):
    ids = _nodes_param(node, nodes)
    if quantity in SCALAR_ANALYSIS:
        params = {"window": window} if quantity == "ACCUM" else \
                 {"lead": lead} if quantity == "NOWCAST" else {}
        return store.product_value(quantity, ids, time, lat, lon, **params)
    is_column = quantity in composite.COLUMN_PRODUCTS
    use_cmax = cmax and quantity in composite.CMAX_MOMENTS
    if len(ids) == 1:
        frame = store.frame_at(ids[0], time)
        if frame is None:
            raise HTTPException(404, "frame not available")
        try:
            if is_column:
                out = render_mod.sample_point_column(frame.path, quantity, lat, lon, dealias)
            elif use_cmax:
                out = render_mod.sample_point_cmax(frame.path, quantity, lat, lon, dealias)
            else:
                out = render_mod.sample_point(
                    frame.path, quantity, lat, lon, sweep, dealias=dealias
                )
        except (KeyError, IndexError):
            raise HTTPException(404, f"{quantity} not available for {ids[0]}")
        out["units"] = out.get("units", "")
        out["source"] = ids[0]
        return out
    target = store_mod._parse_key(time)
    pairs = store.composite_paths(ids, target) if target else []
    if not pairs:
        raise HTTPException(404, "frame not available")
    return composite.composite_value(
        pairs, quantity, lat, lon, sweep, dealias, cmax=use_cmax
    )


@app.get("/api/motion")
def motion(
    time: str = Query(...),
    node: str | None = Query(None),
    nodes: str | None = Query(None),
    lag: int = Query(15, description="lookback for motion (min)"),
):
    ids = _nodes_param(node, nodes)
    return {"vectors": store.motion_vectors(ids, time, lag)}


@app.get("/api/windvectors")
def windvectors(
    time: str = Query(...),
    nodes: str = Query(..., description="csv, needs >=2 radars"),
    step: int = Query(45),
):
    ids = _nodes_param(None, nodes)
    return {"vectors": store.wind_vectors(ids, time, step)}


# Serve the Phase 3 frontend if present.
_static = Path(__file__).resolve().parent / "static"
if _static.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
