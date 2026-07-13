"""Benchmark every product's Python (NumPy) path vs its native (Rust) kernel.

Run:  .venv/bin/python scripts/bench_native.py
Needs a few real ODIM files in data/ (run scripts/test_fetch.py first).

For each product it prepares identical inputs, runs both paths, verifies the
outputs match, and prints Python ms / Rust ms / speedup. Products whose Rust
kernel is still a stub print "TODO (NumPy fallback)". As kernels get written the
same harness starts reporting their numbers — no changes needed here.
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from radar import _native, analysis, composite, config, geo, odim, process, render

NODES = ["silis", "sipas", "hrpun", "hrbil"]
PATHS = {n: sorted(glob.glob(f"data/{n}*.h5"))[-1:] for n in NODES}
if not all(PATHS[n] for n in NODES):
    print("Missing local data — run scripts/test_fetch.py first.")
    sys.exit(1)
PATHS = {n: PATHS[n][0] for n in NODES}


def timed(fn, n=7):
    best = 1e18
    for _ in range(n):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1000.0


def match(a, b):
    a, b = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    if a.shape != b.shape:
        return False
    ma, mb = np.isnan(a), np.isnan(b)
    return bool(np.array_equal(ma, mb) and np.allclose(a[~ma], b[~mb], atol=1e-2))


def report(name, py_ms, rust):
    if rust is None:
        print(f"  {name:22s} py {py_ms:7.1f} ms   rust: TODO (NumPy fallback)")
        return
    rust_ms, ok = rust
    sp = py_ms / rust_ms if rust_ms else 0
    tick = "match" if ok else "!!MISMATCH!!"
    print(f"  {name:22s} py {py_ms:7.1f} ms   rust {rust_ms:7.1f} ms   {sp:4.1f}x   {tick}")


# --- shared prep: per-radar grid geometry + per-sweep polar/edges -----------
def radar_ctx(path, quantity, grid):
    vol = odim.summarize(path)
    geom0 = odim.sweep_geometry(path, vol.sweeps[0].index)
    dist_f, ray_idx = render._radar_grid_geom(vol, grid, geom0)
    dist_flat = np.ascontiguousarray(dist_f.ravel(), np.float32)
    ray_flat = np.ascontiguousarray(ray_idx.ravel(), np.int32)
    sweeps = []
    for s in vol.sweeps:
        data = render.prepare_sweep(path, quantity, s.index)
        polar = np.ascontiguousarray(np.ma.filled(data, np.nan), np.float32)
        geom = odim.sweep_geometry(path, s.index)
        edges = np.ascontiguousarray(render._bin_edges(geom), np.float64)
        r = geom.slant_ranges
        h = (np.sqrt(r * r + geo.A_E ** 2 + 2 * r * geo.A_E * np.sin(geom.elangle_rad))
             - geo.A_E + vol.site_height_m).astype(np.float32)
        sweeps.append((polar, edges, np.ascontiguousarray(h)))
    return dist_flat, ray_flat, sweeps


print("Preparing inputs…")
GW = config.grid_for("DBZH")           # wide grid (reflectivity)
GB = config.grid_for("VRADH")          # base grid (everything else)
NPIXW = GW.width * GW.height
refl_ctx = [radar_ctx(PATHS[n], "DBZH", GW) for n in NODES]
print(f"wide grid {GW.width}x{GW.height} = {NPIXW/1e6:.2f}M px, {len(NODES)} radars\n")

results = []

# --- 1. reflectivity CMAX composite (implemented) --------------------------
def py_cmax():
    best = None
    for dist_f, ray, sweeps in refl_ctx:
        nb = sweeps[0][0].shape[1]
        for polar, edges, _ in sweeps:
            bi = np.searchsorted(edges, dist_f).astype(np.int32) - 1
            inr = (bi >= 0) & (bi < polar.shape[1])
            np.clip(bi, 0, polar.shape[1] - 1, out=bi)
            v = polar.ravel().take(ray * np.int32(polar.shape[1]) + bi)
            v[~inr] = np.nan
            best = v if best is None else np.fmax(best, v, out=best)
    return best

def rs_cmax():
    best = np.full(NPIXW, np.nan, np.float32)
    ok = True
    for dist_f, ray, sweeps in refl_ctx:
        for polar, edges, _ in sweeps:
            ok = _native.cmax_sweep(ray, dist_f, polar, edges, best) and ok
    return best, ok

o = py_cmax(); r = rs_cmax()
report("reflectivity_cmax", timed(py_cmax), (timed(lambda: rs_cmax()), match(o, r[0])) if r[1] else None)

# --- 2. single-sweep sample (velocity base grid, implemented) --------------
vctx = radar_ctx(PATHS["silis"], "VRADH", GB)
vdist, vray, vsweeps = vctx
vpolar, vedges, _ = vsweeps[0]
NPIXB = GB.width * GB.height

def py_sample():
    bi = np.searchsorted(vedges, vdist).astype(np.int32) - 1
    inr = (bi >= 0) & (bi < vpolar.shape[1])
    np.clip(bi, 0, vpolar.shape[1] - 1, out=bi)
    out = vpolar.ravel().take(vray * np.int32(vpolar.shape[1]) + bi)
    out[~inr] = np.nan
    return out

def rs_sample():
    out = np.empty(NPIXB, np.float32)
    ok = _native.sample_sweep(vray, vdist, vpolar, vedges, out)
    return out, ok

o = py_sample(); r = rs_sample()
report("sample_sweep", timed(py_sample), (timed(lambda: rs_sample()), match(o, r[0])) if r[1] else None)

# --- 3. echo tops (column, stub) -------------------------------------------
def py_et():
    return render.sample_column_to_grid(PATHS["silis"], "ET", GB)[0]

def rs_et():
    et = np.full(NPIXB, np.nan, np.float32)
    ok = True
    for polar, edges, heights in radar_ctx(PATHS["silis"], "DBZH", GB)[2]:
        ok = _native.echo_top_sweep(vray, vdist, polar, edges, heights, 18.0, et) and ok
    return et, ok

report("echo_top", timed(py_et), (timed(lambda: rs_et()), None) if rs_et()[1] else None)

# --- 4. VIL (full product, NumPy vs native kernel) ---------------------------
_IMPL = _native.IMPLEMENTED

def py_vil():
    _native.IMPLEMENTED = _IMPL - {"vil_column"}      # force the NumPy path
    try:
        return render.sample_column_to_grid(PATHS["silis"], "VIL", GB)[0]
    finally:
        _native.IMPLEMENTED = _IMPL

def rs_vil():
    return render.sample_column_to_grid(PATHS["silis"], "VIL", GB)[0]

if "vil_column" in _IMPL and _native.HAVE_NATIVE:
    o = py_vil().filled(0.0); r = rs_vil().filled(0.0)
    # The native path reuses sweep-0 azimuths (cached geometry); real antenna
    # jitter between sweeps (~0.2 deg) flips a handful of ray-boundary pixels.
    # Same approximation the production CMAX path uses — allow <0.005% of px.
    ok = np.mean(np.abs(o - r) > 1e-2) < 5e-5
    report("vil", timed(py_vil, 5), (timed(rs_vil, 5), ok))
else:
    report("vil", timed(py_vil, 5), None)

# --- 5. nearest-radar velocity mosaic (stub) -------------------------------
def py_nearest():
    return composite.build_composite([PATHS[n] for n in NODES], "VRADH", GB, 0, True)

report("nearest_mosaic", timed(py_nearest),
       None if "nearest_mosaic" not in _native.IMPLEMENTED else ("todo", None))

# --- 6. velocity dealiasing (stub) -----------------------------------------
vel_sweep = render.odim.read_sweep(PATHS["silis"], "VRADH", 0)
nyq = odim.sweep_geometry(PATHS["silis"], 0).nyquist or 8.0

def py_dealias():
    return process.dealias(vel_sweep, nyq)

report("dealias", timed(py_dealias),
       None if "dealias_rays" not in _native.IMPLEMENTED else ("todo", None))

# --- 7. dual-Doppler wind (stub) -------------------------------------------
def py_dd():
    v1, _ = render.sample_to_grid(PATHS["silis"], "VRADH", GB, 0)
    v2, _ = render.sample_to_grid(PATHS["sipas"], "VRADH", GB, 0)
    s1, s2 = config.NODES["silis"], config.NODES["sipas"]
    return analysis.dual_doppler(v1, v2, (s1.lat, s1.lon), (s2.lat, s2.lon), GB)

report("dual_doppler", timed(py_dd, 3),
       None if "dual_doppler" not in _native.IMPLEMENTED else ("todo", None))

# --- 8. storm motion (stub) ------------------------------------------------
prev, _ = render.sample_cmax_to_grid(PATHS["silis"], "DBZH", GB)
curr, _ = render.sample_cmax_to_grid(PATHS["sipas"], "DBZH", GB)

def py_motion():
    return analysis.motion_field(prev, curr, GB, 300.0)

report("motion_ssd", timed(py_motion, 3),
       None if "motion_ssd" not in _native.IMPLEMENTED else ("todo", None))

print(f"\nnative lib loaded: {_native.HAVE_NATIVE}   implemented: "
      f"{sorted(_native.IMPLEMENTED)}")
