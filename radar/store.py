"""Rolling frame store + background fetch worker.

Keeps the last ~90 minutes of ODIM volumes per active node on disk, refreshing
every minute to track the ~5-minute radar cadence. Rendered PNGs (single-radar
and composite) are cached in memory. Composites match each radar's frame nearest
to a reference time, since Slovenian and Croatian radars scan on different
schedules.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import analysis, composite, config, fetcher, opera, render

WINDOW = timedelta(minutes=90)
POLL_SECONDS = 60
RENDER_CACHE_MAX = 512
NEAREST_TOL = timedelta(minutes=6)   # max gap when matching radars in a composite
OPERA_WAIT = timedelta(minutes=15)   # hold archiving a frame this long for OPERA
GRID = config.BASE_GRID          # default (velocity / heavy / analysis products)
ARCHIVE = timedelta(hours=config.ARCHIVE_HOURS)
# At most this many renders run at once, so a burst of requests (or the eager
# frame preload) can never peg CPU / exhaust memory on a shared server.
_RENDER_SEMAPHORE = threading.Semaphore(2)


@dataclass
class Frame:
    node: str
    time: datetime
    path: Path
    rf: fetcher.RadarFile


class Store:
    def __init__(self) -> None:
        self._frames: dict[str, "OrderedDict[str, Frame]"] = {
            n: OrderedDict() for n in config.ACTIVE_NODES
        }
        self._render_cache: "OrderedDict[tuple, render.RenderResult]" = OrderedDict()
        self._lock = threading.Lock()
        # OPERA European composite frames: key -> (time, path).
        self._opera: "OrderedDict[str, tuple[datetime, Path]]" = OrderedDict()

    # --- frame access -----------------------------------------------------
    def frames(self, node: str) -> list[Frame]:
        with self._lock:
            return list(self._frames.get(node, {}).values())

    def latest(self, node: str) -> Frame | None:
        fr = self.frames(node)
        return fr[-1] if fr else None

    def frame_at(self, node: str, key: str) -> Frame | None:
        with self._lock:
            return self._frames.get(node, {}).get(key)

    def frame_near(self, node: str, target: datetime) -> Frame | None:
        """Frame for a node closest to `target` within NEAREST_TOL."""
        best, best_gap = None, NEAREST_TOL
        for f in self.frames(node):
            gap = abs(f.time - target)
            if gap <= best_gap:
                best, best_gap = f, gap
        return best

    def ref_node(self, nodes: list[str]) -> str:
        """First of `nodes` that currently has frames (radars drop out)."""
        return next((n for n in nodes if self.frames(n)), nodes[0])

    # --- OPERA composite fallback -------------------------------------------
    def opera_near(self, target: datetime) -> tuple[datetime, Path] | None:
        if not config.OPERA_FILL:
            return None
        with self._lock:
            files = dict(self._opera)
        return opera.nearest(files, target, NEAREST_TOL)

    def _opera_fill(self, target: datetime, quantity: str, use_cmax: bool,
                    grid) -> tuple[np.ndarray | None, str | None]:
        """(dBZ field on `grid`, frame key) for CMAX reflectivity products."""
        if not (use_cmax and quantity in composite.CMAX_MOMENTS):
            return None, None
        hit = self.opera_near(target)
        if hit is None:
            return None, None
        return opera.dbz_grid(hit[1], grid), hit[0].strftime("%Y%m%dT%H%M")

    def _fetch_opera(self) -> int:
        now = datetime.now(timezone.utc)
        cands: list[opera.OperaFile] = []
        for off in (0, 1):
            cands += opera.list_files(now - timedelta(days=off))
        with self._lock:
            have = set(self._opera)
        added = 0
        for of in cands:
            key = of.time.strftime("%Y%m%dT%H%M%S")
            if of.time < now - WINDOW or key in have:
                continue
            path = opera.download(of)
            with self._lock:
                self._opera[key] = (of.time, path)
                self._opera = OrderedDict(sorted(self._opera.items()))
            added += 1
        return added

    def _add(self, node: str, rf: fetcher.RadarFile, path: Path) -> None:
        key = rf.time.strftime("%Y%m%dT%H%M%S")
        with self._lock:
            self._frames[node][key] = Frame(node, rf.time, path, rf)
            self._frames[node] = OrderedDict(sorted(self._frames[node].items()))

    def _prune(self) -> None:
        cutoff = datetime.now(timezone.utc) - WINDOW
        with self._lock:
            for frames in self._frames.values():
                for key in [k for k, f in frames.items() if f.time < cutoff]:
                    fr = frames.pop(key)
                    try:
                        fr.path.unlink(missing_ok=True)
                    except OSError:
                        pass
            for key in [k for k, (t, _) in self._opera.items() if t < cutoff]:
                _, p = self._opera.pop(key)
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass

    # --- rendering --------------------------------------------------------
    def _cached(self, key, produce):
        with self._lock:
            if key in self._render_cache:
                self._render_cache.move_to_end(key)
                return self._render_cache[key]
        with _RENDER_SEMAPHORE:
            # Re-check: another thread may have rendered this key while we waited.
            with self._lock:
                if key in self._render_cache:
                    self._render_cache.move_to_end(key)
                    return self._render_cache[key]
            res = produce()
        if res is not None:
            with self._lock:
                self._render_cache[key] = res
                while len(self._render_cache) > RENDER_CACHE_MAX:
                    self._render_cache.popitem(last=False)
        return res

    def render_single(
        self, node: str, key: str, quantity: str, sweep: int,
        dealias: bool, cmax: bool = False,
    ) -> render.RenderResult | None:
        frame = self.frame_at(node, key)
        if frame is None:
            return None
        is_column = quantity in composite.COLUMN_PRODUCTS
        use_cmax = cmax and quantity in composite.CMAX_MOMENTS
        grid = config.grid_for(quantity)      # reflectivity -> wide, else base
        mode = "col" if is_column else ("cmax" if use_cmax else sweep)
        ck = ("single", node, key, quantity, mode, dealias)
        if is_column:
            produce = lambda: render.render_column(frame.path, quantity, grid, dealias)
        elif use_cmax:
            produce = lambda: render.render_cmax(frame.path, quantity, grid, dealias)
        else:
            produce = lambda: render.render_sweep(
                frame.path, quantity, grid, sweep, dealias=dealias
            )
        try:
            return self._cached(ck, produce)
        except (KeyError, IndexError):
            return None   # this radar lacks the moment (e.g. RHOHV on SI radars)

    def render_composite(
        self, nodes: list[str], key: str, quantity: str, sweep: int,
        dealias: bool, cmax: bool = False,
    ) -> render.RenderResult | None:
        """Composite the given nodes at the reference time `key` (from any node)."""
        use_cmax = cmax and quantity in composite.CMAX_MOMENTS
        # Flagship (composite CMAX reflectivity): serve the pre-rendered archive
        # PNG if present -- instant, and works even after the raw window expires.
        if self._is_flagship(nodes, quantity, use_cmax):
            arc = self.archive_result(key)
            if arc is not None:
                return arc
        target = _parse_key(key)
        if target is None:
            return None
        paths = []
        for n in nodes:
            fr = self.frame_near(n, target)
            if fr is not None:
                paths.append(fr.path)
        is_column = quantity in composite.COLUMN_PRODUCTS
        grid = config.grid_for(quantity)      # reflectivity -> wide, else base
        fill, okey = self._opera_fill(target, quantity, use_cmax, grid)
        if not paths and fill is None:
            return None
        mode = "col" if is_column else ("cmax" if use_cmax else sweep)
        # The OPERA key is part of the cache key: it lands ~4 min after the
        # radars, and a render made before it arrived must not stick.
        ck = ("comp", tuple(sorted(nodes)), key, quantity, mode, dealias, okey)
        try:
            return self._cached(
                ck,
                lambda: render.render_grid(
                    composite.build_composite(
                        paths, quantity, grid, sweep, dealias, cmax=use_cmax,
                        fill_dbz=fill,
                    ),
                    quantity, grid,
                ),
            )
        except (KeyError, IndexError):
            return None

    def composite_paths(self, nodes: list[str], target: datetime):
        """(path, node) pairs for a composite at a reference time."""
        out = []
        for n in nodes:
            fr = self.frame_near(n, target)
            if fr is not None:
                out.append((fr.path, n))
        return out

    # --- flagship archive (24h of composite CMAX reflectivity) ------------
    @staticmethod
    def _is_flagship(nodes, quantity, use_cmax) -> bool:
        return (use_cmax and quantity == config.ARCHIVE_QUANTITY
                and sorted(nodes) == sorted(config.ARCHIVE_NODES))

    @staticmethod
    def _archive_path(key: str) -> Path:
        return config.ARCHIVE_DIR / f"{key}.png"

    def archive_result(self, key: str) -> render.RenderResult | None:
        """A RenderResult backed by the on-disk archive PNG, if it exists."""
        p = self._archive_path(key)
        if not p.exists():
            return None
        ag = config.grid_for(config.ARCHIVE_QUANTITY)     # flagship is on WIDE
        return render.RenderResult(
            png=p.read_bytes(), bounds=ag.leaflet_bounds,
            width=ag.width, height=ag.height,
        )

    def archive_keys(self) -> list[str]:
        """Frame keys with an archived flagship PNG, within the 24h window."""
        cutoff = datetime.now(timezone.utc) - ARCHIVE
        keys = []
        for p in config.ARCHIVE_DIR.glob("*.png"):
            t = _parse_key(p.stem)
            if t is not None and t >= cutoff:
                keys.append(p.stem)
        return sorted(keys)

    def _update_archive(self) -> int:
        """Render + persist the flagship for any frame not yet archived."""
        seen: set[str] = set()
        for n in config.ACTIVE_NODES:
            for f in self.frames(n):
                seen.add(f.time.strftime("%Y%m%dT%H%M%S"))
        made = 0
        wait_until = datetime.now(timezone.utc) - OPERA_WAIT
        for key in sorted(seen):
            if self._archive_path(key).exists():
                continue
            t = _parse_key(key)
            # Archived PNGs are permanent: give the OPERA composite (which lands
            # a few minutes after the radars) a chance before freezing a frame.
            if (config.OPERA_FILL and t is not None and t > wait_until
                    and self.opera_near(t) is None):
                continue
            res = self.render_composite(
                list(config.ARCHIVE_NODES), key, config.ARCHIVE_QUANTITY,
                0, True, cmax=True,
            )
            if res is not None:
                try:
                    self._archive_path(key).write_bytes(res.png)
                    made += 1
                except OSError:
                    pass
        self._prune_archive()
        return made

    def _prune_archive(self) -> None:
        cutoff = datetime.now(timezone.utc) - ARCHIVE
        for p in config.ARCHIVE_DIR.glob("*.png"):
            t = _parse_key(p.stem)
            if t is None or t < cutoff:
                try:
                    p.unlink()
                except OSError:
                    pass

    # --- analysis products (temporal / multi-radar) -----------------------
    def _grid_for(self, nodes, key, quantity, sweep=0, cmax=False):
        """Composite masked grid for a product at a reference time (no PNG)."""
        target = _parse_key(key)
        if target is None:
            return None
        paths = [fr.path for n in nodes
                 if (fr := self.frame_near(n, target)) is not None]
        fill, _ = self._opera_fill(target, quantity, cmax, GRID)
        if not paths and fill is None:
            return None
        try:
            return composite.build_composite(paths, quantity, GRID, sweep,
                                             True, cmax=cmax, fill_dbz=fill)
        except (KeyError, IndexError):
            return None

    def accum_grid(self, nodes, key, window_min=60):
        target = _parse_key(key)
        if target is None:
            return None
        lo = target - timedelta(minutes=window_min)
        times = [f.time for f in self.frames(self.ref_node(nodes))
                 if lo <= f.time <= target]
        grids = []
        for t in times:
            g = self._grid_for(nodes, t.strftime("%Y%m%dT%H%M%S"), "RATE")
            if g is not None:
                grids.append(g)
        return analysis.accumulate(grids, 5.0) if grids else None

    def motion_vectors(self, nodes, key, lag_min=15):
        target = _parse_key(key)
        if target is None:
            return []
        ref = self.frame_near(self.ref_node(nodes),
                              target - timedelta(minutes=lag_min))
        if ref is None or abs(ref.time - target) < timedelta(minutes=4):
            return []
        curr = self._grid_for(nodes, key, "DBZH", cmax=True)
        prev = self._grid_for(nodes, ref.time.strftime("%Y%m%dT%H%M%S"),
                              "DBZH", cmax=True)
        if curr is None or prev is None:
            return []
        dt = (target - ref.time).total_seconds()
        ck = ("motion", tuple(sorted(nodes)), key, lag_min)
        return self._cached(ck, lambda: analysis.motion_field(prev, curr, GRID, dt))

    def nowcast_grid(self, nodes, key, lead_min=30):
        curr = self._grid_for(nodes, key, "DBZH", cmax=True)
        if curr is None:
            return None
        u, v = analysis.mean_motion(self.motion_vectors(nodes, key))
        return analysis.advect(curr, u, v, lead_min * 60, GRID)

    def wind_grids(self, nodes, key):
        """(speed, u, v) dual-Doppler grids from the first two nodes."""
        if len(nodes) < 2:
            return None
        target = _parse_key(key)
        f1 = self.frame_near(nodes[0], target) if target else None
        f2 = self.frame_near(nodes[1], target) if target else None
        if f1 is None or f2 is None:
            return None
        v1, _ = render.sample_to_grid(f1.path, "VRADH", GRID, 0, True)
        v2, _ = render.sample_to_grid(f2.path, "VRADH", GRID, 0, True)
        s1, s2 = config.NODES[nodes[0]], config.NODES[nodes[1]]
        return analysis.dual_doppler(v1, v2, (s1.lat, s1.lon), (s2.lat, s2.lon), GRID)

    def wind_vectors(self, nodes, key, step=45):
        """Subsampled dual-Doppler wind arrows for overlay."""
        wg = self.wind_grids(nodes, key)
        if wg is None:
            return []
        speed, u, v = wg
        lon, lat = GRID.lonlat()
        H, W = speed.shape
        out = []
        for y in range(step // 2, H, step):
            for x in range(step // 2, W, step):
                if np.ma.is_masked(speed[y, x]):
                    continue
                out.append({
                    "lat": round(float(lat[y, x]), 4),
                    "lon": round(float(lon[y, x]), 4),
                    "u": round(float(u[y, x]), 2), "v": round(float(v[y, x]), 2),
                    "speed": round(float(speed[y, x]), 1),
                })
        return out

    def render_product(self, product, nodes, key, **params):
        """Render an analysis product (ACCUM/NOWCAST/WIND) to a PNG."""
        ck = ("prod", product, tuple(sorted(nodes)), key, tuple(sorted(params.items())))

        def produce():
            if product == "ACCUM":
                g = self.accum_grid(nodes, key, params.get("window", 60))
            elif product == "NOWCAST":
                g = self.nowcast_grid(nodes, key, params.get("lead", 30))
            elif product == "WIND":
                wg = self.wind_grids(nodes, key)
                g = wg[0] if wg else None
            else:
                g = None
            if g is None or g.count() == 0:      # nothing valid -> 404 -> "No data"
                return None
            return render.render_grid(g, product, GRID)

        return self._cached(ck, produce)

    def product_value(self, product, nodes, key, lat, lon, **params):
        """Point value of an analysis product for the hover tool."""
        if product == "ACCUM":
            g = self.accum_grid(nodes, key, params.get("window", 60))
        elif product == "NOWCAST":
            g = self.nowcast_grid(nodes, key, params.get("lead", 30))
        elif product == "WIND":
            wg = self.wind_grids(nodes, key)
            g = wg[0] if wg else None
        else:
            g = None
        idx = GRID.index(lat, lon)
        if g is None or idx is None:
            return {"value": None, "in_range": idx is not None}
        val = g[idx[0], idx[1]]
        from . import colormaps
        return {
            "value": None if np.ma.is_masked(val) else round(float(val), 1),
            "units": colormaps.UNITS.get(product, ""),
            "in_range": True,
        }

    # --- worker -----------------------------------------------------------
    def _fetch_node(self, node: str) -> int:
        now = datetime.now(timezone.utc)
        added = 0
        candidates: list[fetcher.RadarFile] = []
        for off in (0, 1):
            day = now - timedelta(days=off)
            candidates += fetcher.list_node_files(node, day)
        with self._lock:
            have = set(self._frames[node].keys())
        for rf in candidates:
            if rf.time < now - WINDOW:
                continue
            key = rf.time.strftime("%Y%m%dT%H%M%S")
            if key in have:
                continue
            path = fetcher.download(rf)
            self._add(node, rf, path)
            added += 1
        return added

    def refresh(self) -> dict[str, int]:
        result = {}
        for node in config.ACTIVE_NODES:
            try:
                result[node] = self._fetch_node(node)
            except Exception as exc:
                result[node] = -1
                print(f"[store] fetch error for {node}: {exc}")
        if config.OPERA_FILL:
            try:
                result["opera"] = self._fetch_opera()
            except Exception as exc:
                result["opera"] = -1
                print(f"[store] fetch error for opera: {exc}")
        self._prune()
        try:
            self._archived = self._update_archive()   # flagship 24h archive
        except Exception as exc:
            self._archived = -1
            print(f"[store] archive error: {exc}")
        return result


def _parse_key(key: str) -> datetime | None:
    try:
        return datetime.strptime(key, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


store = Store()


async def worker_loop(stop: asyncio.Event) -> None:
    while not stop.is_set():
        t0 = time.perf_counter()
        added = await asyncio.to_thread(store.refresh)
        dt = time.perf_counter() - t0
        total = {n: len(store.frames(n)) for n in config.ACTIVE_NODES}
        total["opera"] = len(store._opera)
        arch = len(store.archive_keys())
        print(f"[store] refresh +{added} in {dt:.1f}s; holding {total}; "
              f"archive={arch} frames", flush=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=POLL_SECONDS)
        except asyncio.TimeoutError:
            pass
