"""EUMETNET OPERA European composite (CIRRUS MAX reflectivity) as a fallback.

Since ~Oct 2026 ARSO no longer publishes the Slovenian radars' raw volumes to
Open Radar Data, but both (silis, sipas) still feed the pan-European OPERA
composite in the same bucket:

    YYYY/MM/DD/OPERA/COMP/OPERA@<UTC>@0@DBZH.h5

It is a 1 km column-maximum reflectivity mosaic on a Lambert azimuthal
equal-area grid (3800x4400, every 5 min, ~4 min latency). We reproject it onto
our Web Mercator grids with a cached nearest-pixel index map and read only the
hyperslab that covers the grid (a few gzip chunks, not the whole continent).

Reflectivity only -- the composite has no velocity / per-tilt data.
"""

from __future__ import annotations

import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import h5py
import numpy as np
import requests

from . import config, geo

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_NAME_RE = re.compile(r"^OPERA@(?P<ts>\d{8}T\d{4})@0@DBZH\.h5$")


@dataclass(frozen=True)
class OperaFile:
    key: str
    time: datetime
    size: int

    @property
    def url(self) -> str:
        return f"{config.S3_BASE}/{self.key}"

    @property
    def filename(self) -> str:
        return self.key.rsplit("/", 1)[-1]


def list_files(day: datetime) -> list[OperaFile]:
    """DBZH composite files for one UTC day, time-sorted."""
    out: list[OperaFile] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": f"{day:%Y/%m/%d}/OPERA/COMP/",
                  "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        resp = requests.get(f"{config.S3_BASE}/", params=params,
                            timeout=config.HTTP_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for c in root.findall(f"{_S3_NS}Contents"):
            key = c.findtext(f"{_S3_NS}Key", default="")
            m = _NAME_RE.match(key.rsplit("/", 1)[-1])
            if m:
                t = datetime.strptime(m.group("ts"), "%Y%m%dT%H%M").replace(
                    tzinfo=timezone.utc)
                out.append(OperaFile(key, t,
                                     int(c.findtext(f"{_S3_NS}Size", default="0"))))
        if root.findtext(f"{_S3_NS}IsTruncated") == "true":
            token = root.findtext(f"{_S3_NS}NextContinuationToken")
        else:
            break
    out.sort(key=lambda f: f.time)
    return out


def download(of: OperaFile, dest_dir: Path | None = None) -> Path:
    dest_dir = (dest_dir or config.DATA_DIR) / "opera"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / of.filename
    if dest.exists() and dest.stat().st_size == of.size:
        return dest
    tmp = dest.with_suffix(".part")
    with requests.get(of.url, stream=True, timeout=config.HTTP_TIMEOUT) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
    tmp.replace(dest)            # never leave a half-written file under the real name
    return dest


# --- Lambert azimuthal equal-area, ellipsoidal (Snyder 1987, eqs 24-28) -----
_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
_E = math.sqrt(_E2)


def _q(sinphi):
    es = _E * sinphi
    return (1 - _E2) * (sinphi / (1 - _E2 * sinphi * sinphi)
                        - (1 / (2 * _E)) * np.log((1 - es) / (1 + es)))


def _parse_projdef(projdef: str) -> dict[str, float]:
    out = {}
    for tok in projdef.split():
        if tok.startswith("+") and "=" in tok:
            k, v = tok[1:].split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def laea_forward(lon, lat, lat0: float, lon0: float, x0: float = 0.0,
                 y0: float = 0.0):
    """WGS84 lon/lat (deg) -> LAEA x/y (m), matching proj's +proj=laea."""
    phi = np.radians(lat)
    lam = np.radians(lon) - math.radians(lon0)
    phi1 = math.radians(lat0)
    qp = float(_q(1.0))
    rq = _A * math.sqrt(qp / 2)
    beta = np.arcsin(_q(np.sin(phi)) / qp)
    beta1 = math.asin(float(_q(math.sin(phi1))) / qp)
    m1 = math.cos(phi1) / math.sqrt(1 - _E2 * math.sin(phi1) ** 2)
    d = _A * m1 / (rq * math.cos(beta1))
    b = rq * np.sqrt(2 / (1 + math.sin(beta1) * np.sin(beta)
                          + math.cos(beta1) * np.cos(beta) * np.cos(lam)))
    x = b * d * np.cos(beta) * np.sin(lam) + x0
    y = (b / d) * (math.cos(beta1) * np.sin(beta)
                   - math.sin(beta1) * np.cos(beta) * np.cos(lam)) + y0
    return x, y


@dataclass(frozen=True)
class _Geom:
    """How a target grid maps into the composite."""
    r0: int                 # hyperslab rows [r0, r1)
    r1: int
    c0: int                 # hyperslab cols [c0, c1)
    c1: int
    flat: np.ndarray        # [npix] int32 index into the slab (ravelled)
    valid: np.ndarray       # [npix] bool, pixel falls inside the composite


_GEOM_CACHE: dict = {}
_GEOM_LOCK = threading.Lock()


def _where_sig(where) -> tuple:
    return (str(where["projdef"]), int(where["xsize"]), int(where["ysize"]),
            float(where["xscale"]), float(where["yscale"]),
            round(float(where["LL_lon"]), 6), round(float(where["LL_lat"]), 6))


def _geom_for(where, grid: geo.Grid) -> _Geom:
    sig = (_where_sig(where), grid.west, grid.south, grid.east, grid.north,
           grid.width)
    with _GEOM_LOCK:
        hit = _GEOM_CACHE.get(sig)
    if hit is not None:
        return hit
    p = _parse_projdef(where["projdef"])
    xs, ys = int(where["xsize"]), int(where["ysize"])
    dx, dy = float(where["xscale"]), float(where["yscale"])
    lat0, lon0 = p.get("lat_0", 0.0), p.get("lon_0", 0.0)
    fx, fy = p.get("x_0", 0.0), p.get("y_0", 0.0)
    # Image origin from the file's own LL/UL corners (centre of corner pixels
    # are not guaranteed, so anchor on the outer edge: LL corner = (x, y_min)).
    llx, lly = laea_forward(float(where["LL_lon"]), float(where["LL_lat"]),
                            lat0, lon0, fx, fy)
    top = lly + ys * dy
    lon, lat = grid.lonlat()
    x, y = laea_forward(lon.ravel(), lat.ravel(), lat0, lon0, fx, fy)
    col = np.floor((x - llx) / dx).astype(np.int64)
    row = np.floor((top - y) / dy).astype(np.int64)
    valid = (col >= 0) & (col < xs) & (row >= 0) & (row < ys)
    if not valid.any():
        g = _Geom(0, 0, 0, 0, np.zeros(lon.size, np.int32), valid)
    else:
        r0, r1 = int(row[valid].min()), int(row[valid].max()) + 1
        c0, c1 = int(col[valid].min()), int(col[valid].max()) + 1
        flat = np.where(valid, (row - r0) * (c1 - c0) + (col - c0), 0)
        g = _Geom(r0, r1, c0, c1, flat.astype(np.int32), valid)
    with _GEOM_LOCK:
        if len(_GEOM_CACHE) > 8:
            _GEOM_CACHE.clear()
        _GEOM_CACHE[sig] = g
    return g


# Decoded grids are reused by every panel/product at the same time step.
_FIELD_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_FIELD_LOCK = threading.Lock()
_FIELD_MAX = 6


def dbz_grid(path: Path, grid: geo.Grid) -> np.ndarray | None:
    """Composite reflectivity (dBZ, float32 [H,W], NaN = no echo / no data)."""
    ck = (str(path), grid.west, grid.south, grid.east, grid.north, grid.width)
    with _FIELD_LOCK:
        if ck in _FIELD_CACHE:
            _FIELD_CACHE.move_to_end(ck)
            return _FIELD_CACHE[ck]
    try:
        with h5py.File(path, "r") as f:
            where = {k: (v.decode() if isinstance(v, bytes) else v)
                     for k, v in f["where"].attrs.items()}
            what = f["dataset1/data1/what"].attrs
            gain, offset = float(what["gain"]), float(what["offset"])
            nodata, undetect = float(what["nodata"]), float(what["undetect"])
            g = _geom_for(where, grid)
            if g.r1 <= g.r0:
                return None
            slab = f["dataset1/data1/data"][g.r0:g.r1, g.c0:g.c1]
    except (OSError, KeyError):
        return None
    raw = slab.ravel().take(g.flat)
    bad = ~g.valid | (raw == nodata) | (raw == undetect)
    out = (raw * gain + offset).astype(np.float32)
    out[bad] = np.nan
    out = out.reshape(grid.height, grid.width)
    with _FIELD_LOCK:
        _FIELD_CACHE[ck] = out
        while len(_FIELD_CACHE) > _FIELD_MAX:
            _FIELD_CACHE.popitem(last=False)
    return out


def point_dbz(path: Path, lat: float, lon: float) -> float | None:
    """Composite dBZ at one location (hover inspector)."""
    try:
        with h5py.File(path, "r") as f:
            where = {k: (v.decode() if isinstance(v, bytes) else v)
                     for k, v in f["where"].attrs.items()}
            what = f["dataset1/data1/what"].attrs
            p = _parse_projdef(where["projdef"])
            lat0, lon0 = p.get("lat_0", 0.0), p.get("lon_0", 0.0)
            fx, fy = p.get("x_0", 0.0), p.get("y_0", 0.0)
            llx, lly = laea_forward(float(where["LL_lon"]), float(where["LL_lat"]),
                                    lat0, lon0, fx, fy)
            x, y = laea_forward(lon, lat, lat0, lon0, fx, fy)
            dy = float(where["yscale"])
            col = int(math.floor((float(x) - llx) / float(where["xscale"])))
            row = int(math.floor((lly + int(where["ysize"]) * dy - float(y)) / dy))
            if not (0 <= col < int(where["xsize"]) and 0 <= row < int(where["ysize"])):
                return None
            v = float(f["dataset1/data1/data"][row, col])
            if v in (float(what["nodata"]), float(what["undetect"])):
                return None
            return v * float(what["gain"]) + float(what["offset"])
    except (OSError, KeyError):
        return None


def nearest(files: dict[str, tuple[datetime, Path]], target: datetime,
            tol: timedelta) -> tuple[datetime, Path] | None:
    best, gap = None, tol
    for t, p in files.values():
        d = abs(t - target)
        if d <= gap:
            best, gap = (t, p), d
    return best
