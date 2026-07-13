"""Minimal, dependency-light ODIM_H5 reader.

Reads EUMETNET OPERA polar volumes (PVOL) straight from HDF5 using h5py and the
ODIM information model, so we understand every byte before layering Py-ART/
wradlib on top. Exposes:

  * summarize(path)                -> RadarVolume metadata tree (for inspection)
  * read_sweep(path, quantity, i)  -> physically-decoded sweep (masked array)

ODIM layout (relevant bits):
  /                 root, Conventions attr
  /what             object, version, date, time, source
  /where            lat, lon, height              (site location)
  /datasetN         one per elevation sweep
    /datasetN/where elangle, nbins, nrays, rstart, rscale, a1gate
    /datasetN/what  startdate/time, enddate/time
    /datasetN/dataM one per moment (DBZH, TH, VRADH, ...)
      .../what      quantity, gain, offset, nodata, undetect
      .../data      raw uint array [nrays x nbins]
Physical value = offset + gain * raw ; raw==nodata|undetect are not real echoes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np


def _attrs(group) -> dict:
    """Decode an HDF5 group's attributes into plain Python (bytes -> str)."""
    out = {}
    for k, v in group.attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


@dataclass
class MomentInfo:
    quantity: str
    gain: float
    offset: float
    nodata: float
    undetect: float
    shape: tuple[int, int]
    dtype: str
    dataset_path: str  # e.g. "/dataset1/data3"


@dataclass
class SweepInfo:
    index: int
    dataset_path: str
    elangle: float
    nrays: int
    nbins: int
    rstart_km: float
    rscale_m: float
    moments: dict[str, MomentInfo] = field(default_factory=dict)

    @property
    def max_range_km(self) -> float:
        return (self.rstart_km * 1000 + self.nbins * self.rscale_m) / 1000.0


@dataclass
class RadarVolume:
    path: str
    object_type: str
    source: str
    nominal_time: str          # "YYYY-MM-DD HH:MM:SSZ"
    site_lat: float
    site_lon: float
    site_height_m: float
    sweeps: list[SweepInfo] = field(default_factory=list)

    @property
    def quantities(self) -> list[str]:
        seen: list[str] = []
        for s in self.sweeps:
            for q in s.moments:
                if q not in seen:
                    seen.append(q)
        return seen


def summarize(path: str | Path) -> RadarVolume:
    """Parse the full metadata tree of an ODIM_H5 volume (no bulk data read).

    Cached: ODIM files are immutable once written, and CMAX/composite paths call
    this dozens of times per render.
    """
    return _summarize_cached(str(path))


@lru_cache(maxsize=256)
def _summarize_cached(path: str) -> RadarVolume:
    with h5py.File(path, "r") as h5:
        root_what = _attrs(h5["what"]) if "what" in h5 else {}
        where = _attrs(h5["where"]) if "where" in h5 else {}

        date = root_what.get("date", "")
        time = root_what.get("time", "")
        nominal = "unknown"
        if len(date) == 8 and len(time) >= 4:
            nominal = (
                f"{date[:4]}-{date[4:6]}-{date[6:8]} "
                f"{time[:2]}:{time[2:4]}:{time[4:6] or '00'}Z"
            )

        vol = RadarVolume(
            path=path,
            object_type=root_what.get("object", "?"),
            source=root_what.get("source", "?"),
            nominal_time=nominal,
            site_lat=float(where.get("lat", float("nan"))),
            site_lon=float(where.get("lon", float("nan"))),
            site_height_m=float(where.get("height", float("nan"))),
        )

        # dataset1, dataset2, ... sorted numerically.
        ds_names = sorted(
            (k for k in h5 if k.startswith("dataset")),
            key=lambda s: int(s.replace("dataset", "")),
        )
        for i, ds_name in enumerate(ds_names):
            ds = h5[ds_name]
            dwhere = _attrs(ds["where"]) if "where" in ds else {}
            sweep = SweepInfo(
                index=i,
                dataset_path=f"/{ds_name}",
                elangle=float(dwhere.get("elangle", float("nan"))),
                nrays=int(dwhere.get("nrays", 0)),
                nbins=int(dwhere.get("nbins", 0)),
                rstart_km=float(dwhere.get("rstart", 0.0)),
                rscale_m=float(dwhere.get("rscale", 0.0)),
            )
            for data_name in (k for k in ds if k.startswith("data")):
                dgrp = ds[data_name]
                if "data" not in dgrp:
                    continue
                dwhat = _attrs(dgrp["what"]) if "what" in dgrp else {}
                arr = dgrp["data"]
                q = dwhat.get("quantity", "?")
                sweep.moments[q] = MomentInfo(
                    quantity=q,
                    gain=float(dwhat.get("gain", 1.0)),
                    offset=float(dwhat.get("offset", 0.0)),
                    nodata=float(dwhat.get("nodata", float("nan"))),
                    undetect=float(dwhat.get("undetect", float("nan"))),
                    shape=tuple(arr.shape),
                    dtype=str(arr.dtype),
                    dataset_path=f"/{ds_name}/{data_name}",
                )
            vol.sweeps.append(sweep)
    return vol


@dataclass
class SweepGeometry:
    """Everything needed to place a sweep's gates on the ground."""

    elangle_rad: float
    az_mid: np.ndarray       # [nrays] mid azimuth of each ray, degrees (0=N cw)
    slant_ranges: np.ndarray  # [nbins] slant range to bin centres, metres
    nyquist: float | None    # unambiguous velocity (m/s), if known


def sweep_geometry(path: str | Path, sweep_index: int = 0) -> SweepGeometry:
    """Read true per-ray azimuths + slant ranges for one sweep (cached)."""
    return _sweep_geometry_cached(str(path), sweep_index)


@lru_cache(maxsize=1024)
def _sweep_geometry_cached(path: str, sweep_index: int) -> SweepGeometry:
    vol = summarize(path)
    sweep = vol.sweeps[sweep_index]
    ds = sweep.dataset_path.lstrip("/")
    with h5py.File(str(path), "r") as h5:
        grp = h5[ds]
        how = _attrs(grp["how"]) if "how" in grp else {}
        start = how.get("startazA")
        stop = how.get("stopazA")
        if start is not None and stop is not None:
            start = np.asarray(start, dtype=np.float64)
            stop = np.asarray(stop, dtype=np.float64)
            # Handle rays that wrap through north (stop < start).
            stop_adj = np.where(stop < start, stop + 360.0, stop)
            az_mid = ((start + stop_adj) / 2.0) % 360.0
        else:
            # Fallback: assume 360 rays of equal width starting at north.
            az_mid = (np.arange(sweep.nrays) + 0.5) * (360.0 / sweep.nrays)
        nyq = how.get("NI")
    slant = sweep.rstart_km * 1000.0 + (np.arange(sweep.nbins) + 0.5) * sweep.rscale_m
    return SweepGeometry(
        elangle_rad=np.radians(sweep.elangle),
        az_mid=az_mid,
        slant_ranges=slant,
        nyquist=float(nyq) if nyq is not None else None,
    )


def read_sweep(
    path: str | Path, quantity: str, sweep_index: int = 0
) -> np.ma.MaskedArray:
    """Return one sweep of a moment as a physically-decoded masked array.

    Shape is [nrays x nbins]; masked where the raw value was nodata/undetect.
    Units: dBZ for DBZH/TH, m/s for VRADH.
    """
    vol = summarize(path)
    if sweep_index >= len(vol.sweeps):
        raise IndexError(f"sweep {sweep_index} out of range (have {len(vol.sweeps)})")
    sweep = vol.sweeps[sweep_index]
    if quantity not in sweep.moments:
        raise KeyError(
            f"{quantity!r} not in sweep {sweep_index}; have {list(sweep.moments)}"
        )
    m = sweep.moments[quantity]
    with h5py.File(str(path), "r") as h5:
        raw = h5[m.dataset_path.lstrip("/") + "/data"][:].astype(np.float64)
    invalid = np.zeros(raw.shape, dtype=bool)
    if not np.isnan(m.nodata):
        invalid |= raw == m.nodata
    if not np.isnan(m.undetect):
        invalid |= raw == m.undetect
    phys = m.offset + m.gain * raw
    return np.ma.array(phys, mask=invalid)
