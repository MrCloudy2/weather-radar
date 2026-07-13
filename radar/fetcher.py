"""Anonymous S3 fetcher for the openradar-24h bucket.

Talks to the CloudFerro S3 REST endpoint over plain HTTPS (no auth needed for
this public bucket), lists ODIM_H5 keys, decodes the self-describing filenames
and downloads volumes to the local data dir.

Key schema (confirmed against live bucket):
    YYYY/MM/DD/<CC>/<node>/<TYPE>/<node>@<UTC>@<elev1_elev2_...>@<MOM1_MOM2_...>.h5
e.g.
    2026/07/10/SI/silis/PVOL/silis@20260710T1730@0.5_1.0_...@DBZH_TH_VRADH.h5
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

from . import config

# S3 ListBucketResult uses this XML namespace on every element.
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# Parses the "<node>@<timestamp>@<elevations>@<moments>.h5" tail of a key.
_NAME_RE = re.compile(
    r"^(?P<node>[a-z0-9]+)@"
    r"(?P<ts>\d{8}T\d{4,6})@"
    r"(?P<elevs>[0-9._]+)@"
    r"(?P<moments>[A-Z0-9_]+)\.h5$"
)


@dataclass(frozen=True)
class RadarFile:
    """One ODIM_H5 object in the bucket, decoded from its key + metadata."""

    key: str                    # full S3 key
    node: str                   # e.g. "silis"
    scan_type: str              # e.g. "PVOL"
    time: datetime              # scan time (UTC)
    elevations: list[float]     # elevation angles in degrees
    moments: list[str]          # e.g. ["DBZH", "TH", "VRADH"]
    size: int                   # bytes

    @property
    def url(self) -> str:
        return f"{config.S3_BASE}/{self.key}"

    @property
    def filename(self) -> str:
        return self.key.rsplit("/", 1)[-1]

    def __str__(self) -> str:
        return (
            f"{self.node} {self.scan_type} {self.time:%Y-%m-%d %H:%M}Z "
            f"| {len(self.elevations)} tilts | {'+'.join(self.moments)} "
            f"| {self.size/1024:.0f} KiB"
        )


def _decode_key(key: str, size: int) -> RadarFile | None:
    """Turn an S3 key into a RadarFile, or None if it doesn't match schema."""
    parts = key.split("/")
    if len(parts) < 7:
        return None
    scan_type = parts[5]
    m = _NAME_RE.match(parts[-1])
    if not m:
        return None
    ts = m.group("ts")
    # Timestamps are either YYYYMMDDThhmm or ...hhmmss.
    fmt = "%Y%m%dT%H%M%S" if len(ts) == 15 else "%Y%m%dT%H%M"
    when = datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
    elevs = [float(x) for x in m.group("elevs").split("_") if x]
    moments = m.group("moments").split("_")
    return RadarFile(
        key=key,
        node=m.group("node"),
        scan_type=scan_type,
        time=when,
        elevations=elevs,
        moments=moments,
        size=size,
    )


def list_keys(prefix: str) -> list[RadarFile]:
    """List every decodable ODIM file under an S3 prefix (handles pagination)."""
    out: list[RadarFile] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        resp = requests.get(
            f"{config.S3_BASE}/", params=params, timeout=config.HTTP_TIMEOUT
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for c in root.findall(f"{_S3_NS}Contents"):
            key = c.findtext(f"{_S3_NS}Key", default="")
            size = int(c.findtext(f"{_S3_NS}Size", default="0"))
            rf = _decode_key(key, size)
            if rf:
                out.append(rf)
        if root.findtext(f"{_S3_NS}IsTruncated") == "true":
            token = root.findtext(f"{_S3_NS}NextContinuationToken")
        else:
            break
    return out


def _prefix_for(node: str, day: datetime) -> str:
    country = config.NODES[node].country
    return f"{day:%Y/%m/%d}/{country}/{node}/"


def list_node_files(node: str, day: datetime | None = None) -> list[RadarFile]:
    """All files for a node on a given UTC day (default: today), time-sorted."""
    day = day or datetime.now(timezone.utc)
    files = list_keys(_prefix_for(node, day))
    files.sort(key=lambda f: f.time)
    return files


def latest_file(node: str) -> RadarFile | None:
    """Most recent file for a node, checking today and rolling back a day."""
    now = datetime.now(timezone.utc)
    for day_offset in (0, 1):  # bucket is rolling 24h; today may be sparse early
        day = now.fromordinal(now.toordinal() - day_offset).replace(
            tzinfo=timezone.utc
        )
        files = list_node_files(node, day)
        if files:
            return files[-1]
    return None


def download(rf: RadarFile, dest_dir: Path | None = None) -> Path:
    """Download a RadarFile to the local data dir; returns the local path."""
    dest_dir = dest_dir or config.DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / rf.filename
    if dest.exists() and dest.stat().st_size == rf.size:
        return dest  # already have it
    with requests.get(rf.url, stream=True, timeout=config.HTTP_TIMEOUT) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
    return dest
