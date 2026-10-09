"""Central configuration for the Slovenia radar backend.

Single source of truth for endpoints, radar nodes, the shared output grid and
paths, so fetcher / parser / render / API all agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .geo import Grid

# --- Open Radar Data (ORD) S3, anonymous / public read ----------------------
S3_ENDPOINT = "https://s3.waw3-1.cloudferro.com"
S3_BUCKET = "openradar-24h"
S3_BASE = f"{S3_ENDPOINT}/{S3_BUCKET}"


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    country: str          # ISO code / bucket partition (e.g. "SI", "HR")
    lat: float            # nominal site lat (authoritative value read per-file)
    lon: float
    composite: bool       # part of the default Slovenia composite?


# Slovenian radars (DBZH, TH, VRADH) + nearby Croatian radars (also polarimetric
# ZDR/RHOHV). `composite=True` = included in the default Slovenia mosaic; the far
# Croatian radars stay individually selectable but out of the default composite.
NODES: dict[str, Node] = {
    "silis": Node("silis", "Lisca",       "SI", 46.068, 15.284, True),
    "sipas": Node("sipas", "Pasja Ravan", "SI", 46.099, 14.234, True),
    "hrpun": Node("hrpun", "Puntijarka",  "HR", 45.908, 15.968, True),
    "hrbil": Node("hrbil", "Bilogora",    "HR", 45.884, 17.200, True),
    "hrgol": Node("hrgol", "Goli",        "HR", 45.020, 14.122, True),
    "hrdeb": Node("hrdeb", "Debeljak",    "HR", 44.045, 15.376, False),
    "hrgra": Node("hrgra", "Gradiste",    "HR", 45.159, 18.703, False),
    "hrulj": Node("hrulj", "Uljenje",     "HR", 42.894, 17.478, False),
}

# Nodes the worker actively fetches + keeps history for. Keep this to the ones
# that actually cover Slovenia so we don't hoard the whole network.
# (silis/sipas/hrpun absent from the ORD feed since ~Oct 2026; kept so they come
# back automatically if republished -- the timeline falls back to a live node.)
ACTIVE_NODES = ["silis", "sipas", "hrpun", "hrbil", "hrgol"]

# Blend the EUMETNET OPERA European composite (1 km column-max reflectivity,
# includes silis/sipas) into composite CMAX reflectivity / rain rate. Fills
# Slovenia while ARSO's raw volumes are missing from the feed. See opera.py.
OPERA_FILL = True

# Two Web Mercator output grids:
#  * BASE  — Slovenia-focused; used by all the heavier products (velocity, VIL,
#    echo tops, etc.) so their render cost/memory stay bounded.
#  * WIDE  — extended toward the ~250 km radar range, so you can see storms
#    forming farther out. Reflectivity only (light + Rust-accelerated). Edges are
#    single-radar / long-range and less reliable — that's expected.
BASE_GRID = Grid(west=12.4, south=45.2, east=16.8, north=47.0, width=1100)
WIDE_GRID = Grid(west=11.6, south=44.4, east=18.0, north=48.2, width=1400)
SLOVENIA_GRID = BASE_GRID          # back-compat alias

WIDE_PRODUCTS = {"DBZH", "TH", "RATE"}   # reflectivity family -> extended grid


def grid_for(quantity: str) -> Grid:
    return WIDE_GRID if quantity in WIDE_PRODUCTS else BASE_GRID

# --- Local storage ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Persistent archive of the flagship product (composite CMAX reflectivity):
# one PNG per frame, kept for ARCHIVE_HOURS. Bind-mounted so it survives restarts
# and the 90-min raw window, giving a full-day reflectivity loop.
ARCHIVE_DIR = PROJECT_ROOT / "archive"
ARCHIVE_DIR.mkdir(exist_ok=True)
ARCHIVE_HOURS = 24
ARCHIVE_QUANTITY = "DBZH"          # composite CMAX reflectivity
ARCHIVE_NODES = [n for n in ACTIVE_NODES]   # the default Slovenia composite

HTTP_TIMEOUT = 30
