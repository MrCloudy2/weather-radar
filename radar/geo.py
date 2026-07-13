"""Radar georeferencing: polar (range, azimuth, elevation) <-> geographic.

Uses the standard 4/3-effective-earth-radius beam-propagation model
(Doviak & Zrunic) so gates land at their true ground projection, and Web
Mercator helpers so rendered images align pixel-perfect under Leaflet.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

R_EARTH = 6_371_000.0          # mean earth radius, metres
K_E = 4.0 / 3.0                # effective-radius factor (standard refraction)
A_E = K_E * R_EARTH            # effective earth radius
_MERC_R = 6_378_137.0          # Web Mercator (EPSG:3857) sphere radius


def beam_ground_range(slant_r: np.ndarray, elev_rad: float) -> np.ndarray:
    """Ground (great-circle) range for gates at slant range `slant_r` (m).

    Accounts for beam height gained along the tilt, so a 250 km slant gate on
    the 0.5 deg sweep maps slightly inside 250 km on the ground.
    """
    r = np.asarray(slant_r, dtype=np.float64)
    h = np.sqrt(r * r + A_E * A_E + 2.0 * r * A_E * np.sin(elev_rad)) - A_E
    return A_E * np.arcsin(r * np.cos(elev_rad) / (A_E + h))


def haversine_bearing(
    lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ground distance (m) and compass bearing (deg, 0=N cw) from site to points."""
    lat0r, lon0r = np.radians(lat0), np.radians(lon0)
    latr, lonr = np.radians(lat), np.radians(lon)
    dlat = latr - lat0r
    dlon = lonr - lon0r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0r) * np.cos(latr) * np.sin(dlon / 2) ** 2
    dist = 2 * R_EARTH * np.arcsin(np.sqrt(a))
    y = np.sin(dlon) * np.cos(latr)
    x = np.cos(lat0r) * np.sin(latr) - np.sin(lat0r) * np.cos(latr) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return dist, bearing


def destination(
    lat0: float, lon0: float, bearing_deg: np.ndarray, dist_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Geographic point reached from (lat0,lon0) along a bearing for dist_m."""
    lat0r, lon0r = np.radians(lat0), np.radians(lon0)
    br = np.radians(bearing_deg)
    dr = np.asarray(dist_m) / R_EARTH
    lat = np.arcsin(
        np.sin(lat0r) * np.cos(dr) + np.cos(lat0r) * np.sin(dr) * np.cos(br)
    )
    lon = lon0r + np.arctan2(
        np.sin(br) * np.sin(dr) * np.cos(lat0r),
        np.cos(dr) - np.sin(lat0r) * np.sin(lat),
    )
    return np.degrees(lat), np.degrees(lon)


# --- Web Mercator (EPSG:3857) --------------------------------------------
def lonlat_to_merc(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.radians(lon) * _MERC_R
    y = _MERC_R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def merc_to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.degrees(np.asarray(x) / _MERC_R)
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y) / _MERC_R)) - np.pi / 2)
    return lon, lat


def site_bbox_lonlat(
    lat0: float, lon0: float, radius_m: float
) -> tuple[float, float, float, float]:
    """(west, south, east, north) lon/lat box enclosing a radius around a site."""
    lats, lons = destination(
        lat0, lon0, np.array([0.0, 90.0, 180.0, 270.0]), np.full(4, radius_m)
    )
    return float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())


@dataclass(frozen=True)
class Grid:
    """A fixed Web Mercator output grid shared by all panels/composites.

    Rendering in EPSG:3857 with a common extent means one screen pixel is the
    same geographic point in every panel, so multi-panel hover-inspection lines
    up and composites overlay cleanly.
    """

    west: float
    south: float
    east: float
    north: float
    width: int

    @property
    def height(self) -> int:
        x0, y0 = lonlat_to_merc(np.array([self.west]), np.array([self.south]))
        x1, y1 = lonlat_to_merc(np.array([self.east]), np.array([self.north]))
        return int(round(self.width * (y1[0] - y0[0]) / (x1[0] - x0[0])))

    @property
    def leaflet_bounds(self) -> list[list[float]]:
        return [[self.south, self.west], [self.north, self.east]]

    @lru_cache(maxsize=None)
    def _lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        x0, y0 = lonlat_to_merc(np.array([self.west]), np.array([self.south]))
        x1, y1 = lonlat_to_merc(np.array([self.east]), np.array([self.north]))
        mx = np.linspace(float(x0[0]), float(x1[0]), self.width)
        my = np.linspace(float(y1[0]), float(y0[0]), self.height)  # top=north
        gx, gy = np.meshgrid(mx, my)
        return merc_to_lonlat(gx, gy)

    def lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-pixel (lon, lat) arrays for this grid, shape [height x width]."""
        return self._lonlat()

    def index(self, lat: float, lon: float) -> tuple[int, int] | None:
        """(row, col) pixel for a geographic point, or None if outside."""
        if not (self.west <= lon <= self.east and self.south <= lat <= self.north):
            return None
        col = int((lon - self.west) / (self.east - self.west) * self.width)
        _, yN = lonlat_to_merc(np.array([lon]), np.array([self.north]))
        _, yS = lonlat_to_merc(np.array([lon]), np.array([self.south]))
        _, y = lonlat_to_merc(np.array([lon]), np.array([lat]))
        row = int((yN[0] - y[0]) / (yN[0] - yS[0]) * self.height)
        row = min(max(row, 0), self.height - 1)
        col = min(max(col, 0), self.width - 1)
        return row, col
