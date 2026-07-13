"""Colour maps + legends for radar moments.

Each colour map is a list of (physical_value, (r,g,b)) stops. `apply()` linearly
interpolates a masked physical array into an RGBA uint8 image (transparent where
masked). Legends are exported for the frontend so the map and sidebar agree.
"""

from __future__ import annotations

import numpy as np

# NWS-style reflectivity ramp (dBZ). Green->yellow->red->magenta for hail.
REFLECTIVITY = [
    (-30, (0, 0, 0)),
    (-10, (100, 100, 100)),
    (5, (4, 233, 231)),
    (10, (1, 159, 244)),
    (15, (3, 0, 244)),
    (20, (2, 253, 2)),
    (25, (1, 197, 1)),
    (30, (0, 142, 0)),
    (35, (253, 248, 2)),
    (40, (229, 188, 0)),
    (45, (253, 149, 0)),
    (50, (253, 0, 0)),
    (55, (212, 0, 0)),
    (60, (188, 0, 0)),
    (65, (248, 0, 253)),
    (70, (152, 84, 198)),
    (75, (255, 255, 255)),
]

# Diverging velocity ramp (m/s): inbound (toward radar, negative) = green/blue,
# outbound (positive) = red/orange, near-zero = grey.
VELOCITY = [
    (-30, (0, 224, 224)),
    (-20, (0, 176, 176)),
    (-10, (0, 224, 0)),
    (-5, (0, 144, 0)),
    (-1, (120, 190, 120)),
    (0, (150, 150, 150)),
    (1, (190, 120, 120)),
    (5, (160, 0, 0)),
    (10, (224, 0, 0)),
    (20, (224, 128, 0)),
    (30, (224, 224, 0)),
]

# Rainfall rate (mm/h), ARSO-style thresholds. Stops are non-uniform, so the
# legend is drawn on a log scale (see `scale` below).
RAINRATE = [
    (0.1, (180, 220, 255)),
    (0.5, (0, 150, 255)),
    (1, (0, 90, 230)),
    (2, (0, 200, 120)),
    (5, (0, 175, 0)),
    (10, (180, 220, 0)),
    (20, (255, 230, 0)),
    (35, (255, 150, 0)),
    (50, (240, 0, 0)),
    (100, (190, 0, 130)),
]

# Echo top height (km): blue (low) -> red (towering).
ECHOTOP = [
    (0, (40, 60, 120)),
    (2, (0, 120, 220)),
    (4, (0, 200, 200)),
    (6, (0, 200, 60)),
    (8, (200, 220, 0)),
    (10, (255, 150, 0)),
    (12, (255, 40, 0)),
    (15, (200, 0, 130)),
]

# Vertically Integrated Liquid (kg/m^2): hail/severe proxy.
VILMAP = [
    (0, (40, 60, 120)),
    (2, (0, 140, 220)),
    (5, (0, 200, 120)),
    (10, (0, 190, 0)),
    (20, (200, 220, 0)),
    (30, (255, 160, 0)),
    (45, (255, 40, 0)),
    (60, (190, 0, 130)),
]

# Azimuthal shear (per second): diverging about zero. Reds = one rotation sense.
AZSHEAR = [
    (-0.02, (0, 180, 180)),
    (-0.008, (0, 120, 220)),
    (-0.002, (120, 150, 180)),
    (0.0, (150, 150, 150)),
    (0.002, (200, 150, 120)),
    (0.008, (220, 80, 0)),
    (0.02, (200, 0, 0)),
]

# Correlation coefficient rho_HV (particle uniformity, 0..1.05):
# high ~1 = uniform (rain, blue/green); low = mixed hail / debris (yellow/red).
RHOHV = [
    (0.2, (120, 0, 130)),
    (0.7, (220, 0, 0)),
    (0.85, (255, 150, 0)),
    (0.92, (230, 220, 0)),
    (0.96, (0, 200, 60)),
    (0.98, (0, 180, 220)),
    (1.0, (0, 80, 220)),
    (1.05, (120, 120, 120)),
]

# Differential reflectivity ZDR (dB): shape. ~0 spherical(hail), + = oblate(rain).
ZDR = [
    (-2, (60, 60, 60)),
    (0, (0, 120, 220)),
    (1, (0, 200, 80)),
    (2, (200, 220, 0)),
    (4, (255, 120, 0)),
    (6, (200, 0, 0)),
]

# Rainfall accumulation (mm) over a time window; log-spaced legend.
ACCUM = [
    (0.5, (180, 220, 255)),
    (1, (0, 150, 255)),
    (2, (0, 90, 230)),
    (5, (0, 200, 120)),
    (10, (0, 175, 0)),
    (20, (200, 220, 0)),
    (35, (255, 160, 0)),
    (50, (240, 0, 0)),
    (100, (190, 0, 130)),
]

# Horizontal wind speed (m/s), dual-Doppler.
WINDSPEED = [
    (0, (40, 60, 120)),
    (5, (0, 140, 220)),
    (10, (0, 200, 120)),
    (15, (0, 190, 0)),
    (20, (200, 220, 0)),
    (28, (255, 150, 0)),
    (36, (255, 40, 0)),
    (50, (190, 0, 130)),
]

CMAPS = {
    "DBZH": REFLECTIVITY,
    "TH": REFLECTIVITY,
    "VRADH": VELOCITY,
    "RATE": RAINRATE,
    "ET": ECHOTOP,
    "VIL": VILMAP,
    "AZSHEAR": AZSHEAR,
    "RHOHV": RHOHV,
    "ZDR": ZDR,
    "NOWCAST": REFLECTIVITY,
    "ACCUM": ACCUM,
    "WIND": WINDSPEED,
}

UNITS = {
    "DBZH": "dBZ", "TH": "dBZ", "VRADH": "m/s", "RATE": "mm/h",
    "ET": "km", "VIL": "kg/m²", "AZSHEAR": "1/s", "RHOHV": "", "ZDR": "dB",
    "NOWCAST": "dBZ", "ACCUM": "mm", "WIND": "m/s",
}
LABELS = {
    "DBZH": "Reflectivity (corrected)",
    "TH": "Total power (uncorrected)",
    "VRADH": "Radial velocity",
    "RATE": "Rainfall rate",
    "ET": "Echo top height",
    "VIL": "Vertically integrated liquid",
    "AZSHEAR": "Azimuthal shear (rotation)",
    "RHOHV": "Correlation ρHV (particle uniformity)",
    "ZDR": "Differential reflectivity",
    "NOWCAST": "Nowcast (extrapolated echo)",
    "ACCUM": "Rainfall accumulation",
    "WIND": "Dual-Doppler wind speed",
}
# Which quantities use a log-spaced legend bar.
LOG_SCALE = {"RATE", "ACCUM"}


def _ramp(stops):
    vals = np.array([s[0] for s in stops], dtype=np.float64)
    cols = np.array([s[1] for s in stops], dtype=np.float64)
    return vals, cols


def apply(data: np.ma.MaskedArray, quantity: str) -> np.ndarray:
    """Map a masked physical array -> RGBA uint8 (transparent where masked)."""
    stops = CMAPS.get(quantity, REFLECTIVITY)
    vals, cols = _ramp(stops)
    arr = np.ma.filled(data, np.nan).astype(np.float64)
    valid = ~np.isnan(arr)
    safe = np.where(valid, arr, vals[0])  # avoid NaN in interp/cast
    r = np.interp(safe, vals, cols[:, 0])
    g = np.interp(safe, vals, cols[:, 1])
    b = np.interp(safe, vals, cols[:, 2])
    rgba = np.zeros(arr.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = np.clip(r, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(g, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(b, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def legend(quantity: str) -> dict:
    """Legend spec (stops + units) for the frontend colour bar."""
    stops = CMAPS.get(quantity, REFLECTIVITY)
    return {
        "quantity": quantity,
        "label": LABELS.get(quantity, quantity),
        "units": UNITS.get(quantity, ""),
        "scale": "log" if quantity in LOG_SCALE else "linear",
        "stops": [
            {"value": v, "color": f"#{r:02x}{g:02x}{b:02x}"}
            for v, (r, g, b) in stops
        ],
    }
