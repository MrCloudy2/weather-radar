"""Velocity dealiasing + clutter gating (pure NumPy).

The Slovenian VRADH is folded at the Nyquist velocity (NI ~= 8 m/s) and is not
pre-unfolded. We use a continuity-based unfold: march outward along each ray and
whenever a gate jumps by more than one Nyquist interval from its running
reference, add/subtract multiples of 2*Vn to restore continuity. This is a
first-order dealiaser -- good for moderate folds, not a substitute for a full
2D region-growing scheme (that's the point where we'd reach for Py-ART).
"""

from __future__ import annotations

import numpy as np


# Marshall-Palmer Z-R: Z = a * R^b  ->  R = (z_lin / a)^(1/b).
# Hail cap limits absurd rates from hail-contaminated high reflectivity.
ZR_A = 200.0
ZR_B = 1.6
HAIL_CAP_DBZ = 53.0


def dbz_to_rate(dbz: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """Convert reflectivity (dBZ) to rainfall rate (mm/h) via Marshall-Palmer."""
    capped = np.ma.minimum(dbz, HAIL_CAP_DBZ)
    z_lin = np.ma.power(10.0, capped / 10.0)          # dBZ -> linear Z (mm^6/m^3)
    return np.ma.power(z_lin / ZR_A, 1.0 / ZR_B)      # mm/h


def floor(data: np.ma.MaskedArray, min_value: float) -> np.ma.MaskedArray:
    """Mask gates below a display threshold (e.g. drop <5 dBZ reflectivity haze)."""
    mask = np.ma.getmaskarray(data) | (np.ma.getdata(data) < min_value)
    return np.ma.array(np.ma.getdata(data), mask=mask)


def despeckle(data: np.ma.MaskedArray, min_valid: int = 3) -> np.ma.MaskedArray:
    """Remove isolated gates: mask any valid gate with < min_valid valid
    neighbours in its 3x3 polar neighbourhood (azimuth wraps at north).

    Kills the salt-and-pepper clutter/anom-prop speckle that survives near the
    radar without touching coherent storm bodies.
    """
    valid = (~np.ma.getmaskarray(data)).astype(np.int8)
    neigh = np.zeros_like(valid, dtype=np.int16)
    for da in (-1, 0, 1):          # azimuth axis wraps
        rolled = np.roll(valid, da, axis=0)
        for db in (-1, 0, 1):      # range axis: shift, no wrap
            if da == 0 and db == 0:
                continue
            shifted = np.zeros_like(rolled)
            if db == 0:
                shifted = rolled
            elif db > 0:
                shifted[:, db:] = rolled[:, :-db]
            else:
                shifted[:, :db] = rolled[:, -db:]
            neigh += shifted
    keep = valid.astype(bool) & (neigh >= min_valid)
    return np.ma.array(np.ma.getdata(data), mask=~keep)


def azimuthal_shear(
    velocity: np.ma.MaskedArray, slant_ranges: np.ndarray, nrays: int,
    min_range_m: float = 5000.0,
) -> np.ma.MaskedArray:
    """Azimuthal shear of radial velocity (per second).

    d(Vr)/ds across azimuth, where the arc step ds = range * d(azimuth). Strong
    localised couplets (inbound next to outbound) => high |shear| => rotation /
    mesocyclone. Sign encodes rotation sense. Undefined near the radar (tiny arc)
    so we blank the inner few km.
    """
    v = velocity
    daz = 2.0 * np.pi / nrays
    ahead = np.roll(v, -1, axis=0)      # ray j+1 (azimuth wraps at north)
    behind = np.roll(v, +1, axis=0)     # ray j-1
    arc = slant_ranges[np.newaxis, :] * daz          # metres per ray step
    with np.errstate(divide="ignore", invalid="ignore"):
        shear = (ahead - behind) / (2.0 * arc)
    mask = (
        np.ma.getmaskarray(ahead) | np.ma.getmaskarray(behind)
        | np.ma.getmaskarray(v)
        | (slant_ranges[np.newaxis, :] < min_range_m)
    )
    return np.ma.array(np.ma.getdata(shear), mask=mask)


def gate_by_reflectivity(
    velocity: np.ma.MaskedArray, reflectivity: np.ma.MaskedArray
) -> np.ma.MaskedArray:
    """Mask velocity where corrected reflectivity has no echo (clutter/noise)."""
    mask = np.ma.getmaskarray(velocity) | np.ma.getmaskarray(reflectivity)
    return np.ma.array(np.ma.getdata(velocity), mask=mask)


def dealias(velocity: np.ma.MaskedArray, nyquist: float) -> np.ma.MaskedArray:
    """Continuity-based along-ray velocity unfolding.

    velocity : [nrays x nbins] masked array of folded radial velocity (m/s)
    nyquist  : unambiguous velocity Vn (m/s); folding interval is 2*Vn
    """
    if nyquist is None or nyquist <= 0:
        return velocity
    vn2 = 2.0 * nyquist
    data = np.ma.getdata(velocity).astype(np.float64)
    mask = np.ma.getmaskarray(velocity)
    out = data.copy()

    for ray in range(data.shape[0]):
        ref = None
        for b in range(data.shape[1]):
            if mask[ray, b]:
                continue
            if ref is None:
                ref = out[ray, b]  # seed: first valid gate assumed unfolded
                continue
            # Shift by whole folding intervals to sit closest to the reference.
            n = round((ref - out[ray, b]) / vn2)
            out[ray, b] += n * vn2
            # Slowly track the reference to follow smooth shear.
            ref = 0.5 * ref + 0.5 * out[ray, b]
    return np.ma.array(out, mask=mask)
