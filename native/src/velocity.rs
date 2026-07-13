//! Velocity products: nearest-radar mosaic + along-ray dealiasing (STUBS).
//!
//! Python references: `radar/composite.py` `build_composite` (nearest branch)
//! and `radar/process.py` `dealias`.

use crate::NOT_IMPLEMENTED;

/// Nearest-radar mosaic: per grid pixel choose the value from the closest radar
/// that has a valid (non-NaN) sample. Radial velocity is radar-relative and
/// can't be averaged across sites, so we pick nearest instead of max.
///
/// `values` and `dists` are `[nradar*npix]` row-major (radar-major); `dists` is
/// ground range from each radar to each pixel (metres).
///
/// TODO(implement): for each pixel `p`, over radars `r`, track the value with
/// the smallest `dists[r*npix+p]` among those where `values[r*npix+p]` is not
/// NaN; write it to `out[p]` (NaN if none valid).
///
/// # Safety
/// Pointers valid for the stated lengths.
#[no_mangle]
pub unsafe extern "C" fn nearest_mosaic(
    _npix: i32,
    _nradar: i32,
    _values: *const f32, // [nradar*npix], NaN = invalid
    _dists: *const f32,  // [nradar*npix], metres
    _out: *mut f32,      // [npix]
) -> i32 {
    NOT_IMPLEMENTED
}

/// Continuity-based along-ray velocity dealiasing, in place on a polar sweep.
///
/// `vel` is `[nrays*nbins]` row-major (ray-major), NaN = invalid gate; folding
/// interval is `2*nyquist`.
///
/// TODO(implement): for each ray, march outward over valid gates keeping a
/// running reference `ref` (seed = first valid gate). For each subsequent valid
/// gate: `n = round((ref - v)/(2*nyquist)); v += n*2*nyquist;` then
/// `ref = 0.5*ref + 0.5*v`. Mirrors `radar/process.py::dealias`. Consider also a
/// second pass across azimuth for the P3 "2D region-growing" upgrade.
///
/// # Safety
/// `vel` valid for `nrays*nbins`.
#[no_mangle]
pub unsafe extern "C" fn dealias_rays(
    _nrays: i32,
    _nbins: i32,
    _vel: *mut f32, // [nrays*nbins], in/out
    _nyquist: f32,
) -> i32 {
    NOT_IMPLEMENTED
}
