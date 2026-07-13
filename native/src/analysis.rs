//! Temporal / multi-radar analysis: storm motion + dual-Doppler (STUBS).
//!
//! Python reference: `radar/analysis.py` (`motion_field`, `dual_doppler`).

use crate::NOT_IMPLEMENTED;

/// Block cross-correlation storm motion between two reflectivity grids.
///
/// For each `block`x`block` tile of `curr` whose max >= `min_dbz`, search
/// offsets `dx,dy` in `[-search, search]` (step `step`) minimising SSD against
/// `prev`; the best offset (negated) is the tile's pixel displacement. Write one
/// vector per moving tile into the output arrays (up to `max_out`), return the
/// count (>=0), or `NOT_IMPLEMENTED`.
///
/// Outputs (caller-allocated, length `max_out`): `cx,cy` = tile-centre pixel;
/// `dx,dy` = displacement in pixels (prev->curr).
///
/// # Safety
/// `prev`/`curr` valid for `h*w`; output arrays valid for `max_out`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn motion_ssd(
    _h: i32,
    _w: i32,
    _prev: *const f32,
    _curr: *const f32,
    _block: i32,
    _search: i32,
    _step: i32,
    _min_dbz: f32,
    _cx: *mut i32,
    _cy: *mut i32,
    _dx: *mut f32,
    _dy: *mut f32,
    _max_out: i32,
) -> i32 {
    NOT_IMPLEMENTED
}

/// Dual-Doppler horizontal wind from two radars' radial velocities, per pixel.
///
/// `az1`,`az2` are the compass bearings (degrees) from each radar to each pixel.
/// Solve `Vr = u*sin(az) + v*cos(az)` as a 2x2 system:
///   `det = sin(az1)cos(az2) - sin(az2)cos(az1)`
///   `u = (v1*cos(az2) - v2*cos(az1))/det`,  `v = (sin(az1)*v2 - sin(az2)*v1)/det`
/// Mask (NaN) where either input is NaN or `|sin(az1-az2)| < sin(min_cross_deg)`.
/// Write `u`,`v` (m/s) and `speed = hypot(u,v)`.
///
/// # Safety
/// All arrays valid for `npix`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn dual_doppler(
    _npix: i32,
    _v1: *const f32,
    _v2: *const f32,
    _az1: *const f32, // degrees
    _az2: *const f32,
    _min_cross_deg: f32,
    _u: *mut f32,
    _v: *mut f32,
    _speed: *mut f32,
) -> i32 {
    NOT_IMPLEMENTED
}
