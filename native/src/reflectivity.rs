//! Reflectivity / max-composite kernels (IMPLEMENTED).
//!
//! Python reference: `radar/render.py` `sample_cmax_to_grid` / `_sample_cached`.

use crate::grid::{find_bin, is_nan};
use crate::OK;

/// NaN-aware column-max accumulation for ONE sweep into `best`.
///
/// For each grid pixel: bin-search its ground distance, gather the polar sample,
/// keep the running max. Call once per sweep with `best` NaN-initialised; the
/// result is the CMAX over all sweeps for one radar. Composite across radars by
/// continuing to accumulate into the same `best` per radar.
///
/// # Safety
/// All pointers must be valid for the stated lengths (see module docs).
#[no_mangle]
pub unsafe extern "C" fn cmax_sweep(
    npix: i32,
    ray_idx: *const i32,
    dist: *const f32,
    data: *const f32, // [nrays*nbins], NaN = invalid
    nbins: i32,
    edges: *const f64, // [nbins+1]
    best: *mut f32,    // [npix], in/out
) -> i32 {
    let nb = nbins as isize;
    for p in 0..npix as isize {
        let bin = find_bin(edges, nbins, *dist.offset(p) as f64);
        if bin < 0 {
            continue;
        }
        let v = *data.offset(*ray_idx.offset(p) as isize * nb + bin as isize);
        if !is_nan(v) {
            let cur = *best.offset(p);
            if is_nan(cur) || v > cur {
                *best.offset(p) = v;
            }
        }
    }
    OK
}

/// Sample ONE sweep onto the grid into `out` (no max): `out[p]` = gate value or
/// NaN. Used for single-tilt products (velocity, ZDR, RHOHV, single-radar DBZH).
///
/// # Safety
/// All pointers must be valid for the stated lengths.
#[no_mangle]
pub unsafe extern "C" fn sample_sweep(
    npix: i32,
    ray_idx: *const i32,
    dist: *const f32,
    data: *const f32,
    nbins: i32,
    edges: *const f64,
    out: *mut f32, // [npix]
) -> i32 {
    let nb = nbins as isize;
    for p in 0..npix as isize {
        let bin = find_bin(edges, nbins, *dist.offset(p) as f64);
        if bin < 0 {
            *out.offset(p) = f32::NAN;
            continue;
        }
        *out.offset(p) = *data.offset(*ray_idx.offset(p) as isize * nb + bin as isize);
    }
    OK
}
