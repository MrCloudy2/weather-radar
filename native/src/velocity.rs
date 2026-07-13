//! Velocity products: nearest-radar mosaic + along-ray dealiasing.
//!
//! Python references: `radar/composite.py` `build_composite` (nearest branch)
//! and `radar/process.py` `dealias`.

/// Nearest-radar mosaic: per grid pixel choose the value from the closest radar
/// that has a valid (non-NaN) sample. Radial velocity is radar-relative and
/// can't be averaged across sites, so we pick nearest instead of max.
///
/// `values` and `dists` are `[nradar*npix]` row-major (radar-major); `dists` is
/// ground range from each radar to each pixel (metres). `out[p]` = value of the
/// nearest valid radar, NaN if no radar has an echo there. Replaces the NumPy
/// masked-inf `argmin` + fancy-index gather with one streaming pass.
///
/// # Safety
/// Pointers valid for the stated lengths.
#[no_mangle]
pub unsafe extern "C" fn nearest_mosaic(
    npix: i32,
    nradar: i32,
    values: *const f32, // [nradar*npix], NaN = invalid
    dists: *const f32,  // [nradar*npix], metres
    out: *mut f32,      // [npix]
) -> i32 {
    let npix = npix as usize;
    let nr = nradar as usize;
    for p in 0..npix {
        let mut best_d = f32::INFINITY;
        let mut best_v = f32::NAN;
        for r in 0..nr {
            let v = *values.add(r * npix + p);
            if v != v {
                continue;
            }
            let d = *dists.add(r * npix + p);
            if d < best_d {
                best_d = d;
                best_v = v;
            }
        }
        *out.add(p) = best_v;
    }
    crate::OK
}

/// Continuity-based along-ray velocity dealiasing, in place on a polar sweep.
///
/// `vel` is `[nrays*nbins]` row-major (ray-major), NaN = invalid gate; folding
/// interval is `2*nyquist`. For each ray, march outward over valid gates
/// keeping a running reference `ref` (seed = first valid gate): shift each gate
/// by whole folding intervals to sit closest to `ref`, then track
/// `ref = 0.5*ref + 0.5*v` to follow smooth shear. Mirrors
/// `radar/process.py::dealias` (f64 internally, like the NumPy path).
///
/// # Safety
/// `vel` valid for `nrays*nbins`.
#[no_mangle]
pub unsafe extern "C" fn dealias_rays(
    nrays: i32,
    nbins: i32,
    vel: *mut f32, // [nrays*nbins], in/out
    nyquist: f32,
) -> i32 {
    let nb = nbins as usize;
    let vn2 = 2.0 * nyquist as f64;
    for ray in 0..nrays as usize {
        let row = vel.add(ray * nb);
        let mut reference = f64::NAN;
        for b in 0..nb {
            let v = *row.add(b);
            if v != v {
                continue;
            }
            if reference != reference {
                reference = v as f64; // seed: first valid gate assumed unfolded
                continue;
            }
            let n = ((reference - v as f64) / vn2).round();
            let unfolded = v as f64 + n * vn2;
            *row.add(b) = unfolded as f32;
            reference = 0.5 * reference + 0.5 * unfolded;
        }
    }
    crate::OK
}
