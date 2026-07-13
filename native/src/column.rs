//! Vertical-column products: echo tops + VIL (STUBS — to implement).
//!
//! Python reference: `radar/render.py` `sample_column_to_grid`.

use crate::NOT_IMPLEMENTED;

/// Echo-top accumulation for ONE sweep into `et` (metres). NaN-initialise `et`,
/// call per sweep; result = highest beam altitude with reflectivity >= `thr`.
///
/// TODO(implement): for each pixel `p`
///   * `bin = find_bin(edges, nbins, dist[p])`; skip if < 0
///   * `z = data[ray_idx[p]*nbins + bin]`; skip if NaN or `z < thr`
///   * `h = heights_bin[bin]` (beam height for this sweep's bins, precomputed
///     in Python from slant range + elevation + site height)
///   * `if et[p].is_nan() || h > et[p] { et[p] = h }`
/// Return `OK`.
///
/// # Safety
/// Pointers valid for the stated lengths.
#[no_mangle]
pub unsafe extern "C" fn echo_top_sweep(
    _npix: i32,
    _ray_idx: *const i32,
    _dist: *const f32,
    _data: *const f32,   // [nrays*nbins]
    _nbins: i32,
    _edges: *const f64,  // [nbins+1]
    _heights_bin: *const f32, // [nbins] beam height per bin (metres)
    _thr: f32,           // echo-top reflectivity threshold (e.g. 18 dBZ)
    _et: *mut f32,       // [npix], in/out (metres, NaN = none)
) -> i32 {
    NOT_IMPLEMENTED
}

/// Vertically Integrated Liquid over the column, per pixel, into `vil`.
///
/// Inputs are pre-sampled per-pixel stacks (Python samples each sweep to the
/// grid first): `dbz` and `height` are `[nsweeps*npix]` row-major (sweep-major),
/// NaN where a sweep has no gate at that pixel.
///
/// TODO(implement): for each pixel `p`, gather the valid `(height, dbz)` pairs
/// across sweeps, sort by height, then trapezoid-integrate liquid water content
///   `M(Z) = 3.44e-6 * (10^(min(Z,56)/10))^(4/7)`  [kg/m^3]
/// over height:  `vil[p] = sum_i (M_i + M_{i+1})/2 * (h_{i+1}-h_i)` [kg/m^2].
/// Set `vil[p] = NaN` if < 2 valid layers or result <= 0.05.
///
/// # Safety
/// Pointers valid for the stated lengths.
#[no_mangle]
pub unsafe extern "C" fn vil_column(
    _npix: i32,
    _nsweeps: i32,
    _dbz: *const f32,    // [nsweeps*npix], NaN = empty
    _height: *const f32, // [nsweeps*npix], metres
    _vil: *mut f32,      // [npix]
) -> i32 {
    NOT_IMPLEMENTED
}
