//! Vertical-column products: echo tops + VIL.
//!
//! Python reference: `radar/render.py` `sample_column_to_grid`.

use crate::NOT_IMPLEMENTED;

/// Echo-top accumulation for ONE sweep into `et` (metres). NaN-initialise `et`,
/// call per sweep; result = highest beam altitude with reflectivity >= `thr`.
///
/// Mirrors the "ET" branch of `render.sample_column_to_grid`: only in-range
/// gates with `z >= thr` contribute their beam height; the running max across
/// sweeps is kept in `et` (NaN = no echo top yet).
///
/// # Safety
/// Pointers valid for the stated lengths.
#[no_mangle]
pub unsafe extern "C" fn echo_top_sweep(
    npix: i32,
    ray_idx: *const i32,
    dist: *const f32,
    data: *const f32,   // [nrays*nbins], NaN = invalid gate
    nbins: i32,
    edges: *const f64,  // [nbins+1]
    heights_bin: *const f32, // [nbins] beam height per bin (metres)
    thr: f32,           // echo-top reflectivity threshold (e.g. 18 dBZ)
    et: *mut f32,       // [npix], in/out (metres, NaN = none)
) -> i32 {
    let npix = npix as usize;
    let nb = nbins as usize;
    for p in 0..npix {
        let bin = crate::grid::find_bin(edges, nbins, *dist.add(p) as f64);
        if bin < 0 {
            continue;
        }
        let z = *data.add(*ray_idx.add(p) as usize * nb + bin as usize);
        if z != z || z < thr {
            continue;
        }
        let h = *heights_bin.add(bin as usize);
        let cur = *et.add(p);
        if cur != cur || h > cur {
            *et.add(p) = h;
        }
    }
    crate::OK
}

/// Gather ONE sweep's per-pixel reflectivity + beam height for the VIL stack.
///
/// Fused replacement for the Python `_lookup_indices` + fancy-index + height
/// gather in `_reflectivity_layers`: one pass over the grid fills both
/// `z_out[p]` (dBZ, NaN where masked/out-of-range) and `h_out[p]` (metres).
/// Out-of-range pixels keep the CLIPPED bin's beam height — exactly like the
/// NumPy path's `np.clip(bin_idx, ...)` — so the column trapezoid weights match.
///
/// # Safety
/// Pointers valid for the stated lengths.
#[no_mangle]
pub unsafe extern "C" fn vil_gather_sweep(
    npix: i32,
    ray_idx: *const i32,
    dist: *const f32,
    data: *const f32,        // [nrays*nbins], NaN = invalid gate
    nbins: i32,
    edges: *const f64,       // [nbins+1]
    heights_bin: *const f32, // [nbins] beam height per bin (metres)
    z_out: *mut f32,         // [npix]
    h_out: *mut f32,         // [npix]
) -> i32 {
    let npix = npix as usize;
    let nb = nbins as usize;
    for p in 0..npix {
        let x = *dist.add(p) as f64;
        let bin = crate::grid::find_bin(edges, nbins, x);
        if bin < 0 {
            let cb = if x <= *edges { 0 } else { nb - 1 };
            *z_out.add(p) = f32::NAN;
            *h_out.add(p) = *heights_bin.add(cb);
        } else {
            let b = bin as usize;
            *z_out.add(p) = *data.add(*ray_idx.add(p) as usize * nb + b);
            *h_out.add(p) = *heights_bin.add(b);
        }
    }
    crate::OK
}

/// Cap applied to reflectivity before Z→M conversion (matches render.VIL_CAP_DBZ).
const VIL_CAP_DBZ: f32 = 56.0;
/// M(Z) = 3.44e-6 * (10^(Z/10))^(4/7) = 3.44e-6 * exp(Z * ln10 * 4/70).
const M_EXP_COEF: f64 = core::f64::consts::LN_10 * 4.0 / 70.0;

/// Liquid-water content [kg/m^3] for one gate; empty (NaN) gates weigh 0,
/// exactly like the NumPy path's `nan_to_num`.
#[inline(always)]
fn lwc(z: f32) -> f64 {
    if z != z {
        0.0
    } else {
        3.44e-6 * ((z.min(VIL_CAP_DBZ) as f64) * M_EXP_COEF).exp()
    }
}

/// Vertically Integrated Liquid over the column, per pixel, into `vil`.
///
/// Inputs are pre-sampled per-pixel stacks (Python samples each sweep to the
/// grid first): `dbz` and `height` are `[nsweeps*npix]` row-major (sweep-major),
/// NaN where a sweep has no gate at that pixel.
///
/// Mirrors `render.sample_column_to_grid` ("VIL" branch) exactly: sort ALL
/// layers by height (empty gates contribute M = 0 but keep their height, like
/// the NumPy `nan_to_num` + full-stack trapezoid), then
///   `vil[p] = sum_i (M_i + M_{i+1})/2 * (h_{i+1}-h_i)`  [kg/m^2].
/// `vil[p] = NaN` when no gate has an echo or the integral is <= 0.05 —
/// the caller masks NaN, matching Python's `vil > 0.05` validity test.
///
/// Fast path: a pixel with no valid dBZ at any tilt (the vast majority of a
/// clear-sky grid) is NaN'd after `nsweeps` loads, skipping sort + exp.
/// Heights arrive nearly sorted (tilts scan low→high), so insertion sort is
/// ~O(n). exp() runs only on gates with echo.
///
/// # Safety
/// Pointers valid for the stated lengths; `nsweeps <= 64`.
#[no_mangle]
pub unsafe extern "C" fn vil_column(
    npix: i32,
    nsweeps: i32,
    dbz: *const f32,    // [nsweeps*npix], NaN = empty
    height: *const f32, // [nsweeps*npix], metres
    vil: *mut f32,      // [npix]
) -> i32 {
    let npix = npix as usize;
    let ns = nsweeps as usize;
    if ns > 64 {
        return NOT_IMPLEMENTED; // stack buffers below are fixed-size
    }
    if ns < 2 {
        // dh is empty in the NumPy path -> vil = 0 -> masked everywhere.
        for p in 0..npix {
            *vil.add(p) = f32::NAN;
        }
        return crate::OK;
    }

    let mut zs = [0.0f32; 64];
    let mut hs = [0.0f32; 64];
    for p in 0..npix {
        // Gather the column; consecutive p makes each sweep a sequential stream.
        let mut any = false;
        for s in 0..ns {
            let z = *dbz.add(s * npix + p);
            any |= z == z;
            *zs.get_unchecked_mut(s) = z;
        }
        if !any {
            *vil.add(p) = f32::NAN;
            continue;
        }
        for s in 0..ns {
            *hs.get_unchecked_mut(s) = *height.add(s * npix + p);
        }

        // Insertion sort (z,h) pairs by ascending height.
        for i in 1..ns {
            let (h, z) = (*hs.get_unchecked(i), *zs.get_unchecked(i));
            let mut j = i;
            while j > 0 && *hs.get_unchecked(j - 1) > h {
                *hs.get_unchecked_mut(j) = *hs.get_unchecked(j - 1);
                *zs.get_unchecked_mut(j) = *zs.get_unchecked(j - 1);
                j -= 1;
            }
            *hs.get_unchecked_mut(j) = h;
            *zs.get_unchecked_mut(j) = z;
        }

        // Trapezoid integration over the full sorted column.
        let mut sum = 0.0f64;
        let mut m_prev = lwc(*zs.get_unchecked(0));
        for i in 1..ns {
            let m = lwc(*zs.get_unchecked(i));
            let dh = (*hs.get_unchecked(i) - *hs.get_unchecked(i - 1)) as f64;
            sum += (m_prev + m) * 0.5 * dh;
            m_prev = m;
        }
        *vil.add(p) = if sum > 0.05 { sum as f32 } else { f32::NAN };
    }
    crate::OK
}
