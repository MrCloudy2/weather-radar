//! Native accelerated kernels for the Slovenia radar backend.
//!
//! Every kernel is exposed with the C ABI (`extern "C"`, `#[no_mangle]`) and
//! loaded from Python via `ctypes` (see `radar/_native.py`). Each returns an
//! `i32` status: **0 = computed, -1 = not implemented** (Python then falls back
//! to the pure-NumPy path — identical result, slower).
//!
//! Conventions shared by all kernels:
//!   * grids are flat row-major `[height*width]` (call it `npix`)
//!   * polar sweeps are flat row-major `[nrays*nbins]`, f32, **NaN = invalid gate**
//!   * `ray_idx[npix]` (i32) and `dist[npix]` (f32, ground metres) are the
//!     precomputed per-pixel radar geometry (constant per radar+grid)
//!   * `edges[nbins+1]` (f64) are monotonic ground-range bin edges
//!   * outputs use NaN for "no value".
//!
//! STATUS: `reflectivity` implemented; the rest are specified stubs to be
//! written next (each doc-comment states the exact algorithm + the Python
//! reference in `radar/`).

pub mod grid;
pub mod reflectivity;
pub mod column;
pub mod velocity;
pub mod analysis;

/// Returned by an unimplemented kernel; Python treats it as "use NumPy".
pub const NOT_IMPLEMENTED: i32 = -1;
pub const OK: i32 = 0;
