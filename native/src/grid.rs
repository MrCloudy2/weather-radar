//! Shared low-level helpers for the polar->grid kernels.

/// Largest bin index `i` with `edges[i] < x`, matching NumPy `searchsorted-1`.
/// Returns -1 if `x` is outside `(edges[0], edges[nbins]]` (out of range).
///
/// # Safety
/// `edges` must point to at least `nbins + 1` valid f64 values.
#[inline]
pub unsafe fn find_bin(edges: *const f64, nbins: i32, x: f64) -> i32 {
    let nb = nbins as isize;
    if x <= *edges.offset(0) || x > *edges.offset(nb) {
        return -1;
    }
    let (mut lo, mut hi) = (0isize, nb);
    while lo < hi - 1 {
        let mid = (lo + hi) >> 1;
        if *edges.offset(mid) < x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo as i32
}

#[inline]
pub fn is_nan(v: f32) -> bool {
    v != v
}
