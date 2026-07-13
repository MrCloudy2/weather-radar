// NaN-aware column-max accumulation for ONE sweep into `best`. Same algorithm
// as cmax.c, exposed with the C ABI so it can be loaded via ctypes.
#[no_mangle]
pub unsafe extern "C" fn cmax_sweep(
    npix: i32,
    ray_idx: *const i32,
    dist: *const f32,
    data: *const f32,
    nbins: i32,
    edges: *const f64,
    best: *mut f32,
) {
    let npix = npix as isize;
    let nbins = nbins as isize;
    let e0 = *edges.offset(0);
    let en = *edges.offset(nbins);
    for p in 0..npix {
        let x = *dist.offset(p) as f64;
        if x <= e0 || x > en {
            continue;
        }
        let (mut lo, mut hi) = (0isize, nbins);
        while lo < hi - 1 {
            let mid = (lo + hi) >> 1;
            if *edges.offset(mid) < x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let r = *ray_idx.offset(p) as isize;
        let v = *data.offset(r * nbins + lo);
        if v == v {
            let cur = *best.offset(p);
            if !(cur == cur) || v > cur {
                *best.offset(p) = v;
            }
        }
    }
}
