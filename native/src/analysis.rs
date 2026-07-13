//! Temporal / multi-radar analysis: storm motion + dual-Doppler.
//!
//! Python reference: `radar/analysis.py` (`motion_field`, `dual_doppler`).

/// Block cross-correlation storm motion between two reflectivity grids.
///
/// For each `block`x`block` tile of `curr` whose max >= `min_dbz`, search
/// offsets `dy,dx` in `[-search, search]` (step `step`, dy-outer / dx-inner
/// with strict `<` — same first-minimum tie-break as the Python loop)
/// minimising SSD against `prev`. Tiles whose best offset is (0,0) are
/// skipped, like Python. Writes one vector per moving tile (up to `max_out`)
/// and returns the count; the caller negates the offset and converts to m/s.
///
/// Outputs (caller-allocated, length `max_out`): `cx,cy` = tile-centre pixel;
/// `dx,dy` = raw best offset in pixels (curr tile matched prev at +dx,+dy).
///
/// # Safety
/// `prev`/`curr` valid for `h*w`; output arrays valid for `max_out`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn motion_ssd(
    h: i32,
    w: i32,
    prev: *const f32,
    curr: *const f32,
    block: i32,
    search: i32,
    step: i32,
    min_dbz: f32,
    cx: *mut i32,
    cy: *mut i32,
    dx: *mut f32,
    dy: *mut f32,
    max_out: i32,
) -> i32 {
    let (h, w) = (h as usize, w as usize);
    let block = block as usize;
    if block == 0 || step <= 0 || h < block || w < block {
        return 0;
    }
    let inv_n = 1.0 / (block * block) as f64;
    let mut count: i32 = 0;

    let mut by = 0;
    while by + block < h {
        let mut bx = 0;
        while bx + block < w {
            // Skip tiles with no echo (the vast majority of the grid).
            let mut tile_max = f32::NEG_INFINITY;
            for y in 0..block {
                let row = curr.add((by + y) * w + bx);
                for x in 0..block {
                    let v = *row.add(x);
                    if v > tile_max {
                        tile_max = v;
                    }
                }
            }
            if tile_max < min_dbz {
                bx += block;
                continue;
            }

            let mut best = (f64::INFINITY, 0i32, 0i32); // (ssd, dx, dy)
            let mut ody = -search;
            while ody <= search {
                let y0 = by as i64 + ody as i64;
                if y0 < 0 || y0 as usize + block > h {
                    ody += step;
                    continue;
                }
                let mut odx = -search;
                while odx <= search {
                    let x0 = bx as i64 + odx as i64;
                    if x0 < 0 || x0 as usize + block > w {
                        odx += step;
                        continue;
                    }
                    let mut ssd = 0.0f64;
                    for y in 0..block {
                        let pr = curr.add((by + y) * w + bx);
                        let rr = prev.add((y0 as usize + y) * w + x0 as usize);
                        for x in 0..block {
                            let d = (*pr.add(x) - *rr.add(x)) as f64;
                            ssd += d * d;
                        }
                    }
                    ssd *= inv_n;
                    if ssd < best.0 {
                        best = (ssd, odx, ody);
                    }
                    odx += step;
                }
                ody += step;
            }
            let (_, bdx, bdy) = best;
            if !(bdx == 0 && bdy == 0) && count < max_out {
                *cx.add(count as usize) = (bx + block / 2) as i32;
                *cy.add(count as usize) = (by + block / 2) as i32;
                *dx.add(count as usize) = bdx as f32;
                *dy.add(count as usize) = bdy as f32;
                count += 1;
            }
            bx += block;
        }
        by += block;
    }
    count
}

/// Dual-Doppler horizontal wind from two radars' radial velocities, per pixel.
///
/// `az1`,`az2` are the compass bearings (degrees) from each radar to each pixel.
/// Solve `Vr = u*sin(az) + v*cos(az)` as a 2x2 system:
///   `det = sin(az1)cos(az2) - sin(az2)cos(az1)`
///   `u = (v1*cos(az2) - v2*cos(az1))/det`,  `v = (sin(az1)*v2 - sin(az2)*v1)/det`
/// NaN where either input is NaN or `|sin(az1-az2)| < sin(min_cross_deg)`
/// (beam-crossing too shallow — inversion unstable). Writes `u`,`v` (m/s) and
/// `speed = hypot(u,v)`. Trig runs in f64 only on pixels with echo.
///
/// # Safety
/// All arrays valid for `npix`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn dual_doppler(
    npix: i32,
    v1: *const f32,
    v2: *const f32,
    az1: *const f32, // degrees
    az2: *const f32,
    min_cross_deg: f32,
    u: *mut f32,
    v: *mut f32,
    speed: *mut f32,
) -> i32 {
    let npix = npix as usize;
    let min_cross = (min_cross_deg as f64).to_radians().sin();
    for p in 0..npix {
        let r1 = *v1.add(p);
        let r2 = *v2.add(p);
        // Fast path: any NaN input -> masked pixel, no trig.
        if r1 != r1 || r2 != r2 {
            *u.add(p) = f32::NAN;
            *v.add(p) = f32::NAN;
            *speed.add(p) = f32::NAN;
            continue;
        }
        let t1 = (*az1.add(p) as f64).to_radians();
        let t2 = (*az2.add(p) as f64).to_radians();
        let (s1, c1) = t1.sin_cos();
        let (s2, c2) = t2.sin_cos();
        let det = s1 * c2 - s2 * c1; // == sin(az1 - az2)
        if det.abs() < min_cross {
            *u.add(p) = f32::NAN;
            *v.add(p) = f32::NAN;
            *speed.add(p) = f32::NAN;
            continue;
        }
        let uu = (r1 as f64 * c2 - r2 as f64 * c1) / det;
        let vv = (s1 * r2 as f64 - s2 * r1 as f64) / det;
        *u.add(p) = uu as f32;
        *v.add(p) = vv as f32;
        *speed.add(p) = uu.hypot(vv) as f32;
    }
    crate::OK
}
