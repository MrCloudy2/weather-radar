"""Validate velocity gating + dealiasing and the point sampler."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from radar import config, odim, render  # noqa: E402


def main() -> int:
    path = sorted(config.DATA_DIR.glob("sipas*.h5"))[-1]
    print("file:", path.name)

    raw = render.prepare_sweep(path, "VRADH", 0, dealias=False, gate_clutter=False)
    gated = render.prepare_sweep(path, "VRADH", 0, dealias=False, gate_clutter=True)
    deal = render.prepare_sweep(path, "VRADH", 0, dealias=True, gate_clutter=True)
    geom = odim.sweep_geometry(path, 0)

    print(f"Nyquist NI = {geom.nyquist:.2f} m/s (fold interval {2*geom.nyquist:.1f})")
    print(f"raw    valid={raw.count():6d}  range [{raw.min():+.1f},{raw.max():+.1f}]")
    print(f"gated  valid={gated.count():6d}  (clutter removed via DBZH)")
    print(f"dealia valid={deal.count():6d}  range [{deal.min():+.1f},{deal.max():+.1f}]"
          "  <- should exceed +/-Nyquist if folds were unwrapped")

    # Render dealiased velocity so we can eyeball the improvement.
    res = render.render_sweep(path, "VRADH", 0, width=900, dealias=True)
    out = config.DATA_DIR / "render_sipas_VRADH_dealias.png"
    out.write_bytes(res.png)
    print(f"wrote {out.name}")

    # Sample the site itself and a point ~30 km NE.
    vol = odim.summarize(path)
    for name, (la, lo) in {
        "at site": (vol.site_lat, vol.site_lon),
        "~30km NE": (vol.site_lat + 0.2, vol.site_lon + 0.3),
    }.items():
        s = render.sample_point(path, "DBZH", la, lo, 0)
        print(f"DBZH {name}: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
