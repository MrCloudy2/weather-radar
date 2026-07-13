"""Phase 1 / Layer 2 validation: parse a downloaded ODIM volume and print its
metadata + radar quantities, then decode one sweep to real physical units.

Usage:
    .venv/bin/python scripts/inspect_odim.py [path-to.h5]
If no path is given, the newest file in data/ is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from radar import config, odim  # noqa: E402


def _newest_local() -> Path | None:
    files = sorted(config.DATA_DIR.glob("*.h5"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _newest_local()
    if not path or not path.exists():
        print("No file to inspect. Run scripts/test_fetch.py first.")
        return 1

    vol = odim.summarize(path)
    print(f"File     : {Path(vol.path).name}")
    print(f"Object   : {vol.object_type}   Source: {vol.source}")
    print(f"Time     : {vol.nominal_time}")
    print(
        f"Site     : lat {vol.site_lat:.4f}, lon {vol.site_lon:.4f}, "
        f"height {vol.site_height_m:.0f} m"
    )
    print(f"Sweeps   : {len(vol.sweeps)}")
    print(f"Quantities present: {', '.join(vol.quantities)}")

    print("\nElevation sweeps:")
    print(f"  {'#':>2}  {'elev':>5}  {'rays':>4}  {'bins':>4}  "
          f"{'range':>7}  moments")
    for s in vol.sweeps:
        print(f"  {s.index:>2}  {s.elangle:>5.1f}  {s.nrays:>4}  {s.nbins:>4}  "
              f"{s.max_range_km:>5.0f}km  {', '.join(s.moments)}")

    # Show the raw->physical decode for each quantity on the lowest tilt.
    print("\nLowest tilt (sweep 0) decoded stats:")
    s0 = vol.sweeps[0]
    for q, m in s0.moments.items():
        data = odim.read_sweep(path, q, 0)
        valid = data.compressed()
        unit = {"DBZH": "dBZ", "TH": "dBZ", "VRADH": "m/s"}.get(q, "")
        pct = 100.0 * valid.size / data.size if data.size else 0.0
        if valid.size:
            print(
                f"  {q:<6} gain={m.gain:+.4f} offset={m.offset:+.2f} "
                f"nodata={m.nodata:.0f} undetect={m.undetect:.0f} | "
                f"valid {pct:4.1f}%  "
                f"min {valid.min():+7.2f}  max {valid.max():+7.2f}  "
                f"mean {valid.mean():+6.2f} {unit}"
            )
        else:
            print(f"  {q:<6} (no valid gates this sweep)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
