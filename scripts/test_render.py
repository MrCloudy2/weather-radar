"""Phase 2 / Step A validation: render sweeps to PNGs and print their bounds.

Writes data/render_<node>_<quantity>.png for the lowest tilt so you can eyeball
that the storm lands over Slovenia and the colours make sense.

    .venv/bin/python scripts/test_render.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar import config, render  # noqa: E402


def main() -> int:
    files = sorted(config.DATA_DIR.glob("*.h5"))
    if not files:
        print("No files. Run scripts/test_fetch.py first.")
        return 1

    # One file per node, lowest tilt, all three moments.
    seen: dict[str, Path] = {}
    for f in files:
        node = f.name.split("@")[0]
        seen[node] = f  # keep latest per node (sorted ascending)

    for node, path in seen.items():
        for q in ("DBZH", "VRADH"):
            t0 = time.perf_counter()
            res = render.render_sweep(path, q, sweep_index=0, width=900)
            dt = (time.perf_counter() - t0) * 1000
            out = config.DATA_DIR / f"render_{node}_{q}.png"
            out.write_bytes(res.png)
            (s, w), (n, e) = res.bounds
            print(
                f"{node} {q:<6} {res.width}x{res.height}  "
                f"bounds S{s:.3f} W{w:.3f} N{n:.3f} E{e:.3f}  "
                f"{len(res.png)/1024:.0f} KiB  {dt:.0f} ms -> {out.name}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
