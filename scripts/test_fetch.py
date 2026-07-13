"""Phase 1 / Layer 1 validation: list + download latest Slovenian radar files.

Run from the project root with the venv active:
    .venv/bin/python scripts/test_fetch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script (add project root to path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar import config, fetcher  # noqa: E402


def main() -> int:
    for node, name in config.NODES.items():
        print(f"\n=== {name} ({node}) ===")
        latest = fetcher.latest_file(node)
        if not latest:
            print("  no files found (bucket may be empty for this node)")
            continue
        print(f"  latest: {latest}")
        print(f"  key   : {latest.key}")

        # Show a few recent scans to confirm the ~5 min cadence.
        recent = fetcher.list_node_files(node)[-4:]
        print("  recent cadence:")
        for f in recent:
            print(f"    {f.time:%H:%M:%S}Z  {'+'.join(f.moments)}")

        local = fetcher.download(latest)
        print(f"  downloaded -> {local}  ({local.stat().st_size/1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
