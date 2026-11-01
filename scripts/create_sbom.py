"""Write a JSON inventory of installed Python distributions and licenses.

Run this inside the clean AegisVoice virtual environment after installation.
The inventory reflects that environment only; it does not audit packages.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def build_inventory() -> dict:
    packages = []
    for distribution in metadata.distributions():
        package_metadata = distribution.metadata
        name = package_metadata.get("Name") or "unknown"
        packages.append({
            "name": name,
            "version": distribution.version,
            "license": package_metadata.get("License-Expression") or package_metadata.get("License") or "UNKNOWN",
            "requires_dist": sorted(distribution.requires or []),
        })
    packages.sort(key=lambda item: item["name"].casefold())
    return {
        "format": "AegisVoice installed Python package inventory v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "notice": "Lists distributions visible to this Python environment. Review each license and the upstream security notices before redistribution.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/software-inventory.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory()
    args.output.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote inventory for {len(inventory['packages'])} installed distributions to {args.output}")


if __name__ == "__main__":
    main()
