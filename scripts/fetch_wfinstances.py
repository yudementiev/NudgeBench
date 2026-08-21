"""Fetch a representative sample of real scientific-workflow executions.

Source: WfCommons WfInstances, https://github.com/wfcommons/WfInstances - open
traces of production runs in the WfFormat schema, released under LGPL-3.0. The
full Pegasus collection is about 275 MB, most of it in a handful of very large
runs, so this takes the smaller instances of each workflow type: enough shape
diversity to measure, without the bulk.

    .venv/bin/python scripts/fetch_wfinstances.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

TREE_API = "https://api.github.com/repos/wfcommons/WfInstances/git/trees/HEAD?recursive=1"
RAW = "https://raw.githubusercontent.com/wfcommons/WfInstances/main"
DEST = Path(__file__).resolve().parents[1] / "data" / "workflows"

PER_TYPE = 6
SIZE_CAP = 2_000_000


def listing() -> list[dict]:
    with urllib.request.urlopen(TREE_API, timeout=60) as response:
        tree = json.load(response)["tree"]
    return [
        entry
        for entry in tree
        if entry["type"] == "blob"
        and entry["path"].startswith("pegasus/")
        and entry["path"].endswith(".json")
    ]


def select(entries: list[dict]) -> list[dict]:
    """The smallest few runs of each workflow type, under the size cap."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_type[entry["path"].split("/")[1]].append(entry)

    chosen: list[dict] = []
    for workflow_type in sorted(by_type):
        candidates = sorted(by_type[workflow_type], key=lambda e: e["size"])
        chosen.extend([c for c in candidates if c["size"] <= SIZE_CAP][:PER_TYPE])
    return chosen


def main() -> None:
    entries = select(listing())
    total = sum(entry["size"] for entry in entries)
    print(f"Fetching {len(entries)} workflow instances ({total / 1024 / 1024:.1f} MB)")

    DEST.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        workflow_type, filename = entry["path"].split("/")[1:]
        target = DEST / workflow_type / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        url = f"{RAW}/{entry['path'].replace(' ', '%20')}"
        with urllib.request.urlopen(url, timeout=120) as response:
            target.write_bytes(response.read())
        print(f"  {entry['path']}  ({entry['size'] / 1024:.0f} KB)")

    print(f"Done. {sum(1 for _ in DEST.rglob('*.json'))} instances in {DEST}")


if __name__ == "__main__":
    sys.exit(main())
