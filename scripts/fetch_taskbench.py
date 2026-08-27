"""Fetch TaskBench agent task-decomposition graphs.

Source: https://huggingface.co/datasets/microsoft/Taskbench (MIT licensed).
Roughly 24 MB across three tool domains.

    .venv/bin/python scripts/fetch_taskbench.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/microsoft/Taskbench/resolve/main"
DOMAINS = ("dailylifeapis", "multimedia", "huggingface")
DEST = Path(__file__).resolve().parents[1] / "data" / "taskbench"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        target = DEST / f"{domain}.json"
        if target.exists():
            print(f"  {domain}: already present")
            continue
        url = f"{BASE}/data_{domain}/data.json"
        with urllib.request.urlopen(url, timeout=300) as response:
            target.write_bytes(response.read())
        print(f"  {domain}: {target.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Done, in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
