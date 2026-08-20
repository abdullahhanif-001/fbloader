#!/usr/bin/env python3
"""Fail CI when fbload throughput regresses against PyTorch baseline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_smoke.json"
MIN_RATIO = 0.90


def main() -> int:
    cmd = [
        sys.executable,
        str(ROOT / "tests" / "benchmark_soak.py"),
        "--steps",
        "200",
        "--quick",
        "--json-out",
        str(OUT),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    data = json.loads(OUT.read_text(encoding="utf-8"))
    fb = data["fbload"]
    pt = data["pytorch"]
    ratio = fb["throughput_ips"] / max(1e-9, pt["throughput_ips"])
    print(f"perf gate: fbload={fb['throughput_ips']:.1f} pytorch={pt['throughput_ips']:.1f} ratio={ratio:.2f}")
    if ratio < MIN_RATIO:
        print(f"FAIL: fbload throughput below {MIN_RATIO:.0%} of tuned PyTorch baseline")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
