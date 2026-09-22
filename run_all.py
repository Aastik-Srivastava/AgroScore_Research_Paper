#!/usr/bin/env python3
"""Run the complete AgroScore reproduction pipeline from a clean checkout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    print(f"\n=== Running {script} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


def main() -> None:
    run("agroscore_extended.py")
    run("part4.py")
    run("regime_benchmark.py")
    run("make_figures2.py")
    run("stress_test.py")
    run("shap_conformance.py")
    run("manifold_diagnostics.py")
    run("task_diversity.py")
    run("test_agroscore.py")
    print("\nReproduction complete. See results2/ and figures/.", flush=True)


if __name__ == "__main__":
    main()
