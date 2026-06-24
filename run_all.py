"""Windows-compatible one-shot pipeline runner.

Replaces `make all` on systems without GNU Make (e.g., Windows).
Usage: python run_all.py
"""

import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 mode for child processes on Windows before any heavy imports.
os.environ.setdefault("PYTHONUTF8", "1")


def run(cmd: list[str], cwd: Path | None = None):
    print(f"\n{'=' * 60}")
    print(f">>> {' '.join(cmd)}")
    print("=" * 60)
    # Pass cmd as a list and avoid shell=True — no shell injection surface and
    # correct argv splitting even if a path contains spaces.
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"WARNING: Command failed with exit code {result.returncode}")
        return False
    return True


def main():
    here = Path(__file__).resolve().parent

    steps = [
        ("Preprocessing", ["python", "scripts/preprocess.py"]),
        ("Feature Engineering", ["python", "scripts/feature_engineering.py"]),
        ("Model Training", ["python", "scripts/train_models.py"]),
        ("Evaluation", ["python", "scripts/evaluate.py"]),
        ("SHAP Analysis", ["python", "scripts/shap_analysis.py"]),
    ]

    print("Credit Risk Scoring - Full Pipeline")
    print("=" * 60)

    for name, cmd in steps:
        if not run(cmd, cwd=here):
            print(f"\nPipeline stopped at step: {name}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
