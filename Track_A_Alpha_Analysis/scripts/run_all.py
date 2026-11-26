"""Run the whole Track A pipeline in order; stops at the first failing step."""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_data_audit.py", "02_visualize_data.py", "03_rotation_invariance.py", "04_alpha_persistence.py",
         "05_bottleneck_distances.py", "06_class_analysis.py", "07_robustness.py"]

if __name__ == "__main__":
    extra = sys.argv[1:]                          # e.g. --force
    for s in STEPS:
        print(f"\n{'=' * 70}\n>>> {s}\n{'=' * 70}", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, str(HERE / s), *extra], cwd=HERE)
        if r.returncode != 0:
            sys.exit(f"step {s} FAILED (exit code {r.returncode})")
        print(f"--- {s} done in {time.time() - t:.1f}s", flush=True)
    print("\nall steps completed")
