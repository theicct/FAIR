"""Run the complete ICCT FaIR pipeline end-to-end.

Steps:
  1. Preprocess SLCP emissions  (preprocessing/preprocess_slcp_ems.py)
  2. Create custom FaIR scenarios  (create_custom_scenarios.py)
  3. Run FaIR  (examples/run_cmip6.py)

User parameters defined here are passed to all three scripts automatically.
"""

import os
import sys
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── User parameters ──────────────────────────────────────────────────────────
BASE_YR         = 2020               # First year of the scenario period
END_YR          = 2050               # Last year of the scenario period
BASELINE_SCEN   = 'BAU'             # Scenario name used as the baseline in FaIR inputs
BASE_SCEN       = 'BAU'  # Display label for BASELINE_SCEN in output files
BATCH_SIZE      = 100               # Batch size for sensitivity runs
EMS_IN          = 'preprocessing/final/PACE_inventory_long.csv'  # SLCP emissions input for step 2
# ─────────────────────────────────────────────────────────────────────────────

ENV = {
    **os.environ,
    'FAIR_BASE_YR':       str(BASE_YR),
    'FAIR_END_YR':        str(END_YR),
    'FAIR_BASELINE_SCEN': BASELINE_SCEN,
    'FAIR_BASE_SCEN':     BASE_SCEN,
    'FAIR_BATCH_SIZE':    str(BATCH_SIZE),
    'FAIR_EMS_IN':        EMS_IN,
}


def run_step(label, script_path, cwd):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    t0 = time.time()
    subprocess.run([sys.executable, script_path], cwd=cwd, env=ENV, check=True)
    print(f"  Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == '__main__':
    total_start = time.time()

    run_step(
        'Step 1/3 — Preprocess SLCP emissions',
        os.path.join(REPO_ROOT, 'preprocessing', 'preprocess_slcp_ems.py'),
        os.path.join(REPO_ROOT, 'preprocessing'),
    )
    run_step(
        'Step 2/3 — Create custom FaIR scenarios',
        os.path.join(REPO_ROOT, 'create_custom_scenarios.py'),
        REPO_ROOT,
    )
    run_step(
        'Step 3/3 — Run FaIR',
        os.path.join(REPO_ROOT, 'examples', 'run_cmip6.py'),
        os.path.join(REPO_ROOT, 'examples'),
    )

    print(f"\nFull pipeline completed in {(time.time() - total_start) / 60:.1f} min")
