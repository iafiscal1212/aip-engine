"""
PHP-E(5) d=10 — Column Generation Solver for Colab
====================================================

Runs the column generation solver for PHP-Entangled(5) at degree 10.
Estimated: ~110 GB RAM, ~4-8 hours on Colab (227 GB).

Usage:
  pip install aip-engine numba psutil
  python run_phpe5_colgen.py

Author: Carmen Esteban
"""

import time
import json
import os
import sys
import gc
import threading

# Install deps if needed (Colab)
try:
    import psutil
except ImportError:
    os.system("pip install -q psutil")
    import psutil

try:
    from numba import njit
except ImportError:
    os.system("pip install -q numba")

from aip.colgen_solver import solve_phpe_colgen, build_axioms_phpe, enumerate_column_groups
from aip.accordion import PascalIndex

# ── Keep-alive ──
_keepalive_stop = threading.Event()
def _keepalive_worker():
    while not _keepalive_stop.is_set():
        _keepalive_stop.wait(60)
        if not _keepalive_stop.is_set():
            ram = psutil.virtual_memory()
            print(f"  [keepalive] {time.strftime('%H:%M:%S')} - "
                  f"RAM: {ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB ({ram.percent}%)")
            sys.stdout.flush()

_keepalive_thread = threading.Thread(target=_keepalive_worker, daemon=True)
_keepalive_thread.start()

# ── Config ──
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ram_total = psutil.virtual_memory().total / 1e9
print(f"RAM total: {ram_total:.1f} GB")
print(f"RAM required: ~110 GB")

if ram_total < 100:
    print(f"WARNING: RAM ({ram_total:.0f} GB) may be insufficient. Need ~110 GB.")

# ── Pre-analysis ──
print("\nPre-analysis...")
axioms, num_vars = build_axioms_phpe(5)
pidx = PascalIndex(num_vars, 10)
groups, total_cols = enumerate_column_groups(axioms, num_vars, 10, pidx)
print(f"  Variables: {num_vars}")
print(f"  Axioms: {len(axioms)}")
print(f"  Rows (monomials): {pidx.total_monomials():,}")
print(f"  Total columns: {total_cols:,}")
print(f"  Column groups: {len(groups)}")
del axioms, pidx, groups
gc.collect()

# ── Solve ──
print(f"\n{'='*70}")
print(f"  PHP-E(5) d=10 — Column Generation")
print(f"{'='*70}")

t0 = time.time()
result = solve_phpe_colgen(
    n=5,
    max_degree=10,
    initial_mult_degree=3,
    lsqr_max_iter=5000,
    pricing_top_k=100_000,
    pricing_batch_size=5_000_000,
    feasibility_tol=1e-6,
    max_cg_iters=30,
    verbose=True,
)
total_time = time.time() - t0

# ── Save results ──
result_save = {k: v for k, v in result.items() if k != 'x'}
result_save['total_time_hours'] = total_time / 3600
result_save['ram_total_gb'] = ram_total

results_file = os.path.join(RESULTS_DIR, "phpe5_d10_colgen.json")
with open(results_file, 'w') as f:
    json.dump(result_save, f, indent=2, default=str)
print(f"\nResults saved to {results_file}")

# ── Summary ──
print(f"\n{'='*70}")
print(f"  RESULT: {'FEASIBLE' if result['feasible'] else 'INFEASIBLE'}")
print(f"  Residual: {result['residual']:.2e}")
print(f"  CG iterations: {result['iterations']}")
print(f"  Active columns: {result['active_cols']:,}")
print(f"  Non-zeros: {result['size_l2']:,}")
print(f"  Time: {total_time/3600:.2f} hours")
print(f"{'='*70}")

_keepalive_stop.set()
