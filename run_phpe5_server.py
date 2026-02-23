#!/usr/bin/env python3
"""
PHP-E(5) d=10 — Column Generation con float32.

Runner para servidor iafiscal (125 GB RAM, EPYC 24T).
Usa float32 para reducir memoria de ~105 GB a ~53 GB.

Límites de seguridad:
  - Max 90 GB RAM (quedan 35 GB para servicios)
  - nice -n 19 (prioridad baja)
  - 8 threads Numba (de 24 disponibles)

Uso:
  nice -n 19 /opt/aip/venv/bin/python run_phpe5_server.py

Autora: Carmen Esteban
"""

import os
import sys
import gc
import time
import json
import resource
import signal
import numpy as np
from datetime import datetime
from itertools import combinations
from scipy.sparse.linalg import lsqr, LinearOperator

# ── Límites de seguridad ──
MAX_RAM_GB = 90
MAX_RAM_BYTES = MAX_RAM_GB * 1024**3

# Limitar RAM con ulimit
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (MAX_RAM_BYTES, hard))
print(f"  RAM limit: {MAX_RAM_GB} GB (soft ulimit)")

# Limitar threads
os.environ['NUMBA_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

# Dtype global
DTYPE = np.float32
DTYPE_NAME = 'float32'

# ── Importar Numba ──
try:
    from numba import njit, prange
    HAS_NUMBA = True
    print(f"  Numba: OK (8 threads)")
except ImportError:
    HAS_NUMBA = False
    print(f"  Numba: NOT FOUND (will be slow)")
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator
    def prange(*args):
        return range(*args)


# ============================================================
# PascalIndex — standalone (no dependency on aip package)
# ============================================================

class PascalIndex:
    """Combinatorial number system indexer."""

    def __init__(self, num_vars, max_degree):
        self.num_vars = num_vars
        self.max_degree = max_degree
        self._build_pascal()
        self._build_offsets()

    def _build_pascal(self):
        n = self.num_vars + 1
        k = self.max_degree + 1
        self.pascal = np.zeros((n, k), dtype=np.int64)
        for i in range(n):
            self.pascal[i, 0] = 1
            for j in range(1, min(i + 1, k)):
                self.pascal[i, j] = self.pascal[i-1, j-1] + self.pascal[i-1, j]

    def _build_offsets(self):
        self.offsets = np.zeros(self.max_degree + 2, dtype=np.int64)
        total = 0
        for d in range(self.max_degree + 1):
            self.offsets[d] = total
            total += int(self.pascal[self.num_vars, d])
        self.offsets[self.max_degree + 1] = total
        self._total = int(total)

    def total_monomials(self):
        return self._total

    def pack_for_numba(self):
        return {
            'pascal': self.pascal,
            'offsets': self.offsets,
            'num_vars': self.num_vars,
            'max_degree': self.max_degree,
        }


# ============================================================
# Numba kernels
# ============================================================

@njit(cache=True)
def _lex_rank_jit(combo, pascal, num_vars):
    rank = np.int64(0)
    prev = np.int64(-1)
    for i in range(len(combo)):
        c = combo[i]
        remaining = len(combo) - 1 - i
        for a in range(prev + 1, c):
            available = num_vars - 1 - a
            if available >= remaining >= 0:
                rank += pascal[available, remaining]
        prev = c
    return rank


@njit(cache=True)
def union_sorted(a, b):
    result = np.empty(len(a) + len(b), dtype=np.int64)
    ia, ib, ir = 0, 0, 0
    while ia < len(a) and ib < len(b):
        if a[ia] < b[ib]:
            result[ir] = a[ia]; ia += 1; ir += 1
        elif a[ia] > b[ib]:
            result[ir] = b[ib]; ib += 1; ir += 1
        else:
            result[ir] = a[ia]; ia += 1; ib += 1; ir += 1
    while ia < len(a):
        result[ir] = a[ia]; ia += 1; ir += 1
    while ib < len(b):
        result[ir] = b[ib]; ib += 1; ir += 1
    return result[:ir]


@njit(cache=True)
def pack_combos(combo_list_flat, combo_list_starts, combo_list_lengths):
    """Numba-compatible pack_combos that takes pre-flattened data."""
    return combo_list_flat, combo_list_starts, combo_list_lengths


def pack_combos_py(combos):
    """Pack list of tuple combos into flat arrays."""
    flat = []
    starts = []
    lengths = []
    for c in combos:
        starts.append(len(flat))
        lengths.append(len(c))
        flat.extend(c)
    return (np.array(flat, dtype=np.int64),
            np.array(starts, dtype=np.int64),
            np.array(lengths, dtype=np.int64))


@njit(cache=True)
def _unrank_combination(rank, k, num_vars, pascal):
    combo = np.empty(k, dtype=np.int64)
    r = rank
    prev = np.int64(-1)
    for i in range(k):
        remaining = k - 1 - i
        for a in range(prev + 1, num_vars):
            available = num_vars - 1 - a
            if available >= remaining >= 0:
                count = pascal[available, remaining]
            else:
                count = np.int64(0)
            if r < count:
                combo[i] = a
                prev = a
                break
            r -= count
    return combo


@njit(cache=True)
def _price_group_batch_kernel(
    batch_start, batch_size,
    ax_flat, ax_starts, ax_lengths, ax_coeffs,
    k_mult, num_vars,
    pascal, offsets, max_degree,
    residual
):
    n_terms = len(ax_starts)
    correlations = np.empty(batch_size, dtype=np.float64)

    for local_j in range(batch_size):
        rank = batch_start + local_j
        mult_combo = _unrank_combination(rank, k_mult, num_vars, pascal)

        corr = 0.0
        for i in range(n_terms):
            a_start = ax_starts[i]
            a_len = ax_lengths[i]
            coeff = ax_coeffs[i]
            a_combo = ax_flat[a_start:a_start + a_len]
            product = union_sorted(a_combo, mult_combo)

            if len(product) > max_degree:
                continue

            row_idx = offsets[len(product)] + _lex_rank_jit(
                product, pascal, num_vars
            )
            corr += coeff * residual[row_idx]

        correlations[local_j] = abs(corr)

    return correlations


# ============================================================
# Build PHP-E axioms
# ============================================================

def build_axioms_phpe(n):
    pigeons = list(range(1, n + 2))
    holes = list(range(1, n + 1))

    var_x = {}
    idx = 0
    for p in pigeons:
        for h in holes:
            var_x[(p, h)] = idx
            idx += 1

    var_y = {}
    for i, p in enumerate(pigeons):
        for p2 in pigeons[i + 1:]:
            var_y[(p, p2)] = idx
            idx += 1
    num_vars = idx

    axioms = []

    # 1. Existence
    for p in pigeons:
        hvars = [var_x[(p, h)] for h in holes]
        terms = []
        for k in range(len(hvars) + 1):
            for subset in combinations(hvars, k):
                terms.append(((-1.0) ** k, tuple(sorted(subset))))
        axioms.append(terms)

    # 2. Hole exclusion
    for h in holes:
        for i, p in enumerate(pigeons):
            for p2 in pigeons[i + 1:]:
                axioms.append([(1.0, tuple(sorted([var_x[(p, h)], var_x[(p2, h)]])))])

    # 3. Functionality
    for p in pigeons:
        for j, h in enumerate(holes):
            for h2 in holes[j + 1:]:
                axioms.append([(1.0, tuple(sorted([var_x[(p, h)], var_x[(p, h2)]])))])

    # 4. Ordering
    for i_p, p in enumerate(pigeons):
        for p2 in pigeons[i_p + 1:]:
            y_idx = var_y[(p, p2)]
            for h in holes:
                for h2 in holes:
                    if h == h2:
                        continue
                    x1, x2 = var_x[(p, h)], var_x[(p2, h2)]
                    if h < h2:
                        m1 = tuple(sorted([x1, x2]))
                        m2 = tuple(sorted([x1, x2, y_idx]))
                        axioms.append([(1.0, m1), (-1.0, m2)])
                    else:
                        axioms.append([(1.0, tuple(sorted([x1, x2, y_idx])))])

    # 5. Transitivity
    for i_p, p in enumerate(pigeons):
        for j_p, p2 in enumerate(pigeons[i_p + 1:], i_p + 1):
            for p3 in pigeons[j_p + 1:]:
                y12, y23, y13 = var_y[(p, p2)], var_y[(p2, p3)], var_y[(p, p3)]
                m_a = tuple(sorted([y12, y23]))
                m_b = tuple(sorted([y12, y23, y13]))
                axioms.append([(1.0, m_a), (-1.0, m_b)])
                m_c = tuple(sorted([y13]))
                m_d = tuple(sorted([y12, y13]))
                m_e = tuple(sorted([y23, y13]))
                axioms.append([(1.0, m_c), (-1.0, m_d), (-1.0, m_e), (1.0, m_b)])

    return axioms, num_vars


# ============================================================
# ActiveSet with float32 support
# ============================================================

class ActiveSet:
    def __init__(self, axioms, num_vars, max_degree, pidx, dtype=np.float32):
        self.axioms = axioms
        self.num_vars = num_vars
        self.max_degree = max_degree
        self.pidx = pidx
        self.dtype = dtype
        self.pack = pidx.pack_for_numba()

        self._ax_packs = []
        for ax_terms in axioms:
            combos = [m for c, m in ax_terms]
            coeffs = np.array([c for c, m in ax_terms], dtype=np.float64)
            flat, starts, lengths = pack_combos_py(combos)
            self._ax_packs.append((flat, starts, lengths, coeffs))

        self.col_ax_idx = []
        self.col_mult_combo = []
        self.n_active = 0
        self._rows_per_col = None
        self._vals_per_col = None
        self._dirty = True

    def add_initial_columns(self, max_mult_degree=3, verbose=True):
        t0 = time.time()
        count = 0
        for ax_idx, ax_terms in enumerate(self.axioms):
            deg_ax = max(len(m) for c, m in ax_terms)
            max_md = min(max_mult_degree, self.max_degree - deg_ax)
            for d in range(max_md + 1):
                for combo in combinations(range(self.num_vars), d):
                    self.col_ax_idx.append(ax_idx)
                    self.col_mult_combo.append(combo)
                    count += 1

        self.n_active = len(self.col_ax_idx)
        self._dirty = True
        if verbose:
            print(f"  ActiveSet: {self.n_active:,} initial columns "
                  f"(mult_deg <= {max_mult_degree}) [{time.time()-t0:.1f}s]")
            sys.stdout.flush()

    def add_columns_from_pricing(self, pricing_results, verbose=True):
        pascal = self.pack['pascal']
        added = 0
        existing = set()
        for i in range(self.n_active):
            existing.add((self.col_ax_idx[i], self.col_mult_combo[i]))

        for corr, ax_idx, k_mult, mult_rank in pricing_results:
            combo_arr = _unrank_combination(
                np.int64(mult_rank), k_mult, self.num_vars, pascal
            )
            combo = tuple(combo_arr)
            key = (ax_idx, combo)
            if key in existing:
                continue
            self.col_ax_idx.append(ax_idx)
            self.col_mult_combo.append(combo)
            existing.add(key)
            added += 1

        self.n_active = len(self.col_ax_idx)
        self._dirty = True
        if verbose:
            print(f"  ActiveSet: added {added:,} columns, total {self.n_active:,}")
            sys.stdout.flush()
        return added

    def _build_column_data(self):
        if not self._dirty:
            return
        pascal = self.pack['pascal']
        num_vars = self.pack['num_vars']
        offsets = self.pack['offsets']
        max_degree = self.pack['max_degree']

        rows_per_col = []
        vals_per_col = []

        t0 = time.time()
        for j in range(self.n_active):
            ax_idx = self.col_ax_idx[j]
            mult_combo = self.col_mult_combo[j]
            ax_flat, ax_starts, ax_lengths, ax_coeffs = self._ax_packs[ax_idx]
            mult_arr = np.array(mult_combo, dtype=np.int64)

            rows = []
            vals = []
            for i in range(len(ax_starts)):
                a_start = ax_starts[i]
                a_len = ax_lengths[i]
                coeff = ax_coeffs[i]
                a_combo = ax_flat[a_start:a_start + a_len]
                product = union_sorted(a_combo, mult_arr)
                if len(product) > max_degree:
                    continue
                row_idx = int(offsets[len(product)] + _lex_rank_jit(
                    product, pascal, num_vars
                ))
                rows.append(row_idx)
                vals.append(coeff)

            rows_per_col.append(np.array(rows, dtype=np.int64))
            vals_per_col.append(np.array(vals, dtype=self.dtype))

        self._rows_per_col = rows_per_col
        self._vals_per_col = vals_per_col
        self._dirty = False
        elapsed = time.time() - t0
        print(f"  Column data built [{elapsed:.1f}s]")
        sys.stdout.flush()

    def make_linear_operator(self, num_rows):
        self._build_column_data()
        n = self.n_active
        m = num_rows
        rows_per_col = self._rows_per_col
        vals_per_col = self._vals_per_col
        dtype = self.dtype

        def matvec(x):
            result = np.zeros(m, dtype=dtype)
            for j in range(n):
                if x[j] == 0.0:
                    continue
                rows = rows_per_col[j]
                vals = vals_per_col[j]
                xj = x[j]
                for idx in range(len(rows)):
                    result[rows[idx]] += vals[idx] * xj
            return result

        def rmatvec(y):
            result = np.zeros(n, dtype=dtype)
            for j in range(n):
                rows = rows_per_col[j]
                vals = vals_per_col[j]
                dot = dtype(0)
                for idx in range(len(rows)):
                    dot += vals[idx] * y[rows[idx]]
                result[j] = dot
            return result

        return LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=dtype)


# ============================================================
# Pricing
# ============================================================

def price_all_columns(axioms, num_vars, max_degree, residual, pidx,
                      top_k=100_000, batch_size=5_000_000, verbose=True):
    import heapq
    pack = pidx.pack_for_numba()
    pascal = pack['pascal']
    offsets = pack['offsets']

    # Use float64 residual for pricing accuracy
    residual_f64 = residual.astype(np.float64) if residual.dtype != np.float64 else residual

    heap = []
    total = 0
    t0 = time.time()

    for ax_idx, ax_terms in enumerate(axioms):
        deg_ax = max(len(m) for c, m in ax_terms)
        max_mult_deg = max(0, max_degree - deg_ax)

        ax_combos = [m for c, m in ax_terms]
        ax_coeffs = np.array([c for c, m in ax_terms], dtype=np.float64)
        ax_flat, ax_starts, ax_lengths = pack_combos_py(ax_combos)

        for k_mult in range(max_mult_deg + 1):
            n_mult = int(pascal[num_vars, k_mult])
            if n_mult == 0:
                continue

            for b_start in range(0, n_mult, batch_size):
                b_size = min(batch_size, n_mult - b_start)
                correlations = _price_group_batch_kernel(
                    b_start, b_size,
                    ax_flat, ax_starts, ax_lengths, ax_coeffs,
                    k_mult, num_vars,
                    pascal, offsets, max_degree,
                    residual_f64
                )

                for local_j in range(b_size):
                    corr_val = correlations[local_j]
                    if corr_val < 1e-15:
                        continue
                    entry = (corr_val, ax_idx, k_mult, b_start + local_j)
                    if len(heap) < top_k:
                        heapq.heappush(heap, entry)
                    elif corr_val > heap[0][0]:
                        heapq.heapreplace(heap, entry)

                total += b_size

        if verbose and (ax_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = total / max(elapsed, 1e-6)
            print(f"    Pricing: axiom {ax_idx+1}/{len(axioms)}, "
                  f"cols={total:,.0f}, rate={rate:,.0f}/s, heap={len(heap)}")
            sys.stdout.flush()

    if verbose:
        elapsed = time.time() - t0
        print(f"    Pricing done: {total:,.0f} cols in {elapsed:.1f}s, "
              f"top-{len(heap)} selected")
        sys.stdout.flush()

    heap.sort(key=lambda x: -x[0])
    return heap


# ============================================================
# Column Generation Solver
# ============================================================

class ColumnGenSolver:
    def __init__(self, axioms, num_vars, max_degree, dtype=np.float32,
                 initial_mult_degree=3, lsqr_max_iter=5000,
                 lsqr_tol=1e-6, pricing_top_k=100_000,
                 pricing_batch_size=5_000_000, feasibility_tol=1e-5,
                 max_cg_iters=30, stagnation_patience=3, verbose=True):
        self.axioms = axioms
        self.num_vars = num_vars
        self.max_degree = max_degree
        self.dtype = dtype
        self.initial_mult_degree = initial_mult_degree
        self.lsqr_max_iter = lsqr_max_iter
        self.lsqr_tol = lsqr_tol
        self.pricing_top_k = pricing_top_k
        self.pricing_batch_size = pricing_batch_size
        self.feasibility_tol = feasibility_tol
        self.max_cg_iters = max_cg_iters
        self.stagnation_patience = stagnation_patience
        self.verbose = verbose

        self.pidx = PascalIndex(num_vars, max_degree)
        self.num_rows = self.pidx.total_monomials()

        # RHS in float32
        self.b = np.zeros(self.num_rows, dtype=dtype)
        self.b[0] = 1.0

        self.active_set = ActiveSet(axioms, num_vars, max_degree, self.pidx,
                                    dtype=dtype)

    def _save_checkpoint(self, cg_iter, x, residual, rel_residual,
                         history, elapsed):
        """Guardar checkpoint después de cada iteración CG."""
        ckpt_dir = '/opt/aip/results'
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            ckpt = {
                'cg_iter': cg_iter,
                'residual': residual,
                'rel_residual': rel_residual,
                'active_cols': self.active_set.n_active,
                'elapsed_s': elapsed,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'history': history,
            }
            ckpt_file = f'{ckpt_dir}/phpe5_checkpoint.json'
            with open(ckpt_file, 'w') as f:
                json.dump(ckpt, f, indent=2, default=str)

            if x is not None:
                np.save(f'{ckpt_dir}/phpe5_checkpoint_x.npy', x)

            if self.verbose:
                print(f"  Checkpoint guardado (iter {cg_iter}, "
                      f"res={residual:.2e})")
                sys.stdout.flush()
        except Exception as e:
            print(f"  WARNING: checkpoint failed: {e}")
            sys.stdout.flush()

    def solve(self):
        if self.verbose:
            ram_est_gb = self.num_rows * 4 * 3 / 1e9
            print(f"\n{'='*70}")
            print(f"  Column Generation Solver (dtype={self.dtype.__name__})")
            print(f"  Rows: {self.num_rows:,}")
            print(f"  Vars: {self.num_vars}, Max degree: {self.max_degree}")
            print(f"  Axioms: {len(self.axioms)}")
            print(f"  Estimated RAM (vectors): {ram_est_gb:.1f} GB")
            print(f"  Hora: {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*70}")
            sys.stdout.flush()

        t_total = time.time()

        self.active_set.add_initial_columns(
            max_mult_degree=self.initial_mult_degree, verbose=self.verbose
        )

        x = None
        best_residual = np.inf
        stagnation_count = 0
        history = []

        for cg_iter in range(1, self.max_cg_iters + 1):
            t_iter = time.time()

            if self.verbose:
                print(f"\n  --- CG iteration {cg_iter} "
                      f"[{datetime.now().strftime('%H:%M:%S')}] ---")
                print(f"  Active columns: {self.active_set.n_active:,}")
                sys.stdout.flush()

            # Build operator
            A_op = self.active_set.make_linear_operator(self.num_rows)

            # Warm-start
            x0 = None
            if x is not None:
                x0 = np.zeros(self.active_set.n_active, dtype=self.dtype)
                x0[:len(x)] = x

            if self.verbose:
                print(f"  LSQR (max_iter={self.lsqr_max_iter})...")
                sys.stdout.flush()

            t_lsqr = time.time()
            result = lsqr(A_op, self.b,
                          atol=self.lsqr_tol, btol=self.lsqr_tol,
                          iter_lim=self.lsqr_max_iter, x0=x0)
            x = result[0]
            lsqr_iters = result[2]
            lsqr_rnorm = result[3]
            t_lsqr = time.time() - t_lsqr

            # Actual residual
            residual_vec = self.b - A_op.matvec(x)
            residual_norm = float(np.linalg.norm(residual_vec))
            b_norm = float(np.linalg.norm(self.b))
            rel_residual = residual_norm / max(b_norm, 1e-15)

            if self.verbose:
                print(f"  LSQR: {lsqr_iters} iters, |r|={residual_norm:.2e}, "
                      f"rel={rel_residual:.2e} [{t_lsqr:.1f}s]")
                sys.stdout.flush()

            history.append({
                'cg_iter': cg_iter,
                'residual': residual_norm,
                'rel_residual': rel_residual,
                'active_cols': self.active_set.n_active,
                'lsqr_iters': lsqr_iters,
                'time': time.time() - t_iter,
            })

            # ── CHECKPOINT: guardar después de cada iteración CG ──
            self._save_checkpoint(cg_iter, x, residual_norm, rel_residual,
                                  history, time.time() - t_total)

            # Check feasibility
            if rel_residual < self.feasibility_tol:
                t_elapsed = time.time() - t_total
                size_l2 = int(np.sum(x != 0))
                if self.verbose:
                    print(f"\n  *** FEASIBLE ***")
                    print(f"  Residual: {residual_norm:.2e} (rel: {rel_residual:.2e})")
                    print(f"  Non-zeros: {size_l2:,}")
                    print(f"  CG iterations: {cg_iter}")
                    print(f"  Total time: {t_elapsed:.1f}s ({t_elapsed/3600:.2f}h)")
                    sys.stdout.flush()

                return {
                    'feasible': True, 'residual': residual_norm,
                    'rel_residual': rel_residual, 'x': x,
                    'size_l2': size_l2, 'iterations': cg_iter,
                    'lsqr_iterations': lsqr_iters,
                    'active_cols': self.active_set.n_active,
                    'time_total': t_elapsed, 'history': history,
                }

            # Stagnation check
            if best_residual == np.inf:
                improvement = 1.0
            else:
                improvement = (best_residual - residual_norm) / max(best_residual, 1e-15)
            if improvement < 0.001:
                stagnation_count += 1
                if self.verbose:
                    print(f"  Stagnation: {stagnation_count}/{self.stagnation_patience}")
            else:
                stagnation_count = 0
                best_residual = residual_norm

            if stagnation_count >= self.stagnation_patience:
                if self.verbose:
                    print(f"\n  Stopped: stagnation")
                break

            # Pricing
            if self.verbose:
                print(f"  Pricing (top-{self.pricing_top_k:,})...")
                sys.stdout.flush()

            t_price = time.time()
            top_cols = price_all_columns(
                self.axioms, self.num_vars, self.max_degree,
                residual_vec, self.pidx,
                top_k=self.pricing_top_k,
                batch_size=self.pricing_batch_size,
                verbose=self.verbose,
            )
            t_price = time.time() - t_price

            if self.verbose:
                if top_cols:
                    print(f"  Best corr: {top_cols[0][0]:.4e}")
                print(f"  Pricing: {t_price:.1f}s")
                sys.stdout.flush()

            if not top_cols:
                if self.verbose:
                    print(f"  No new columns found")
                break

            added = self.active_set.add_columns_from_pricing(
                top_cols, verbose=self.verbose
            )
            if added == 0:
                if self.verbose:
                    print(f"  All top columns already active")
                break

            gc.collect()

        t_elapsed = time.time() - t_total
        if self.verbose:
            print(f"\n  Final: res={residual_norm:.2e}, CG={cg_iter}, "
                  f"time={t_elapsed:.1f}s ({t_elapsed/3600:.2f}h)")
        return {
            'feasible': False, 'residual': residual_norm,
            'rel_residual': rel_residual, 'x': x,
            'size_l2': int(np.sum(x != 0)), 'iterations': cg_iter,
            'active_cols': self.active_set.n_active,
            'time_total': t_elapsed, 'history': history,
        }


# ============================================================
# Main
# ============================================================

def main():
    N = 5
    D = 10

    print(f"\n{'='*70}")
    print(f"  PHP-E({N}) d={D} — Column Generation (float32)")
    print(f"  Servidor: iafiscal (EPYC 24T, 125 GB)")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    sys.stdout.flush()

    print(f"\n  Construyendo axiomas PHP-E({N})...")
    t0 = time.time()
    axioms, num_vars = build_axioms_phpe(N)
    print(f"  Variables: {num_vars}")
    print(f"  Axiomas: {len(axioms)}")

    pidx = PascalIndex(num_vars, D)
    print(f"  Monomios (filas): {pidx.total_monomials():,}")
    print(f"  Build axiomas: {time.time()-t0:.1f}s")

    # Estimate memory
    num_rows = pidx.total_monomials()
    mem_vectors_gb = num_rows * 4 * 3 / 1e9  # b, u, residual in float32
    print(f"\n  Memoria estimada vectores: {mem_vectors_gb:.1f} GB (float32)")
    print(f"  Memoria disponible: ~108 GB")
    print(f"  Margen: ~{108 - mem_vectors_gb:.0f} GB")
    sys.stdout.flush()

    if mem_vectors_gb > MAX_RAM_GB:
        print(f"\n  ERROR: no cabe en {MAX_RAM_GB} GB!")
        sys.exit(1)

    # Warm up Numba
    print(f"\n  Calentando Numba JIT...")
    t0 = time.time()
    _test = _unrank_combination(np.int64(0), 2, 10, pidx.pascal[:11, :3])
    _test2 = _lex_rank_jit(np.array([0, 1], dtype=np.int64), pidx.pascal, num_vars)
    _test3 = union_sorted(np.array([0, 1], dtype=np.int64), np.array([2], dtype=np.int64))
    print(f"  JIT warmup: {time.time()-t0:.1f}s")
    sys.stdout.flush()

    # Solve
    solver = ColumnGenSolver(
        axioms=axioms,
        num_vars=num_vars,
        max_degree=D,
        dtype=DTYPE,
        initial_mult_degree=3,
        lsqr_max_iter=3000,
        lsqr_tol=1e-6,
        pricing_top_k=50_000,
        pricing_batch_size=2_000_000,
        feasibility_tol=1e-5,
        max_cg_iters=30,
        stagnation_patience=5,
        verbose=True,
    )

    result = solver.solve()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = '/opt/aip/results'
    os.makedirs(results_dir, exist_ok=True)

    meta = {
        'n': N, 'd_max': D,
        'feasible': result['feasible'],
        'residual': result['residual'],
        'rel_residual': result['rel_residual'],
        'iterations': result['iterations'],
        'size_l2': result['size_l2'],
        'active_cols': result['active_cols'],
        'time_total_s': result['time_total'],
        'num_vars': num_vars,
        'num_axioms': len(axioms),
        'num_monomials': pidx.total_monomials(),
        'dtype': DTYPE_NAME,
        'method': 'column_generation',
        'server': 'iafiscal_82.223.71.108',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'history': result['history'],
    }

    meta_file = f'{results_dir}/phpe5_d10_{timestamp}_meta.json'
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Guardado: {meta_file}")

    if result['x'] is not None:
        x_file = f'{results_dir}/phpe5_d10_{timestamp}_x.npy'
        np.save(x_file, result['x'])
        print(f"  Guardado: {x_file}")

    print(f"\n{'='*70}")
    print(f"  RESULTADO: {'FEASIBLE' if result['feasible'] else 'NO FEASIBLE'}")
    print(f"  Residual: {result['residual']:.2e}")
    print(f"  Tiempo: {result['time_total']:.1f}s ({result['time_total']/3600:.2f}h)")
    print(f"  Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
