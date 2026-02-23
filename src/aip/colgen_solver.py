"""
Column Generation Solver for PHP-Entangled IPS.

Solves ultra-large systems (e.g., PHP-E(5) d=10: 4.35B rows × 92B cols)
by iteratively growing a subset of active columns instead of working with
all columns at once.

Algorithm:
  1. Start with columns of low multiplier degree (≤ 3) → ~7M columns
  2. Solve restricted LSQR on active columns
  3. If feasible (residual < tol) → done
  4. Pricing: stream through ALL possible columns, find top-K by correlation
  5. Add top-K to active set, warm-start LSQR
  6. Repeat until feasible or stagnation

Memory: ~110 GB for PHP-E(5) d=10 (vs ~4,700 GB for full system).

Author: Carmen Esteban
"""

import numpy as np
from dataclasses import dataclass, field
from itertools import combinations
from scipy.sparse.linalg import lsqr, LinearOperator
import time
import sys
import gc

from aip.accordion import PascalIndex
from aip.accordion import fast as _fast
from aip.colgen_pricing import price_all_columns, _unrank_combination


# ============================================================
# ColumnGroup: metadata for a group of columns
# ============================================================

@dataclass
class ColumnGroup:
    """Represents a group of columns: all multiplier monomials of a given
    degree for a given axiom.

    Fields
    ------
    axiom_idx : int
        Index into the axioms list.
    mult_degree : int
        Degree of the multiplier monomials in this group.
    global_col_start : int
        First global column index of this group.
    num_cols : int
        Number of columns = C(num_vars, mult_degree).
    """
    axiom_idx: int
    mult_degree: int
    global_col_start: int
    num_cols: int


# ============================================================
# ActiveSet: manages the subset of active columns
# ============================================================

class ActiveSet:
    """Manages the active subset of columns for column generation.

    Stores packed metadata compatible with Numba kernels:
      - col_ax_idx: axiom index per column
      - col_mult_flat, col_mult_starts, col_mult_lengths: packed multiplier combos

    Supports efficient matvec/rmatvec via the Numba kernels from fast.py.
    """

    def __init__(self, axioms, num_vars, max_degree, pidx):
        """
        Parameters
        ----------
        axioms : list of list of (coeff, combo_tuple)
            Axiom definitions.
        num_vars : int
            Number of variables.
        max_degree : int
            Maximum polynomial degree.
        pidx : PascalIndex
            Index object.
        """
        self.axioms = axioms
        self.num_vars = num_vars
        self.max_degree = max_degree
        self.pidx = pidx
        self.pack = pidx.pack_for_numba()

        # Packed axiom data (precomputed once)
        self._ax_packs = []
        for ax_terms in axioms:
            combos = [m for c, m in ax_terms]
            coeffs = np.array([c for c, m in ax_terms], dtype=np.float64)
            flat, starts, lengths = _fast.pack_combos(combos)
            self._ax_packs.append((flat, starts, lengths, coeffs))

        # Active column metadata
        self.col_ax_idx = []      # axiom index per column
        self.col_mult_combo = []  # multiplier combo (tuple) per column
        self.n_active = 0

        # Precomputed for matvec: sparse entries per column
        # Built lazily on first matvec call
        self._rows_per_col = None
        self._vals_per_col = None
        self._dirty = True

    def add_initial_columns(self, max_mult_degree=3, verbose=True):
        """Add all columns with multiplier degree ≤ max_mult_degree."""
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
        elapsed = time.time() - t0
        if verbose:
            print(f"  ActiveSet: {self.n_active:,} initial columns "
                  f"(mult_deg ≤ {max_mult_degree}) [{elapsed:.1f}s]")
            sys.stdout.flush()

    def add_columns_from_pricing(self, pricing_results, verbose=True):
        """Add columns identified by pricing.

        Parameters
        ----------
        pricing_results : list of (corr, axiom_idx, mult_degree, mult_rank)
            From price_all_columns().
        """
        pascal = self.pack['pascal']
        added = 0

        # Build set of existing columns for dedup
        existing = set()
        for i in range(self.n_active):
            existing.add((self.col_ax_idx[i], self.col_mult_combo[i]))

        for corr, ax_idx, k_mult, mult_rank in pricing_results:
            # Unrank multiplier combo
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
            print(f"  ActiveSet: added {added:,} columns, "
                  f"total {self.n_active:,}")
            sys.stdout.flush()

        return added

    def _build_column_data(self):
        """Precompute sparse entries for all active columns."""
        if not self._dirty:
            return

        pascal = self.pack['pascal']
        num_vars = self.pack['num_vars']
        offsets = self.pack['offsets']
        max_degree = self.pack['max_degree']

        rows_per_col = []
        vals_per_col = []

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

                product = _fast.union_sorted(a_combo, mult_arr)
                if len(product) > max_degree:
                    continue

                row_idx = int(offsets[len(product)] + _fast._lex_rank_jit(
                    product, pascal, num_vars
                ))
                rows.append(row_idx)
                vals.append(coeff)

            rows_per_col.append(np.array(rows, dtype=np.int64))
            vals_per_col.append(np.array(vals, dtype=np.float64))

        self._rows_per_col = rows_per_col
        self._vals_per_col = vals_per_col
        self._dirty = False

    def make_linear_operator(self, num_rows):
        """Create a LinearOperator for the active columns.

        Parameters
        ----------
        num_rows : int
            Number of rows (= total monomials).

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
        """
        self._build_column_data()
        n = self.n_active
        m = num_rows
        rows_per_col = self._rows_per_col
        vals_per_col = self._vals_per_col

        def matvec(x):
            """y = A_active @ x"""
            result = np.zeros(m, dtype=np.float64)
            for j in range(n):
                if x[j] == 0.0:
                    continue
                rows = rows_per_col[j]
                vals = vals_per_col[j]
                for idx in range(len(rows)):
                    result[rows[idx]] += vals[idx] * x[j]
            return result

        def rmatvec(y):
            """x = A_active^T @ y"""
            result = np.zeros(n, dtype=np.float64)
            for j in range(n):
                rows = rows_per_col[j]
                vals = vals_per_col[j]
                dot = 0.0
                for idx in range(len(rows)):
                    dot += vals[idx] * y[rows[idx]]
                result[j] = dot
            return result

        return LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec,
                              dtype=np.float64)


# ============================================================
# ColumnGenSolver: main solver
# ============================================================

class ColumnGenSolver:
    """Column generation solver for PHP-Entangled IPS.

    Parameters
    ----------
    axioms : list of list of (coeff, combo_tuple)
        Axiom definitions.
    num_vars : int
        Number of variables.
    max_degree : int
        Maximum polynomial degree.
    initial_mult_degree : int
        Maximum multiplier degree for initial column set.
    lsqr_max_iter : int
        Max iterations per LSQR solve.
    lsqr_tol : float
        LSQR convergence tolerance.
    pricing_top_k : int
        Number of columns to add per CG iteration.
    pricing_batch_size : int
        Columns per pricing batch.
    feasibility_tol : float
        Residual threshold for declaring feasibility.
    max_cg_iters : int
        Maximum column generation iterations.
    stagnation_patience : int
        Stop after this many iterations without improvement.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        axioms,
        num_vars,
        max_degree,
        initial_mult_degree=3,
        lsqr_max_iter=5000,
        lsqr_tol=1e-10,
        pricing_top_k=100_000,
        pricing_batch_size=5_000_000,
        feasibility_tol=1e-6,
        max_cg_iters=30,
        stagnation_patience=3,
        verbose=True,
    ):
        self.axioms = axioms
        self.num_vars = num_vars
        self.max_degree = max_degree
        self.initial_mult_degree = initial_mult_degree
        self.lsqr_max_iter = lsqr_max_iter
        self.lsqr_tol = lsqr_tol
        self.pricing_top_k = pricing_top_k
        self.pricing_batch_size = pricing_batch_size
        self.feasibility_tol = feasibility_tol
        self.max_cg_iters = max_cg_iters
        self.stagnation_patience = stagnation_patience
        self.verbose = verbose

        # Build PascalIndex
        self.pidx = PascalIndex(num_vars, max_degree)
        self.num_rows = self.pidx.total_monomials()

        # RHS vector: b[0] = 1, rest = 0
        self.b = np.zeros(self.num_rows, dtype=np.float64)
        self.b[0] = 1.0

        # Active set
        self.active_set = ActiveSet(axioms, num_vars, max_degree, self.pidx)

    def solve(self):
        """Run column generation.

        Returns
        -------
        dict with:
            feasible : bool
            residual : float
            x : numpy array (solution for active columns)
            iterations : int (CG iterations)
            active_cols : int (final active set size)
            history : list of (cg_iter, residual, active_cols, time)
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  Column Generation Solver")
            print(f"  Rows: {self.num_rows:,}, Vars: {self.num_vars}")
            print(f"  Max degree: {self.max_degree}")
            print(f"  Axioms: {len(self.axioms)}")
            print(f"{'='*70}")
            sys.stdout.flush()

        t_total = time.time()

        # Step 1: Initialize active set
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
                print(f"\n  --- CG iteration {cg_iter} ---")
                print(f"  Active columns: {self.active_set.n_active:,}")
                sys.stdout.flush()

            # Step 2: Build LinearOperator and solve LSQR
            if self.verbose:
                print(f"  Building LinearOperator...")
                sys.stdout.flush()

            A_op = self.active_set.make_linear_operator(self.num_rows)

            # Warm-start: extend x0 with zeros for new columns
            x0 = None
            if x is not None:
                x0 = np.zeros(self.active_set.n_active, dtype=np.float64)
                x0[:len(x)] = x

            if self.verbose:
                print(f"  Solving LSQR (max_iter={self.lsqr_max_iter})...")
                sys.stdout.flush()

            t_lsqr = time.time()
            result = lsqr(A_op, self.b,
                          atol=self.lsqr_tol, btol=self.lsqr_tol,
                          iter_lim=self.lsqr_max_iter, x0=x0)
            x = result[0]
            lsqr_iters = result[2]
            lsqr_rnorm = result[3]
            t_lsqr = time.time() - t_lsqr

            # Compute actual residual
            residual_vec = self.b - A_op.matvec(x)
            residual_norm = np.linalg.norm(residual_vec)
            rel_residual = residual_norm / max(np.linalg.norm(self.b), 1e-15)

            if self.verbose:
                print(f"  LSQR: {lsqr_iters} iters, "
                      f"rnorm={lsqr_rnorm:.2e}, "
                      f"|r|={residual_norm:.2e}, "
                      f"rel={rel_residual:.2e} [{t_lsqr:.1f}s]")
                sys.stdout.flush()

            history.append({
                'cg_iter': cg_iter,
                'residual': float(residual_norm),
                'rel_residual': float(rel_residual),
                'active_cols': self.active_set.n_active,
                'lsqr_iters': lsqr_iters,
                'time': time.time() - t_iter,
            })

            # Step 3: Check feasibility
            if rel_residual < self.feasibility_tol:
                if self.verbose:
                    t_elapsed = time.time() - t_total
                    size_l2 = int(np.sum(x != 0))
                    print(f"\n  *** FEASIBLE ***")
                    print(f"  Residual: {residual_norm:.2e} "
                          f"(rel: {rel_residual:.2e})")
                    print(f"  Non-zeros: {size_l2:,}")
                    print(f"  CG iterations: {cg_iter}")
                    print(f"  Total time: {t_elapsed:.1f}s")
                    sys.stdout.flush()

                return {
                    'feasible': True,
                    'residual': float(residual_norm),
                    'rel_residual': float(rel_residual),
                    'x': x,
                    'size_l2': int(np.sum(x != 0)),
                    'iterations': cg_iter,
                    'lsqr_iterations': lsqr_iters,
                    'active_cols': self.active_set.n_active,
                    'time_total': time.time() - t_total,
                    'history': history,
                }

            # Step 4: Check stagnation
            if best_residual == np.inf:
                improvement = 1.0
            else:
                improvement = (best_residual - residual_norm) / max(best_residual, 1e-15)
            if improvement < 0.001:
                stagnation_count += 1
                if self.verbose:
                    print(f"  Stagnation: {stagnation_count}/{self.stagnation_patience} "
                          f"(improvement: {improvement:.4f})")
            else:
                stagnation_count = 0
                best_residual = residual_norm

            if stagnation_count >= self.stagnation_patience:
                if self.verbose:
                    print(f"\n  Stopped: stagnation after {cg_iter} iterations")
                    sys.stdout.flush()
                break

            # Step 5: Pricing — find best new columns
            if self.verbose:
                print(f"  Pricing (streaming top-{self.pricing_top_k:,})...")
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
                    print(f"  Best correlation: {top_cols[0][0]:.4e}")
                print(f"  Pricing time: {t_price:.1f}s")
                sys.stdout.flush()

            if not top_cols:
                if self.verbose:
                    print(f"  No new columns found, stopping")
                break

            # Step 6: Add columns to active set
            added = self.active_set.add_columns_from_pricing(
                top_cols, verbose=self.verbose
            )

            if added == 0:
                if self.verbose:
                    print(f"  All top columns already active, stopping")
                break

            gc.collect()

        # Final result (not feasible)
        t_elapsed = time.time() - t_total

        if self.verbose:
            print(f"\n  Final: residual={residual_norm:.2e}, "
                  f"CG iters={cg_iter}, time={t_elapsed:.1f}s")
            sys.stdout.flush()

        return {
            'feasible': False,
            'residual': float(residual_norm),
            'rel_residual': float(rel_residual),
            'x': x,
            'size_l2': int(np.sum(x != 0)),
            'iterations': cg_iter,
            'lsqr_iterations': lsqr_iters,
            'active_cols': self.active_set.n_active,
            'time_total': t_elapsed,
            'history': history,
        }


# ============================================================
# Convenience function
# ============================================================

def build_axioms_phpe(n):
    """Build PHP-Entangled axioms for n holes (n+1 pigeons, n holes).

    Variables:
      x_{p,h} : pigeon p goes to hole h (p=1..n+1, h=1..n)
      y_{p,q} : pigeon p < pigeon q in ordering (p<q)

    Axioms:
      1. Existence: each pigeon goes to some hole
      2. Hole exclusion: two pigeons don't share a hole
      3. Functionality: each pigeon goes to one hole
      4. Ordering: consistency of x_{p,h} * x_{q,h'} with y_{p,q}
      5. Transitivity: y_{p,q} * y_{q,r} => y_{p,r}

    Returns
    -------
    axioms : list of list of (coeff, combo_tuple)
    num_vars : int
    """
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


def enumerate_column_groups(axioms, num_vars, max_degree, pidx):
    """Enumerate all column groups (axiom × mult_degree) without generating columns.

    Returns
    -------
    groups : list of ColumnGroup
    total_cols : int
    """
    pascal = pidx.pack_for_numba()['pascal']
    groups = []
    total_cols = 0

    for ax_idx, ax_terms in enumerate(axioms):
        deg_ax = max(len(m) for c, m in ax_terms)
        max_mult_deg = max(0, max_degree - deg_ax)

        for d in range(max_mult_deg + 1):
            n_cols = int(pascal[num_vars, d])
            if n_cols == 0:
                continue
            groups.append(ColumnGroup(
                axiom_idx=ax_idx,
                mult_degree=d,
                global_col_start=total_cols,
                num_cols=n_cols,
            ))
            total_cols += n_cols

    return groups, total_cols


def solve_phpe_colgen(
    n,
    max_degree,
    initial_mult_degree=3,
    lsqr_max_iter=5000,
    pricing_top_k=100_000,
    pricing_batch_size=5_000_000,
    feasibility_tol=1e-6,
    max_cg_iters=30,
    verbose=True,
):
    """Convenience function: build axioms and solve PHP-E(n) at given degree.

    Parameters
    ----------
    n : int
        Number of holes (n+1 pigeons).
    max_degree : int
        Maximum polynomial degree.
    initial_mult_degree : int
        Max multiplier degree for initial active set.
    lsqr_max_iter : int
        Max LSQR iterations per CG round.
    pricing_top_k : int
        Columns to add per CG iteration.
    pricing_batch_size : int
        Pricing batch size.
    feasibility_tol : float
        Residual threshold for feasibility.
    max_cg_iters : int
        Max CG iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with feasible, residual, x, iterations, etc.
    """
    if verbose:
        print(f"\n  PHP-E(n={n}) d={max_degree}")
        print(f"  Building axioms...")
        sys.stdout.flush()

    axioms, num_vars = build_axioms_phpe(n)

    if verbose:
        pidx = PascalIndex(num_vars, max_degree)
        groups, total_cols = enumerate_column_groups(
            axioms, num_vars, max_degree, pidx
        )
        print(f"  Variables: {num_vars}")
        print(f"  Axioms: {len(axioms)}")
        print(f"  Total monomials (rows): {pidx.total_monomials():,}")
        print(f"  Total columns: {total_cols:,}")
        print(f"  Column groups: {len(groups)}")
        sys.stdout.flush()

    solver = ColumnGenSolver(
        axioms=axioms,
        num_vars=num_vars,
        max_degree=max_degree,
        initial_mult_degree=initial_mult_degree,
        lsqr_max_iter=lsqr_max_iter,
        pricing_top_k=pricing_top_k,
        pricing_batch_size=pricing_batch_size,
        feasibility_tol=feasibility_tol,
        max_cg_iters=max_cg_iters,
        verbose=verbose,
    )

    result = solver.solve()
    result['n'] = n
    result['max_degree'] = max_degree
    result['num_vars'] = num_vars
    result['num_axioms'] = len(axioms)
    result['num_rows'] = solver.num_rows
    return result
