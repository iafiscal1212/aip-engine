"""
Column Generation Pricing: Numba kernels for streaming column evaluation.

The key insight: we never store all ~92B columns. Instead, we:
1. Enumerate column GROUPS (axiom × mult_degree) — ~4000 groups
2. For each group, stream through columns in batches of 10M
3. For each column, unrank to get the multiplier combo, compute correlation
4. Keep a global top-K heap of best columns

Core kernels:
  - _unrank_combination: rank → sorted combination (inverse of _lex_rank_jit)
  - _price_group_batch_kernel: compute correlations for a batch of columns
  - price_all_columns: streaming top-K over all groups

Author: Carmen Esteban
"""

import numpy as np
import heapq
import time
import sys

from aip.accordion import fast as _fast
from aip.accordion.indexing import PascalIndex

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

    def prange(*args):
        return range(*args)


# ============================================================
# Unranking: rank → sorted combination
# ============================================================

@njit(cache=True)
def _unrank_combination(rank, k, num_vars, pascal):
    """Convert a lexicographic rank back to the sorted combination.

    This is the inverse of _lex_rank_jit from fast.py.

    Parameters
    ----------
    rank : int64
        Lexicographic rank within C(num_vars, k).
    k : int
        Degree (size of the combination).
    num_vars : int
        Number of variables.
    pascal : 2D array of int64
        Precomputed Pascal triangle.

    Returns
    -------
    combo : numpy array of int64, length k
        Sorted variable indices.
    """
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


# ============================================================
# Pricing kernel: correlations for a batch of columns
# ============================================================

@njit(cache=True)
def _price_group_batch_kernel(
    batch_start, batch_size,
    ax_flat, ax_starts, ax_lengths, ax_coeffs,
    k_mult, num_vars,
    pascal, offsets, max_degree,
    residual
):
    """Compute |correlation| for a batch of columns within one group.

    Each column corresponds to a multiplier monomial of degree k_mult.
    Column rank j maps to the j-th combination of C(num_vars, k_mult).

    Parameters
    ----------
    batch_start : int
        Starting rank within C(num_vars, k_mult).
    batch_size : int
        Number of columns in this batch.
    ax_flat, ax_starts, ax_lengths, ax_coeffs :
        Packed axiom terms (from fast.pack_combos).
    k_mult : int
        Degree of multiplier monomials in this group.
    num_vars, pascal, offsets, max_degree :
        PascalIndex data.
    residual : 1D float64 array
        Current residual vector r = b - A_active @ x.

    Returns
    -------
    correlations : 1D float64 array of length batch_size
        |A[:,j]^T @ residual| for each column j in the batch.
    """
    n_terms = len(ax_starts)
    correlations = np.empty(batch_size, dtype=np.float64)

    for local_j in range(batch_size):
        rank = batch_start + local_j

        # Unrank to get multiplier combo
        mult_combo = _unrank_combination(rank, k_mult, num_vars, pascal)

        # Compute A[:,j]^T @ residual = sum_i coeff_i * residual[row_of(product_i)]
        corr = 0.0
        for i in range(n_terms):
            a_start = ax_starts[i]
            a_len = ax_lengths[i]
            coeff = ax_coeffs[i]
            a_combo = ax_flat[a_start:a_start + a_len]

            # Boolean product: union of variables
            product = _fast.union_sorted(a_combo, mult_combo)

            if len(product) > max_degree:
                continue

            # Row index via PascalIndex
            row_idx = offsets[len(product)] + _fast._lex_rank_jit(
                product, pascal, num_vars
            )

            corr += coeff * residual[row_idx]

        correlations[local_j] = abs(corr)

    return correlations


# ============================================================
# Streaming top-K pricing over all groups
# ============================================================

def price_all_columns(
    axioms,
    num_vars,
    max_degree,
    residual,
    pidx,
    top_k=100_000,
    batch_size=5_000_000,
    verbose=True
):
    """Stream through all possible columns, returning the top-K by correlation.

    Parameters
    ----------
    axioms : list of list of (coeff, combo_tuple)
        The axiom definitions.
    num_vars : int
        Number of variables.
    max_degree : int
        Maximum polynomial degree.
    residual : 1D float64 array
        Current residual vector.
    pidx : PascalIndex
        Index for monomial computations.
    top_k : int
        Number of best columns to return.
    batch_size : int
        Columns per batch (controls memory).
    verbose : bool
        Print progress.

    Returns
    -------
    list of (neg_corr, axiom_idx, mult_degree, mult_rank)
        Top-K columns sorted by descending |correlation|.
        neg_corr is negative for heap ordering.
    """
    pack = pidx.pack_for_numba()
    pascal = pack['pascal']
    offsets = pack['offsets']

    # Min-heap of (-|corr|, axiom_idx, mult_degree, mult_rank)
    # We use negative correlation so heappush keeps the smallest |corr|
    # at the top and we can pop it when a better one arrives.
    heap = []

    total_cols_processed = 0
    t0 = time.time()

    for ax_idx, ax_terms in enumerate(axioms):
        deg_ax = max(len(m) for c, m in ax_terms)
        max_mult_deg = max(0, max_degree - deg_ax)

        # Pack axiom for Numba
        ax_combos = [m for c, m in ax_terms]
        ax_coeffs = np.array([c for c, m in ax_terms], dtype=np.float64)
        ax_flat, ax_starts, ax_lengths = _fast.pack_combos(ax_combos)

        for k_mult in range(max_mult_deg + 1):
            # Number of multiplier monomials of degree k_mult
            n_mult = int(pascal[num_vars, k_mult])
            if n_mult == 0:
                continue

            # Process in batches
            for b_start in range(0, n_mult, batch_size):
                b_size = min(batch_size, n_mult - b_start)

                correlations = _price_group_batch_kernel(
                    b_start, b_size,
                    ax_flat, ax_starts, ax_lengths, ax_coeffs,
                    k_mult, num_vars,
                    pascal, offsets, max_degree,
                    residual
                )

                # Update top-K heap
                for local_j in range(b_size):
                    corr_val = correlations[local_j]
                    if corr_val < 1e-15:
                        continue

                    entry = (corr_val, ax_idx, k_mult, b_start + local_j)

                    if len(heap) < top_k:
                        heapq.heappush(heap, entry)
                    elif corr_val > heap[0][0]:
                        heapq.heapreplace(heap, entry)

                total_cols_processed += b_size

        if verbose and (ax_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = total_cols_processed / max(elapsed, 1e-6)
            print(f"    Pricing: axiom {ax_idx+1}/{len(axioms)}, "
                  f"cols={total_cols_processed:,.0f}, "
                  f"rate={rate:,.0f}/s, "
                  f"heap={len(heap)}")
            sys.stdout.flush()

    if verbose:
        elapsed = time.time() - t0
        print(f"    Pricing done: {total_cols_processed:,.0f} cols in {elapsed:.1f}s, "
              f"top-{len(heap)} selected")
        sys.stdout.flush()

    # Sort by descending correlation
    heap.sort(key=lambda x: -x[0])
    return heap


def unrank_combination(rank, k, num_vars, pascal):
    """Public wrapper for _unrank_combination (for testing)."""
    return _unrank_combination(np.int64(rank), k, num_vars, pascal)
