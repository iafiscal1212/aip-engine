"""
Accordion Fast: Numba JIT-compiled kernels for construction hot loops.

When numba is installed, these functions replace the pure-Python versions,
giving 50-100x speedup on the construction phase.

If numba is not installed, everything falls back to pure Python transparently.

Install: pip install numba

Author: Carmen Esteban
"""

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Fallback decorators that do nothing
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

    def prange(*args):
        return range(*args)


# ============================================================
# Core kernel: lexicographic rank (the hottest loop)
# ============================================================

@njit(cache=True)
def _lex_rank_jit(combo, pascal, num_vars):
    """Compute lexicographic rank of a sorted combination.

    Parameters
    ----------
    combo : numpy array of int64
        Sorted variable indices.
    pascal : 2D numpy array of int64
        Precomputed Pascal triangle.
    num_vars : int
        Number of variables.

    Returns
    -------
    int64
        Lexicographic rank.
    """
    k = len(combo)
    if k == 0:
        return np.int64(0)
    rank = np.int64(0)
    prev = np.int64(-1)
    for i in range(k):
        a = combo[i]
        remaining = k - 1 - i
        for j in range(prev + 1, a):
            available = num_vars - 1 - j
            if available >= remaining >= 0:
                rank += pascal[available, remaining]
        prev = a
    return rank


@njit(cache=True)
def combo_to_index_jit(combo, pascal, num_vars, offsets):
    """Convert sorted combo to global monomial index.

    Parameters
    ----------
    combo : numpy array of int64
        Sorted variable indices.
    pascal : 2D numpy array of int64
        Precomputed Pascal triangle.
    num_vars : int
        Number of variables.
    offsets : numpy array of int64
        Degree offset table.

    Returns
    -------
    int64
        Global monomial index.
    """
    d = len(combo)
    return offsets[d] + _lex_rank_jit(combo, pascal, num_vars)


# ============================================================
# Batch kernel: process many combos at once
# ============================================================

@njit(cache=True)
def batch_combo_to_index_jit(combos_flat, starts, lengths, pascal, num_vars, offsets):
    """Convert multiple combos to indices in one compiled loop.

    Parameters
    ----------
    combos_flat : 1D numpy array of int64
        All combos concatenated.
    starts : numpy array of int64
        Start position of each combo in combos_flat.
    lengths : numpy array of int64
        Length (degree) of each combo.
    pascal : 2D numpy array of int64
        Precomputed Pascal triangle.
    num_vars : int
        Number of variables.
    offsets : numpy array of int64
        Degree offset table.

    Returns
    -------
    numpy array of int64
        Global monomial indices.
    """
    n = len(starts)
    result = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = starts[i]
        k = lengths[i]
        combo = combos_flat[s:s + k]
        result[i] = offsets[k] + _lex_rank_jit(combo, pascal, num_vars)
    return result


# ============================================================
# Monomial product: merge two sorted combos
# ============================================================

@njit(cache=True)
def merge_sorted(a, b):
    """Merge two sorted arrays into one sorted array.

    This is the monomial product: x_i * x_j = sorted merge of indices.

    Parameters
    ----------
    a, b : numpy arrays of int64
        Sorted variable indices.

    Returns
    -------
    numpy array of int64
        Merged sorted array.
    """
    na = len(a)
    nb = len(b)
    out = np.empty(na + nb, dtype=np.int64)
    i = 0
    j = 0
    k = 0
    while i < na and j < nb:
        if a[i] <= b[j]:
            out[k] = a[i]
            i += 1
        else:
            out[k] = b[j]
            j += 1
        k += 1
    while i < na:
        out[k] = a[i]
        i += 1
        k += 1
    while j < nb:
        out[k] = b[j]
        j += 1
        k += 1
    return out


# ============================================================
# Construction kernel: build sparse entries for one axiom batch
# ============================================================

@njit(cache=True)
def build_axiom_entries(
    axiom_flat, axiom_starts, axiom_lengths, axiom_coeffs,
    mono_flat, mono_starts, mono_lengths,
    pascal, num_vars, offsets, max_degree, axiom_row_offset
):
    """Build sparse matrix entries for axiom × monomial products.

    For each axiom term and each target monomial, computes the product
    monomial index and generates (row, col, val) triples.

    Parameters
    ----------
    axiom_flat : 1D array
        All axiom monomial combos concatenated.
    axiom_starts : 1D array
        Start of each axiom term in axiom_flat.
    axiom_lengths : 1D array
        Degree of each axiom term.
    axiom_coeffs : 1D float array
        Coefficient of each axiom term.
    mono_flat : 1D array
        All target monomial combos concatenated.
    mono_starts : 1D array
        Start of each monomial in mono_flat.
    mono_lengths : 1D array
        Degree of each target monomial.
    pascal : 2D array
        Pascal triangle.
    num_vars : int
        Number of variables.
    offsets : 1D array
        Degree offsets.
    max_degree : int
        Maximum degree (products beyond this are skipped).
    axiom_row_offset : int
        Row offset for this axiom in the matrix.

    Returns
    -------
    rows : 1D int64 array
    cols : 1D int64 array
    vals : 1D float64 array
    count : int
        Number of valid entries.
    """
    n_axiom_terms = len(axiom_starts)
    n_monos = len(mono_starts)

    # Pre-allocate max possible entries
    max_entries = n_axiom_terms * n_monos
    rows = np.empty(max_entries, dtype=np.int64)
    cols = np.empty(max_entries, dtype=np.int64)
    vals = np.empty(max_entries, dtype=np.float64)
    count = 0

    for i in range(n_axiom_terms):
        a_start = axiom_starts[i]
        a_len = axiom_lengths[i]
        coeff = axiom_coeffs[i]
        a_combo = axiom_flat[a_start:a_start + a_len]

        for j in range(n_monos):
            m_start = mono_starts[j]
            m_len = mono_lengths[j]

            # Check if product degree exceeds max
            prod_deg = a_len + m_len
            if prod_deg > max_degree:
                continue

            m_combo = mono_flat[m_start:m_start + m_len]

            # Merge sorted arrays (monomial product)
            product = merge_sorted(a_combo, m_combo)

            # Compute global index
            col_idx = offsets[prod_deg] + _lex_rank_jit(product, pascal, num_vars)

            rows[count] = j  # monomial row
            cols[count] = col_idx
            vals[count] = coeff
            count += 1

    return rows[:count], cols[:count], vals[:count], count


# ============================================================
# Utility: pack combos into flat arrays for numba
# ============================================================

def pack_combos(combos):
    """Convert list of tuples to flat array + starts + lengths.

    Parameters
    ----------
    combos : list of tuple of int
        List of sorted variable index tuples.

    Returns
    -------
    flat : numpy array of int64
    starts : numpy array of int64
    lengths : numpy array of int64
    """
    if not combos:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64))

    total_len = sum(len(c) for c in combos)
    flat = np.empty(total_len, dtype=np.int64)
    starts = np.empty(len(combos), dtype=np.int64)
    lengths = np.empty(len(combos), dtype=np.int64)

    pos = 0
    for i, combo in enumerate(combos):
        starts[i] = pos
        lengths[i] = len(combo)
        for v in combo:
            flat[pos] = v
            pos += 1

    return flat, starts, lengths


def is_available():
    """Check if numba acceleration is available."""
    return HAS_NUMBA


def warmup(pascal, num_vars, offsets):
    """Trigger JIT compilation with a small dummy call.

    Call this once after creating a PascalIndex to avoid
    compilation overhead during the first real computation.

    Parameters
    ----------
    pascal : numpy array
        Pascal triangle from PascalIndex.
    num_vars : int
        Number of variables.
    offsets : numpy array of int64
        Degree offsets.
    """
    if not HAS_NUMBA:
        return
    dummy = np.array([0], dtype=np.int64)
    combo_to_index_jit(dummy, pascal, num_vars, offsets)
    batch_combo_to_index_jit(dummy, np.array([0], dtype=np.int64),
                              np.array([1], dtype=np.int64),
                              pascal, num_vars, offsets)
    merge_sorted(dummy, dummy)
