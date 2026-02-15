"""
AIP Solver: Auto-routing linear system solver.

Automatically detects matrix structure and routes to the optimal solver:
  - Square sparse -> spsolve (LU)
  - Rectangular sparse -> LSQR (iterative)
  - Square dense -> numpy.linalg.solve (LAPACK)
  - Rectangular dense -> numpy.linalg.lstsq

Usage:
    import aip
    x = aip.solve(A, b)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, lsqr

from aip.detector import detect_matrix


def solve(A, b, verbose=False):
    """
    Solve Ax = b with automatic strategy selection.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
        Coefficient matrix.
    b : numpy.ndarray
        Right-hand side vector.
    verbose : bool
        Print strategy and timing info.

    Returns
    -------
    numpy.ndarray
        Solution vector x.
    """
    report = detect_matrix(A)
    strategy = report["strategy"]

    if verbose:
        print(f"  [AIP] {report['shape'][0]:,} x {report['shape'][1]:,}, "
              f"density={report['density']:.4%}, strategy={strategy}")

    if strategy in ("sparse_iterative", "sparse_lsqr"):
        if not sparse.issparse(A):
            A = sparse.csr_matrix(A)
        result = lsqr(A, b, atol=1e-10, btol=1e-10)
        return result[0]

    elif strategy == "sparse_direct":
        if not sparse.issparse(A):
            A = sparse.csr_matrix(A)
        return spsolve(A.tocsc(), b)

    elif strategy == "dense_lstsq":
        if sparse.issparse(A):
            A = A.toarray()
        result = np.linalg.lstsq(A, b, rcond=None)
        return result[0]

    else:  # dense
        if sparse.issparse(A):
            A = A.toarray()
        return np.linalg.solve(A, b)
