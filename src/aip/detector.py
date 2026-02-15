"""
AIP Detector: Auto-detection of matrix structure.

Analyzes a matrix and returns a report with:
  - Shape, density, sparsity pattern
  - Recommended computation strategy (dense/sparse/iterative)
  - Memory estimates

Usage:
    import aip
    report = aip.detect_matrix(A)
    print(report)
"""

import numpy as np
from scipy import sparse


def detect_matrix(A):
    """
    Analyze matrix structure and recommend computation strategy.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
        The matrix to analyze.

    Returns
    -------
    dict
        Structure report with shape, density, nnz, recommendations.
    """
    if sparse.issparse(A):
        A_sp = A.tocsr()
        m, n = A_sp.shape
        nnz = A_sp.nnz
        total = m * n
        density = nnz / total if total > 0 else 0
        ram_sparse = A_sp.data.nbytes + A_sp.indices.nbytes + A_sp.indptr.nbytes
        ram_dense = m * n * 8
    else:
        A_arr = np.asarray(A, dtype=float)
        m, n = A_arr.shape
        nnz = int(np.count_nonzero(A_arr))
        total = m * n
        density = nnz / total if total > 0 else 0
        ram_sparse = None
        ram_dense = A_arr.nbytes

    is_square = (m == n)
    is_rectangular = not is_square

    # Strategy recommendation
    if density < 0.01:
        strategy = "sparse_iterative"  # LSQR / CG
        reason = f"Very sparse ({density:.4%}), iterative solver optimal"
    elif density < 0.1:
        strategy = "sparse_direct"  # spsolve / SuperLU
        reason = f"Sparse ({density:.2%}), direct sparse solver"
    elif density < 0.5:
        strategy = "dense"  # LAPACK
        reason = f"Moderate density ({density:.1%}), dense solver"
    else:
        strategy = "dense"
        reason = f"Dense ({density:.1%}), LAPACK"

    if is_rectangular:
        if density < 0.1:
            strategy = "sparse_lsqr"
            reason = f"Rectangular sparse ({density:.4%}), LSQR"
        else:
            strategy = "dense_lstsq"
            reason = f"Rectangular dense ({density:.1%}), lstsq"

    report = {
        "shape": (m, n),
        "nnz": nnz,
        "density": round(density, 6),
        "is_square": is_square,
        "strategy": strategy,
        "reason": reason,
        "ram_dense_mb": round(ram_dense / 1e6, 1),
    }

    if ram_sparse is not None:
        report["ram_sparse_mb"] = round(ram_sparse / 1e6, 1)
        report["compression"] = round(ram_dense / max(ram_sparse, 1), 1)

    # Accordion recommendation
    report["recommend_accordion"] = (
        ram_dense > 500_000_000  # >500 MB dense
        or (is_rectangular and nnz > 10_000_000)  # >10M nnz rectangular
    )

    return report
