"""
Accordion Solver: Streaming LSQR over column chunks.

Instead of assembling a full sparse matrix, uses a LinearOperator
that performs matvec/rmatvec by iterating over CSR column-chunks.
This allows solving systems with billions of unknowns.

v0.4.0: Added diagonal (Jacobi) preconditioning and float32 support.

Author: Carmen Esteban
"""

import numpy as np
from scipy.sparse.linalg import lsqr, LinearOperator
import time
import sys


def accordion_info(chunks):
    """
    Analyze memory usage of chunked matrix.

    Parameters
    ----------
    chunks : list of scipy.sparse.csr_matrix
        Column chunks of the matrix.

    Returns
    -------
    dict
        Shape, nnz, RAM, compression ratio.
    """
    if not chunks:
        return {"shape": (0, 0), "nnz": 0, "ram_mb": 0}

    m = chunks[0].shape[0]
    n = sum(c.shape[1] for c in chunks)
    nnz = sum(c.nnz for c in chunks)
    ram = sum(c.data.nbytes + c.indices.nbytes + c.indptr.nbytes for c in chunks)
    dense = m * n * 8

    return {
        "shape": (m, n),
        "nnz": nnz,
        "ram_mb": round(ram / 1e6, 1),
        "dense_would_be_mb": round(dense / 1e6, 1),
        "compression": round(dense / max(ram, 1), 1),
        "num_chunks": len(chunks),
    }


def _compute_col_norms(chunks):
    """Compute column norms across chunks for diagonal preconditioning."""
    norms = []
    for chunk in chunks:
        # For each column, compute sum of squares
        col_sq = chunk.multiply(chunk)  # element-wise square
        col_norms = np.sqrt(np.asarray(col_sq.sum(axis=0)).ravel())
        norms.append(col_norms)
    return np.concatenate(norms)


def solve_chunks(chunks, b, max_iter=5000, tol=1e-10, verbose=True,
                 precondition=True, dtype=None):
    """
    Solve Ax = b where A is represented as column chunks.

    Uses LSQR with a LinearOperator that streams over chunks,
    never assembling the full matrix.

    Parameters
    ----------
    chunks : list of scipy.sparse.csr_matrix
        Column chunks of matrix A. Each chunk has the same number of rows.
    b : numpy.ndarray
        Right-hand side vector.
    max_iter : int
        Maximum LSQR iterations.
    tol : float
        Convergence tolerance (atol and btol).
    verbose : bool
        Print progress and results.
    precondition : bool
        Apply diagonal (Jacobi) preconditioning. Reduces iterations
        significantly for ill-conditioned systems. Default True.
    dtype : numpy dtype, optional
        Force float32 or float64 for internal vectors. If None, uses
        the dtype of the first chunk.

    Returns
    -------
    dict
        Solution with keys: x, residual, size_l2, iterations, feasible, time.
    """
    if not chunks:
        raise ValueError("No chunks provided")

    m = chunks[0].shape[0]
    n = sum(c.shape[1] for c in chunks)

    # Determine dtype
    if dtype is None:
        dtype = chunks[0].dtype
    use_dtype = np.dtype(dtype)

    if verbose:
        info = accordion_info(chunks)
        dtype_name = 'f32' if use_dtype == np.float32 else 'f64'
        print(f"  [Accordion] Solve {m:,} x {n:,} in {len(chunks)} chunks ({dtype_name})")
        print(f"  [Accordion] RAM: {info['ram_mb']} MB "
              f"(dense: {info['dense_would_be_mb']} MB, "
              f"compression: {info['compression']}x)")
        sys.stdout.flush()

    # Diagonal preconditioning
    diag_inv = None
    if precondition:
        t_pre = time.time()
        col_norms = _compute_col_norms(chunks)
        # Avoid division by zero
        col_norms[col_norms < 1e-15] = 1.0
        diag_inv = (1.0 / col_norms).astype(use_dtype)
        if verbose:
            print(f"  [Accordion] Preconditioner computed [{time.time() - t_pre:.1f}s]")
            sys.stdout.flush()

    def matvec(x):
        x_work = x.astype(use_dtype) if x.dtype != use_dtype else x
        if diag_inv is not None:
            x_work = x_work * diag_inv
        result = np.zeros(m, dtype=use_dtype)
        offset = 0
        for chunk in chunks:
            cols = chunk.shape[1]
            result += chunk @ x_work[offset:offset + cols]
            offset += cols
        return result.astype(np.float64)  # LSQR needs float64

    def rmatvec(y):
        y_work = y.astype(use_dtype) if y.dtype != use_dtype else y
        result = np.zeros(n, dtype=use_dtype)
        offset = 0
        for chunk in chunks:
            cols = chunk.shape[1]
            result[offset:offset + cols] = chunk.T @ y_work
            offset += cols
        if diag_inv is not None:
            result *= diag_inv
        return result.astype(np.float64)  # LSQR needs float64

    op = LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

    if verbose:
        print(f"  [Accordion] LSQR max_iter={max_iter}, precond={'ON' if precondition else 'OFF'}...")
        sys.stdout.flush()

    t0 = time.time()
    result = lsqr(op, b.astype(np.float64), atol=tol, btol=tol, iter_lim=max_iter)
    x_raw = result[0]
    elapsed = time.time() - t0

    # Undo preconditioning to get real x
    if diag_inv is not None:
        x = (x_raw * diag_inv.astype(np.float64))
    else:
        x = x_raw

    # Compute residual (always in float64 for accuracy)
    offset = 0
    Ax = np.zeros(m, dtype=np.float64)
    for chunk in chunks:
        cols = chunk.shape[1]
        Ax += chunk @ x[offset:offset + cols]
        offset += cols
    residual = float(np.linalg.norm(Ax - b))
    del Ax

    size_l2 = int(np.sum(np.abs(x) > 1e-8))
    feasible = residual < 1e-6

    if verbose:
        status = "FEASIBLE" if feasible else "INFEASIBLE"
        print(f"  [Accordion] res={residual:.2e}, iters={result[2]}, "
              f"SIZE_L2={size_l2:,}, {status} [{elapsed:.1f}s]")
        sys.stdout.flush()

    return {
        "x": x,
        "residual": residual,
        "size_l2": size_l2,
        "iterations": result[2],
        "feasible": feasible,
        "time": elapsed,
    }
