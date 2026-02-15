"""
Accordion Solver: Streaming LSQR over column chunks.

Instead of assembling a full sparse matrix, uses a LinearOperator
that performs matvec/rmatvec by iterating over CSR column-chunks.
This allows solving systems with billions of unknowns.

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


def solve_chunks(chunks, b, max_iter=5000, tol=1e-10, verbose=True):
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

    Returns
    -------
    dict
        Solution with keys: x, residual, size_l2, iterations, feasible, time.
    """
    if not chunks:
        raise ValueError("No chunks provided")

    m = chunks[0].shape[0]
    n = sum(c.shape[1] for c in chunks)

    if verbose:
        info = accordion_info(chunks)
        print(f"  [Accordion] Solve {m:,} x {n:,} in {len(chunks)} chunks")
        print(f"  [Accordion] RAM: {info['ram_mb']} MB "
              f"(dense: {info['dense_would_be_mb']} MB, "
              f"compression: {info['compression']}x)")
        print(f"  [Accordion] LSQR max_iter={max_iter}...")
        sys.stdout.flush()

    def matvec(x):
        result = np.zeros(m)
        offset = 0
        for chunk in chunks:
            cols = chunk.shape[1]
            result += chunk @ x[offset:offset + cols]
            offset += cols
        return result

    def rmatvec(y):
        result = np.zeros(n)
        offset = 0
        for chunk in chunks:
            cols = chunk.shape[1]
            result[offset:offset + cols] = chunk.T @ y
            offset += cols
        return result

    op = LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec)

    t0 = time.time()
    result = lsqr(op, b, atol=tol, btol=tol, iter_lim=max_iter)
    x = result[0]
    elapsed = time.time() - t0

    # Compute residual
    Ax = matvec(x)
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
