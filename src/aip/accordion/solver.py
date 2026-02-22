"""
Accordion Solver: Streaming LSQR over column chunks.

Instead of assembling a full sparse matrix, uses a LinearOperator
that performs matvec/rmatvec by iterating over CSR column-chunks.
This allows solving systems with billions of unknowns.

v0.4.0: Added diagonal (Jacobi) preconditioning and float32 support.
v0.5.0: Early stopping on residual stagnation.

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


def _compute_residual(chunks, x, b):
    """Compute ||Ax - b|| streaming over chunks. Always float64."""
    m = b.shape[0]
    Ax = np.zeros(m, dtype=np.float64)
    offset = 0
    for chunk in chunks:
        cols = chunk.shape[1]
        Ax += chunk @ x[offset:offset + cols]
        offset += cols
    residual = float(np.linalg.norm(Ax - b))
    del Ax
    return residual


def solve_chunks(chunks, b, max_iter=5000, tol=1e-10, verbose=True,
                 precondition=True, dtype=None, early_stop=True,
                 check_every=200, stagnation_patience=3,
                 stagnation_threshold=0.001):
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
    early_stop : bool
        Enable early stopping when residual stagnates. Default True.
        When the relative improvement between checks is below
        stagnation_threshold for stagnation_patience consecutive checks,
        LSQR stops early.
    check_every : int
        Check residual every N iterations for early stopping. Default 200.
    stagnation_patience : int
        Number of consecutive stagnant checks before stopping. Default 3.
    stagnation_threshold : float
        Minimum relative improvement to not be considered stagnant.
        Default 0.001 (0.1% improvement).

    Returns
    -------
    dict
        Solution with keys: x, residual, size_l2, iterations, feasible,
        time, early_stopped, residual_history.
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
    b64 = b.astype(np.float64)

    if verbose:
        es_label = f", early_stop=ON (check={check_every}, patience={stagnation_patience})" if early_stop else ""
        print(f"  [Accordion] LSQR max_iter={max_iter}, precond={'ON' if precondition else 'OFF'}{es_label}...")
        sys.stdout.flush()

    t0 = time.time()
    total_iters = 0
    x_raw = np.zeros(n, dtype=np.float64)
    early_stopped = False
    residual_history = []
    stagnant_count = 0

    if early_stop and max_iter > check_every:
        # Run LSQR in segments, checking residual between segments
        remaining = max_iter
        while remaining > 0:
            segment = min(check_every, remaining)
            result = lsqr(op, b64, x0=x_raw, atol=tol, btol=tol, iter_lim=segment)
            x_raw = result[0]
            total_iters += result[2]
            remaining -= segment

            # Check if LSQR converged internally (istop 1 or 2)
            if result[1] in (1, 2):
                if verbose:
                    print(f"  [Accordion] LSQR converged at iter {total_iters} (istop={result[1]})")
                    sys.stdout.flush()
                break

            # Compute actual residual for stagnation check
            if diag_inv is not None:
                x_check = x_raw * diag_inv.astype(np.float64)
            else:
                x_check = x_raw
            res_now = _compute_residual(chunks, x_check, b64)
            residual_history.append((total_iters, res_now))

            if verbose:
                elapsed_now = time.time() - t0
                print(f"  [Accordion] iter {total_iters}: res={res_now:.2e} [{elapsed_now:.0f}s]")
                sys.stdout.flush()

            # Already feasible?
            if res_now < 1e-6:
                if verbose:
                    print(f"  [Accordion] FEASIBLE at iter {total_iters}!")
                    sys.stdout.flush()
                break

            # Check stagnation
            if len(residual_history) >= 2:
                prev_res = residual_history[-2][1]
                if prev_res > 0:
                    improvement = (prev_res - res_now) / prev_res
                else:
                    improvement = 0.0

                if improvement < stagnation_threshold:
                    stagnant_count += 1
                    if verbose and stagnant_count > 0:
                        print(f"  [Accordion] Stagnant ({stagnant_count}/{stagnation_patience}): "
                              f"improvement={improvement:.4f} < {stagnation_threshold}")
                        sys.stdout.flush()
                    if stagnant_count >= stagnation_patience:
                        early_stopped = True
                        if verbose:
                            print(f"  [Accordion] EARLY STOP: residual stagnated at {res_now:.2e} "
                                  f"after {total_iters} iters")
                            sys.stdout.flush()
                        break
                else:
                    stagnant_count = 0
    else:
        # Original behavior: single LSQR call
        result = lsqr(op, b64, atol=tol, btol=tol, iter_lim=max_iter)
        x_raw = result[0]
        total_iters = result[2]

    elapsed = time.time() - t0

    # Undo preconditioning to get real x
    if diag_inv is not None:
        x = (x_raw * diag_inv.astype(np.float64))
    else:
        x = x_raw

    # Compute final residual (always in float64 for accuracy)
    residual = _compute_residual(chunks, x, b64)

    size_l2 = int(np.sum(np.abs(x) > 1e-8))
    feasible = residual < 1e-6

    if verbose:
        status = "FEASIBLE" if feasible else "INFEASIBLE"
        es_tag = " [EARLY_STOP]" if early_stopped else ""
        print(f"  [Accordion] res={residual:.2e}, iters={total_iters}, "
              f"SIZE_L2={size_l2:,}, {status}{es_tag} [{elapsed:.1f}s]")
        sys.stdout.flush()

    return {
        "x": x,
        "residual": residual,
        "size_l2": size_l2,
        "iterations": total_iters,
        "feasible": feasible,
        "time": elapsed,
        "early_stopped": early_stopped,
        "residual_history": residual_history,
    }
