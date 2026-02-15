"""Tests for aip-engine."""
import numpy as np
from scipy import sparse
import pytest

import aip
from aip.accordion import PascalIndex, AccordionBuilder, solve_chunks


def test_version():
    assert aip.__version__ == "0.3.0"


def test_detect_dense():
    A = np.random.randn(10, 10)
    report = aip.detect_matrix(A)
    assert report["shape"] == (10, 10)
    assert report["is_square"]
    assert report["strategy"] == "dense"


def test_detect_sparse():
    A = sparse.random(1000, 1000, density=0.001, format="csr")
    report = aip.detect_matrix(A)
    assert report["density"] < 0.01
    assert "sparse" in report["strategy"]


def test_detect_rectangular():
    A = sparse.random(100, 500, density=0.01, format="csr")
    report = aip.detect_matrix(A)
    assert not report["is_square"]


def test_solve_dense():
    A = np.array([[2, 1], [1, 3]], dtype=float)
    b = np.array([5, 7], dtype=float)
    x = aip.solve(A, b)
    assert np.allclose(A @ x, b)


def test_solve_sparse():
    A = sparse.eye(100, format="csr") * 2.0
    b = np.ones(100)
    x = aip.solve(A, b)
    assert np.allclose(x, 0.5)


def test_solve_rectangular():
    A = sparse.random(50, 100, density=0.1, format="csr")
    b = A @ np.ones(100)
    x = aip.solve(A, b)
    assert np.linalg.norm(A @ x - b) < 1e-6


# === Accordion tests ===

def test_pascal_index_basic():
    idx = PascalIndex(6, 4)
    # Constant monomial
    assert idx.combo_to_index(()) == 0
    # First variable
    assert idx.combo_to_index((0,)) == 1
    # Total monomials for C(6,0)+C(6,1)+C(6,2)+C(6,3)+C(6,4) = 1+6+15+20+15 = 57
    assert idx.total_monomials() == 57


def test_pascal_index_unique():
    """All indices should be unique."""
    from itertools import combinations
    idx = PascalIndex(6, 3)
    seen = set()
    for d in range(4):
        for combo in combinations(range(6), d):
            i = idx.combo_to_index(combo)
            assert i not in seen, f"Duplicate index {i} for {combo}"
            seen.add(i)
    assert len(seen) == idx.total_monomials()


def test_pascal_memory():
    idx = PascalIndex(30, 8)
    # Pascal table should be tiny
    assert idx.memory_bytes() < 10000  # <10 KB


def test_accordion_builder():
    builder = AccordionBuilder(num_rows=10)
    builder.add_entry(0, 0, 1.0)
    builder.add_entry(1, 1, 2.0)
    builder.add_entry(2, 2, 3.0)
    builder._batch_cols = 3
    builder.flush()
    assert len(builder.chunks) == 1
    assert builder.total_nnz == 3


def test_accordion_builder_batches():
    builder = AccordionBuilder(num_rows=5)

    # Batch 1
    builder.add_entries([0, 1], [0, 1], [1.0, 2.0], num_cols=2)
    builder.flush()

    # Batch 2
    builder.add_entries([2, 3], [0, 1], [3.0, 4.0], num_cols=2)
    builder.flush()

    assert len(builder.chunks) == 2
    assert builder.total_cols == 4
    assert builder.total_nnz == 4


def test_solve_chunks():
    """Solve a simple system using chunks."""
    # Create identity-like system in 2 chunks
    n = 10
    builder = AccordionBuilder(num_rows=n)

    # Chunk 1: first 5 columns
    for i in range(5):
        builder.add_entry(i, i, 2.0)
    builder._batch_cols = 5
    builder.flush()

    # Chunk 2: last 5 columns
    for i in range(5):
        builder.add_entry(i + 5, i, 2.0)
    builder._batch_cols = 5
    builder.flush()

    chunks = builder.finalize()
    b = np.ones(n)

    result = solve_chunks(chunks, b, verbose=False)
    assert result["feasible"]
    assert result["residual"] < 1e-6
    assert np.allclose(result["x"], 0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
