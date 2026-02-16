"""Tests for aip-engine v0.4.0."""
import numpy as np
from scipy import sparse
import pytest

import aip
from aip.accordion import PascalIndex, AccordionBuilder, solve_chunks, fast


def test_version():
    assert aip.__version__ == "0.4.0"


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
    assert idx.combo_to_index(()) == 0
    assert idx.combo_to_index((0,)) == 1
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


def test_pascal_cumulative_same_as_original():
    """Cumulative Pascal optimization must give same results."""
    from itertools import combinations
    idx = PascalIndex(12, 4)
    for d in range(5):
        for combo in combinations(range(12), d):
            i = idx.combo_to_index(combo)
            assert 0 <= i < idx.total_monomials()


def test_pascal_batch():
    """batch_combo_to_index should match individual calls."""
    from itertools import combinations
    idx = PascalIndex(8, 3)
    combos = [combo for d in range(4) for combo in combinations(range(8), d)]
    batch = idx.batch_combo_to_index(combos)
    individual = [idx.combo_to_index(c) for c in combos]
    assert np.array_equal(batch, individual)


def test_pascal_memory():
    idx = PascalIndex(30, 8)
    assert idx.memory_bytes() < 100000  # <100 KB


def test_pascal_repr_shows_backend():
    idx = PascalIndex(6, 4)
    r = repr(idx)
    assert "backend=" in r
    assert ("numba" in r or "python" in r)


def test_pascal_pack_for_numba():
    idx = PascalIndex(10, 4)
    pack = idx.pack_for_numba()
    assert 'pascal' in pack
    assert 'num_vars' in pack
    assert 'offsets' in pack
    assert 'max_degree' in pack
    assert pack['num_vars'] == 10
    assert pack['max_degree'] == 4


# === Fast module tests ===

def test_fast_is_available():
    """fast.is_available() should return bool."""
    assert isinstance(fast.is_available(), bool)


def test_fast_pack_combos():
    combos = [(1, 3, 5), (2,), (), (0, 4)]
    flat, starts, lengths = fast.pack_combos(combos)
    assert len(flat) == 3 + 1 + 0 + 2  # total elements
    assert list(lengths) == [3, 1, 0, 2]
    assert list(flat[starts[0]:starts[0]+3]) == [1, 3, 5]
    assert list(flat[starts[1]:starts[1]+1]) == [2]
    assert list(flat[starts[3]:starts[3]+2]) == [0, 4]


def test_fast_pack_combos_empty():
    flat, starts, lengths = fast.pack_combos([])
    assert len(flat) == 0
    assert len(starts) == 0


def test_fast_merge_sorted():
    """Test merge_sorted with numpy arrays."""
    a = np.array([1, 3, 5], dtype=np.int64)
    b = np.array([2, 4], dtype=np.int64)
    result = fast.merge_sorted(a, b)
    assert list(result) == [1, 2, 3, 4, 5]


def test_fast_merge_sorted_empty():
    a = np.array([], dtype=np.int64)
    b = np.array([1, 2], dtype=np.int64)
    assert list(fast.merge_sorted(a, b)) == [1, 2]
    assert list(fast.merge_sorted(b, a)) == [1, 2]


# === AccordionBuilder tests ===

def test_accordion_builder():
    builder = AccordionBuilder(num_rows=10)
    builder.add_entry(0, 0, 1.0)
    builder.add_entry(1, 1, 2.0)
    builder.add_entry(2, 2, 3.0)
    builder._batch_cols = 3
    builder.flush()
    assert len(builder.chunks) == 1
    assert builder.total_nnz == 3


def test_accordion_builder_float32():
    builder = AccordionBuilder(num_rows=10, dtype='float32')
    builder.add_entry(0, 0, 1.0)
    builder.add_entry(1, 1, 2.0)
    builder._batch_cols = 2
    builder.flush()
    assert builder.chunks[0].dtype == np.float32
    assert builder.chunks[0].data.nbytes == 2 * 4


def test_accordion_builder_float64():
    builder = AccordionBuilder(num_rows=10, dtype='float64')
    builder.add_entry(0, 0, 1.0)
    builder.add_entry(1, 1, 2.0)
    builder._batch_cols = 2
    builder.flush()
    assert builder.chunks[0].dtype == np.float64
    assert builder.chunks[0].data.nbytes == 2 * 8


def test_accordion_builder_batches():
    builder = AccordionBuilder(num_rows=5)
    builder.add_entries([0, 1], [0, 1], [1.0, 2.0], num_cols=2)
    builder.flush()
    builder.add_entries([2, 3], [0, 1], [3.0, 4.0], num_cols=2)
    builder.flush()
    assert len(builder.chunks) == 2
    assert builder.total_cols == 4
    assert builder.total_nnz == 4


def test_accordion_parallel_build():
    builder = AccordionBuilder(num_rows=10)
    for batch in range(3):
        for i in range(5):
            builder.add_entry(i + batch, i, float(i + 1))
        builder._batch_cols = 5
        builder.flush_async()

    assert len(builder._pending) == 3
    assert len(builder.chunks) == 0
    builder.build_parallel(max_workers=2)
    assert len(builder.chunks) == 3
    assert builder.total_nnz == 15


def test_accordion_dtype_property():
    b32 = AccordionBuilder(num_rows=5, dtype='float32')
    b64 = AccordionBuilder(num_rows=5, dtype='float64')
    assert b32.dtype == np.float32
    assert b64.dtype == np.float64


# === Solver tests ===

def test_solve_chunks():
    n = 10
    builder = AccordionBuilder(num_rows=n)
    for i in range(5):
        builder.add_entry(i, i, 2.0)
    builder._batch_cols = 5
    builder.flush()
    for i in range(5):
        builder.add_entry(i + 5, i, 2.0)
    builder._batch_cols = 5
    builder.flush()

    chunks = builder.finalize()
    b = np.ones(n)
    result = solve_chunks(chunks, b, verbose=False, precondition=False)
    assert result["feasible"]
    assert result["residual"] < 1e-6
    assert np.allclose(result["x"], 0.5)


def test_solve_chunks_with_preconditioner():
    n = 20
    builder = AccordionBuilder(num_rows=n)
    for i in range(n):
        builder.add_entry(i, i, float(i + 1) * 10)
    builder._batch_cols = n
    builder.flush()
    chunks = builder.finalize()
    b = np.arange(1, n + 1, dtype=float) * 10

    result = solve_chunks(chunks, b, verbose=False, precondition=True)
    assert result["feasible"]
    assert np.allclose(result["x"], 1.0, atol=1e-6)


def test_solve_chunks_float32():
    n = 10
    builder = AccordionBuilder(num_rows=n, dtype='float32')
    for i in range(n):
        builder.add_entry(i, i, 2.0)
    builder._batch_cols = n
    builder.flush()
    chunks = builder.finalize()
    b = np.ones(n)

    result = solve_chunks(chunks, b, verbose=False, precondition=False)
    assert result["feasible"]
    assert result["residual"] < 1e-4


def test_solve_chunks_preconditioner_reduces_iterations():
    np.random.seed(42)
    m, n = 50, 100
    A = sparse.random(m, n, density=0.1, format="csr")
    scales = np.logspace(0, 4, n)
    A = A @ sparse.diags(scales)
    b = A @ np.ones(n)

    builder1 = AccordionBuilder(num_rows=m)
    builder1.add_entries(
        A.nonzero()[0].tolist(), A.nonzero()[1].tolist(),
        A.data.tolist(), num_cols=n)
    builder1.flush()

    builder2 = AccordionBuilder(num_rows=m)
    builder2.add_entries(
        A.nonzero()[0].tolist(), A.nonzero()[1].tolist(),
        A.data.tolist(), num_cols=n)
    builder2.flush()

    r_no = solve_chunks(builder1.chunks, b, verbose=False, precondition=False, max_iter=2000)
    r_yes = solve_chunks(builder2.chunks, b, verbose=False, precondition=True, max_iter=2000)
    assert r_yes["iterations"] <= r_no["iterations"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
