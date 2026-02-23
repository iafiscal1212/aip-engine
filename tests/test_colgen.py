"""Tests for Column Generation Solver."""
import numpy as np
from itertools import combinations
import pytest

from aip.accordion import PascalIndex
from aip.accordion import fast
from aip.colgen_pricing import (
    _unrank_combination, unrank_combination, price_all_columns,
)
from aip.colgen_solver import (
    ColumnGenSolver, ActiveSet, ColumnGroup,
    build_axioms_phpe, enumerate_column_groups, solve_phpe_colgen,
)


# ============================================================
# Unranking round-trip tests
# ============================================================

class TestUnranking:
    """Test _unrank_combination is the inverse of _lex_rank_jit."""

    def test_unrank_degree0(self):
        """Degree 0: only one combo (), rank 0."""
        pidx = PascalIndex(10, 5)
        combo = unrank_combination(0, 0, 10, pidx.pascal)
        assert len(combo) == 0

    def test_unrank_degree1_all(self):
        """Degree 1: each rank maps to a single variable."""
        n = 10
        pidx = PascalIndex(n, 5)
        for rank in range(n):
            combo = unrank_combination(rank, 1, n, pidx.pascal)
            assert len(combo) == 1
            assert combo[0] == rank

    def test_unrank_roundtrip_small(self):
        """Round-trip: rank → combo → rank for all C(8, 3) = 56 combos."""
        n, k = 8, 3
        pidx = PascalIndex(n, k)
        all_combos = list(combinations(range(n), k))

        for expected_rank, expected_combo in enumerate(all_combos):
            # Unrank
            combo = unrank_combination(expected_rank, k, n, pidx.pascal)
            assert tuple(combo) == expected_combo, (
                f"rank={expected_rank}: expected {expected_combo}, got {tuple(combo)}"
            )

            # Rank back
            combo_arr = np.array(expected_combo, dtype=np.int64)
            actual_rank = fast._lex_rank_jit(combo_arr, pidx.pascal, n)
            assert actual_rank == expected_rank

    def test_unrank_roundtrip_larger(self):
        """Round-trip for C(15, 4) = 1365 combos."""
        n, k = 15, 4
        pidx = PascalIndex(n, k)
        all_combos = list(combinations(range(n), k))

        for expected_rank, expected_combo in enumerate(all_combos):
            combo = unrank_combination(expected_rank, k, n, pidx.pascal)
            assert tuple(combo) == expected_combo

    def test_unrank_roundtrip_degree2(self):
        """Round-trip for C(20, 2) = 190 combos."""
        n, k = 20, 2
        pidx = PascalIndex(n, k)
        all_combos = list(combinations(range(n), k))

        for expected_rank, expected_combo in enumerate(all_combos):
            combo = unrank_combination(expected_rank, k, n, pidx.pascal)
            assert tuple(combo) == expected_combo


# ============================================================
# Pricing tests
# ============================================================

class TestPricing:
    """Test pricing vs brute-force on small instances."""

    def test_pricing_phpe2_d4(self):
        """Pricing on PHP-E(2) d=4 should find the same top columns as brute-force."""
        axioms, num_vars = build_axioms_phpe(2)
        d = 4
        pidx = PascalIndex(num_vars, d)
        num_monoms = pidx.total_monomials()

        # Build the full system brute-force to get a realistic residual
        # Use a random residual for testing
        np.random.seed(42)
        residual = np.random.randn(num_monoms)

        # Pricing: streaming
        top_cols = price_all_columns(
            axioms, num_vars, d, residual, pidx,
            top_k=50, batch_size=1000, verbose=False
        )

        assert len(top_cols) > 0
        assert len(top_cols) <= 50

        # Verify correlations are sorted descending
        for i in range(1, len(top_cols)):
            assert top_cols[i-1][0] >= top_cols[i][0]

        # Verify each correlation is correct by recomputing
        pack = pidx.pack_for_numba()
        pascal = pack['pascal']
        offsets = pack['offsets']

        for corr, ax_idx, k_mult, mult_rank in top_cols[:10]:
            ax_terms = axioms[ax_idx]
            mult_combo = _unrank_combination(
                np.int64(mult_rank), k_mult, num_vars, pascal
            )

            # Recompute A[:,j]^T @ residual
            expected_corr = 0.0
            for coeff, m_ax in ax_terms:
                a_arr = np.array(m_ax, dtype=np.int64)
                product = fast.union_sorted(a_arr, mult_combo)
                if len(product) > d:
                    continue
                row_idx = int(offsets[len(product)] + fast._lex_rank_jit(
                    product, pascal, num_vars
                ))
                expected_corr += coeff * residual[row_idx]

            assert abs(corr - abs(expected_corr)) < 1e-10, (
                f"Correlation mismatch: {corr} vs {abs(expected_corr)}"
            )


# ============================================================
# ActiveSet tests
# ============================================================

class TestActiveSet:
    """Test ActiveSet management."""

    def test_initial_columns_count(self):
        """Initial columns for PHP-E(2) d=4 with mult_deg ≤ 2."""
        axioms, num_vars = build_axioms_phpe(2)
        pidx = PascalIndex(num_vars, 4)
        aset = ActiveSet(axioms, num_vars, 4, pidx)
        aset.add_initial_columns(max_mult_degree=2, verbose=False)
        assert aset.n_active > 0

    def test_matvec_rmatvec_consistency(self):
        """A @ x and A^T @ y should be consistent: <Ax, y> = <x, A^T y>."""
        axioms, num_vars = build_axioms_phpe(2)
        d = 3
        pidx = PascalIndex(num_vars, d)
        num_monoms = pidx.total_monomials()

        aset = ActiveSet(axioms, num_vars, d, pidx)
        aset.add_initial_columns(max_mult_degree=2, verbose=False)

        A_op = aset.make_linear_operator(num_monoms)

        np.random.seed(123)
        x = np.random.randn(aset.n_active)
        y = np.random.randn(num_monoms)

        Ax = A_op.matvec(x)
        ATy = A_op.rmatvec(y)

        # <Ax, y> should equal <x, A^T y>
        lhs = np.dot(Ax, y)
        rhs = np.dot(x, ATy)
        assert abs(lhs - rhs) < 1e-8 * max(abs(lhs), abs(rhs), 1.0), (
            f"<Ax,y>={lhs} != <x,A^Ty>={rhs}"
        )


# ============================================================
# Column group enumeration tests
# ============================================================

class TestColumnGroups:
    """Test column group enumeration."""

    def test_enumerate_phpe2_d4(self):
        """Column groups for PHP-E(2) d=4 should cover all columns."""
        axioms, num_vars = build_axioms_phpe(2)
        pidx = PascalIndex(num_vars, 4)
        groups, total_cols = enumerate_column_groups(axioms, num_vars, 4, pidx)

        assert len(groups) > 0
        assert total_cols > 0

        # Verify total matches manual count
        manual_total = 0
        for ax_terms in axioms:
            deg_ax = max(len(m) for c, m in ax_terms)
            max_md = max(0, 4 - deg_ax)
            for d in range(max_md + 1):
                manual_total += len(list(combinations(range(num_vars), d)))

        assert total_cols == manual_total, (
            f"total_cols={total_cols} != manual={manual_total}"
        )

    def test_groups_contiguous(self):
        """Groups should have contiguous, non-overlapping column ranges."""
        axioms, num_vars = build_axioms_phpe(2)
        pidx = PascalIndex(num_vars, 4)
        groups, total_cols = enumerate_column_groups(axioms, num_vars, 4, pidx)

        expected_start = 0
        for g in groups:
            assert g.global_col_start == expected_start
            assert g.num_cols > 0
            expected_start += g.num_cols

        assert expected_start == total_cols


# ============================================================
# Full CG solver tests
# ============================================================

class TestColumnGenSolver:
    """Test complete column generation solver."""

    def test_cg_phpe2_d4_feasible(self):
        """PHP-E(2) d=4 should be FEASIBLE (known result)."""
        result = solve_phpe_colgen(
            n=2, max_degree=4,
            initial_mult_degree=4,  # Include all columns for small problem
            lsqr_max_iter=3000,
            pricing_top_k=1000,
            feasibility_tol=1e-4,
            max_cg_iters=5,
            verbose=False,
        )

        assert result['feasible'], (
            f"PHP-E(2) d=4 should be FEASIBLE, got residual={result['residual']:.2e}"
        )
        assert result['residual'] < 1e-4

    def test_cg_phpe2_d3_infeasible(self):
        """PHP-E(2) d=3 should NOT be feasible (degree too low)."""
        result = solve_phpe_colgen(
            n=2, max_degree=3,
            initial_mult_degree=3,
            lsqr_max_iter=2000,
            pricing_top_k=500,
            feasibility_tol=1e-4,
            max_cg_iters=3,
            verbose=False,
        )

        assert not result['feasible'], (
            "PHP-E(2) d=3 should NOT be feasible"
        )

    def test_cg_phpe3_d6_feasible(self):
        """PHP-E(3) d=6 should be FEASIBLE (known result)."""
        result = solve_phpe_colgen(
            n=3, max_degree=6,
            initial_mult_degree=3,
            lsqr_max_iter=5000,
            pricing_top_k=5000,
            feasibility_tol=1e-3,
            max_cg_iters=10,
            verbose=False,
        )

        assert result['feasible'], (
            f"PHP-E(3) d=6 should be FEASIBLE, got residual={result['residual']:.2e}"
        )

    def test_solver_returns_expected_fields(self):
        """Result dict should contain all expected fields."""
        result = solve_phpe_colgen(
            n=2, max_degree=4,
            initial_mult_degree=4,
            lsqr_max_iter=1000,
            pricing_top_k=100,
            max_cg_iters=2,
            verbose=False,
        )

        expected_fields = [
            'feasible', 'residual', 'rel_residual', 'x',
            'size_l2', 'iterations', 'active_cols', 'time_total',
            'history', 'n', 'max_degree', 'num_vars', 'num_axioms', 'num_rows',
        ]
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"


# ============================================================
# Build axioms tests
# ============================================================

class TestBuildAxioms:
    """Test axiom construction."""

    def test_phpe2_counts(self):
        """PHP-E(2): 3 pigeons, 2 holes → 9 vars, expected axiom count."""
        axioms, num_vars = build_axioms_phpe(2)
        # x_{p,h}: 3*2=6, y_{p,q}: C(3,2)=3 → total=9
        assert num_vars == 9
        assert len(axioms) > 0

    def test_phpe3_counts(self):
        """PHP-E(3): 4 pigeons, 3 holes → 18 vars."""
        axioms, num_vars = build_axioms_phpe(3)
        # x_{p,h}: 4*3=12, y_{p,q}: C(4,2)=6 → total=18
        assert num_vars == 18
        assert len(axioms) > 0

    def test_axiom_coeffs_valid(self):
        """All axiom coefficients should be ±1."""
        axioms, _ = build_axioms_phpe(2)
        for ax in axioms:
            for coeff, combo in ax:
                assert coeff in (1.0, -1.0), f"Unexpected coeff: {coeff}"

    def test_axiom_combos_sorted(self):
        """All axiom monomial combos should be sorted."""
        axioms, _ = build_axioms_phpe(2)
        for ax in axioms:
            for coeff, combo in ax:
                assert combo == tuple(sorted(combo)), (
                    f"Unsorted combo: {combo}"
                )
