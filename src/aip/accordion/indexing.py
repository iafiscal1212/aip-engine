"""
Accordion Indexing: Combinatorial Number System for monomial indices.

Computes the index of any k-subset of {0, 1, ..., n-1} in O(k) time
using a precomputed Pascal table. Replaces dictionaries that would
consume gigabytes of RAM for large problems.

v0.4.0: Cumulative Pascal sums + Numba JIT acceleration.

Author: Carmen Esteban
"""

import numpy as np
from aip.accordion import fast as _fast


class PascalIndex:
    """
    Mathematical monomial indexing via Combinatorial Number System.

    Automatically uses Numba JIT if available (50-100x faster).
    Falls back to pure Python otherwise.

    Parameters
    ----------
    num_vars : int
        Number of variables (n).
    max_degree : int
        Maximum monomial degree (d).

    Examples
    --------
    >>> idx = PascalIndex(30, 8)
    >>> idx.total_monomials()
    263950
    >>> idx.combo_to_index((3, 7, 12))  # degree-3 monomial
    1584
    >>> idx.combo_to_index(())  # constant monomial
    0
    """

    def __init__(self, num_vars, max_degree):
        self.num_vars = num_vars
        self.max_degree = max_degree
        self.pascal = self._build_pascal(num_vars + 1, max_degree + 1)

        # Precompute degree offsets
        self._offsets_list = []
        offset = 0
        for d in range(max_degree + 1):
            self._offsets_list.append(offset)
            offset += int(self.pascal[num_vars, d])
        self._total = offset

        # Numpy version for numba kernels
        self._offsets_np = np.array(self._offsets_list, dtype=np.int64)

        # Precompute cumulative Pascal sums for pure-Python fast path
        self._cum_pascal = {}
        for remaining in range(max_degree + 1):
            cum = np.zeros(num_vars + 1, dtype=np.int64)
            for j in range(num_vars):
                available = num_vars - 1 - j
                if available >= remaining >= 0:
                    cum[j + 1] = cum[j] + int(self.pascal[available, remaining])
                else:
                    cum[j + 1] = cum[j]
            self._cum_pascal[remaining] = cum

        # Warm up Numba JIT if available
        self._use_numba = _fast.is_available()
        if self._use_numba:
            _fast.warmup(self.pascal, self.num_vars, self._offsets_np)

    @staticmethod
    def _build_pascal(n_max, k_max):
        """Precompute Pascal's triangle C(n, k)."""
        table = np.zeros((n_max + 1, k_max + 1), dtype=np.int64)
        for n in range(n_max + 1):
            table[n, 0] = 1
            for k in range(1, min(n, k_max) + 1):
                table[n, k] = table[n-1, k-1] + table[n-1, k]
        return table

    def total_monomials(self):
        """Total number of monomials up to max_degree."""
        return self._total

    def combo_to_index(self, combo):
        """
        Convert a sorted tuple of variable indices to a global monomial index.

        Uses Numba JIT if available, pure Python otherwise.

        Parameters
        ----------
        combo : tuple of int
            Sorted variable indices, e.g., (3, 7, 12) for x3*x7*x12.
            Empty tuple () represents the constant monomial (degree 0).

        Returns
        -------
        int
            Global index in [0, total_monomials).
        """
        d = len(combo)
        if d > self.max_degree:
            raise ValueError(f"Degree {d} exceeds max_degree {self.max_degree}")

        if self._use_numba:
            combo_arr = np.array(combo, dtype=np.int64)
            return int(_fast.combo_to_index_jit(
                combo_arr, self.pascal, self.num_vars, self._offsets_np))

        offset = self._offsets_list[d]
        rank = self._lex_rank(combo)
        return offset + rank

    def _lex_rank(self, combo):
        """Lexicographic rank using precomputed cumulative Pascal sums."""
        k = len(combo)
        if k == 0:
            return 0
        rank = 0
        prev = -1
        for i, a in enumerate(combo):
            remaining = k - 1 - i
            cum = self._cum_pascal[remaining]
            rank += int(cum[a] - cum[prev + 1])
            prev = a
        return rank

    def batch_combo_to_index(self, combos):
        """
        Convert multiple combos to indices at once.

        Uses Numba JIT batch kernel if available (fastest path).

        Parameters
        ----------
        combos : list of tuple
            List of sorted variable index tuples.

        Returns
        -------
        numpy.ndarray of int64
            Array of global indices.
        """
        if not combos:
            return np.empty(0, dtype=np.int64)

        if self._use_numba:
            flat, starts, lengths = _fast.pack_combos(combos)
            return _fast.batch_combo_to_index_jit(
                flat, starts, lengths,
                self.pascal, self.num_vars, self._offsets_np)

        result = np.empty(len(combos), dtype=np.int64)
        for i, combo in enumerate(combos):
            result[i] = self.combo_to_index(combo)
        return result

    def pack_for_numba(self):
        """Return arrays needed for direct numba kernel calls.

        Returns
        -------
        dict with pascal, num_vars, offsets, max_degree
        """
        return {
            'pascal': self.pascal,
            'num_vars': self.num_vars,
            'offsets': self._offsets_np,
            'max_degree': self.max_degree,
        }

    def memory_bytes(self):
        """Memory used by Pascal table + cumulative sums."""
        total = self.pascal.nbytes + self._offsets_np.nbytes
        for cum in self._cum_pascal.values():
            total += cum.nbytes
        return total

    def __repr__(self):
        accel = "numba" if self._use_numba else "python"
        return (f"PascalIndex(vars={self.num_vars}, deg={self.max_degree}, "
                f"monomials={self._total:,}, backend={accel})")
