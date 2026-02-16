"""
Accordion Builder: Batch construction of ultra-large sparse matrices.

Instead of accumulating billions of entries in Python lists (~28 bytes each),
uses array.array (C native, 4-8 bytes each) and converts each batch to
CSR immediately, freeing raw data.

v0.4.0: Added float32 support (dtype parameter) and parallel chunk building.

Author: Carmen Esteban
"""

import numpy as np
from scipy import sparse
import array as pyarray
import gc
from concurrent.futures import ProcessPoolExecutor
import os


def _build_chunk_from_data(args):
    """Worker function for parallel chunk building."""
    rows_bytes, cols_bytes, vals_bytes, num_rows, num_cols, dtype_str = args
    rows_np = np.frombuffer(rows_bytes, dtype=np.int32).copy()
    cols_np = np.frombuffer(cols_bytes, dtype=np.int32).copy()
    np_dtype = np.float32 if dtype_str == 'f' else np.float64
    vals_np = np.frombuffer(vals_bytes, dtype=np_dtype).copy()

    chunk = sparse.csr_matrix(
        (vals_np, (rows_np, cols_np)),
        shape=(num_rows, num_cols),
        dtype=np_dtype,
    )
    return chunk


class AccordionBuilder:
    """
    Build a sparse matrix in column-chunks for memory efficiency.

    Parameters
    ----------
    num_rows : int
        Number of rows in the final matrix.
    dtype : str
        'float32' for half memory, 'float64' for full precision (default).

    Examples
    --------
    >>> builder = AccordionBuilder(num_rows=1000, dtype='float32')
    >>> builder.add_entries(row_indices, col_indices, values, num_cols=50)
    >>> builder.flush()
    >>> chunks = builder.finalize()
    """

    def __init__(self, num_rows, dtype='float64'):
        self.num_rows = num_rows
        self.chunks = []
        self.total_cols = 0
        self.total_nnz = 0

        # dtype config
        if dtype in ('float32', 'f', np.float32):
            self._array_code = 'f'  # float32: 4 bytes/elem
            self._np_dtype = np.float32
        else:
            self._array_code = 'd'  # float64: 8 bytes/elem
            self._np_dtype = np.float64

        # Current batch (C-native arrays for memory efficiency)
        self._rows = pyarray.array('i')  # int32: 4 bytes/elem
        self._cols = pyarray.array('i')
        self._vals = pyarray.array(self._array_code)
        self._batch_cols = 0

        # Pending chunks for parallel building
        self._pending = []

    def add_entry(self, row, col, val):
        """Add a single entry to the current batch."""
        self._rows.append(row)
        self._cols.append(col)
        self._vals.append(val)
        if col >= self._batch_cols:
            self._batch_cols = col + 1

    def add_entries(self, rows, cols, vals, num_cols=None):
        """
        Add multiple entries to the current batch.

        Parameters
        ----------
        rows, cols : array-like of int
            Row and column indices.
        vals : array-like of float
            Values.
        num_cols : int, optional
            Number of columns in this batch. If None, inferred from max(cols)+1.
        """
        self._rows.extend(rows)
        self._cols.extend(cols)
        self._vals.extend(vals)
        if num_cols is not None:
            self._batch_cols = max(self._batch_cols, num_cols)
        elif len(cols) > 0:
            self._batch_cols = max(self._batch_cols, max(cols) + 1)

    def flush(self):
        """Convert current batch to CSR chunk and free raw arrays."""
        if self._batch_cols == 0:
            return

        rows_np = np.frombuffer(self._rows, dtype=np.int32).copy()
        cols_np = np.frombuffer(self._cols, dtype=np.int32).copy()
        vals_np = np.frombuffer(self._vals, dtype=self._np_dtype).copy()

        chunk = sparse.csr_matrix(
            (vals_np, (rows_np, cols_np)),
            shape=(self.num_rows, self._batch_cols),
            dtype=self._np_dtype,
        )

        self.chunks.append(chunk)
        self.total_cols += self._batch_cols
        self.total_nnz += chunk.nnz

        # Free C-native arrays
        del rows_np, cols_np, vals_np
        self._rows = pyarray.array('i')
        self._cols = pyarray.array('i')
        self._vals = pyarray.array(self._array_code)
        self._batch_cols = 0
        gc.collect()

    def flush_async(self):
        """Queue current batch for parallel building (call build_parallel later)."""
        if self._batch_cols == 0:
            return

        # Store raw bytes for pickling to workers
        self._pending.append((
            bytes(self._rows),
            bytes(self._cols),
            bytes(self._vals),
            self.num_rows,
            self._batch_cols,
            self._array_code,
        ))
        self.total_cols += self._batch_cols

        # Reset batch
        self._rows = pyarray.array('i')
        self._cols = pyarray.array('i')
        self._vals = pyarray.array(self._array_code)
        self._batch_cols = 0
        gc.collect()

    def build_parallel(self, max_workers=None):
        """Build all pending chunks in parallel using multiprocessing."""
        if not self._pending:
            return

        if max_workers is None:
            max_workers = min(len(self._pending), os.cpu_count() or 1)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_build_chunk_from_data, self._pending))

        for chunk in results:
            self.chunks.append(chunk)
            self.total_nnz += chunk.nnz

        self._pending = []

    def finalize(self):
        """Flush remaining entries and return list of CSR chunks."""
        if self._batch_cols > 0:
            self.flush()
        if self._pending:
            self.build_parallel()
        return self.chunks

    def memory_bytes(self):
        """Total memory of all CSR chunks."""
        total = 0
        for c in self.chunks:
            total += c.data.nbytes + c.indices.nbytes + c.indptr.nbytes
        return total

    @property
    def dtype(self):
        return self._np_dtype

    def __repr__(self):
        return (f"AccordionBuilder(rows={self.num_rows:,}, "
                f"dtype={self._np_dtype.__name__}, "
                f"chunks={len(self.chunks)}, "
                f"cols={self.total_cols:,}, "
                f"nnz={self.total_nnz:,}, "
                f"ram={self.memory_bytes() / 1e6:.1f} MB)")
