"""
AIP Accordion: Memory-efficient computation for ultra-large sparse systems.

The Accordion engine processes matrices that would normally require
terabytes of RAM by:
  1. Computing indices mathematically instead of storing dictionaries
  2. Building sparse matrices in batches (never all in RAM at once)
  3. Solving with streaming LinearOperator over column chunks

v0.4.0: Numba JIT acceleration, float32, preconditioner, parallel build.
v0.4.1: Add union_sorted for Boolean IPS, improve build_axiom_entries col_offset.

Example:
    from aip.accordion import PascalIndex, AccordionBuilder, solve_chunks

    # Mathematical indexing (replaces dict, saves GB of RAM)
    index = PascalIndex(num_vars=30, max_degree=10)
    idx = index.combo_to_index((3, 7, 12))  # O(k) time, 0 extra memory

    # Batch construction (float32 = half memory)
    builder = AccordionBuilder(num_rows=53_000_000, dtype='float32')
    for batch_rows, batch_cols, batch_vals in my_batches():
        builder.add_batch(batch_rows, batch_cols, batch_vals)
    chunks = builder.finalize()

    # Streaming solve with preconditioner
    result = solve_chunks(chunks, b, precondition=True)

Author: Carmen Esteban
"""

from aip.accordion import fast
from aip.accordion.indexing import PascalIndex
from aip.accordion.builder import AccordionBuilder
from aip.accordion.solver import solve_chunks, accordion_info

__all__ = [
    "PascalIndex", "AccordionBuilder", "solve_chunks", "accordion_info",
    "fast",
]
