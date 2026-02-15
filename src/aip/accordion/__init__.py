"""
AIP Accordion: Memory-efficient computation for ultra-large sparse systems.

The Accordion engine processes matrices that would normally require
terabytes of RAM by:
  1. Computing indices mathematically instead of storing dictionaries
  2. Building sparse matrices in batches (never all in RAM at once)
  3. Solving with streaming LinearOperator over column chunks

Example:
    from aip.accordion import PascalIndex, AccordionBuilder, solve_chunks

    # Mathematical indexing (replaces dict, saves GB of RAM)
    index = PascalIndex(num_vars=30, max_degree=10)
    idx = index.combo_to_index((3, 7, 12))  # O(k) time, 0 extra memory

    # Batch construction
    builder = AccordionBuilder(num_rows=53_000_000)
    for batch_rows, batch_cols, batch_vals in my_batches():
        builder.add_batch(batch_rows, batch_cols, batch_vals)
    chunks = builder.finalize()

    # Streaming solve (never assembles full matrix)
    result = solve_chunks(chunks, b)

Author: Carmen Esteban
"""

from aip.accordion.indexing import PascalIndex
from aip.accordion.builder import AccordionBuilder
from aip.accordion.solver import solve_chunks, accordion_info

__all__ = ["PascalIndex", "AccordionBuilder", "solve_chunks", "accordion_info"]
