"""
AIP Engine - Algebraic Independence Processor
==============================================

Auto-detection of matrix structure + memory-efficient computation.

Quick start:
    import aip

    # Detect matrix structure
    report = aip.detect_matrix(A)

    # Solve linear system (auto-routes sparse/dense)
    x = aip.solve(A, b)

    # Build ultra-large sparse systems with Accordion memory
    from aip.accordion import AccordionBuilder
    builder = AccordionBuilder(num_rows=53_000_000)
    builder.add_batch(rows, cols, vals)
    chunks = builder.finalize()
    x = aip.accordion.solve(chunks, b)

Author: Carmen Esteban
License: MIT
"""

__version__ = "0.4.0"
__author__ = "Carmen Esteban"

from aip.detector import detect_matrix
from aip.solver import solve
from aip import accordion

__all__ = ["detect_matrix", "solve", "accordion"]
