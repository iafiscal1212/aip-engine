# AIP Engine

**Algebraic Independence Processor** — auto-detection of matrix structure + memory-efficient computation for ultra-large sparse systems.

```
pip install aip-engine
```

## What it does

AIP Engine solves a real problem: building and solving sparse linear systems that are **too large for conventional tools**. It does three things:

1. **Detects** matrix structure automatically (sparse/dense, square/rectangular)
2. **Routes** to the optimal solver (LSQR, spsolve, LAPACK)
3. **Accordion Memory**: builds and solves ultra-large systems without running out of RAM

## Quick start

```python
import aip

# Auto-detect structure and solve
report = aip.detect_matrix(A)
x = aip.solve(A, b)
```

## Accordion Memory

For systems with millions or billions of entries that don't fit in RAM:

```python
from aip.accordion import PascalIndex, AccordionBuilder, solve_chunks

# 1. Mathematical indexing (replaces dictionary, saves GB of RAM)
index = PascalIndex(num_vars=20, max_degree=6)
idx = index.combo_to_index((3, 7, 12))  # O(k) time, 0 extra memory

# 2. Batch construction (never all in RAM at once)
builder = AccordionBuilder(num_rows=1_000_000)
# ... add entries in batches ...
builder.flush()  # converts to CSR chunk, frees raw data
chunks = builder.finalize()

# 3. Streaming solve (never assembles full matrix)
result = solve_chunks(chunks, b, max_iter=10000)
print(result['residual'], result['size_l2'])
```

## Why Accordion?

| | Without Accordion | With Accordion |
|---|---|---|
| Monomial index (millions of entries) | ~2 GB dictionary | 0 MB (computed mathematically) |
| Matrix construction (hundreds of millions) | ~12 GB Python lists | ~2.4 GB array.array per batch |
| Full matrix (millions x billions) | petabytes dense | GB-scale sparse chunks |
| Solve | needs full matrix in RAM | streaming over chunks |

Real-world results:

| Problem | Matrix size | Dense would be | Accordion uses | Compression |
|---|---|---|---|---|
| Large sparse system A | 8.6M x 78M | 5.4 PB | 1.2 GB | 4,640,586x |
| Ultra-large sparse system B | 53M x 1.17B | 496,052 TB | 14.5 GB | 34,215,310x |

## How it works

### PascalIndex

Uses the [Combinatorial Number System](https://en.wikipedia.org/wiki/Combinatorial_number_system) to compute the index of any monomial in O(k) time using a precomputed Pascal table. No dictionary needed.

```python
index = PascalIndex(num_vars=20, max_degree=6)
print(index)  # PascalIndex(vars=20, deg=6, monomials=60,459, pascal=336 bytes)
# 60K monomials indexed with 336 bytes of memory
```

### AccordionBuilder

Builds sparse matrices in batches using `array.array` (C-native, 4-8 bytes/element) instead of Python lists (28 bytes/element). Each batch is converted to CSR immediately and raw arrays are freed.

### solve_chunks

LSQR solver that operates on a `LinearOperator` built from column chunks. The matvec/rmatvec operations iterate over chunks sequentially, never needing the full matrix in memory.

## Requirements

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7

## License

MIT License - Carmen Esteban, 2025-2026
