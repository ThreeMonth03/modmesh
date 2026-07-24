# Apple Silicon Cartesian matmul summary

## Environment

The clean run used revision `6120aec8`, macOS 26.5.1 on an Apple M1 with
8 GB of memory, native arm64 Python 3.11.6, NumPy 2.2.4, and one thread.
NumPy and `_solvcon` both link to Accelerate.

The correctness run used side 4 and batch 3.  The timing run used side 32,
batch 4, one call per route, seven paired samples, and two warmups.  The
[machine-readable summary](macos-matmul-cartesian-summary.json) retains the
exact coverage counts and aggregate ratios.  The complete raw timing JSON is
96 MiB and is not committed to the branch.  It can be reproduced with the
command below.

## Complete Cartesian result

All 31,825 declared lhs by rhs layout pairs match NumPy.  No row classifies
NumPy as conclusively faster.

| Comparison with NumPy | Cases |
| --- | ---: |
| Planned faster | 29,939 |
| Parity | 378 |
| Inconclusive | 1,508 |
| NumPy faster | 0 |

The overall NumPy/planned median is 21.410x, with q10 and q90 values of
1.224x and 25.073x.  The large ratios are concentrated in non-contiguous
matrix and cross-broadcast layouts.  They should be read as same-backend
route behavior for this small matrix size, not as a portable speed claim.

| Topology | Cases | Planned faster | Parity | Inconclusive | NumPy faster | Median NumPy/planned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1d-1d` | 25 | 15 | 0 | 10 | 0 | 1.232x |
| `1d-2d` | 50 | 46 | 0 | 4 | 0 | 1.309x |
| `2d-1d` | 50 | 47 | 0 | 3 | 0 | 1.302x |
| `2d-2d` | 100 | 100 | 0 | 0 | 0 | 6.343x |
| `1d-nd` | 200 | 199 | 0 | 1 | 0 | 1.207x |
| `nd-1d` | 200 | 200 | 0 | 0 | 0 | 1.151x |
| `2d-nd` | 400 | 389 | 0 | 11 | 0 | 12.676x |
| `nd-2d` | 400 | 389 | 0 | 11 | 0 | 12.562x |
| `nd-nd-same-batch` | 1,600 | 1,529 | 0 | 71 | 0 | 10.312x |
| `nd-nd-lhs-broadcast` | 1,600 | 1,493 | 0 | 107 | 0 | 12.496x |
| `nd-nd-rhs-broadcast` | 1,600 | 1,531 | 0 | 69 | 0 | 12.430x |
| `nd-nd-cross-broadcast` | 25,600 | 24,001 | 378 | 1,221 | 0 | 21.557x |

## Stable vector control

The focused run retains all 400 `1D @ ND` and `ND @ 1D` layout pairs.  Each
route executes 100 calls in each of 15 paired samples after five warmups.
The [complete notebook](macos-matmul-vector-cartesian-results.md) and
[raw JSON](macos-matmul-vector-cartesian-results.json) retain every row.

`Generic/current` above one means current dispatch is faster.  A
`BLAS/current` value below one means the forced BLAS route is faster.

| Layout class | Pairs | Generic/current | BLAS/current | Current route |
| --- | ---: | ---: | ---: | --- |
| Positive vector stride and direct GEMV matrix | 48 | 2.093x (1.470..2.352) | 0.996x (0.992..0.999) | BLAS |
| Negative or zero vector and direct GEMV matrix | 72 | 0.997x (0.995..0.999) | 0.535x (0.490..0.582) | Generic |
| Matrix not directly describable by GEMV | 280 | 0.997x (0.995..0.999) | 0.999x (0.997..1.001) | Generic |

All 48 positive-stride rows make forced generic execution conclusively
slower.  Forced BLAS is at parity in 45 rows and inconclusive in three, so
the existing direct fast path passes the Apple gate.

All 72 negative or zero vector rows make pack-once forced BLAS conclusively
faster than current dispatch.  Its median time is 0.535 of current, or about
1.87 times faster.  This supports widening the automatic pack-once vector
path on Accelerate at the measured work and batch size.  It does not justify
sending the 280 non-describable matrix layouts through BLAS.

## Reproduce

```console
$ source /path/to/devenv/scripts/init
$ devenv use prime
$ PYTHONPATH=.:profiling python3 \
    profiling/profile_matmul_cartesian.py \
    --side 4 --batch 3 --check-only
$ PYTHONPATH=.:profiling python3 \
    profiling/profile_matmul_cartesian.py \
    --side 32 --batch 4 --number 1 \
    --repeat 7 --warmup 2 \
    --output /tmp/macos-matmul-cartesian.json
$ PYTHONPATH=.:profiling python3 \
    profiling/profile_matmul_cartesian.py \
    --side 32 --batch 4 --number 100 \
    --repeat 15 --warmup 5 \
    --filter 1d-nd/ --filter nd-1d/ \
    --output /tmp/macos-matmul-vector-cartesian.json
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
