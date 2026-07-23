# Linux detailed execution benchmark

## Recorded environment

- Code revision: `c26a2bbc71f786c9614bbc77affebd25310d5e0d`.
- Dirty tree: `false`.
- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`.
- Machine: `x86_64`.
- Python: `3.12.7`.
- NumPy: `2.3.0`.
- Seed: `20260722`.
- Fixed cases: `5`.
- HPC matmul suite: `false`.
- Optional slow matmul case: `false`.
- Samples per route: `31`.
- Warmups per route: `3`.
- Threads: `1`.
- CPU affinity: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23`.

The machine-readable JSON also records NumPy build configuration,
extension linkage, thread-control variables, raw timing samples,
paired ratios, and q10/q90 ratio quantiles.

## Reproduction

```console
$ source /path/to/devenv/scripts/init
$ devenv use prime
$ python3 profiling/profile_execution_prototype.py \
    --benchmark-only \
    --filter matrix-matrix-c \
    --filter matrix-matrix-small-direct \
    --filter vector-vector \
    --filter vector-matrix \
    --filter matrix-vector \
    --repeat 31 \
    --warmup 3 \
    --output /tmp/solvcon-execution-c26a2bbc.json
```

The profiler silently checks every route against NumPy before
timing. Mutable destinations are reset before every route and
sample. Call order rotates within each paired sample.

## Reading the tables

- Strides are measured in elements, not bytes.
- `C` and `F` mean C- and F-contiguous.
- `N` means at least one negative stride.
- `S` means a general non-dense strided view.
- `Z` means at least one zero stride.
- A ratio greater than one means planned is faster.
- Every ratio shows `median (q10..q90)` from paired samples.
- Faster or slower requires the full interval to clear five
  percent. A crossing interval is `inconclusive`.
- Existing BLAS is `SimpleArray::matmul_blas()` and is measured
  only for unbatched rank-1 and rank-2 operations.
- A control is another planned route measured in the same rotating
  sample. Its label identifies the compared layout.
- `legacy-incorrect` means planned matched NumPy but legacy did
  not. Incorrect legacy results are not timed.

## Coverage inventory

| Family | Cases | Improved | Parity | Regression | Legacy incorrect | New only | Inconclusive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matmul | 5 | 3 | 0 | 0 | 0 | 0 | 2 |

## Complete results

### matmul

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Existing BLAS ms | Planned ms | Legacy/planned (q10..q90) | BLAS/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs BLAS | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| matmul | matrix-matrix-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 26.411656 | 7.084522 | 0.568881 | 0.574241 | 12.232x (10.826..13.536) | 0.965x (0.845..1.126) | 46.121x (39.674..48.724) | improved | inconclusive | planned-faster |
| matmul | matrix-matrix-small-direct | lhs: shape=(8,16), strides=(16,1), flags=C<br>rhs: shape=(16,8), strides=(8,1), flags=C | 2000 | 0.003082 | 0.000723 | 0.000657 | 0.000705 | 1.032x (0.804..1.119) | 0.931x (0.737..1.099) | 4.311x (3.988..4.580) | inconclusive | inconclusive | planned-faster |
| matmul | vector-vector | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256), strides=(1), flags=CF | 2000 | 0.001396 | 0.000456 | 0.000425 | 0.000452 | 0.964x (0.908..1.065) | 0.948x (0.848..0.992) | 3.007x (2.743..3.296) | inconclusive | inconclusive | planned-faster |
| matmul | vector-matrix | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(256,1), flags=C | 200 | 0.105018 | 0.028869 | 0.006130 | 0.006411 | 4.399x (3.698..4.935) | 0.939x (0.892..1.065) | 16.367x (14.555..17.737) | improved | inconclusive | planned-faster |
| matmul | matrix-vector | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.101180 | 0.020650 | 0.004459 | 0.004485 | 4.624x (4.098..5.279) | 0.991x (0.861..1.158) | 22.743x (20.892..25.560) | improved | inconclusive | planned-faster |

## Legacy correctness failures

These rows are not performance regressions. Planned matched
NumPy, while legacy returned a different value.

| Family | Operation | Scenario | Error | Mismatch summary |
| --- | --- | --- | --- | --- |

## Interpretation boundary

This notebook records every row, including inconclusive results.
It is evidence for choosing reusable routes, not a claim that the
prototype is the final implementation. Results from another OS,
architecture, or BLAS backend require a separate run.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
