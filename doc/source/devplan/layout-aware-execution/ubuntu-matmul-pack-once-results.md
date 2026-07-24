# Linux detailed execution benchmark

## Recorded environment

- Code revision: `b66516c888fc5204573eacdc3b8b063f214d825b`.
- Dirty tree: `false`.
- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`.
- Machine: `x86_64`.
- Python: `3.12.7`.
- NumPy: `2.3.0`.
- Seed: `20260722`.
- Fixed cases: `4`.
- HPC matmul suite: `true`.
- Optional slow matmul case: `false`.
- Samples per route: `5`.
- Warmups per route: `1`.
- Threads: `1`.
- CPU affinity: `0`.

The machine-readable JSON also records NumPy build configuration,
extension linkage, thread-control variables, raw timing samples,
paired ratios, and q10/q90 ratio quantiles.

## Reproduction

```console
$ source /path/to/devenv/scripts/init
$ devenv use prime
$ python3 profiling/profile_execution_prototype.py \
    --benchmark-only \
    --hpc-matmul \
    --filter dense-lhs-256-b64 \
    --filter negative-lhs-256-b64 \
    --filter step2-lhs-256-b64 \
    --repeat 5 \
    --warmup 1 --cpu 0 \
    --output /tmp/solvcon-execution-b66516c8.json
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
- `legacy-incorrect` means planned matched NumPy but legacy did
  not. Incorrect legacy results are not timed.

## Coverage inventory

| Family | Cases | Improved | Parity | Regression | Legacy incorrect | New only | Inconclusive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matmul-hpc | 4 | 0 | 0 | 0 | 0 | 4 | 0 |

## Complete results

### matmul-hpc

| Operation | Scenario | Workload | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matmul | broadcast-dense-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=32.5 MiB, backing input=32.5 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(1,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 1891.610643 | n/a | 49.166506 | n/a | 38.794x (33.627..39.898) | new-only | planned-faster |
| matmul | materialized-dense-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=64.0 MiB, backing input=64.0 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(64,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 1892.548844 | n/a | 51.299132 | n/a | 36.368x (33.985..38.830) | new-only | planned-faster |
| matmul | broadcast-negative-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=32.5 MiB, backing input=32.5 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(1,256,256), strides=(65536,256,-1), flags=N<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 1866.253721 | n/a | 51.149032 | n/a | 37.101x (34.914..40.539) | new-only | planned-faster |
| matmul | broadcast-step2-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=32.5 MiB, backing input=33.0 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(1,256,256), strides=(131072,512,2), flags=S<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 1867.539108 | n/a | 51.019284 | n/a | 36.215x (32.697..38.093) | new-only | planned-faster |

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
