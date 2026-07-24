# macOS detailed execution benchmark

## Recorded environment

- Code revision: `72283a19e9c44b48f6fbac6c1ecb4a444989c99c`.
- Dirty tree: `false`.
- Platform: `macOS-26.5.1-arm64-arm-64bit`.
- Machine: `arm64`.
- Python: `3.11.6`.
- NumPy: `2.2.4`.
- Seed: `20260722`.
- Fixed cases: `14`.
- HPC matmul suite: `true`.
- Optional slow matmul case: `false`.
- Samples per route: `5`.
- Warmups per route: `1`.
- Threads: `1`.
- CPU affinity: `n/a`.

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
    --repeat 5 \
    --warmup 1 \
    --output /tmp/solvcon-execution-72283a19.json
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
| matmul-hpc | 14 | 1 | 0 | 0 | 0 | 13 | 0 |

## Complete results

### matmul-hpc

| Operation | Scenario | Workload | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matmul | large-square-1024-c | B=1, R=0, MxKxN=1024x1024x1024<br>MAC=1.074G, logical input=16.0 MiB, backing input=16.0 MiB, expanded input=16.0 MiB, output=8.0 MiB | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 1 | 19.883958 | 3477.480083 | 18.777875 | 185.854x (182.817..195.645) | 1.029x (0.961..1.063) | improved | inconclusive |
| matmul | large-broadcast-one-sided-1024-b8 | B=8, R=1, MxKxN=1024x1024x1024<br>MAC=8.590G, logical input=72.0 MiB, backing input=72.0 MiB, expanded input=128.0 MiB, output=64.0 MiB | lhs: shape=(1,1024,1024), strides=(1048576,1024,1), flags=C<br>rhs: shape=(8,1024,1024), strides=(1048576,1024,1), flags=C | 1 | 207.378500 | n/a | 201.155500 | n/a | 1.061x (0.994..1.115) | new-only | inconclusive |
| matmul | large-broadcast-transposed-lhs-1024-b8 | B=8, R=1, MxKxN=1024x1024x1024<br>MAC=8.590G, logical input=72.0 MiB, backing input=72.0 MiB, expanded input=128.0 MiB, output=64.0 MiB | lhs: shape=(1,1024,1024), strides=(1048576,1,1024), flags=F<br>rhs: shape=(8,1024,1024), strides=(1048576,1024,1), flags=C | 1 | 196.201708 | n/a | 196.970541 | n/a | 1.005x (0.890..1.099) | new-only | inconclusive |
| matmul | large-broadcast-cross-512-b64 | B=64, R=2, MxKxN=512x512x512<br>MAC=8.590G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=256.0 MiB, output=128.0 MiB | lhs: shape=(8,1,512,512), strides=(262144,262144,512,1), flags=C<br>rhs: shape=(1,8,512,512), strides=(2097152,262144,512,1), flags=C | 1 | 292.236958 | n/a | 282.220708 | n/a | 1.014x (0.932..1.163) | new-only | inconclusive |
| matmul | large-batch-same-32-b4096 | B=4096, R=1, MxKxN=32x32x32<br>MAC=0.134G, logical input=64.0 MiB, backing input=64.0 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(4096,32,32), strides=(1024,32,1), flags=C<br>rhs: shape=(4096,32,32), strides=(1024,32,1), flags=C | 1 | 7.846958 | n/a | 7.511791 | n/a | 1.109x (0.849..1.152) | new-only | inconclusive |
| matmul | large-batch-one-sided-32-b16384 | B=16384, R=1, MxKxN=32x32x32<br>MAC=0.537G, logical input=128.0 MiB, backing input=128.0 MiB, expanded input=256.0 MiB, output=128.0 MiB | lhs: shape=(1,32,32), strides=(1024,32,1), flags=C<br>rhs: shape=(16384,32,32), strides=(1024,32,1), flags=C | 1 | 36.729417 | n/a | 36.575417 | n/a | 1.140x (0.816..1.171) | new-only | inconclusive |
| matmul | large-batch-cross-32-b16384 | B=16384, R=2, MxKxN=32x32x32<br>MAC=0.537G, logical input=2.0 MiB, backing input=2.0 MiB, expanded input=256.0 MiB, output=128.0 MiB | lhs: shape=(128,1,32,32), strides=(1024,1024,32,1), flags=C<br>rhs: shape=(1,128,32,32), strides=(131072,1024,32,1), flags=C | 1 | 51.173375 | n/a | 38.397083 | n/a | 0.994x (0.920..1.753) | new-only | inconclusive |
| matmul | large-batch-high-rank-32-b16384 | B=16384, R=4, MxKxN=32x32x32<br>MAC=0.537G, logical input=2.0 MiB, backing input=2.0 MiB, expanded input=256.0 MiB, output=128.0 MiB | lhs: shape=(8,1,16,1,32,32), strides=(16384,16384,1024,1024,32,1), flags=C<br>rhs: shape=(1,8,1,16,32,32), strides=(131072,16384,16384,1024,32,1), flags=C | 1 | 35.916000 | n/a | 32.128416 | n/a | 1.075x (1.005..1.158) | new-only | inconclusive |
| matmul | broadcast-dense-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=32.5 MiB, backing input=32.5 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(1,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 26.408792 | n/a | 26.759958 | n/a | 1.001x (0.988..1.016) | new-only | parity |
| matmul | materialized-dense-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=64.0 MiB, backing input=64.0 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(64,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 24.735459 | n/a | 24.534042 | n/a | 0.962x (0.910..1.012) | new-only | inconclusive |
| matmul | broadcast-negative-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=32.5 MiB, backing input=32.5 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(1,256,256), strides=(65536,256,-1), flags=N<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 2961.193125 | n/a | 17.748333 | n/a | 160.191x (140.671..168.659) | new-only | planned-faster |
| matmul | broadcast-step2-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=32.5 MiB, backing input=33.0 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(1,256,256), strides=(131072,512,2), flags=S<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 3233.025958 | n/a | 18.893834 | n/a | 170.221x (123.605..179.899) | new-only | planned-faster |
| matmul | large-batch-dense-axis-256-b128 | B=128, R=1, MxKxN=256x256x256<br>MAC=2.147G, logical input=64.5 MiB, backing input=64.5 MiB, expanded input=128.0 MiB, output=64.0 MiB | lhs: shape=(128,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(1,256,256), strides=(65536,256,1), flags=C | 1 | 49.780250 | n/a | 49.141000 | n/a | 1.014x (0.972..1.172) | new-only | inconclusive |
| matmul | large-batch-step2-axis-256-b128 | B=128, R=1, MxKxN=256x256x256<br>MAC=2.147G, logical input=64.5 MiB, backing input=128.5 MiB, expanded input=128.0 MiB, output=64.0 MiB | lhs: shape=(128,256,256), strides=(131072,256,1), flags=S<br>rhs: shape=(1,256,256), strides=(65536,256,1), flags=C | 1 | 66.747625 | n/a | 61.492875 | n/a | 0.931x (0.685..1.083) | new-only | inconclusive |

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
