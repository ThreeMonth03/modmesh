# Linux detailed execution benchmark

## Recorded environment

- Code revision: `3804672ef7b68d1651cfe5c30881ac8263871b45`.
- Dirty tree: `false`.
- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`.
- Machine: `x86_64`.
- Python: `3.12.7`.
- NumPy: `2.3.0`.
- Seed: `20260722`.
- Fixed cases: `14`.
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
    --repeat 5 \
    --warmup 1 --cpu 0 \
    --output /tmp/solvcon-execution-3804672e.json
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
| matmul | large-square-1024-c | B=1, R=0, MxKxN=1024x1024x1024<br>MAC=1.074G, logical input=16.0 MiB, backing input=16.0 MiB, expanded input=16.0 MiB, output=8.0 MiB | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 1 | 4442.407078 | 3074.690639 | 33.972061 | 92.318x (81.199..96.505) | 130.418x (121.579..140.533) | improved | planned-faster |
| matmul | large-broadcast-one-sided-1024-b8 | B=8, R=1, MxKxN=1024x1024x1024<br>MAC=8.590G, logical input=72.0 MiB, backing input=72.0 MiB, expanded input=128.0 MiB, output=64.0 MiB | lhs: shape=(1,1024,1024), strides=(1048576,1024,1), flags=C<br>rhs: shape=(8,1024,1024), strides=(1048576,1024,1), flags=C | 1 | 34831.094537 | n/a | 271.235646 | n/a | 128.416x (96.294..146.000) | new-only | planned-faster |
| matmul | large-broadcast-transposed-lhs-1024-b8 | B=8, R=1, MxKxN=1024x1024x1024<br>MAC=8.590G, logical input=72.0 MiB, backing input=72.0 MiB, expanded input=128.0 MiB, output=64.0 MiB | lhs: shape=(1,1024,1024), strides=(1048576,1,1024), flags=F<br>rhs: shape=(8,1024,1024), strides=(1048576,1024,1), flags=C | 1 | 37789.056425 | n/a | 261.329835 | n/a | 141.991x (129.751..145.474) | new-only | planned-faster |
| matmul | large-broadcast-cross-512-b64 | B=64, R=2, MxKxN=512x512x512<br>MAC=8.590G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=256.0 MiB, output=128.0 MiB | lhs: shape=(8,1,512,512), strides=(262144,262144,512,1), flags=C<br>rhs: shape=(1,8,512,512), strides=(2097152,262144,512,1), flags=C | 1 | 14910.756117 | n/a | 314.942408 | n/a | 48.300x (45.884..50.792) | new-only | planned-faster |
| matmul | large-batch-same-32-b4096 | B=4096, R=1, MxKxN=32x32x32<br>MAC=0.134G, logical input=64.0 MiB, backing input=64.0 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(4096,32,32), strides=(1024,32,1), flags=C<br>rhs: shape=(4096,32,32), strides=(1024,32,1), flags=C | 1 | 226.105973 | n/a | 27.336669 | n/a | 8.373x (7.989..8.617) | new-only | planned-faster |
| matmul | large-batch-one-sided-32-b16384 | B=16384, R=1, MxKxN=32x32x32<br>MAC=0.537G, logical input=128.0 MiB, backing input=128.0 MiB, expanded input=256.0 MiB, output=128.0 MiB | lhs: shape=(1,32,32), strides=(1024,32,1), flags=C<br>rhs: shape=(16384,32,32), strides=(1024,32,1), flags=C | 1 | 835.929125 | n/a | 87.218298 | n/a | 9.674x (9.589..9.841) | new-only | planned-faster |
| matmul | large-batch-cross-32-b16384 | B=16384, R=2, MxKxN=32x32x32<br>MAC=0.537G, logical input=2.0 MiB, backing input=2.0 MiB, expanded input=256.0 MiB, output=128.0 MiB | lhs: shape=(128,1,32,32), strides=(1024,1024,32,1), flags=C<br>rhs: shape=(1,128,32,32), strides=(131072,1024,32,1), flags=C | 1 | 805.737330 | n/a | 81.413711 | n/a | 9.854x (9.787..9.894) | new-only | planned-faster |
| matmul | large-batch-high-rank-32-b16384 | B=16384, R=4, MxKxN=32x32x32<br>MAC=0.537G, logical input=2.0 MiB, backing input=2.0 MiB, expanded input=256.0 MiB, output=128.0 MiB | lhs: shape=(8,1,16,1,32,32), strides=(16384,16384,1024,1024,32,1), flags=C<br>rhs: shape=(1,8,1,16,32,32), strides=(131072,16384,16384,1024,32,1), flags=C | 1 | 832.073199 | n/a | 86.080420 | n/a | 9.666x (9.542..10.054) | new-only | planned-faster |
| matmul | broadcast-dense-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=32.5 MiB, backing input=32.5 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(1,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 1686.423258 | n/a | 48.879152 | n/a | 34.653x (33.985..35.010) | new-only | planned-faster |
| matmul | materialized-dense-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=64.0 MiB, backing input=64.0 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(64,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 1698.012541 | n/a | 48.936076 | n/a | 34.616x (33.353..35.374) | new-only | planned-faster |
| matmul | broadcast-negative-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=32.5 MiB, backing input=32.5 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(1,256,256), strides=(65536,256,-1), flags=N<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 1577.250287 | n/a | 460.702483 | n/a | 3.573x (3.461..3.602) | new-only | planned-faster |
| matmul | broadcast-step2-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=32.5 MiB, backing input=33.0 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(1,256,256), strides=(131072,512,2), flags=S<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 1654.910026 | n/a | 459.865131 | n/a | 3.600x (3.518..3.615) | new-only | planned-faster |
| matmul | large-batch-dense-axis-256-b128 | B=128, R=1, MxKxN=256x256x256<br>MAC=2.147G, logical input=64.5 MiB, backing input=64.5 MiB, expanded input=128.0 MiB, output=64.0 MiB | lhs: shape=(128,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(1,256,256), strides=(65536,256,1), flags=C | 1 | 3312.934394 | n/a | 92.880133 | n/a | 35.713x (35.288..35.892) | new-only | planned-faster |
| matmul | large-batch-step2-axis-256-b128 | B=128, R=1, MxKxN=256x256x256<br>MAC=2.147G, logical input=64.5 MiB, backing input=128.5 MiB, expanded input=128.0 MiB, output=64.0 MiB | lhs: shape=(128,256,256), strides=(131072,256,1), flags=S<br>rhs: shape=(1,256,256), strides=(65536,256,1), flags=C | 1 | 3304.650512 | n/a | 94.388743 | n/a | 35.150x (34.454..35.564) | new-only | planned-faster |

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
