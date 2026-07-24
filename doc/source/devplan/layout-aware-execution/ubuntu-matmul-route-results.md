# Linux detailed execution benchmark

## Recorded environment

- Code revision: `a8dacc01a77a62b0518198b28a4b987d20ded1cc`.
- Dirty tree: `false`.
- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`.
- Machine: `x86_64`.
- Python: `3.12.7`.
- NumPy: `2.3.0`.
- Seed: `20260722`.
- Fixed cases: `4`.
- HPC matmul suite: `true`.
- Optional slow matmul case: `false`.
- Samples per route: `7`.
- Warmups per route: `2`.
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
    --hpc-matmul \
    --filter large-square-1024-c \
    --filter broadcast-dense-lhs-256-b64 \
    --filter broadcast-negative-lhs-256-b64 \
    --filter broadcast-step2-lhs-256-b64 \
    --repeat 7 \
    --warmup 2 \
    --output /tmp/solvcon-execution-a8dacc01.json
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
| matmul-hpc | 4 | 1 | 0 | 0 | 0 | 3 | 0 |

## Complete results

### matmul-hpc

| Operation | Scenario | Workload | Operands | Calls/sample | NumPy ms | Legacy ms | Existing BLAS ms | Control ms | Planned ms | Legacy/planned (q10..q90) | BLAS/planned (q10..q90) | Control/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs BLAS | Planned vs control | Planned vs NumPy |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| matmul | large-square-1024-c | B=1, R=0, MxKxN=1024x1024x1024<br>MAC=1.074G, logical input=16.0 MiB, backing input=16.0 MiB, expanded input=16.0 MiB, output=8.0 MiB | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 1 | 5272.865246 | 3673.318302 | 33.556144 | n/a | 32.025459 | 112.423x (104.588..151.732) | 1.062x (0.995..1.128) | n/a | 161.910x (150.116..191.713) | improved | inconclusive | not-measured | planned-faster |
| matmul | broadcast-dense-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=32.5 MiB, backing input=32.5 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(1,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 1734.432615 | n/a | n/a | n/a | 47.085455 | n/a | n/a | n/a | 36.335x (34.495..37.797) | new-only | not-measured | not-measured | planned-faster |
| matmul | broadcast-negative-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=32.5 MiB, backing input=32.5 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(1,256,256), strides=(65536,256,-1), flags=N<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 1755.808584 | n/a | n/a | 47.486683 (dense planned) | 47.855660 | n/a | n/a | 0.996x (0.964..1.071) | 36.708x (35.272..38.196) | new-only | not-measured | inconclusive | planned-faster |
| matmul | broadcast-step2-lhs-256-b64 | B=64, R=1, MxKxN=256x256x256<br>MAC=1.074G, logical input=32.5 MiB, backing input=33.0 MiB, expanded input=64.0 MiB, output=32.0 MiB | lhs: shape=(1,256,256), strides=(131072,512,2), flags=S<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 1728.479774 | n/a | n/a | 48.532845 (dense planned) | 51.361129 | n/a | n/a | 0.934x (0.882..0.966) | 33.179x (30.958..34.856) | new-only | not-measured | inconclusive | planned-faster |

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
