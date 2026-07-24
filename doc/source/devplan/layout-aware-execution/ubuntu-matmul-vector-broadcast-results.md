# Linux detailed execution benchmark

## Recorded environment

- Code revision: `ea09ea2a5822a841650af84a6d7a919c97466d9a`.
- Dirty tree: `false`.
- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`.
- Machine: `x86_64`.
- Python: `3.12.7`.
- NumPy: `2.3.0`.
- Seed: `20260722`.
- Fixed cases: `8`.
- HPC matmul suite: `true`.
- Optional slow matmul case: `false`.
- Samples per route: `15`.
- Warmups per route: `5`.
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
    --filter vector-batch \
    --filter batch-matrix-vector-256 \
    --filter batch-matrix-negative-vector \
    --filter batch-matrix-step2-vector \
    --filter batch-step2-matrix-vector \
    --repeat 15 \
    --warmup 5 --cpu 0 \
    --output /tmp/solvcon-execution-ea09ea2a.json
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
| matmul-hpc | 8 | 0 | 0 | 0 | 0 | 8 | 0 |

## Complete results

### matmul-hpc

| Operation | Scenario | Purpose | Workload | Operands | Calls/sample | NumPy ms | Legacy ms | Control ms | Planned ms | Legacy/planned (q10..q90) | Control/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs control | Planned vs NumPy |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| matmul | vector-batch-matrix-256-b64 | One vector is reused by 64 matrix-vector contractions. | B=64, R=1, MxKxN=1x256x256<br>MAC=0.004G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 8.246120 | n/a | n/a | 1.424367 | n/a | n/a | 6.228x (4.766..6.773) | new-only | not-measured | planned-faster |
| matmul | negative-vector-batch-matrix-256-b64 | A repeated negative vector may be packed at most once. | B=64, R=1, MxKxN=1x256x256<br>MAC=0.004G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(256), strides=(-1), flags=N<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 8.297421 | n/a | 1.334177 (dense planned) | 1.573469 | n/a | 0.923x (0.711..1.128) | 5.444x (4.165..7.037) | new-only | inconclusive | planned-faster |
| matmul | step2-vector-batch-matrix-256-b64 | A repeated step-two vector retains one physical copy. | B=64, R=1, MxKxN=1x256x256<br>MAC=0.004G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(256), strides=(2), flags=S<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 8.169853 | n/a | 1.346747 (dense planned) | 1.355567 | n/a | 0.952x (0.736..1.224) | 6.157x (4.570..6.515) | new-only | inconclusive | planned-faster |
| matmul | vector-batch-step2-matrix-256-b64 | Every rhs matrix has a step-two column stride. | B=64, R=1, MxKxN=1x256x256<br>MAC=0.004G, logical input=32.0 MiB, backing input=64.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(64,256,256), strides=(131072,512,2), flags=S | 1 | 15.696044 | n/a | 1.244190 (dense planned) | 12.277971 | n/a | 0.103x (0.096..0.116) | 1.282x (1.226..1.402) | new-only | control-faster | planned-faster |
| matmul | batch-matrix-vector-256-b64 | One vector is reused by 64 matrix-vector contractions. | B=64, R=1, MxKxN=256x256x1<br>MAC=0.004G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(64,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 1 | 6.571519 | n/a | n/a | 1.343578 | n/a | n/a | 5.023x (3.000..5.946) | new-only | not-measured | planned-faster |
| matmul | batch-matrix-negative-vector-256-b64 | A repeated negative vector may be packed at most once. | B=64, R=1, MxKxN=256x256x1<br>MAC=0.004G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(64,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(256), strides=(-1), flags=N | 1 | 9.481682 | n/a | 3.548127 (dense planned) | 3.401740 | n/a | 0.990x (0.745..1.407) | 2.718x (1.926..4.463) | new-only | inconclusive | planned-faster |
| matmul | batch-matrix-step2-vector-256-b64 | A repeated step-two vector retains one physical copy. | B=64, R=1, MxKxN=256x256x1<br>MAC=0.004G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(64,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(256), strides=(2), flags=S | 1 | 6.483437 | n/a | 1.452139 (dense planned) | 1.547711 | n/a | 1.003x (0.754..1.484) | 4.183x (3.443..5.050) | new-only | inconclusive | planned-faster |
| matmul | batch-step2-matrix-vector-256-b64 | Every lhs matrix has a step-two inner stride. | B=64, R=1, MxKxN=256x256x1<br>MAC=0.004G, logical input=32.0 MiB, backing input=64.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(64,256,256), strides=(131072,512,2), flags=S<br>rhs: shape=(256), strides=(1), flags=CF | 1 | 7.424426 | n/a | 1.554996 (dense planned) | 3.881835 | n/a | 0.380x (0.344..0.435) | 1.893x (1.749..2.014) | new-only | control-faster | planned-faster |

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
