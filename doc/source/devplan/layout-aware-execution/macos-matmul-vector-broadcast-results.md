# macOS detailed execution benchmark

## Recorded environment

- Code revision: `0d906a8fb8f1b6e9ea57b5bae650c14e28a8b519`.
- Dirty tree: `false`.
- Platform: `macOS-26.5.1-arm64-arm-64bit`.
- Machine: `arm64`.
- Python: `3.11.6`.
- NumPy: `2.2.4`.
- Seed: `20260722`.
- Fixed cases: `8`.
- HPC matmul suite: `true`.
- Optional slow matmul case: `false`.
- Samples per route: `15`.
- Warmups per route: `5`.
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
    --filter vector-batch \
    --filter batch-matrix-vector-256 \
    --filter batch-matrix-negative-vector \
    --filter batch-matrix-step2-vector \
    --filter batch-step2-matrix-vector \
    --repeat 15 \
    --warmup 5 \
    --output /tmp/solvcon-execution-0d906a8f.json
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
| matmul | vector-batch-matrix-256-b64 | One vector is reused by 64 matrix-vector contractions. | B=64, R=1, MxKxN=1x256x256<br>MAC=0.004G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 1.912584 | n/a | n/a | 1.920291 | n/a | n/a | 1.021x (0.977..1.119) | new-only | not-measured | inconclusive |
| matmul | negative-vector-batch-matrix-256-b64 | A repeated negative vector may be packed at most once. | B=64, R=1, MxKxN=1x256x256<br>MAC=0.004G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(256), strides=(-1), flags=N<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 8.148000 | n/a | 1.787542 (dense planned) | 1.778958 | n/a | 1.006x (0.962..1.078) | 4.591x (4.214..4.728) | new-only | inconclusive | planned-faster |
| matmul | step2-vector-batch-matrix-256-b64 | A repeated step-two vector retains one physical copy. | B=64, R=1, MxKxN=1x256x256<br>MAC=0.004G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(256), strides=(2), flags=S<br>rhs: shape=(64,256,256), strides=(65536,256,1), flags=C | 1 | 1.887583 | n/a | 1.905834 (dense planned) | 1.901625 | n/a | 1.000x (0.891..1.048) | 0.993x (0.881..1.060) | new-only | inconclusive | inconclusive |
| matmul | vector-batch-step2-matrix-256-b64 | Every rhs matrix has a step-two column stride. | B=64, R=1, MxKxN=1x256x256<br>MAC=0.004G, logical input=32.0 MiB, backing input=64.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(64,256,256), strides=(131072,512,2), flags=S | 1 | 10.024083 | n/a | 2.124250 (dense planned) | 9.182875 | n/a | 0.232x (0.226..0.236) | 1.076x (1.029..1.107) | new-only | control-faster | inconclusive |
| matmul | batch-matrix-vector-256-b64 | One vector is reused by 64 matrix-vector contractions. | B=64, R=1, MxKxN=256x256x1<br>MAC=0.004G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(64,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 1 | 0.840625 | n/a | n/a | 0.830500 | n/a | n/a | 1.003x (0.938..1.088) | new-only | not-measured | inconclusive |
| matmul | batch-matrix-negative-vector-256-b64 | A repeated negative vector may be packed at most once. | B=64, R=1, MxKxN=256x256x1<br>MAC=0.004G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(64,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(256), strides=(-1), flags=N | 1 | 7.335042 | n/a | 0.845791 (dense planned) | 1.104125 | n/a | 0.977x (0.858..1.038) | 6.656x (6.270..9.021) | new-only | inconclusive | planned-faster |
| matmul | batch-matrix-step2-vector-256-b64 | A repeated step-two vector retains one physical copy. | B=64, R=1, MxKxN=256x256x1<br>MAC=0.004G, logical input=32.0 MiB, backing input=32.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(64,256,256), strides=(65536,256,1), flags=C<br>rhs: shape=(256), strides=(2), flags=S | 1 | 0.856417 | n/a | 0.843375 (dense planned) | 0.938250 | n/a | 0.971x (0.938..1.000) | 0.985x (0.923..1.040) | new-only | inconclusive | inconclusive |
| matmul | batch-step2-matrix-vector-256-b64 | Every lhs matrix has a step-two inner stride. | B=64, R=1, MxKxN=256x256x1<br>MAC=0.004G, logical input=32.0 MiB, backing input=64.0 MiB, expanded input=32.1 MiB, output=0.1 MiB | lhs: shape=(64,256,256), strides=(131072,512,2), flags=S<br>rhs: shape=(256), strides=(1), flags=CF | 1 | 7.393625 | n/a | 1.171875 (dense planned) | 6.159875 | n/a | 0.189x (0.176..0.221) | 1.198x (1.178..1.206) | new-only | control-faster | planned-faster |

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
