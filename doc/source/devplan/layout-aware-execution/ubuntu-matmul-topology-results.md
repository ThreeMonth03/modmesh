# Linux detailed execution benchmark

## Recorded environment

- Code revision: `ea09ea2a5822a841650af84a6d7a919c97466d9a`.
- Dirty tree: `false`.
- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`.
- Machine: `x86_64`.
- Python: `3.12.7`.
- NumPy: `2.3.0`.
- Seed: `20260722`.
- Fixed cases: `47`.
- HPC matmul suite: `false`.
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
    --matmul-only \
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
| matmul | 30 | 22 | 0 | 0 | 4 | 0 | 4 |
| matmul-batch | 17 | 0 | 0 | 0 | 0 | 17 | 0 |

## Complete results

### matmul

| Operation | Scenario | Purpose | Workload | Operands | Calls/sample | NumPy ms | Legacy ms | Existing BLAS ms | Control ms | Planned ms | Legacy/planned (q10..q90) | BLAS/planned (q10..q90) | Control/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs BLAS | Planned vs control | Planned vs NumPy |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| matmul | matrix-matrix-c | Dense GEMM control. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=1.0 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 29.920218 | 7.433613 | 0.590603 | n/a | 0.616921 | 12.104x (9.915..14.285) | 0.952x (0.795..1.065) | n/a | 46.532x (40.206..52.033) | improved | inconclusive | not-measured | planned-faster |
| matmul | matrix-matrix-f | Both matrices have BLAS-describable F layout. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=1.0 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(1,256), flags=F<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 27.230287 | 7.846467 | 8.226024 | 0.627127 (dense planned) | 0.549563 | 14.439x (10.418..17.231) | 14.613x (11.561..18.223) | 1.157x (0.837..1.306) | 48.989x (41.075..54.093) | improved | planned-faster | inconclusive | planned-faster |
| matmul | matrix-matrix-negative | Both matrices require signed-stride handling. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=1.0 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(256,256), strides=(-256,-1), flags=N | 7 | 26.563320 | 7.345474 | 6.828631 | 0.571965 (dense planned) | 0.944017 | 7.982x (5.461..9.967) | 7.251x (6.206..7.715) | 0.598x (0.437..0.881) | 27.286x (23.021..30.820) | improved | planned-faster | control-faster | planned-faster |
| matmul | matrix-matrix-lhs-step2 | The lhs inner stride cannot be described by GEMM. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=1.5 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 25.321556 | 6.834136 | 6.806494 | 0.540206 (dense planned) | 0.815022 | 8.308x (7.541..10.221) | 8.500x (7.447..10.005) | 0.677x (0.594..0.976) | 31.405x (28.523..36.153) | improved | planned-faster | inconclusive | planned-faster |
| matmul | matrix-matrix-rhs-step2 | The rhs column stride cannot be described by GEMM. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=1.5 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 29.028255 | 8.042998 | 7.703910 | 0.584558 (dense planned) | 0.831872 | 9.778x (7.764..15.064) | 9.436x (7.253..10.538) | 0.697x (0.598..0.848) | 34.414x (28.537..40.899) | improved | planned-faster | control-faster | planned-faster |
| matmul | matrix-matrix-both-step2 | Both matrices require packing or generic execution. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=2.0 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 30.128809 | 7.808194 | 8.054797 | 0.565033 (dense planned) | 0.972887 | 8.081x (5.530..9.266) | 8.137x (6.006..10.147) | 0.579x (0.395..0.638) | 29.698x (22.175..32.942) | improved | planned-faster | control-faster | planned-faster |
| matmul | matrix-matrix-mixed-c-f | C lhs and F rhs are both BLAS-describable. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=1.0 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 26.997485 | 5.174974 | 4.950729 | 0.586550 (dense planned) | 0.592281 | 8.406x (7.671..10.082) | 8.374x (7.491..9.733) | 0.996x (0.723..1.169) | 43.807x (35.113..52.635) | improved | planned-faster | inconclusive | planned-faster |
| matmul | matrix-matrix-padded-leading-dimension | Positive padded leading dimensions need no packing. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=2.0 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(512,1), flags=S<br>rhs: shape=(256,256), strides=(512,1), flags=S | 7 | 29.619780 | 7.947653 | 7.965286 | 0.598183 (dense planned) | 0.614697 | 12.896x (8.912..14.645) | 12.879x (10.749..14.937) | 0.979x (0.696..1.350) | 47.732x (36.174..52.599) | improved | planned-faster | inconclusive | planned-faster |
| matmul | matrix-matrix-rectangular | Rectangular dense GEMM control. | B=1, R=0, MxKxN=128x256x64<br>MAC=0.002G, logical input=0.4 MiB, backing input=0.4 MiB, expanded input=0.4 MiB, output=0.1 MiB | lhs: shape=(128,256), strides=(256,1), flags=C<br>rhs: shape=(256,64), strides=(64,1), flags=C | 40 | 3.366891 | 0.847109 | 0.099323 | n/a | 0.105816 | 8.148x (6.849..8.789) | 0.983x (0.840..1.065) | n/a | 31.561x (25.926..35.452) | improved | inconclusive | not-measured | planned-faster |
| matmul | matrix-matrix-small-direct | Work stays below the BLAS dispatch threshold. | B=1, R=0, MxKxN=8x16x8<br>MAC=0.000G, logical input=0.0 MiB, backing input=0.0 MiB, expanded input=0.0 MiB, output=0.0 MiB | lhs: shape=(8,16), strides=(16,1), flags=C<br>rhs: shape=(16,8), strides=(8,1), flags=C | 2000 | 0.002762 | 0.000697 | 0.000617 | n/a | 0.000661 | 1.059x (0.964..1.153) | 0.919x (0.824..0.947) | n/a | 4.212x (3.688..4.566) | inconclusive | blas-faster | not-measured | planned-faster |
| matmul | vector-vector | Dense DOT control. | B=1, R=0, MxKxN=1x256x1<br>MAC=0.000G, logical input=0.0 MiB, backing input=0.0 MiB, expanded input=0.0 MiB, output=0.0 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256), strides=(1), flags=CF | 2000 | 0.001371 | 0.000445 | 0.000402 | n/a | 0.000445 | 1.053x (0.805..1.402) | 0.904x (0.807..1.172) | n/a | 3.259x (2.725..3.850) | inconclusive | inconclusive | not-measured | planned-faster |
| matmul | vector-vector-negative | DOT operands both have negative increments. | B=1, R=0, MxKxN=1x256x1<br>MAC=0.000G, logical input=0.0 MiB, backing input=0.0 MiB, expanded input=0.0 MiB, output=0.0 MiB | lhs: shape=(256), strides=(-1), flags=N<br>rhs: shape=(256), strides=(-1), flags=N | 2000 | 0.001342 | n/a | n/a | 0.000441 (dense planned) | 0.000475 | n/a | n/a | 0.937x (0.891..1.199) | 2.832x (2.680..3.847) | legacy-incorrect | blas-incorrect | inconclusive | planned-faster |
| matmul | vector-vector-step2 | DOT operands both have increment two. | B=1, R=0, MxKxN=1x256x1<br>MAC=0.000G, logical input=0.0 MiB, backing input=0.0 MiB, expanded input=0.0 MiB, output=0.0 MiB | lhs: shape=(256), strides=(2), flags=S<br>rhs: shape=(256), strides=(2), flags=S | 2000 | 0.001308 | n/a | n/a | 0.000415 (dense planned) | 0.000450 | n/a | n/a | 0.920x (0.880..1.034) | 2.880x (2.672..3.192) | legacy-incorrect | blas-incorrect | inconclusive | planned-faster |
| matmul | vector-vector-long-c | Long dense DOT exercises the BLAS route. | B=1, R=0, MxKxN=1x65536x1<br>MAC=0.000G, logical input=1.0 MiB, backing input=1.0 MiB, expanded input=1.0 MiB, output=0.0 MiB | lhs: shape=(65536), strides=(1), flags=CF<br>rhs: shape=(65536), strides=(1), flags=CF | 20 | 0.127519 | 0.028279 | 0.009233 | n/a | 0.012606 | 2.372x (1.464..3.951) | 0.848x (0.661..1.154) | n/a | 9.715x (7.597..15.978) | improved | inconclusive | not-measured | planned-faster |
| matmul | vector-vector-long-negative | Long negative DOT uses signed-stride generic execution. | B=1, R=0, MxKxN=1x65536x1<br>MAC=0.000G, logical input=1.0 MiB, backing input=1.0 MiB, expanded input=1.0 MiB, output=0.0 MiB | lhs: shape=(65536), strides=(-1), flags=N<br>rhs: shape=(65536), strides=(-1), flags=N | 20 | 0.097170 | n/a | n/a | 0.010425 (dense planned) | 0.025405 | n/a | n/a | 0.412x (0.336..0.474) | 3.834x (3.387..4.237) | legacy-incorrect | blas-incorrect | control-faster | planned-faster |
| matmul | vector-vector-long-step2 | Long positive-stride DOT passes increments to BLAS. | B=1, R=0, MxKxN=1x65536x1<br>MAC=0.000G, logical input=1.0 MiB, backing input=2.0 MiB, expanded input=1.0 MiB, output=0.0 MiB | lhs: shape=(65536), strides=(2), flags=S<br>rhs: shape=(65536), strides=(2), flags=S | 20 | 0.097979 | n/a | n/a | 0.010994 (dense planned) | 0.053771 | n/a | n/a | 0.204x (0.179..0.227) | 1.835x (1.668..1.873) | legacy-incorrect | blas-incorrect | control-faster | planned-faster |
| matmul | vector-matrix | Dense vector-matrix control. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(256,1), flags=C | 200 | 0.102172 | 0.026814 | 0.005532 | n/a | 0.005673 | 4.758x (4.183..5.054) | 0.968x (0.854..1.279) | n/a | 17.868x (16.854..20.407) | improved | inconclusive | not-measured | planned-faster |
| matmul | vector-matrix-negative-vector | The GEMV vector has a negative increment. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(-1), flags=N<br>rhs: shape=(256,256), strides=(256,1), flags=C | 200 | 0.100081 | 0.027432 | 0.027715 | 0.005808 (dense planned) | 0.006001 | 4.532x (3.859..4.735) | 4.599x (3.904..4.945) | 0.988x (0.793..1.046) | 16.751x (14.408..17.809) | improved | planned-faster | inconclusive | planned-faster |
| matmul | vector-matrix-step2-vector | The GEMV vector has increment two. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(2), flags=S<br>rhs: shape=(256,256), strides=(256,1), flags=C | 200 | 0.097126 | 0.027095 | 0.026664 | 0.005437 (dense planned) | 0.005807 | 4.638x (4.209..5.119) | 4.629x (4.193..5.028) | 0.933x (0.841..1.093) | 16.979x (15.121..18.412) | improved | planned-faster | inconclusive | planned-faster |
| matmul | vector-matrix-f-matrix | The GEMV matrix is F-contiguous. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(1,256), flags=F | 200 | 0.113640 | 0.022456 | 0.021651 | 0.006840 (dense planned) | 0.006314 | 3.390x (2.242..3.781) | 3.413x (2.715..3.862) | 1.059x (0.804..1.229) | 17.624x (11.987..19.908) | improved | planned-faster | inconclusive | planned-faster |
| matmul | vector-matrix-negative-matrix | The contracted matrix axis has a negative stride. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(-256,1), flags=N | 200 | 0.101257 | 0.029257 | 0.027596 | 0.005658 (dense planned) | 0.021163 | 1.436x (1.137..1.626) | 1.398x (1.130..1.664) | 0.285x (0.217..0.394) | 5.083x (3.872..5.319) | improved | planned-faster | control-faster | planned-faster |
| matmul | vector-matrix-step2-matrix | The matrix column stride cannot be described by GEMV. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=1.0 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(512,2), flags=S | 200 | 0.114001 | 0.032663 | 0.033322 | 0.005626 (dense planned) | 0.024129 | 1.369x (1.201..1.429) | 1.383x (1.274..1.458) | 0.232x (0.203..0.260) | 4.577x (4.343..5.057) | improved | planned-faster | control-faster | planned-faster |
| matmul | vector-matrix-padded-matrix | The matrix has a padded leading dimension. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=1.0 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(512,1), flags=S | 200 | 0.106147 | 0.027816 | 0.029664 | 0.005415 (dense planned) | 0.006080 | 4.621x (4.211..5.052) | 4.927x (4.374..5.445) | 0.926x (0.813..0.989) | 17.413x (15.657..19.567) | improved | planned-faster | inconclusive | planned-faster |
| matmul | matrix-vector | Dense matrix-vector control. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.094181 | 0.017994 | 0.004993 | n/a | 0.005220 | 3.496x (3.235..3.656) | 0.934x (0.790..1.033) | n/a | 17.729x (15.598..18.857) | improved | inconclusive | not-measured | planned-faster |
| matmul | matrix-vector-negative-vector | The GEMV vector has a negative increment. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256), strides=(-1), flags=N | 200 | 0.103093 | 0.026783 | 0.022746 | 0.005793 (dense planned) | 0.006381 | 3.809x (2.903..5.978) | 3.502x (2.936..5.388) | 0.890x (0.706..1.298) | 16.373x (14.685..19.575) | improved | planned-faster | inconclusive | planned-faster |
| matmul | matrix-vector-step2-vector | The GEMV vector has increment two. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256), strides=(2), flags=S | 200 | 0.099959 | 0.021206 | 0.022268 | 0.005078 (dense planned) | 0.005927 | 3.662x (3.168..4.198) | 3.785x (2.834..4.335) | 0.924x (0.717..1.342) | 17.418x (12.905..21.158) | improved | planned-faster | inconclusive | planned-faster |
| matmul | matrix-vector-f-matrix | The GEMV matrix is F-contiguous. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(1,256), flags=F<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.103574 | 0.027322 | 0.027508 | 0.005162 (dense planned) | 0.005996 | 4.554x (4.108..5.006) | 4.593x (4.415..4.857) | 0.846x (0.769..0.974) | 17.082x (16.087..18.621) | improved | planned-faster | inconclusive | planned-faster |
| matmul | matrix-vector-negative-matrix | The contracted matrix axis has a negative stride. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(256,-1), flags=N<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.095445 | 0.021720 | 0.021268 | 0.005463 (dense planned) | 0.019948 | 1.057x (0.967..1.152) | 1.069x (1.031..1.129) | 0.269x (0.232..0.280) | 4.821x (4.412..4.955) | inconclusive | inconclusive | control-faster | planned-faster |
| matmul | matrix-vector-step2-matrix | The matrix inner stride cannot be described by GEMV. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=1.0 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.101483 | 0.022653 | 0.024632 | 0.006344 (dense planned) | 0.025999 | 0.927x (0.752..1.131) | 0.958x (0.859..1.294) | 0.232x (0.200..0.292) | 4.226x (3.496..4.843) | inconclusive | inconclusive | control-faster | planned-faster |
| matmul | matrix-vector-padded-matrix | The matrix has a padded leading dimension. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=1.0 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(512,1), flags=S<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.102844 | 0.020942 | 0.023289 | 0.005395 (dense planned) | 0.005875 | 3.568x (3.037..3.913) | 3.729x (3.127..4.462) | 0.934x (0.751..1.111) | 18.203x (15.864..20.733) | improved | planned-faster | inconclusive | planned-faster |

### matmul-batch

| Operation | Scenario | Purpose | Workload | Operands | Calls/sample | NumPy ms | Legacy ms | Control ms | Planned ms | Legacy/planned (q10..q90) | Control/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs control | Planned vs NumPy |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| matmul | batch-same-shape-c | Dense same-shape matrix batches. | B=8, R=1, MxKxN=32x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.1 MiB | lhs: shape=(8,32,32), strides=(1024,32,1), flags=C<br>rhs: shape=(8,32,32), strides=(1024,32,1), flags=C | 3 | 0.409640 | n/a | n/a | 0.020718 | n/a | n/a | 19.995x (13.763..21.643) | new-only | not-measured | planned-faster |
| matmul | batch-broadcast-c | Leading matrix batch axes broadcast. | B=64, R=2, MxKxN=32x32x32<br>MAC=0.002G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(8,1,32,32), strides=(1024,1024,32,1), flags=C<br>rhs: shape=(1,8,32,32), strides=(8192,1024,32,1), flags=C | 3 | 3.144541 | n/a | n/a | 0.174746 | n/a | n/a | 18.923x (15.686..22.679) | new-only | not-measured | planned-faster |
| matmul | batch-broadcast-negative-matrix | Broadcast matrices have negative inner strides. | B=64, R=2, MxKxN=32x32x32<br>MAC=0.002G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(8,1,32,32), strides=(1024,1024,-32,-1), flags=N<br>rhs: shape=(1,8,32,32), strides=(8192,1024,-32,-1), flags=N | 3 | 3.317826 | n/a | 0.161248 (dense planned) | 0.181866 | n/a | 0.942x (0.752..1.559) | 18.595x (16.108..25.387) | new-only | inconclusive | planned-faster |
| matmul | batch-broadcast-step2-inner | Broadcast matrices have step-two inner strides. | B=64, R=2, MxKxN=32x32x32<br>MAC=0.002G, logical input=0.1 MiB, backing input=0.2 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(8,1,32,32), strides=(2048,2048,64,2), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,64,2), flags=S | 3 | 3.164266 | n/a | 0.152945 (dense planned) | 0.183190 | n/a | 0.929x (0.754..1.110) | 17.893x (16.779..21.786) | new-only | inconclusive | planned-faster |
| matmul | batch-broadcast-step2-batch | Matrix blocks are dense and batch axes are strided. | B=64, R=2, MxKxN=32x32x32<br>MAC=0.002G, logical input=0.1 MiB, backing input=0.2 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(8,1,32,32), strides=(2048,1024,32,1), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,32,1), flags=S | 3 | 3.100379 | n/a | 0.156269 (dense planned) | 0.168866 | n/a | 0.972x (0.852..1.112) | 18.891x (16.874..20.926) | new-only | inconclusive | planned-faster |
| matmul | vector-batch-matrix-c | One dense vector is reused across two batch axes. | B=16, R=2, MxKxN=1x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(32), strides=(1), flags=CF<br>rhs: shape=(2,8,32,32), strides=(8192,1024,32,1), flags=C | 3 | 0.027175 | n/a | n/a | 0.005338 | n/a | n/a | 5.101x (4.702..5.365) | new-only | not-measured | planned-faster |
| matmul | vector-batch-matrix-negative-vector | A negative-stride vector is reused across two batch axes. | B=16, R=2, MxKxN=1x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(32), strides=(-1), flags=N<br>rhs: shape=(2,8,32,32), strides=(8192,1024,32,1), flags=C | 3 | 0.025491 | n/a | 0.004905 (dense planned) | 0.004941 | n/a | 0.994x (0.944..1.032) | 5.141x (4.799..5.383) | new-only | inconclusive | planned-faster |
| matmul | vector-batch-matrix-step2-vector | A step-two vector is reused across two batch axes. | B=16, R=2, MxKxN=1x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(32), strides=(2), flags=S<br>rhs: shape=(2,8,32,32), strides=(8192,1024,32,1), flags=C | 3 | 0.026207 | n/a | 0.005180 (dense planned) | 0.005104 | n/a | 1.016x (0.900..1.393) | 5.129x (4.471..5.323) | new-only | inconclusive | planned-faster |
| matmul | vector-batch-matrix-negative-matrix | The batched matrix contracted axis is negative. | B=16, R=2, MxKxN=1x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(32), strides=(1), flags=CF<br>rhs: shape=(2,8,32,32), strides=(8192,1024,-32,1), flags=N | 3 | 0.026259 | n/a | 0.005013 (dense planned) | 0.005342 | n/a | 0.956x (0.866..0.991) | 4.946x (4.744..5.195) | new-only | inconclusive | planned-faster |
| matmul | vector-batch-matrix-step2-matrix | The batched matrix column stride is two. | B=16, R=2, MxKxN=1x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.3 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(32), strides=(1), flags=CF<br>rhs: shape=(2,8,32,32), strides=(16384,2048,64,2), flags=S | 3 | 0.026151 | n/a | 0.004940 (dense planned) | 0.005276 | n/a | 0.959x (0.875..1.293) | 4.980x (4.682..5.543) | new-only | inconclusive | planned-faster |
| matmul | vector-batch-matrix-padded-matrix | The batched matrix has a padded leading dimension. | B=16, R=2, MxKxN=1x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.3 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(32), strides=(1), flags=CF<br>rhs: shape=(2,8,32,32), strides=(16384,2048,64,1), flags=S | 3 | 0.025701 | n/a | 0.004953 (dense planned) | 0.005031 | n/a | 0.984x (0.725..1.062) | 5.111x (4.741..5.461) | new-only | inconclusive | planned-faster |
| matmul | batch-matrix-vector-c | One dense vector is reused across two batch axes. | B=16, R=2, MxKxN=32x32x1<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(2,8,32,32), strides=(8192,1024,32,1), flags=C<br>rhs: shape=(32), strides=(1), flags=CF | 3 | 0.025944 | n/a | n/a | 0.003608 | n/a | n/a | 7.285x (6.259..8.350) | new-only | not-measured | planned-faster |
| matmul | batch-matrix-vector-negative-vector | A negative-stride vector is reused across two batch axes. | B=16, R=2, MxKxN=32x32x1<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(2,8,32,32), strides=(8192,1024,32,1), flags=C<br>rhs: shape=(32), strides=(-1), flags=N | 3 | 0.025615 | n/a | 0.003567 (dense planned) | 0.005000 | n/a | 0.704x (0.568..0.773) | 5.141x (3.950..5.434) | new-only | control-faster | planned-faster |
| matmul | batch-matrix-vector-step2-vector | A step-two vector is reused across two batch axes. | B=16, R=2, MxKxN=32x32x1<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(2,8,32,32), strides=(8192,1024,32,1), flags=C<br>rhs: shape=(32), strides=(2), flags=S | 3 | 0.025809 | n/a | 0.003565 (dense planned) | 0.005556 | n/a | 0.664x (0.563..0.781) | 4.701x (4.054..5.153) | new-only | control-faster | planned-faster |
| matmul | batch-matrix-vector-negative-matrix | The batched matrix contracted axis is negative. | B=16, R=2, MxKxN=32x32x1<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(2,8,32,32), strides=(8192,1024,32,-1), flags=N<br>rhs: shape=(32), strides=(1), flags=CF | 3 | 0.026560 | n/a | 0.003579 (dense planned) | 0.005655 | n/a | 0.657x (0.569..0.811) | 4.811x (4.135..5.211) | new-only | control-faster | planned-faster |
| matmul | batch-matrix-vector-step2-matrix | The batched matrix inner stride is two. | B=16, R=2, MxKxN=32x32x1<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.3 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(2,8,32,32), strides=(16384,2048,64,2), flags=S<br>rhs: shape=(32), strides=(1), flags=CF | 3 | 0.025945 | n/a | 0.003610 (dense planned) | 0.005551 | n/a | 0.662x (0.571..0.747) | 4.879x (3.971..5.397) | new-only | control-faster | planned-faster |
| matmul | batch-matrix-vector-padded-matrix | The batched matrix has a padded leading dimension. | B=16, R=2, MxKxN=32x32x1<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.3 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(2,8,32,32), strides=(16384,2048,64,1), flags=S<br>rhs: shape=(32), strides=(1), flags=CF | 3 | 0.026499 | n/a | 0.003649 (dense planned) | 0.004241 | n/a | 0.901x (0.778..0.951) | 6.433x (6.111..6.746) | new-only | inconclusive | planned-faster |

## Legacy correctness failures

These rows are not performance regressions. Planned matched
NumPy, while legacy returned a different value.

| Family | Operation | Scenario | Error | Mismatch summary |
| --- | --- | --- | --- | --- |
| matmul | matmul | vector-vector-negative | AssertionError | Mismatched elements: 1 / 1 (100%) |
| matmul | matmul | vector-vector-step2 | AssertionError | Mismatched elements: 1 / 1 (100%) |
| matmul | matmul | vector-vector-long-negative | AssertionError | Mismatched elements: 1 / 1 (100%) |
| matmul | matmul | vector-vector-long-step2 | AssertionError | Mismatched elements: 1 / 1 (100%) |

## Interpretation boundary

This notebook records every row, including inconclusive results.
It is evidence for choosing reusable routes, not a claim that the
prototype is the final implementation. Results from another OS,
architecture, or BLAS backend require a separate run.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
