# macOS detailed execution benchmark

## Recorded environment

- Code revision: `0d906a8fb8f1b6e9ea57b5bae650c14e28a8b519`.
- Dirty tree: `false`.
- Platform: `macOS-26.5.1-arm64-arm-64bit`.
- Machine: `arm64`.
- Python: `3.11.6`.
- NumPy: `2.2.4`.
- Seed: `20260722`.
- Fixed cases: `47`.
- HPC matmul suite: `false`.
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
    --matmul-only \
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
| matmul | 30 | 24 | 2 | 0 | 4 | 0 | 0 |
| matmul-batch | 17 | 0 | 0 | 0 | 0 | 17 | 0 |

## Complete results

### matmul

| Operation | Scenario | Purpose | Workload | Operands | Calls/sample | NumPy ms | Legacy ms | Existing BLAS ms | Control ms | Planned ms | Legacy/planned (q10..q90) | BLAS/planned (q10..q90) | Control/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs BLAS | Planned vs control | Planned vs NumPy |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| matmul | matrix-matrix-c | Dense GEMM control. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=1.0 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 0.158280 | 25.113018 | 0.151399 | n/a | 0.152143 | 165.062x (119.119..169.939) | 0.988x (0.702..1.006) | n/a | 1.035x (0.932..1.203) | improved | inconclusive | not-measured | inconclusive |
| matmul | matrix-matrix-f | Both matrices have BLAS-describable F layout. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=1.0 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(1,256), flags=F<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 0.155399 | 25.291625 | 25.216208 | 0.165702 (dense planned) | 0.151405 | 166.906x (161.766..168.018) | 166.113x (161.433..168.226) | 1.078x (1.030..1.111) | 1.020x (1.010..1.029) | improved | planned-faster | inconclusive | parity |
| matmul | matrix-matrix-negative | Both matrices require signed-stride handling. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=1.0 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(256,256), strides=(-256,-1), flags=N | 7 | 29.720934 | 25.022643 | 24.904012 | 0.164506 (dense planned) | 0.233982 | 107.061x (104.027..109.576) | 106.590x (104.213..108.879) | 0.706x (0.686..0.718) | 127.022x (124.651..129.795) | improved | planned-faster | control-faster | planned-faster |
| matmul | matrix-matrix-lhs-step2 | The lhs inner stride cannot be described by GEMM. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=1.5 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 29.630708 | 24.684679 | 24.659601 | 0.159440 (dense planned) | 0.192554 | 129.005x (126.353..131.227) | 128.191x (127.281..129.614) | 0.827x (0.802..0.855) | 154.165x (152.236..156.704) | improved | planned-faster | control-faster | planned-faster |
| matmul | matrix-matrix-rhs-step2 | The rhs column stride cannot be described by GEMM. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=1.5 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 29.904494 | 26.860333 | 26.996684 | 0.158196 (dense planned) | 0.191732 | 139.957x (137.011..142.848) | 140.903x (138.914..142.213) | 0.821x (0.800..0.848) | 155.666x (154.078..157.724) | improved | planned-faster | control-faster | planned-faster |
| matmul | matrix-matrix-both-step2 | Both matrices require packing or generic execution. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=2.0 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 29.985833 | 27.259911 | 27.065732 | 0.162464 (dense planned) | 0.235345 | 115.969x (114.003..121.876) | 114.883x (114.051..116.846) | 0.698x (0.673..0.711) | 127.941x (126.084..129.969) | improved | planned-faster | control-faster | planned-faster |
| matmul | matrix-matrix-mixed-c-f | C lhs and F rhs are both BLAS-describable. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=1.0 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 0.175054 | 14.786905 | 14.789857 | 0.157083 (dense planned) | 0.171726 | 85.679x (83.967..91.823) | 85.942x (84.493..92.069) | 0.904x (0.878..0.923) | 1.007x (1.002..1.021) | improved | planned-faster | control-faster | parity |
| matmul | matrix-matrix-padded-leading-dimension | Positive padded leading dimensions need no packing. | B=1, R=0, MxKxN=256x256x256<br>MAC=0.017G, logical input=1.0 MiB, backing input=2.0 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(256,256), strides=(512,1), flags=S<br>rhs: shape=(256,256), strides=(512,1), flags=S | 7 | 0.176030 | 26.925655 | 27.043143 | 0.161780 (dense planned) | 0.171720 | 156.616x (154.158..157.993) | 156.502x (154.912..158.752) | 0.941x (0.919..1.011) | 1.020x (1.003..1.041) | improved | planned-faster | inconclusive | parity |
| matmul | matrix-matrix-rectangular | Rectangular dense GEMM control. | B=1, R=0, MxKxN=128x256x64<br>MAC=0.002G, logical input=0.4 MiB, backing input=0.4 MiB, expanded input=0.4 MiB, output=0.1 MiB | lhs: shape=(128,256), strides=(256,1), flags=C<br>rhs: shape=(256,64), strides=(64,1), flags=C | 40 | 0.024573 | 3.032182 | 0.024206 | n/a | 0.024277 | 124.900x (123.575..125.485) | 0.997x (0.983..1.012) | n/a | 1.009x (1.007..1.029) | improved | parity | not-measured | parity |
| matmul | matrix-matrix-small-direct | Work stays below the BLAS dispatch threshold. | B=1, R=0, MxKxN=8x16x8<br>MAC=0.000G, logical input=0.0 MiB, backing input=0.0 MiB, expanded input=0.0 MiB, output=0.0 MiB | lhs: shape=(8,16), strides=(16,1), flags=C<br>rhs: shape=(16,8), strides=(8,1), flags=C | 2000 | 0.001133 | 0.001492 | 0.001041 | n/a | 0.001494 | 0.998x (0.994..1.001) | 0.697x (0.694..0.700) | n/a | 0.759x (0.755..0.760) | parity | blas-faster | not-measured | numpy-faster |
| matmul | vector-vector | Dense DOT control. | B=1, R=0, MxKxN=1x256x1<br>MAC=0.000G, logical input=0.0 MiB, backing input=0.0 MiB, expanded input=0.0 MiB, output=0.0 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256), strides=(1), flags=CF | 2000 | 0.000873 | 0.001143 | 0.000869 | n/a | 0.001148 | 0.997x (0.991..0.999) | 0.759x (0.753..0.763) | n/a | 0.761x (0.759..0.765) | parity | blas-faster | not-measured | numpy-faster |
| matmul | vector-vector-negative | DOT operands both have negative increments. | B=1, R=0, MxKxN=1x256x1<br>MAC=0.000G, logical input=0.0 MiB, backing input=0.0 MiB, expanded input=0.0 MiB, output=0.0 MiB | lhs: shape=(256), strides=(-1), flags=N<br>rhs: shape=(256), strides=(-1), flags=N | 2000 | 0.001300 | n/a | n/a | 0.001149 (dense planned) | 0.001336 | n/a | n/a | 0.860x (0.857..0.866) | 0.970x (0.965..0.977) | legacy-incorrect | blas-incorrect | control-faster | parity |
| matmul | vector-vector-step2 | DOT operands both have increment two. | B=1, R=0, MxKxN=1x256x1<br>MAC=0.000G, logical input=0.0 MiB, backing input=0.0 MiB, expanded input=0.0 MiB, output=0.0 MiB | lhs: shape=(256), strides=(2), flags=S<br>rhs: shape=(256), strides=(2), flags=S | 2000 | 0.000924 | n/a | n/a | 0.001153 (dense planned) | 0.001340 | n/a | n/a | 0.860x (0.853..0.866) | 0.690x (0.682..0.692) | legacy-incorrect | blas-incorrect | control-faster | numpy-faster |
| matmul | vector-vector-long-c | Long dense DOT exercises the BLAS route. | B=1, R=0, MxKxN=1x65536x1<br>MAC=0.000G, logical input=1.0 MiB, backing input=1.0 MiB, expanded input=1.0 MiB, output=0.0 MiB | lhs: shape=(65536), strides=(1), flags=CF<br>rhs: shape=(65536), strides=(1), flags=CF | 20 | 0.021921 | 0.101671 | 0.021767 | n/a | 0.021810 | 4.657x (4.526..4.701) | 0.997x (0.974..1.002) | n/a | 1.004x (0.980..1.011) | improved | parity | not-measured | parity |
| matmul | vector-vector-long-negative | Long negative DOT uses signed-stride generic execution. | B=1, R=0, MxKxN=1x65536x1<br>MAC=0.000G, logical input=1.0 MiB, backing input=1.0 MiB, expanded input=1.0 MiB, output=0.0 MiB | lhs: shape=(65536), strides=(-1), flags=N<br>rhs: shape=(65536), strides=(-1), flags=N | 20 | 0.135342 | n/a | n/a | 0.021785 (dense planned) | 0.135275 | n/a | n/a | 0.161x (0.160..0.161) | 0.999x (0.996..1.005) | legacy-incorrect | blas-incorrect | control-faster | parity |
| matmul | vector-vector-long-step2 | Long positive-stride DOT passes increments to BLAS. | B=1, R=0, MxKxN=1x65536x1<br>MAC=0.000G, logical input=1.0 MiB, backing input=2.0 MiB, expanded input=1.0 MiB, output=0.0 MiB | lhs: shape=(65536), strides=(2), flags=S<br>rhs: shape=(65536), strides=(2), flags=S | 20 | 0.042650 | n/a | n/a | 0.021804 (dense planned) | 0.042575 | n/a | n/a | 0.512x (0.510..0.515) | 1.002x (0.998..1.006) | legacy-incorrect | blas-incorrect | control-faster | parity |
| matmul | vector-matrix | Dense vector-matrix control. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(256,1), flags=C | 200 | 0.017358 | 0.112762 | 0.016633 | n/a | 0.017167 | 6.587x (5.836..6.811) | 0.998x (0.887..1.023) | n/a | 1.006x (0.917..1.147) | improved | inconclusive | not-measured | inconclusive |
| matmul | vector-matrix-negative-vector | The GEMV vector has a negative increment. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(-1), flags=N<br>rhs: shape=(256,256), strides=(256,1), flags=C | 200 | 0.116754 | 0.097062 | 0.096894 | 0.014528 (dense planned) | 0.015837 | 6.120x (5.967..6.167) | 6.118x (5.945..6.171) | 0.918x (0.892..0.945) | 7.383x (7.149..7.460) | improved | planned-faster | control-faster | planned-faster |
| matmul | vector-matrix-step2-vector | The GEMV vector has increment two. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(2), flags=S<br>rhs: shape=(256,256), strides=(256,1), flags=C | 200 | 0.014460 | 0.096851 | 0.096948 | 0.014462 (dense planned) | 0.014512 | 6.669x (6.583..6.724) | 6.678x (6.529..6.689) | 0.994x (0.987..1.007) | 0.995x (0.989..0.998) | improved | planned-faster | parity | parity |
| matmul | vector-matrix-f-matrix | The GEMV matrix is F-contiguous. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(1,256), flags=F | 200 | 0.006987 | 0.058274 | 0.058180 | 0.014879 (dense planned) | 0.006896 | 8.559x (8.345..9.348) | 8.594x (8.346..9.986) | 2.171x (2.116..2.331) | 1.007x (0.974..1.040) | improved | planned-faster | planned-faster | parity |
| matmul | vector-matrix-negative-matrix | The contracted matrix axis has a negative stride. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(-256,1), flags=N | 200 | 0.117660 | 0.097674 | 0.097761 | 0.014787 (dense planned) | 0.052652 | 1.852x (1.743..1.889) | 1.852x (1.746..1.877) | 0.280x (0.263..0.284) | 2.239x (2.215..2.430) | improved | planned-faster | control-faster | planned-faster |
| matmul | vector-matrix-step2-matrix | The matrix column stride cannot be described by GEMV. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=1.0 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(512,2), flags=S | 200 | 0.117941 | 0.104956 | 0.104991 | 0.014726 (dense planned) | 0.052689 | 1.997x (1.983..2.083) | 1.990x (1.985..2.002) | 0.280x (0.278..0.285) | 2.241x (2.230..2.255) | improved | planned-faster | control-faster | planned-faster |
| matmul | vector-matrix-padded-matrix | The matrix has a padded leading dimension. | B=1, R=0, MxKxN=1x256x256<br>MAC=0.000G, logical input=0.5 MiB, backing input=1.0 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(512,1), flags=S | 200 | 0.014603 | 0.103804 | 0.103787 | 0.014683 (dense planned) | 0.014635 | 7.092x (7.015..7.144) | 7.101x (7.012..7.159) | 1.006x (0.994..1.027) | 0.997x (0.990..1.010) | improved | planned-faster | parity | parity |
| matmul | matrix-vector | Dense matrix-vector control. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.006746 | 0.057864 | 0.006673 | n/a | 0.006676 | 8.659x (8.531..8.691) | 0.999x (0.985..1.005) | n/a | 1.010x (0.999..1.027) | improved | parity | not-measured | parity |
| matmul | matrix-vector-negative-vector | The GEMV vector has a negative increment. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256), strides=(-1), flags=N | 200 | 0.115141 | 0.094538 | 0.094561 | 0.006687 (dense planned) | 0.007947 | 11.924x (11.474..12.001) | 11.911x (11.560..12.016) | 0.845x (0.816..0.876) | 14.538x (14.082..15.225) | improved | planned-faster | control-faster | planned-faster |
| matmul | matrix-vector-step2-vector | The GEMV vector has increment two. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256), strides=(2), flags=S | 200 | 0.006955 | 0.095327 | 0.095459 | 0.006733 (dense planned) | 0.006898 | 13.827x (13.638..14.041) | 13.843x (13.600..14.016) | 0.974x (0.958..0.990) | 1.011x (0.994..1.018) | improved | planned-faster | parity | parity |
| matmul | matrix-vector-f-matrix | The GEMV matrix is F-contiguous. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(1,256), flags=F<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.014503 | 0.097839 | 0.097866 | 0.006712 (dense planned) | 0.014536 | 6.730x (6.605..6.762) | 6.725x (6.589..6.768) | 0.460x (0.457..0.465) | 0.996x (0.981..1.009) | improved | planned-faster | control-faster | parity |
| matmul | matrix-vector-negative-matrix | The contracted matrix axis has a negative stride. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=0.5 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(256,-1), flags=N<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.117759 | 0.096616 | 0.096623 | 0.006718 (dense planned) | 0.046092 | 2.093x (2.079..2.111) | 2.101x (2.077..2.109) | 0.146x (0.145..0.150) | 2.551x (2.529..2.575) | improved | planned-faster | control-faster | planned-faster |
| matmul | matrix-vector-step2-matrix | The matrix inner stride cannot be described by GEMV. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=1.0 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.115006 | 0.095281 | 0.095277 | 0.006708 (dense planned) | 0.046286 | 2.058x (2.044..2.066) | 2.058x (2.046..2.064) | 0.144x (0.144..0.146) | 2.486x (2.471..2.495) | improved | planned-faster | control-faster | planned-faster |
| matmul | matrix-vector-padded-matrix | The matrix has a padded leading dimension. | B=1, R=0, MxKxN=256x256x1<br>MAC=0.000G, logical input=0.5 MiB, backing input=1.0 MiB, expanded input=0.5 MiB, output=0.0 MiB | lhs: shape=(256,256), strides=(512,1), flags=S<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.006882 | 0.059369 | 0.059271 | 0.006699 (dense planned) | 0.006855 | 8.651x (8.604..8.742) | 8.654x (8.621..8.703) | 0.978x (0.973..0.990) | 1.004x (0.997..1.031) | improved | planned-faster | parity | parity |

### matmul-batch

| Operation | Scenario | Purpose | Workload | Operands | Calls/sample | NumPy ms | Legacy ms | Control ms | Planned ms | Legacy/planned (q10..q90) | Control/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs control | Planned vs NumPy |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| matmul | batch-same-shape-c | Dense same-shape matrix batches. | B=8, R=1, MxKxN=32x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.1 MiB | lhs: shape=(8,32,32), strides=(1024,32,1), flags=C<br>rhs: shape=(8,32,32), strides=(1024,32,1), flags=C | 3 | 0.007292 | n/a | n/a | 0.007042 | n/a | n/a | 1.033x (1.006..1.062) | new-only | not-measured | inconclusive |
| matmul | batch-broadcast-c | Leading matrix batch axes broadcast. | B=64, R=2, MxKxN=32x32x32<br>MAC=0.002G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(8,1,32,32), strides=(1024,1024,32,1), flags=C<br>rhs: shape=(1,8,32,32), strides=(8192,1024,32,1), flags=C | 3 | 0.043042 | n/a | n/a | 0.042750 | n/a | n/a | 1.005x (0.968..1.015) | new-only | not-measured | parity |
| matmul | batch-broadcast-negative-matrix | Broadcast matrices have negative inner strides. | B=64, R=2, MxKxN=32x32x32<br>MAC=0.002G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(8,1,32,32), strides=(1024,1024,-32,-1), flags=N<br>rhs: shape=(1,8,32,32), strides=(8192,1024,-32,-1), flags=N | 3 | 1.591236 | n/a | 0.041083 (dense planned) | 0.051347 | n/a | 0.803x (0.797..0.845) | 31.007x (30.866..31.307) | new-only | control-faster | planned-faster |
| matmul | batch-broadcast-step2-inner | Broadcast matrices have step-two inner strides. | B=64, R=2, MxKxN=32x32x32<br>MAC=0.002G, logical input=0.1 MiB, backing input=0.2 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(8,1,32,32), strides=(2048,2048,64,2), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,64,2), flags=S | 3 | 1.594695 | n/a | 0.041319 (dense planned) | 0.051500 | n/a | 0.799x (0.794..0.882) | 30.974x (30.012..31.263) | new-only | control-faster | planned-faster |
| matmul | batch-broadcast-step2-batch | Matrix blocks are dense and batch axes are strided. | B=64, R=2, MxKxN=32x32x32<br>MAC=0.002G, logical input=0.1 MiB, backing input=0.2 MiB, expanded input=1.0 MiB, output=0.5 MiB | lhs: shape=(8,1,32,32), strides=(2048,1024,32,1), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,32,1), flags=S | 3 | 0.042569 | n/a | 0.042028 (dense planned) | 0.042153 | n/a | 0.996x (0.979..1.001) | 1.010x (0.986..1.022) | new-only | parity | parity |
| matmul | vector-batch-matrix-c | One dense vector is reused across two batch axes. | B=16, R=2, MxKxN=1x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(32), strides=(1), flags=CF<br>rhs: shape=(2,8,32,32), strides=(8192,1024,32,1), flags=C | 3 | 0.004945 | n/a | n/a | 0.012694 | n/a | n/a | 0.391x (0.382..0.402) | new-only | not-measured | numpy-faster |
| matmul | vector-batch-matrix-negative-vector | A negative-stride vector is reused across two batch axes. | B=16, R=2, MxKxN=1x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(32), strides=(-1), flags=N<br>rhs: shape=(2,8,32,32), strides=(8192,1024,32,1), flags=C | 3 | 0.014361 | n/a | 0.012653 (dense planned) | 0.012694 | n/a | 0.998x (0.989..1.002) | 1.134x (1.128..1.159) | new-only | parity | planned-faster |
| matmul | vector-batch-matrix-step2-vector | A step-two vector is reused across two batch axes. | B=16, R=2, MxKxN=1x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(32), strides=(2), flags=S<br>rhs: shape=(2,8,32,32), strides=(8192,1024,32,1), flags=C | 3 | 0.005028 | n/a | 0.012722 (dense planned) | 0.012764 | n/a | 0.998x (0.984..1.024) | 0.395x (0.388..0.400) | new-only | parity | numpy-faster |
| matmul | vector-batch-matrix-negative-matrix | The batched matrix contracted axis is negative. | B=16, R=2, MxKxN=1x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(32), strides=(1), flags=CF<br>rhs: shape=(2,8,32,32), strides=(8192,1024,-32,1), flags=N | 3 | 0.014625 | n/a | 0.012722 (dense planned) | 0.013070 | n/a | 0.973x (0.971..0.985) | 1.117x (1.103..1.130) | new-only | parity | planned-faster |
| matmul | vector-batch-matrix-step2-matrix | The batched matrix column stride is two. | B=16, R=2, MxKxN=1x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.3 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(32), strides=(1), flags=CF<br>rhs: shape=(2,8,32,32), strides=(16384,2048,64,2), flags=S | 3 | 0.014833 | n/a | 0.012708 (dense planned) | 0.013264 | n/a | 0.957x (0.949..0.971) | 1.119x (1.110..1.129) | new-only | inconclusive | planned-faster |
| matmul | vector-batch-matrix-padded-matrix | The batched matrix has a padded leading dimension. | B=16, R=2, MxKxN=1x32x32<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.3 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(32), strides=(1), flags=CF<br>rhs: shape=(2,8,32,32), strides=(16384,2048,64,1), flags=S | 3 | 0.005139 | n/a | 0.012722 (dense planned) | 0.012889 | n/a | 0.989x (0.981..0.993) | 0.400x (0.395..0.401) | new-only | parity | numpy-faster |
| matmul | batch-matrix-vector-c | One dense vector is reused across two batch axes. | B=16, R=2, MxKxN=32x32x1<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(2,8,32,32), strides=(8192,1024,32,1), flags=C<br>rhs: shape=(32), strides=(1), flags=CF | 3 | 0.005681 | n/a | n/a | 0.009389 | n/a | n/a | 0.606x (0.598..0.618) | new-only | not-measured | numpy-faster |
| matmul | batch-matrix-vector-negative-vector | A negative-stride vector is reused across two batch axes. | B=16, R=2, MxKxN=32x32x1<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(2,8,32,32), strides=(8192,1024,32,1), flags=C<br>rhs: shape=(32), strides=(-1), flags=N | 3 | 0.014736 | n/a | 0.009278 (dense planned) | 0.014055 | n/a | 0.659x (0.651..0.672) | 1.047x (1.040..1.057) | new-only | control-faster | inconclusive |
| matmul | batch-matrix-vector-step2-vector | A step-two vector is reused across two batch axes. | B=16, R=2, MxKxN=32x32x1<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(2,8,32,32), strides=(8192,1024,32,1), flags=C<br>rhs: shape=(32), strides=(2), flags=S | 3 | 0.005681 | n/a | 0.009222 (dense planned) | 0.013931 | n/a | 0.662x (0.658..0.665) | 0.407x (0.403..0.410) | new-only | control-faster | numpy-faster |
| matmul | batch-matrix-vector-negative-matrix | The batched matrix contracted axis is negative. | B=16, R=2, MxKxN=32x32x1<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.1 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(2,8,32,32), strides=(8192,1024,32,-1), flags=N<br>rhs: shape=(32), strides=(1), flags=CF | 3 | 0.014792 | n/a | 0.009333 (dense planned) | 0.013972 | n/a | 0.664x (0.658..0.671) | 1.056x (1.042..1.072) | new-only | control-faster | inconclusive |
| matmul | batch-matrix-vector-step2-matrix | The batched matrix inner stride is two. | B=16, R=2, MxKxN=32x32x1<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.3 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(2,8,32,32), strides=(16384,2048,64,2), flags=S<br>rhs: shape=(32), strides=(1), flags=CF | 3 | 0.014917 | n/a | 0.009417 (dense planned) | 0.014111 | n/a | 0.662x (0.655..0.679) | 1.060x (1.044..1.085) | new-only | control-faster | inconclusive |
| matmul | batch-matrix-vector-padded-matrix | The batched matrix has a padded leading dimension. | B=16, R=2, MxKxN=32x32x1<br>MAC=0.000G, logical input=0.1 MiB, backing input=0.3 MiB, expanded input=0.1 MiB, output=0.0 MiB | lhs: shape=(2,8,32,32), strides=(16384,2048,64,1), flags=S<br>rhs: shape=(32), strides=(1), flags=CF | 3 | 0.005667 | n/a | 0.009403 (dense planned) | 0.009208 | n/a | 1.015x (1.007..1.039) | 0.615x (0.602..0.624) | new-only | parity | numpy-faster |

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
