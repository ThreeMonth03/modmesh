# macOS detailed execution benchmark

## Recorded environment

- Code revision: `e3a0ba930bccab72107eeb8454083940db83d398`.
- Dirty tree: `false`.
- Platform: `macOS-26.5.1-arm64-arm-64bit`.
- Machine: `arm64`.
- Python: `3.11.6`.
- NumPy: `2.2.4`.
- Seed: `20260722`.
- Fixed cases: `154`.
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
    --repeat 15 \
    --warmup 5 \
    --output /tmp/solvcon-execution-e3a0ba93.json
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
| elementwise-array | 20 | 11 | 0 | 0 | 8 | 0 | 1 |
| elementwise-scalar | 16 | 12 | 0 | 0 | 4 | 0 | 0 |
| elementwise-broadcast | 24 | 0 | 0 | 0 | 0 | 24 | 0 |
| inplace-array | 12 | 0 | 4 | 0 | 8 | 0 | 0 |
| inplace-broadcast | 8 | 0 | 0 | 0 | 0 | 8 | 0 |
| inplace-scalar | 12 | 0 | 7 | 0 | 4 | 0 | 1 |
| reduction-axis | 25 | 25 | 0 | 0 | 0 | 0 | 0 |
| reduction-full | 20 | 18 | 1 | 0 | 0 | 0 | 1 |
| matmul | 12 | 10 | 2 | 0 | 0 | 0 | 0 |
| matmul-batch | 5 | 0 | 0 | 0 | 0 | 5 | 0 |

## Complete results

### elementwise-array

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 0.620842 | 1.277277 | 0.604806 | 1.962x (1.618..2.487) | 1.040x (0.955..1.524) | improved | inconclusive |
| sub | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 0.538090 | 1.081862 | 0.539985 | 1.941x (1.844..2.095) | 0.997x (0.896..1.125) | improved | inconclusive |
| mul | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 0.714604 | 1.427433 | 0.724896 | 1.929x (1.873..2.204) | 0.981x (0.880..1.136) | improved | inconclusive |
| div | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 0.707719 | 1.327035 | 0.655408 | 1.959x (1.568..2.226) | 0.977x (0.743..1.055) | improved | inconclusive |
| add | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 0.702227 | 1.417560 | 0.703206 | 1.925x (1.728..2.252) | 0.996x (0.912..1.050) | improved | inconclusive |
| sub | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 0.692263 | 1.297898 | 0.643577 | 1.930x (1.683..2.081) | 1.009x (0.923..1.072) | improved | inconclusive |
| mul | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 0.615879 | 1.186256 | 0.599129 | 1.902x (1.668..2.138) | 1.003x (0.798..1.108) | improved | inconclusive |
| div | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 0.587167 | 1.095898 | 0.584490 | 1.903x (1.872..2.036) | 1.018x (0.994..1.059) | improved | inconclusive |
| add | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 1.067338 | 1.406760 | 0.751971 | 1.850x (0.952..2.060) | 1.509x (0.848..1.846) | inconclusive | inconclusive |
| sub | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 0.796702 | 1.028396 | 0.525696 | 1.929x (1.886..2.032) | 1.518x (1.489..1.561) | improved | planned-faster |
| mul | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 0.807865 | 1.050704 | 0.539083 | 1.941x (1.852..2.173) | 1.521x (1.488..1.549) | improved | planned-faster |
| div | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 0.818706 | 1.051115 | 0.575956 | 1.899x (1.856..1.950) | 1.522x (1.405..1.544) | improved | planned-faster |
| add | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 0.951498 | n/a | 1.172198 | n/a | 0.814x (0.739..0.985) | legacy-incorrect | inconclusive |
| sub | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.102806 | n/a | 1.327623 | n/a | 0.831x (0.816..0.923) | legacy-incorrect | numpy-faster |
| mul | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.101667 | n/a | 1.351321 | n/a | 0.816x (0.807..0.848) | legacy-incorrect | numpy-faster |
| div | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.105498 | n/a | 1.114054 | n/a | 0.993x (0.986..1.035) | legacy-incorrect | parity |
| add | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 0.900090 | n/a | 1.004519 | n/a | 0.896x (0.883..0.919) | legacy-incorrect | numpy-faster |
| sub | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 0.897969 | n/a | 1.005037 | n/a | 0.896x (0.890..0.917) | legacy-incorrect | numpy-faster |
| mul | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 0.898571 | n/a | 1.003200 | n/a | 0.895x (0.866..0.906) | legacy-incorrect | numpy-faster |
| div | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 0.904654 | n/a | 0.901754 | n/a | 1.040x (0.995..1.052) | legacy-incorrect | inconclusive |

### elementwise-scalar

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.372402 | 0.924535 | 0.398229 | 2.322x (2.073..2.546) | 0.999x (0.921..1.025) | improved | inconclusive |
| sub | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.304492 | 0.699881 | 0.303023 | 2.290x (2.272..2.352) | 1.004x (0.994..1.009) | improved | parity |
| mul | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.307517 | 0.732808 | 0.306463 | 2.385x (2.360..2.557) | 1.001x (0.992..1.012) | improved | parity |
| div | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.343371 | 0.822229 | 0.344258 | 2.389x (2.324..2.481) | 0.992x (0.979..1.016) | improved | parity |
| add | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.347892 | 0.781473 | 0.348219 | 2.244x (2.203..2.293) | 0.999x (0.993..1.006) | improved | parity |
| sub | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.344408 | 0.764571 | 0.344183 | 2.250x (2.200..2.404) | 1.006x (0.994..1.065) | improved | inconclusive |
| mul | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.345842 | 0.756913 | 0.347238 | 2.233x (2.180..2.347) | 0.998x (0.987..1.042) | improved | parity |
| div | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.346371 | 0.795344 | 0.347325 | 2.336x (2.274..2.431) | 1.000x (0.979..1.005) | improved | parity |
| add | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.685190 | 0.706315 | 0.303456 | 2.338x (2.291..2.466) | 2.269x (2.231..2.333) | improved | planned-faster |
| sub | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.686158 | 0.713400 | 0.303625 | 2.339x (2.300..2.390) | 2.260x (2.220..2.279) | improved | planned-faster |
| mul | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.685992 | 0.708315 | 0.304027 | 2.330x (2.280..2.398) | 2.254x (2.236..2.272) | improved | planned-faster |
| div | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.775963 | 0.850148 | 0.343317 | 2.322x (2.150..2.556) | 2.281x (2.243..2.511) | improved | planned-faster |
| add | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 0.700652 | n/a | 0.602298 | n/a | 1.163x (1.157..1.169) | legacy-incorrect | planned-faster |
| sub | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 0.933208 | n/a | 0.811821 | n/a | 1.154x (0.968..2.655) | legacy-incorrect | inconclusive |
| mul | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 0.974967 | n/a | 0.815363 | n/a | 1.175x (1.133..1.257) | legacy-incorrect | planned-faster |
| div | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 0.967533 | n/a | 0.831050 | n/a | 1.153x (1.024..1.177) | legacy-incorrect | inconclusive |

### elementwise-broadcast

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.056704 | n/a | 0.037825 | n/a | 1.488x (1.446..1.522) | new-only | planned-faster |
| sub | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.057275 | n/a | 0.038762 | n/a | 1.487x (1.432..1.516) | new-only | planned-faster |
| mul | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.055963 | n/a | 0.037571 | n/a | 1.488x (1.449..1.511) | new-only | planned-faster |
| div | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.048129 | n/a | 0.032487 | n/a | 1.484x (1.470..1.513) | new-only | planned-faster |
| add | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.046446 | n/a | 0.031496 | n/a | 1.471x (1.447..1.486) | new-only | planned-faster |
| sub | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.046829 | n/a | 0.031850 | n/a | 1.472x (1.392..1.509) | new-only | planned-faster |
| mul | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.048029 | n/a | 0.031775 | n/a | 1.484x (1.401..1.623) | new-only | planned-faster |
| div | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.046017 | n/a | 0.031746 | n/a | 1.451x (1.444..1.456) | new-only | planned-faster |
| add | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.056129 | n/a | 0.059263 | n/a | 0.944x (0.919..0.954) | new-only | inconclusive |
| sub | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.049338 | n/a | 0.052792 | n/a | 0.949x (0.886..0.959) | new-only | inconclusive |
| mul | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.048767 | n/a | 0.052108 | n/a | 0.939x (0.883..0.949) | new-only | numpy-faster |
| div | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.051412 | n/a | 0.051775 | n/a | 0.990x (0.976..1.001) | new-only | parity |
| add | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.065775 | n/a | 0.052883 | n/a | 1.245x (1.230..1.261) | new-only | planned-faster |
| sub | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.065125 | n/a | 0.052679 | n/a | 1.241x (1.233..1.250) | new-only | planned-faster |
| mul | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.065025 | n/a | 0.052467 | n/a | 1.239x (1.235..1.247) | new-only | planned-faster |
| div | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.067321 | n/a | 0.054325 | n/a | 1.246x (1.197..1.302) | new-only | planned-faster |
| add | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.085279 | n/a | 0.046108 | n/a | 1.853x (1.840..1.870) | new-only | planned-faster |
| sub | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.085146 | n/a | 0.045963 | n/a | 1.856x (1.841..1.865) | new-only | planned-faster |
| mul | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.085371 | n/a | 0.046408 | n/a | 1.844x (1.830..1.855) | new-only | planned-faster |
| div | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.085342 | n/a | 0.046296 | n/a | 1.845x (1.828..1.853) | new-only | planned-faster |
| add | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.083150 | n/a | 0.090254 | n/a | 0.921x (0.907..0.931) | new-only | numpy-faster |
| sub | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.083275 | n/a | 0.090487 | n/a | 0.927x (0.898..0.941) | new-only | numpy-faster |
| mul | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.083013 | n/a | 0.089854 | n/a | 0.928x (0.909..0.939) | new-only | numpy-faster |
| div | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.088167 | n/a | 0.090704 | n/a | 0.974x (0.968..0.992) | new-only | parity |

### inplace-array

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.102812 | 0.098693 | 0.098322 | 1.004x (1.000..1.011) | 1.045x (1.038..1.057) | parity | inconclusive |
| sub | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.102653 | 0.098280 | 0.098762 | 0.996x (0.992..1.001) | 1.040x (1.034..1.044) | parity | parity |
| mul | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.102618 | 0.098712 | 0.098952 | 0.998x (0.986..1.007) | 1.037x (1.028..1.045) | parity | parity |
| div | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.107681 | 0.099598 | 0.101281 | 0.984x (0.975..0.991) | 1.063x (1.057..1.071) | parity | planned-faster |
| add | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.209145 | n/a | 0.187408 | n/a | 1.116x (1.109..1.129) | legacy-incorrect | planned-faster |
| sub | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.207883 | n/a | 0.196824 | n/a | 1.056x (1.009..1.079) | legacy-incorrect | inconclusive |
| mul | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.208602 | n/a | 0.193949 | n/a | 1.088x (0.850..1.109) | legacy-incorrect | inconclusive |
| div | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.225910 | n/a | 0.219069 | n/a | 1.042x (0.966..1.073) | legacy-incorrect | inconclusive |
| add | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.206153 | n/a | 0.201554 | n/a | 1.026x (0.993..1.049) | legacy-incorrect | parity |
| sub | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.201753 | n/a | 0.200678 | n/a | 1.018x (0.997..1.033) | legacy-incorrect | parity |
| mul | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.202403 | n/a | 0.202237 | n/a | 1.008x (0.987..1.023) | legacy-incorrect | parity |
| div | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.203857 | n/a | 0.209155 | n/a | 0.976x (0.965..0.982) | legacy-incorrect | parity |

### inplace-broadcast

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.154061 | n/a | 0.114296 | n/a | 1.348x (1.336..1.373) | new-only | planned-faster |
| sub | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.153085 | n/a | 0.115148 | n/a | 1.332x (1.314..1.385) | new-only | planned-faster |
| mul | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.153877 | n/a | 0.111865 | n/a | 1.380x (1.346..1.403) | new-only | planned-faster |
| div | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.156276 | n/a | 0.119678 | n/a | 1.310x (1.289..1.329) | new-only | planned-faster |
| add | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.250426 | n/a | 0.179731 | n/a | 1.393x (1.379..1.401) | new-only | planned-faster |
| sub | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.251797 | n/a | 0.180244 | n/a | 1.393x (1.382..1.408) | new-only | planned-faster |
| mul | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.250424 | n/a | 0.179423 | n/a | 1.407x (1.378..1.419) | new-only | planned-faster |
| div | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.255216 | n/a | 0.183599 | n/a | 1.391x (1.375..1.403) | new-only | planned-faster |

### inplace-scalar

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.063205 | 0.063225 | 0.063273 | 1.000x (0.986..1.008) | 0.995x (0.983..1.006) | parity | parity |
| sub | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.063059 | 0.063435 | 0.063325 | 1.002x (0.990..1.011) | 0.996x (0.989..1.001) | parity | parity |
| mul | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.062979 | 0.062883 | 0.062832 | 0.998x (0.986..1.014) | 1.002x (0.988..1.015) | parity | parity |
| div | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.072553 | 0.070968 | 0.071349 | 0.998x (0.981..1.010) | 1.018x (1.003..1.023) | parity | parity |
| add | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.063889 | 0.062903 | 0.062930 | 0.997x (0.991..1.012) | 1.016x (1.007..1.024) | parity | parity |
| sub | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.063921 | 0.062981 | 0.062929 | 1.001x (0.989..1.025) | 1.015x (1.001..1.025) | parity | parity |
| mul | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.073385 | 0.073599 | 0.072452 | 1.011x (0.991..1.257) | 1.016x (1.000..1.051) | inconclusive | inconclusive |
| div | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.084479 | 0.080263 | 0.080453 | 0.997x (0.990..1.002) | 1.047x (1.038..1.061) | parity | inconclusive |
| add | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.179441 | n/a | 0.155637 | n/a | 1.151x (1.146..1.161) | legacy-incorrect | planned-faster |
| sub | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.178734 | n/a | 0.153862 | n/a | 1.162x (1.156..1.166) | legacy-incorrect | planned-faster |
| mul | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.180312 | n/a | 0.153593 | n/a | 1.175x (1.168..1.181) | legacy-incorrect | planned-faster |
| div | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.183612 | n/a | 0.159494 | n/a | 1.150x (1.143..1.162) | legacy-incorrect | planned-faster |

### reduction-axis

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| mean | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.088100 | 3.857279 | 0.195488 | 19.689x (19.090..20.033) | 0.453x (0.442..0.466) | improved | numpy-faster |
| var | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.400504 | 4.282150 | 0.411700 | 10.430x (10.312..10.532) | 0.975x (0.964..1.030) | improved | parity |
| std | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.398725 | 4.296746 | 0.411600 | 10.467x (10.335..10.493) | 0.969x (0.959..0.983) | improved | parity |
| median | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 4.184062 | 4.377742 | 1.931862 | 2.269x (2.236..2.290) | 2.169x (2.151..2.204) | improved | planned-faster |
| average | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.244400 | 3.858792 | 0.213133 | 18.074x (15.982..18.275) | 1.139x (1.049..1.203) | improved | inconclusive |
| mean | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.086450 | 3.882375 | 0.450446 | 8.600x (7.895..8.795) | 0.185x (0.179..0.203) | improved | numpy-faster |
| var | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.396858 | 4.273621 | 0.979988 | 4.351x (4.286..4.415) | 0.405x (0.397..0.410) | improved | numpy-faster |
| std | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.394454 | 4.282942 | 0.975133 | 4.391x (4.372..4.419) | 0.405x (0.402..0.409) | improved | numpy-faster |
| median | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 6.433117 | 4.403100 | 2.129029 | 2.072x (2.039..2.085) | 3.030x (2.975..3.070) | improved | planned-faster |
| average | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.225175 | 3.874158 | 0.534575 | 7.253x (6.809..7.305) | 0.421x (0.416..0.461) | improved | numpy-faster |
| mean | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 0.092992 | 3.856679 | 0.193833 | 19.873x (19.561..19.954) | 0.480x (0.477..0.488) | improved | numpy-faster |
| var | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 0.575287 | 4.279113 | 0.409858 | 10.428x (10.322..10.489) | 1.403x (1.389..1.411) | improved | planned-faster |
| std | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 0.576967 | 4.297042 | 0.411025 | 10.458x (10.415..10.595) | 1.405x (1.392..1.420) | improved | planned-faster |
| median | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 4.372087 | 4.427087 | 2.074067 | 2.144x (2.092..2.174) | 2.106x (2.041..2.240) | improved | planned-faster |
| average | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.418837 | 3.871992 | 0.530750 | 7.312x (7.159..7.385) | 0.789x (0.767..0.812) | improved | numpy-faster |
| mean | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.111871 | 3.890196 | 0.411596 | 9.458x (8.183..9.552) | 0.266x (0.242..0.278) | improved | numpy-faster |
| var | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.509262 | 4.284675 | 0.953667 | 4.488x (4.338..4.533) | 0.536x (0.529..0.562) | improved | numpy-faster |
| std | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.528879 | 4.356396 | 0.960196 | 4.550x (4.307..4.584) | 0.539x (0.530..1.016) | improved | inconclusive |
| median | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 4.307567 | 4.396496 | 2.054283 | 2.145x (2.108..2.299) | 2.092x (2.021..2.115) | improved | planned-faster |
| average | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.336783 | 3.899971 | 0.529913 | 7.354x (7.291..7.450) | 0.636x (0.620..0.685) | improved | numpy-faster |
| mean | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.095812 | 3.866325 | 0.198163 | 19.521x (18.130..19.728) | 0.480x (0.449..0.496) | improved | numpy-faster |
| var | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.469383 | 4.269737 | 0.411946 | 10.366x (10.306..10.419) | 1.139x (1.133..1.160) | improved | planned-faster |
| std | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.475579 | 4.297438 | 0.414704 | 10.374x (10.171..10.793) | 1.150x (1.136..1.173) | improved | planned-faster |
| median | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 4.248800 | 4.397000 | 1.941258 | 2.276x (2.128..2.285) | 2.191x (2.032..2.218) | improved | planned-faster |
| average | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.352375 | 3.913621 | 0.216892 | 18.070x (17.627..18.351) | 1.597x (1.515..1.844) | improved | planned-faster |

### reduction-full

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| mean | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.083488 | 0.403017 | 0.202096 | 1.996x (1.980..2.021) | 0.413x (0.407..0.436) | improved | numpy-faster |
| var | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.341679 | 3.133879 | 0.406904 | 7.680x (7.523..7.751) | 0.833x (0.811..0.876) | improved | numpy-faster |
| std | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.331183 | 3.109079 | 0.405613 | 7.653x (7.537..7.693) | 0.814x (0.802..0.849) | improved | numpy-faster |
| median | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 3.455692 | 3.710875 | 1.861392 | 1.988x (1.873..2.129) | 1.857x (1.534..1.876) | improved | planned-faster |
| average | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.285075 | 7.774763 | 0.409571 | 19.055x (18.124..19.210) | 0.691x (0.661..0.729) | improved | numpy-faster |
| mean | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.088025 | 0.412538 | 0.204450 | 1.994x (1.630..2.395) | 0.421x (0.399..0.549) | improved | numpy-faster |
| var | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.337058 | 3.142392 | 0.406162 | 7.696x (7.409..7.791) | 0.829x (0.790..0.864) | improved | numpy-faster |
| std | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.335767 | 3.107483 | 0.404867 | 7.686x (7.624..9.689) | 0.832x (0.810..0.881) | improved | numpy-faster |
| median | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 3.798729 | 3.736708 | 1.999408 | 1.856x (1.814..2.203) | 1.892x (1.863..2.035) | improved | planned-faster |
| average | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.638979 | 7.737867 | 0.743967 | 10.419x (10.361..10.585) | 0.847x (0.834..0.904) | improved | numpy-faster |
| mean | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.083504 | 0.403054 | 0.202075 | 1.994x (1.966..2.006) | 0.413x (0.409..0.416) | improved | numpy-faster |
| var | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.425533 | 3.091083 | 0.403850 | 7.659x (7.616..7.696) | 1.055x (1.049..1.065) | improved | inconclusive |
| std | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.430571 | 3.096738 | 0.404496 | 7.656x (7.593..7.925) | 1.059x (1.050..1.102) | improved | planned-faster |
| median | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 4.422321 | 3.607704 | 1.866408 | 1.935x (1.926..1.948) | 2.367x (2.353..2.409) | improved | planned-faster |
| average | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.348329 | 7.701842 | 0.741929 | 10.389x (10.311..10.698) | 0.464x (0.457..0.495) | improved | numpy-faster |
| mean | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.099900 | 0.403154 | 0.403467 | 0.999x (0.996..1.003) | 0.247x (0.247..0.251) | parity | numpy-faster |
| var | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.462225 | 3.118033 | 0.944875 | 3.299x (3.249..3.334) | 0.484x (0.475..0.495) | improved | numpy-faster |
| std | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.455950 | 3.105250 | 0.945350 | 3.293x (3.267..3.306) | 0.481x (0.478..0.487) | improved | numpy-faster |
| median | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 3.490363 | 3.666867 | 2.270767 | 1.907x (0.587..2.084) | 1.808x (0.467..1.969) | inconclusive | inconclusive |
| average | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.543225 | 11.594113 | 1.133000 | 10.117x (8.663..11.309) | 0.485x (0.399..0.523) | improved | numpy-faster |

### matmul

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matmul | matrix-matrix-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 0.202929 | 31.611631 | 0.198744 | 159.893x (150.767..171.066) | 1.022x (1.005..1.111) | improved | inconclusive |
| matmul | matrix-matrix-f | lhs: shape=(256,256), strides=(1,256), flags=F<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 0.226542 | 36.348726 | 0.231589 | 167.289x (135.018..183.984) | 1.016x (0.911..1.206) | improved | inconclusive |
| matmul | matrix-matrix-negative | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(256,256), strides=(-256,-1), flags=N | 7 | 29.766345 | 24.632679 | 0.240018 | 103.907x (82.618..105.778) | 125.602x (122.842..139.172) | improved | planned-faster |
| matmul | matrix-matrix-lhs-step2 | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 29.674821 | 24.729946 | 0.195375 | 125.935x (122.634..130.188) | 151.998x (150.193..158.872) | improved | planned-faster |
| matmul | matrix-matrix-rhs-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 29.928851 | 26.988458 | 0.194446 | 138.114x (134.410..142.326) | 154.310x (150.141..157.346) | improved | planned-faster |
| matmul | matrix-matrix-both-step2 | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 30.107548 | 27.202268 | 0.236571 | 114.705x (113.756..119.227) | 127.436x (125.592..131.902) | improved | planned-faster |
| matmul | matrix-matrix-mixed-c-f | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 0.175982 | 14.698482 | 0.172935 | 84.970x (83.037..85.861) | 1.015x (0.991..1.035) | improved | parity |
| matmul | matrix-matrix-rectangular | lhs: shape=(128,256), strides=(256,1), flags=C<br>rhs: shape=(256,64), strides=(64,1), flags=C | 40 | 0.025071 | 3.054314 | 0.024953 | 122.920x (120.628..130.978) | 0.999x (0.994..1.109) | improved | inconclusive |
| matmul | matrix-matrix-small-direct | lhs: shape=(8,16), strides=(16,1), flags=C<br>rhs: shape=(16,8), strides=(8,1), flags=C | 2000 | 0.001146 | 0.001531 | 0.001535 | 0.998x (0.985..1.002) | 0.744x (0.740..0.752) | parity | numpy-faster |
| matmul | vector-vector | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256), strides=(1), flags=CF | 2000 | 0.000883 | 0.001177 | 0.001181 | 0.997x (0.983..1.003) | 0.748x (0.733..0.751) | parity | numpy-faster |
| matmul | vector-matrix | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(256,1), flags=C | 200 | 0.014398 | 0.097119 | 0.014446 | 6.729x (6.682..6.772) | 0.997x (0.991..1.003) | improved | parity |
| matmul | matrix-vector | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.006812 | 0.057970 | 0.006751 | 8.581x (8.549..8.624) | 1.008x (1.002..1.012) | improved | parity |

### matmul-batch

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matmul | batch-same-shape-c | lhs: shape=(8,32,32), strides=(1024,32,1), flags=C<br>rhs: shape=(8,32,32), strides=(1024,32,1), flags=C | 3 | 0.007292 | n/a | 0.007180 | n/a | 1.010x (0.893..1.041) | new-only | inconclusive |
| matmul | batch-broadcast-c | lhs: shape=(8,1,32,32), strides=(1024,1024,32,1), flags=C<br>rhs: shape=(1,8,32,32), strides=(8192,1024,32,1), flags=C | 3 | 0.042431 | n/a | 0.042403 | n/a | 1.001x (0.989..1.011) | new-only | parity |
| matmul | batch-broadcast-negative-matrix | lhs: shape=(8,1,32,32), strides=(1024,1024,-32,-1), flags=N<br>rhs: shape=(1,8,32,32), strides=(8192,1024,-32,-1), flags=N | 3 | 1.588194 | n/a | 1.389986 | n/a | 1.142x (1.135..1.151) | new-only | planned-faster |
| matmul | batch-broadcast-step2-inner | lhs: shape=(8,1,32,32), strides=(2048,2048,64,2), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,64,2), flags=S | 3 | 1.602695 | n/a | 1.400972 | n/a | 1.147x (1.135..1.156) | new-only | planned-faster |
| matmul | batch-broadcast-step2-batch | lhs: shape=(8,1,32,32), strides=(2048,1024,32,1), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,32,1), flags=S | 3 | 0.045042 | n/a | 0.043986 | n/a | 1.020x (0.972..1.049) | new-only | parity |

## Legacy correctness failures

These rows are not performance regressions. Planned matched
NumPy, while legacy returned a different value.

| Family | Operation | Scenario | Error | Mismatch summary |
| --- | --- | --- | --- | --- |
| elementwise-array | add | step2-inner | AssertionError | Mismatched elements: 524288 / 1048576 (50%) |
| elementwise-array | sub | step2-inner | AssertionError | Mismatched elements: 524288 / 1048576 (50%) |
| elementwise-array | mul | step2-inner | AssertionError | Mismatched elements: 524288 / 1048576 (50%) |
| elementwise-array | div | step2-inner | AssertionError | Mismatched elements: 524288 / 1048576 (50%) |
| elementwise-array | add | mixed-c-step2 | AssertionError | Mismatched elements: 1048575 / 1048576 (100%) |
| elementwise-array | sub | mixed-c-step2 | AssertionError | Mismatched elements: 1048575 / 1048576 (100%) |
| elementwise-array | mul | mixed-c-step2 | AssertionError | Mismatched elements: 1048575 / 1048576 (100%) |
| elementwise-array | div | mixed-c-step2 | AssertionError | Not equal to tolerance rtol=1e-11, atol=1e-12 |
| elementwise-scalar | add | step2-inner | AssertionError | Mismatched elements: 524288 / 1048576 (50%) |
| elementwise-scalar | sub | step2-inner | AssertionError | Mismatched elements: 524288 / 1048576 (50%) |
| elementwise-scalar | mul | step2-inner | AssertionError | Mismatched elements: 524288 / 1048576 (50%) |
| elementwise-scalar | div | step2-inner | AssertionError | Mismatched elements: 524288 / 1048576 (50%) |
| inplace-array | add | negative-destination-c-rhs | AssertionError | Mismatched elements: 262144 / 262144 (100%) |
| inplace-array | sub | negative-destination-c-rhs | AssertionError | Mismatched elements: 262144 / 262144 (100%) |
| inplace-array | mul | negative-destination-c-rhs | AssertionError | Mismatched elements: 262144 / 262144 (100%) |
| inplace-array | div | negative-destination-c-rhs | AssertionError | Mismatched elements: 262144 / 262144 (100%) |
| inplace-array | add | step2-destination-c-rhs | AssertionError | Mismatched elements: 262142 / 262144 (100%) |
| inplace-array | sub | step2-destination-c-rhs | AssertionError | Mismatched elements: 262143 / 262144 (100%) |
| inplace-array | mul | step2-destination-c-rhs | AssertionError | Mismatched elements: 262142 / 262144 (100%) |
| inplace-array | div | step2-destination-c-rhs | AssertionError | Mismatched elements: 262142 / 262144 (100%) |
| inplace-scalar | add | step2-destination | AssertionError | Mismatched elements: 131072 / 262144 (50%) |
| inplace-scalar | sub | step2-destination | AssertionError | Mismatched elements: 131072 / 262144 (50%) |
| inplace-scalar | mul | step2-destination | AssertionError | Mismatched elements: 131072 / 262144 (50%) |
| inplace-scalar | div | step2-destination | AssertionError | Mismatched elements: 131072 / 262144 (50%) |

## Interpretation boundary

This notebook records every row, including inconclusive results.
It is evidence for choosing reusable routes, not a claim that the
prototype is the final implementation. Results from another OS,
architecture, or BLAS backend require a separate run.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
