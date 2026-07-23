# macOS detailed execution benchmark

## Recorded environment

- Code revision: `72283a19e9c44b48f6fbac6c1ecb4a444989c99c`.
- Dirty tree: `false`.
- Platform: `macOS-26.5.1-arm64-arm-64bit`.
- Machine: `arm64`.
- Python: `3.11.6`.
- NumPy: `2.2.4`.
- Seed: `20260722`.
- Fixed cases: `154`.
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
    --repeat 15 \
    --warmup 5 \
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
| elementwise-array | 20 | 12 | 0 | 0 | 8 | 0 | 0 |
| elementwise-scalar | 16 | 12 | 0 | 0 | 4 | 0 | 0 |
| elementwise-broadcast | 24 | 0 | 0 | 0 | 0 | 24 | 0 |
| inplace-array | 12 | 0 | 1 | 0 | 8 | 0 | 3 |
| inplace-broadcast | 8 | 0 | 0 | 0 | 0 | 8 | 0 |
| inplace-scalar | 12 | 0 | 0 | 0 | 4 | 0 | 8 |
| reduction-axis | 25 | 25 | 0 | 0 | 0 | 0 | 0 |
| reduction-full | 20 | 19 | 1 | 0 | 0 | 0 | 0 |
| matmul | 12 | 10 | 1 | 0 | 0 | 0 | 1 |
| matmul-batch | 5 | 0 | 0 | 0 | 0 | 5 | 0 |

## Complete results

### elementwise-array

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 1.075927 | 2.144025 | 1.102667 | 1.982x (1.775..2.143) | 0.998x (0.912..1.238) | improved | inconclusive |
| sub | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 1.125042 | 2.214796 | 1.145635 | 1.878x (1.622..2.161) | 1.008x (0.889..1.289) | improved | inconclusive |
| mul | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 1.107677 | 2.064981 | 1.151019 | 1.863x (1.589..1.993) | 0.934x (0.743..1.066) | improved | inconclusive |
| div | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 1.170985 | 2.124438 | 1.181340 | 1.833x (1.566..1.914) | 0.974x (0.836..1.059) | improved | inconclusive |
| add | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 1.084812 | 2.134492 | 1.179194 | 1.799x (1.516..2.075) | 0.914x (0.834..1.042) | improved | inconclusive |
| sub | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 1.105619 | 2.131185 | 1.146831 | 1.766x (1.471..2.104) | 0.930x (0.793..1.156) | improved | inconclusive |
| mul | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 1.116777 | 2.176750 | 1.190496 | 1.883x (1.601..2.162) | 0.951x (0.828..1.109) | improved | inconclusive |
| div | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 1.145635 | 2.178244 | 1.152337 | 1.828x (1.519..2.023) | 0.942x (0.830..1.057) | improved | inconclusive |
| add | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 1.853852 | 2.127304 | 1.204658 | 1.843x (1.626..2.230) | 1.522x (1.130..1.734) | improved | planned-faster |
| sub | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 1.618181 | 2.125619 | 1.110729 | 1.952x (1.524..2.227) | 1.487x (1.140..1.668) | improved | planned-faster |
| mul | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 1.657271 | 2.205860 | 1.279083 | 1.743x (1.443..1.960) | 1.335x (1.117..1.979) | improved | planned-faster |
| div | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 1.774435 | 2.379125 | 1.306910 | 1.773x (1.539..2.058) | 1.343x (1.221..1.508) | improved | planned-faster |
| add | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 2.171429 | n/a | 2.064390 | n/a | 1.069x (0.994..1.349) | legacy-incorrect | inconclusive |
| sub | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 2.000688 | n/a | 2.077754 | n/a | 0.964x (0.695..1.078) | legacy-incorrect | inconclusive |
| mul | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 2.004727 | n/a | 1.998263 | n/a | 1.045x (0.899..1.154) | legacy-incorrect | inconclusive |
| div | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 2.056413 | n/a | 2.046850 | n/a | 1.023x (0.950..1.187) | legacy-incorrect | inconclusive |
| add | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 2.048185 | n/a | 1.984140 | n/a | 1.160x (0.942..1.430) | legacy-incorrect | inconclusive |
| sub | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.933660 | n/a | 1.787481 | n/a | 1.053x (0.824..1.137) | legacy-incorrect | inconclusive |
| mul | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.947454 | n/a | 1.758746 | n/a | 1.084x (0.921..1.178) | legacy-incorrect | inconclusive |
| div | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.894185 | n/a | 1.834013 | n/a | 1.082x (0.873..1.319) | legacy-incorrect | inconclusive |

### elementwise-scalar

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.801360 | 1.881823 | 0.870306 | 2.082x (1.503..2.544) | 0.903x (0.719..1.082) | improved | inconclusive |
| sub | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.834496 | 1.812823 | 0.860915 | 2.232x (1.623..2.646) | 0.889x (0.802..1.171) | improved | inconclusive |
| mul | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.734717 | 1.761392 | 0.810796 | 2.283x (1.906..2.562) | 0.909x (0.743..1.097) | improved | inconclusive |
| div | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.788242 | 1.923958 | 0.826452 | 2.423x (2.037..2.745) | 0.991x (0.819..1.187) | improved | inconclusive |
| add | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.842477 | 1.938419 | 0.791219 | 2.405x (1.800..2.817) | 1.002x (0.768..1.830) | improved | inconclusive |
| sub | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.759960 | 1.768587 | 0.800658 | 2.173x (1.990..2.686) | 0.942x (0.796..1.085) | improved | inconclusive |
| mul | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.805575 | 1.793498 | 0.862033 | 2.126x (1.752..2.570) | 0.928x (0.641..1.125) | improved | inconclusive |
| div | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.793769 | 1.859329 | 0.793096 | 2.405x (2.001..2.572) | 1.040x (0.802..1.192) | improved | inconclusive |
| add | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 1.729840 | 1.760183 | 0.759452 | 2.324x (2.148..2.510) | 2.296x (2.148..2.451) | improved | planned-faster |
| sub | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 1.604056 | 1.756846 | 0.757610 | 2.331x (2.187..2.446) | 2.209x (1.923..2.319) | improved | planned-faster |
| mul | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 1.595771 | 1.774615 | 0.746115 | 2.472x (2.113..2.774) | 2.132x (1.866..2.260) | improved | planned-faster |
| div | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 1.605500 | 1.901894 | 0.811292 | 2.533x (1.822..2.788) | 2.008x (1.568..2.198) | improved | planned-faster |
| add | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 1.852665 | n/a | 1.394467 | n/a | 1.309x (1.155..1.480) | legacy-incorrect | planned-faster |
| sub | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 1.671250 | n/a | 1.390523 | n/a | 1.208x (1.101..1.332) | legacy-incorrect | planned-faster |
| mul | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 1.718069 | n/a | 1.362917 | n/a | 1.227x (1.170..1.490) | legacy-incorrect | planned-faster |
| div | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 1.676435 | n/a | 1.396083 | n/a | 1.187x (1.124..1.266) | legacy-incorrect | planned-faster |

### elementwise-broadcast

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.083262 | n/a | 0.058900 | n/a | 1.413x (1.208..1.592) | new-only | planned-faster |
| sub | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.078333 | n/a | 0.053442 | n/a | 1.489x (1.358..1.621) | new-only | planned-faster |
| mul | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.072883 | n/a | 0.055342 | n/a | 1.460x (1.290..1.823) | new-only | planned-faster |
| div | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.085696 | n/a | 0.062796 | n/a | 1.425x (1.222..1.483) | new-only | planned-faster |
| add | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.082021 | n/a | 0.060496 | n/a | 1.400x (1.271..1.470) | new-only | planned-faster |
| sub | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.080458 | n/a | 0.057921 | n/a | 1.398x (1.291..1.681) | new-only | planned-faster |
| mul | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.080171 | n/a | 0.057092 | n/a | 1.407x (1.334..1.692) | new-only | planned-faster |
| div | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.080879 | n/a | 0.059246 | n/a | 1.382x (1.243..1.402) | new-only | planned-faster |
| add | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.101808 | n/a | 0.108100 | n/a | 0.942x (0.930..0.983) | new-only | inconclusive |
| sub | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.080479 | n/a | 0.084975 | n/a | 0.939x (0.912..1.018) | new-only | inconclusive |
| mul | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.080229 | n/a | 0.085087 | n/a | 0.943x (0.894..0.967) | new-only | inconclusive |
| div | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.083900 | n/a | 0.084704 | n/a | 0.988x (0.916..0.998) | new-only | inconclusive |
| add | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.088292 | n/a | 0.071233 | n/a | 1.244x (1.144..1.269) | new-only | planned-faster |
| sub | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.088529 | n/a | 0.071275 | n/a | 1.252x (1.204..1.482) | new-only | planned-faster |
| mul | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.087996 | n/a | 0.070650 | n/a | 1.249x (1.219..1.260) | new-only | planned-faster |
| div | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.088846 | n/a | 0.071458 | n/a | 1.230x (1.185..1.278) | new-only | planned-faster |
| add | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.138396 | n/a | 0.071750 | n/a | 1.891x (1.815..2.128) | new-only | planned-faster |
| sub | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.131171 | n/a | 0.070400 | n/a | 1.862x (1.811..2.014) | new-only | planned-faster |
| mul | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.159042 | n/a | 0.085754 | n/a | 1.855x (1.795..2.043) | new-only | planned-faster |
| div | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.159658 | n/a | 0.085308 | n/a | 1.842x (1.784..1.890) | new-only | planned-faster |
| add | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.134021 | n/a | 0.141629 | n/a | 0.930x (0.869..0.999) | new-only | inconclusive |
| sub | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.147471 | n/a | 0.148775 | n/a | 0.925x (0.888..0.981) | new-only | inconclusive |
| mul | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.156033 | n/a | 0.183138 | n/a | 0.920x (0.840..1.159) | new-only | inconclusive |
| div | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.166183 | n/a | 0.169725 | n/a | 0.973x (0.934..1.117) | new-only | inconclusive |

### inplace-array

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.194036 | 0.187110 | 0.186422 | 1.000x (0.888..1.158) | 1.038x (0.973..1.093) | inconclusive | inconclusive |
| sub | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.195933 | 0.188552 | 0.192649 | 0.985x (0.902..1.093) | 1.012x (0.929..1.108) | inconclusive | inconclusive |
| mul | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.197467 | 0.187698 | 0.189832 | 0.982x (0.952..1.027) | 1.052x (1.001..1.095) | parity | inconclusive |
| div | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.205702 | 0.187533 | 0.191463 | 0.986x (0.933..1.024) | 1.072x (1.026..1.103) | inconclusive | inconclusive |
| add | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.405644 | n/a | 0.375298 | n/a | 1.083x (1.043..1.109) | legacy-incorrect | inconclusive |
| sub | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.393813 | n/a | 0.374663 | n/a | 1.050x (1.026..1.116) | legacy-incorrect | inconclusive |
| mul | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.392195 | n/a | 0.369729 | n/a | 1.043x (0.955..1.118) | legacy-incorrect | inconclusive |
| div | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.409077 | n/a | 0.403154 | n/a | 1.013x (0.903..1.042) | legacy-incorrect | inconclusive |
| add | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.425862 | n/a | 0.416194 | n/a | 1.022x (0.903..1.126) | legacy-incorrect | inconclusive |
| sub | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.386948 | n/a | 0.378810 | n/a | 1.026x (1.003..1.038) | legacy-incorrect | parity |
| mul | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.423438 | n/a | 0.434199 | n/a | 0.972x (0.894..1.070) | legacy-incorrect | inconclusive |
| div | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.512196 | n/a | 0.521093 | n/a | 1.071x (0.701..1.632) | legacy-incorrect | inconclusive |

### inplace-broadcast

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.281929 | n/a | 0.185988 | n/a | 1.564x (1.319..1.923) | new-only | planned-faster |
| sub | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.322801 | n/a | 0.199597 | n/a | 1.463x (1.289..1.731) | new-only | planned-faster |
| mul | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.317412 | n/a | 0.217538 | n/a | 1.459x (1.014..1.911) | new-only | inconclusive |
| div | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.306018 | n/a | 0.237397 | n/a | 1.298x (1.107..1.558) | new-only | planned-faster |
| add | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.608413 | n/a | 0.397121 | n/a | 1.531x (0.766..1.721) | new-only | inconclusive |
| sub | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.563416 | n/a | 0.420052 | n/a | 1.419x (1.137..1.769) | new-only | planned-faster |
| mul | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.616432 | n/a | 0.433829 | n/a | 1.418x (0.993..1.664) | new-only | inconclusive |
| div | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.669675 | n/a | 0.443332 | n/a | 1.363x (0.971..1.923) | new-only | inconclusive |

### inplace-scalar

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.150942 | 0.156027 | 0.152788 | 1.104x (0.729..1.239) | 0.979x (0.650..1.113) | inconclusive | inconclusive |
| sub | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.118198 | 0.119089 | 0.121810 | 0.967x (0.909..1.107) | 0.966x (0.878..1.044) | inconclusive | inconclusive |
| mul | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.119369 | 0.121793 | 0.121469 | 0.976x (0.875..1.070) | 0.990x (0.897..1.018) | inconclusive | inconclusive |
| div | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.185533 | 0.171569 | 0.169237 | 1.067x (0.933..1.163) | 1.096x (1.024..1.214) | inconclusive | inconclusive |
| add | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.123928 | 0.120801 | 0.120372 | 1.013x (0.967..1.068) | 1.039x (0.967..1.186) | inconclusive | inconclusive |
| sub | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.124744 | 0.123960 | 0.119388 | 1.024x (0.903..1.174) | 1.025x (0.891..1.128) | inconclusive | inconclusive |
| mul | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.156788 | 0.150759 | 0.149378 | 1.007x (0.886..1.086) | 1.029x (0.930..1.135) | inconclusive | inconclusive |
| div | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.142848 | 0.143385 | 0.134670 | 1.034x (0.957..1.120) | 1.064x (1.019..1.111) | inconclusive | inconclusive |
| add | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.389718 | n/a | 0.297800 | n/a | 1.204x (1.148..1.319) | legacy-incorrect | planned-faster |
| sub | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.398500 | n/a | 0.315046 | n/a | 1.180x (1.155..1.360) | legacy-incorrect | planned-faster |
| mul | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.432700 | n/a | 0.359501 | n/a | 1.182x (1.109..1.336) | legacy-incorrect | planned-faster |
| div | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.352382 | n/a | 0.326081 | n/a | 1.122x (0.994..1.251) | legacy-incorrect | inconclusive |

### reduction-axis

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| mean | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.174079 | 7.303737 | 0.377492 | 19.737x (18.113..20.290) | 0.464x (0.446..0.506) | improved | numpy-faster |
| var | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.625546 | 6.599450 | 0.652575 | 10.188x (9.372..10.378) | 0.961x (0.911..1.033) | improved | inconclusive |
| std | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.636567 | 6.671337 | 0.648625 | 10.378x (9.186..10.597) | 0.986x (0.925..1.048) | improved | inconclusive |
| median | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 6.276675 | 6.637642 | 2.659037 | 2.247x (2.204..2.559) | 2.138x (2.094..2.447) | improved | planned-faster |
| average | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.389258 | 6.238438 | 0.343583 | 18.150x (16.887..19.985) | 1.181x (1.089..1.416) | improved | planned-faster |
| mean | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.155417 | 7.391904 | 0.204871 | 35.429x (32.466..42.298) | 0.748x (0.715..0.801) | improved | numpy-faster |
| var | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.633183 | 6.903350 | 0.325683 | 21.074x (19.096..22.482) | 1.949x (1.764..2.047) | improved | planned-faster |
| std | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.806342 | 8.381317 | 0.486492 | 20.686x (18.263..22.092) | 1.968x (1.771..2.087) | improved | planned-faster |
| median | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 9.782271 | 6.878029 | 3.309512 | 2.079x (2.033..2.376) | 3.049x (2.974..3.353) | improved | planned-faster |
| average | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.446150 | 7.094796 | 0.189787 | 36.660x (34.305..38.952) | 2.351x (2.162..2.526) | improved | planned-faster |
| mean | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 0.154171 | 6.062992 | 0.306450 | 20.065x (18.711..21.266) | 0.499x (0.473..0.551) | improved | numpy-faster |
| var | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 0.904550 | 6.604846 | 0.644304 | 10.231x (9.967..10.431) | 1.402x (1.368..1.492) | improved | planned-faster |
| std | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 1.138367 | 8.154729 | 0.669479 | 10.337x (9.872..15.919) | 1.447x (1.364..2.405) | improved | planned-faster |
| median | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 6.535671 | 6.905850 | 3.167079 | 2.129x (1.869..2.469) | 2.096x (1.792..2.330) | improved | planned-faster |
| average | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.811588 | 6.756104 | 1.008208 | 7.233x (6.572..8.784) | 0.793x (0.771..0.931) | improved | numpy-faster |
| mean | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.169146 | 6.526008 | 0.657054 | 9.667x (8.611..10.583) | 0.270x (0.234..0.281) | improved | numpy-faster |
| var | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.821483 | 6.617125 | 1.464892 | 4.503x (4.171..4.613) | 0.543x (0.501..0.589) | improved | numpy-faster |
| std | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.803863 | 6.663696 | 1.473225 | 4.540x (4.501..4.918) | 0.546x (0.530..0.577) | improved | numpy-faster |
| median | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 6.706492 | 6.850054 | 3.191621 | 2.147x (2.031..2.377) | 2.107x (1.974..2.376) | improved | planned-faster |
| average | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.956587 | 8.959600 | 0.906046 | 9.965x (7.213..13.832) | 1.171x (0.714..1.511) | improved | inconclusive |
| mean | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.179733 | 6.710333 | 0.369667 | 19.360x (16.816..20.512) | 0.504x (0.475..0.574) | improved | numpy-faster |
| var | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.780312 | 6.642992 | 0.672463 | 10.068x (9.317..10.547) | 1.180x (1.103..1.235) | improved | planned-faster |
| std | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.831746 | 6.943350 | 0.687129 | 10.241x (9.529..10.495) | 1.197x (1.106..1.284) | improved | planned-faster |
| median | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 6.615558 | 6.862788 | 3.072188 | 2.291x (2.161..2.591) | 2.161x (1.842..2.382) | improved | planned-faster |
| average | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.623579 | 7.014096 | 0.342042 | 18.174x (16.159..21.441) | 1.644x (1.531..1.912) | improved | planned-faster |

### reduction-full

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| mean | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.114475 | 0.544579 | 0.307742 | 1.990x (1.920..2.015) | 0.421x (0.401..0.458) | improved | numpy-faster |
| var | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.508217 | 5.031933 | 0.629167 | 7.745x (7.265..8.020) | 0.807x (0.743..0.872) | improved | numpy-faster |
| std | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.518825 | 4.797963 | 0.627775 | 7.658x (7.292..7.754) | 0.826x (0.796..0.845) | improved | numpy-faster |
| median | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 5.254967 | 5.650638 | 2.891688 | 1.962x (1.825..2.084) | 1.817x (1.750..2.011) | improved | planned-faster |
| average | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.440025 | 13.421708 | 0.631892 | 19.278x (18.923..21.415) | 0.691x (0.658..0.733) | improved | numpy-faster |
| mean | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.132663 | 0.622096 | 0.311708 | 2.004x (1.830..2.037) | 0.423x (0.410..0.446) | improved | numpy-faster |
| var | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.537525 | 4.820892 | 0.639892 | 7.780x (7.553..8.521) | 0.829x (0.778..0.891) | improved | numpy-faster |
| std | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.521058 | 5.284662 | 0.640962 | 7.824x (7.087..8.464) | 0.800x (0.760..0.842) | improved | numpy-faster |
| median | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 5.851225 | 5.680496 | 3.130000 | 1.827x (1.794..1.977) | 1.868x (1.817..1.971) | improved | planned-faster |
| average | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 1.216829 | 14.431583 | 1.292496 | 11.231x (9.928..12.442) | 0.912x (0.840..1.114) | improved | inconclusive |
| mean | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.131871 | 0.627175 | 0.309767 | 1.997x (1.920..2.048) | 0.422x (0.399..0.434) | improved | numpy-faster |
| var | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.687000 | 4.804842 | 0.620425 | 7.751x (7.374..8.173) | 1.091x (1.037..1.220) | improved | inconclusive |
| std | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.720896 | 4.860917 | 0.634671 | 7.745x (7.055..7.796) | 1.082x (1.064..1.140) | improved | planned-faster |
| median | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 6.892250 | 5.603150 | 2.886829 | 1.936x (1.900..2.173) | 2.395x (2.359..2.543) | improved | planned-faster |
| average | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.581971 | 13.486733 | 1.341325 | 10.201x (9.790..12.090) | 0.484x (0.432..0.504) | improved | numpy-faster |
| mean | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.140067 | 0.541975 | 0.542063 | 0.997x (0.973..1.027) | 0.261x (0.250..0.269) | parity | numpy-faster |
| var | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.697667 | 4.795138 | 1.457558 | 3.275x (3.199..3.316) | 0.478x (0.468..0.513) | improved | numpy-faster |
| std | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.752371 | 4.815217 | 1.472325 | 3.290x (3.040..3.516) | 0.487x (0.475..0.543) | improved | numpy-faster |
| median | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 5.455933 | 5.686883 | 2.976171 | 1.924x (1.827..2.410) | 1.841x (1.707..2.045) | improved | planned-faster |
| average | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.761704 | 16.633417 | 1.373404 | 10.509x (9.604..25.822) | 0.509x (0.466..1.026) | improved | inconclusive |

### matmul

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matmul | matrix-matrix-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 0.237738 | 37.639821 | 0.235137 | 160.076x (147.203..176.839) | 1.003x (0.921..1.051) | improved | inconclusive |
| matmul | matrix-matrix-f | lhs: shape=(256,256), strides=(1,256), flags=F<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 0.239827 | 38.806226 | 0.236554 | 164.247x (153.735..176.535) | 1.008x (0.976..1.112) | improved | inconclusive |
| matmul | matrix-matrix-negative | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(256,256), strides=(-256,-1), flags=N | 7 | 45.318762 | 36.967232 | 0.357559 | 107.055x (95.321..121.433) | 126.772x (117.620..140.810) | improved | planned-faster |
| matmul | matrix-matrix-lhs-step2 | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 45.135101 | 38.076583 | 0.301304 | 122.350x (66.208..138.315) | 145.662x (83.496..152.070) | improved | planned-faster |
| matmul | matrix-matrix-rhs-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 45.679155 | 41.634286 | 0.304542 | 131.341x (71.860..155.119) | 144.652x (71.753..178.559) | improved | planned-faster |
| matmul | matrix-matrix-both-step2 | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 44.494732 | 40.463000 | 0.364387 | 110.921x (89.785..126.274) | 123.104x (99.004..139.352) | improved | planned-faster |
| matmul | matrix-matrix-mixed-c-f | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 0.201958 | 16.672875 | 0.198833 | 84.200x (77.585..85.621) | 1.012x (0.916..1.051) | improved | inconclusive |
| matmul | matrix-matrix-rectangular | lhs: shape=(128,256), strides=(256,1), flags=C<br>rhs: shape=(256,64), strides=(64,1), flags=C | 40 | 0.028389 | 3.521154 | 0.029009 | 122.080x (112.817..126.610) | 1.002x (0.894..1.019) | improved | inconclusive |
| matmul | matrix-matrix-small-direct | lhs: shape=(8,16), strides=(16,1), flags=C<br>rhs: shape=(16,8), strides=(8,1), flags=C | 2000 | 0.001543 | 0.001997 | 0.002020 | 0.995x (0.962..1.032) | 0.739x (0.728..0.806) | parity | numpy-faster |
| matmul | vector-vector | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256), strides=(1), flags=CF | 2000 | 0.001354 | 0.001764 | 0.001781 | 0.994x (0.945..1.058) | 0.764x (0.724..0.804) | inconclusive | numpy-faster |
| matmul | vector-matrix | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(256,1), flags=C | 200 | 0.022133 | 0.151721 | 0.022731 | 6.674x (6.199..6.787) | 0.979x (0.926..1.043) | improved | inconclusive |
| matmul | matrix-vector | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.010388 | 0.085879 | 0.010225 | 8.739x (8.432..9.250) | 1.021x (1.004..1.201) | improved | inconclusive |

### matmul-batch

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matmul | batch-same-shape-c | lhs: shape=(8,32,32), strides=(1024,32,1), flags=C<br>rhs: shape=(8,32,32), strides=(1024,32,1), flags=C | 3 | 0.009153 | n/a | 0.009014 | n/a | 1.036x (1.001..1.075) | new-only | inconclusive |
| matmul | batch-broadcast-c | lhs: shape=(8,1,32,32), strides=(1024,1024,32,1), flags=C<br>rhs: shape=(1,8,32,32), strides=(8192,1024,32,1), flags=C | 3 | 0.053986 | n/a | 0.054181 | n/a | 0.998x (0.992..1.139) | new-only | inconclusive |
| matmul | batch-broadcast-negative-matrix | lhs: shape=(8,1,32,32), strides=(1024,1024,-32,-1), flags=N<br>rhs: shape=(1,8,32,32), strides=(8192,1024,-32,-1), flags=N | 3 | 2.423639 | n/a | 0.078972 | n/a | 29.989x (27.735..31.525) | new-only | planned-faster |
| matmul | batch-broadcast-step2-inner | lhs: shape=(8,1,32,32), strides=(2048,2048,64,2), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,64,2), flags=S | 3 | 2.470194 | n/a | 0.082181 | n/a | 30.139x (28.212..32.093) | new-only | planned-faster |
| matmul | batch-broadcast-step2-batch | lhs: shape=(8,1,32,32), strides=(2048,1024,32,1), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,32,1), flags=S | 3 | 0.063430 | n/a | 0.062889 | n/a | 1.008x (0.985..1.026) | new-only | parity |

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
