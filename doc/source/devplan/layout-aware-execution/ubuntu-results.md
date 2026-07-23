# Linux detailed execution benchmark

## Recorded environment

- Code revision: `d320b553baf5fd28cdd36794e3f6d45acd188489`.
- Dirty tree: `false`.
- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`.
- Machine: `x86_64`.
- Python: `3.12.7`.
- NumPy: `2.3.0`.
- Seed: `20260722`.
- Fixed cases: `154`.
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
    --repeat 15 \
    --warmup 5 --cpu 0 \
    --output /tmp/solvcon-execution-d320b553.json
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
| inplace-array | 12 | 0 | 2 | 0 | 8 | 0 | 2 |
| inplace-broadcast | 8 | 0 | 0 | 0 | 0 | 8 | 0 |
| inplace-scalar | 12 | 0 | 0 | 0 | 4 | 0 | 8 |
| reduction-axis | 25 | 25 | 0 | 0 | 0 | 0 | 0 |
| reduction-full | 20 | 16 | 0 | 0 | 0 | 0 | 4 |
| matmul | 12 | 10 | 0 | 0 | 0 | 0 | 2 |
| matmul-batch | 5 | 0 | 0 | 0 | 0 | 5 | 0 |

## Complete results

### elementwise-array

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 1.955768 | 5.459056 | 2.081064 | 2.391x (1.641..3.352) | 1.103x (0.708..1.645) | improved | inconclusive |
| sub | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 2.822099 | 4.262467 | 2.168107 | 2.271x (1.251..2.899) | 1.067x (0.781..1.918) | improved | inconclusive |
| mul | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 2.588673 | 4.095505 | 2.220977 | 1.982x (1.435..2.439) | 1.084x (0.679..1.575) | improved | inconclusive |
| div | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 1.412653 | 3.482569 | 1.361254 | 2.118x (1.641..3.073) | 0.980x (0.649..1.106) | improved | inconclusive |
| add | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 0.665728 | 1.534227 | 0.719122 | 2.138x (1.775..2.576) | 0.986x (0.809..1.190) | improved | inconclusive |
| sub | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 0.543914 | 1.347409 | 0.583435 | 2.302x (1.857..2.551) | 0.976x (0.816..1.119) | improved | inconclusive |
| mul | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 0.683493 | 1.518835 | 0.688398 | 2.197x (1.608..2.607) | 1.007x (0.700..1.313) | improved | inconclusive |
| div | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 0.580274 | 1.435308 | 0.637995 | 2.319x (1.807..2.767) | 0.899x (0.697..1.106) | improved | inconclusive |
| add | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 0.607564 | 1.259789 | 0.550187 | 2.245x (2.057..2.859) | 1.151x (0.926..1.318) | improved | inconclusive |
| sub | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 0.728267 | 1.436371 | 0.615433 | 2.303x (1.952..2.718) | 1.203x (1.000..1.349) | improved | inconclusive |
| mul | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 0.710598 | 1.414697 | 0.589199 | 2.352x (1.928..2.901) | 1.273x (0.913..1.378) | improved | inconclusive |
| div | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 0.650601 | 1.441243 | 0.625723 | 2.298x (2.059..2.520) | 1.122x (0.871..1.288) | improved | inconclusive |
| add | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.788683 | n/a | 1.796626 | n/a | 1.008x (0.938..1.076) | legacy-incorrect | inconclusive |
| sub | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.689106 | n/a | 1.569156 | n/a | 1.000x (0.905..1.209) | legacy-incorrect | inconclusive |
| mul | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.629469 | n/a | 1.643601 | n/a | 0.982x (0.842..1.081) | legacy-incorrect | inconclusive |
| div | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.439330 | n/a | 1.305696 | n/a | 1.069x (0.933..1.173) | legacy-incorrect | inconclusive |
| add | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.365177 | n/a | 1.372000 | n/a | 0.987x (0.849..1.138) | legacy-incorrect | inconclusive |
| sub | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.176391 | n/a | 1.371590 | n/a | 0.962x (0.801..1.070) | legacy-incorrect | inconclusive |
| mul | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.152988 | n/a | 1.098520 | n/a | 0.976x (0.858..1.131) | legacy-incorrect | inconclusive |
| div | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 1.186926 | n/a | 1.244536 | n/a | 0.969x (0.724..1.198) | legacy-incorrect | inconclusive |

### elementwise-scalar

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.308843 | 0.942496 | 0.306994 | 3.175x (2.779..3.408) | 0.981x (0.821..1.212) | improved | inconclusive |
| sub | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.323765 | 1.019823 | 0.304767 | 3.266x (2.707..3.543) | 1.038x (0.856..1.230) | improved | inconclusive |
| mul | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.311764 | 0.937739 | 0.301370 | 3.287x (2.858..3.466) | 1.043x (0.832..1.176) | improved | inconclusive |
| div | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.401957 | 1.288194 | 0.419308 | 3.035x (2.652..3.778) | 0.991x (0.866..1.057) | improved | inconclusive |
| add | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.297846 | 1.197450 | 0.324728 | 3.675x (2.703..4.400) | 0.977x (0.770..1.284) | improved | inconclusive |
| sub | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.279420 | 0.930011 | 0.302997 | 3.109x (2.794..3.683) | 0.956x (0.806..1.171) | improved | inconclusive |
| mul | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.305214 | 1.029213 | 0.302057 | 3.323x (2.747..3.795) | 0.970x (0.863..1.176) | improved | inconclusive |
| div | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.412235 | 1.230573 | 0.407665 | 3.025x (2.801..3.345) | 1.005x (0.918..1.078) | improved | inconclusive |
| add | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.376681 | 0.920978 | 0.307078 | 3.059x (2.611..3.447) | 1.279x (0.899..1.582) | improved | inconclusive |
| sub | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.352671 | 0.985117 | 0.318014 | 3.121x (2.679..3.552) | 1.122x (0.866..1.364) | improved | inconclusive |
| mul | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.438406 | 1.146754 | 0.356920 | 3.398x (2.657..4.063) | 1.230x (0.995..1.453) | improved | inconclusive |
| div | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.454059 | 1.219881 | 0.422410 | 2.962x (2.767..4.022) | 1.158x (1.006..1.812) | improved | inconclusive |
| add | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 0.790613 | n/a | 0.700949 | n/a | 1.046x (0.978..1.225) | legacy-incorrect | inconclusive |
| sub | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 0.687091 | n/a | 0.622099 | n/a | 1.045x (0.821..1.237) | legacy-incorrect | inconclusive |
| mul | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 1.049952 | n/a | 0.935979 | n/a | 1.097x (0.956..1.527) | legacy-incorrect | inconclusive |
| div | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 0.852148 | n/a | 0.768303 | n/a | 1.131x (0.918..1.269) | legacy-incorrect | inconclusive |

### elementwise-broadcast

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.016959 | n/a | 0.012220 | n/a | 1.478x (1.294..1.589) | new-only | planned-faster |
| sub | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.024731 | n/a | 0.010511 | n/a | 2.476x (2.080..2.597) | new-only | planned-faster |
| mul | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.027338 | n/a | 0.012082 | n/a | 2.278x (2.026..2.596) | new-only | planned-faster |
| div | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.030007 | n/a | 0.024296 | n/a | 1.244x (1.168..1.296) | new-only | planned-faster |
| add | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.023376 | n/a | 0.010809 | n/a | 2.153x (1.920..2.563) | new-only | planned-faster |
| sub | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.031696 | n/a | 0.011671 | n/a | 2.718x (2.157..3.327) | new-only | planned-faster |
| mul | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.032523 | n/a | 0.013217 | n/a | 2.460x (2.180..2.739) | new-only | planned-faster |
| div | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.036667 | n/a | 0.024990 | n/a | 1.479x (1.396..1.528) | new-only | planned-faster |
| add | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.029666 | n/a | 0.019154 | n/a | 1.545x (1.457..1.579) | new-only | planned-faster |
| sub | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.038052 | n/a | 0.019612 | n/a | 1.921x (1.707..2.017) | new-only | planned-faster |
| mul | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.037710 | n/a | 0.019391 | n/a | 1.920x (1.831..2.094) | new-only | planned-faster |
| div | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.042814 | n/a | 0.023753 | n/a | 1.782x (1.687..1.896) | new-only | planned-faster |
| add | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.028126 | n/a | 0.020730 | n/a | 1.337x (1.315..1.489) | new-only | planned-faster |
| sub | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.028222 | n/a | 0.020843 | n/a | 1.342x (1.191..1.461) | new-only | planned-faster |
| mul | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.027284 | n/a | 0.020508 | n/a | 1.353x (1.256..1.417) | new-only | planned-faster |
| div | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.030567 | n/a | 0.024181 | n/a | 1.275x (1.229..1.489) | new-only | planned-faster |
| add | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.101821 | n/a | 0.019909 | n/a | 5.064x (4.715..5.351) | new-only | planned-faster |
| sub | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.104307 | n/a | 0.020221 | n/a | 5.143x (4.564..5.311) | new-only | planned-faster |
| mul | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.101644 | n/a | 0.019527 | n/a | 5.258x (4.961..5.318) | new-only | planned-faster |
| div | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.112661 | n/a | 0.024807 | n/a | 4.497x (4.186..4.843) | new-only | planned-faster |
| add | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.072664 | n/a | 0.038106 | n/a | 1.909x (1.861..1.951) | new-only | planned-faster |
| sub | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.057406 | n/a | 0.038514 | n/a | 1.507x (1.363..1.556) | new-only | planned-faster |
| mul | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.074538 | n/a | 0.038862 | n/a | 1.914x (1.657..2.057) | new-only | planned-faster |
| div | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.084399 | n/a | 0.047494 | n/a | 1.786x (1.674..1.828) | new-only | planned-faster |

### inplace-array

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.059948 | 0.060857 | 0.067618 | 0.909x (0.866..0.985) | 0.884x (0.834..0.930) | inconclusive | numpy-faster |
| sub | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.060903 | 0.057675 | 0.056919 | 1.029x (0.960..1.049) | 1.058x (0.982..1.181) | parity | inconclusive |
| mul | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.060702 | 0.057278 | 0.057196 | 0.998x (0.945..1.070) | 1.036x (0.997..1.125) | inconclusive | inconclusive |
| div | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.095154 | 0.094215 | 0.094893 | 0.997x (0.968..1.013) | 0.995x (0.969..1.037) | parity | parity |
| add | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.074946 | n/a | 0.082090 | n/a | 0.907x (0.826..1.051) | legacy-incorrect | inconclusive |
| sub | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.075030 | n/a | 0.077158 | n/a | 0.971x (0.826..1.033) | legacy-incorrect | inconclusive |
| mul | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.076130 | n/a | 0.076858 | n/a | 0.982x (0.937..1.059) | legacy-incorrect | inconclusive |
| div | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.196490 | n/a | 0.190879 | n/a | 1.027x (0.988..1.097) | legacy-incorrect | inconclusive |
| add | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.105600 | n/a | 0.105287 | n/a | 0.998x (0.908..1.142) | legacy-incorrect | inconclusive |
| sub | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.102540 | n/a | 0.100854 | n/a | 1.029x (0.958..1.119) | legacy-incorrect | inconclusive |
| mul | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.102319 | n/a | 0.101635 | n/a | 1.027x (0.958..1.081) | legacy-incorrect | inconclusive |
| div | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.192935 | n/a | 0.191902 | n/a | 1.016x (0.982..1.035) | legacy-incorrect | parity |

### inplace-broadcast

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.066854 | n/a | 0.040759 | n/a | 1.657x (1.537..1.904) | new-only | planned-faster |
| sub | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.073081 | n/a | 0.043083 | n/a | 1.747x (1.584..1.885) | new-only | planned-faster |
| mul | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.072103 | n/a | 0.042836 | n/a | 1.719x (1.565..1.846) | new-only | planned-faster |
| div | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.124307 | n/a | 0.097067 | n/a | 1.267x (1.190..1.351) | new-only | planned-faster |
| add | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.127035 | n/a | 0.090706 | n/a | 1.397x (1.345..1.528) | new-only | planned-faster |
| sub | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.121694 | n/a | 0.090812 | n/a | 1.371x (1.307..1.445) | new-only | planned-faster |
| mul | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.125403 | n/a | 0.089826 | n/a | 1.373x (1.265..1.489) | new-only | planned-faster |
| div | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.226095 | n/a | 0.193221 | n/a | 1.185x (1.060..1.231) | new-only | planned-faster |

### inplace-scalar

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.031811 | 0.026596 | 0.028028 | 0.970x (0.870..1.258) | 1.191x (0.937..1.404) | inconclusive | inconclusive |
| sub | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.029556 | 0.025900 | 0.026684 | 1.025x (0.896..1.194) | 1.121x (0.983..1.260) | inconclusive | inconclusive |
| mul | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.029561 | 0.028214 | 0.027267 | 1.006x (0.938..1.141) | 1.110x (1.024..1.187) | inconclusive | inconclusive |
| div | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.097519 | 0.096934 | 0.096468 | 1.004x (0.931..1.101) | 1.005x (0.957..1.077) | inconclusive | inconclusive |
| add | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.032160 | 0.027136 | 0.027534 | 0.971x (0.915..1.214) | 1.196x (1.087..1.279) | inconclusive | planned-faster |
| sub | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.030994 | 0.027174 | 0.027857 | 0.951x (0.903..1.063) | 1.113x (1.033..1.280) | inconclusive | inconclusive |
| mul | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.031842 | 0.027208 | 0.026928 | 0.958x (0.817..1.133) | 1.134x (0.909..1.247) | inconclusive | inconclusive |
| div | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.101046 | 0.097869 | 0.097017 | 0.990x (0.942..1.084) | 1.048x (0.921..1.109) | inconclusive | inconclusive |
| add | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.091642 | n/a | 0.085326 | n/a | 1.071x (0.972..1.146) | legacy-incorrect | inconclusive |
| sub | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.090581 | n/a | 0.083586 | n/a | 1.072x (0.979..1.147) | legacy-incorrect | inconclusive |
| mul | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.090165 | n/a | 0.084328 | n/a | 1.089x (0.992..1.224) | legacy-incorrect | inconclusive |
| div | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.191373 | n/a | 0.189419 | n/a | 1.005x (0.980..1.039) | legacy-incorrect | parity |

### reduction-axis

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| mean | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.061373 | 1.270467 | 0.098283 | 13.135x (12.105..15.440) | 0.595x (0.492..0.668) | improved | numpy-faster |
| var | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.322527 | 1.392108 | 0.208209 | 6.666x (6.352..8.236) | 1.545x (1.401..1.877) | improved | planned-faster |
| std | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.306396 | 1.349988 | 0.204079 | 6.648x (6.274..6.913) | 1.510x (1.395..1.684) | improved | planned-faster |
| median | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 5.209471 | 1.630395 | 0.755517 | 2.168x (2.028..2.215) | 6.836x (6.546..7.309) | improved | planned-faster |
| average | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.207231 | 1.253856 | 0.102467 | 12.273x (11.293..12.647) | 2.007x (1.836..2.409) | improved | planned-faster |
| mean | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.054846 | 1.415267 | 0.377813 | 3.778x (3.559..3.878) | 0.146x (0.137..0.159) | improved | numpy-faster |
| var | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.278749 | 1.493401 | 0.746637 | 1.980x (1.905..2.158) | 0.379x (0.328..0.441) | improved | numpy-faster |
| std | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.284265 | 1.495646 | 0.762783 | 1.972x (1.891..2.075) | 0.376x (0.352..0.428) | improved | numpy-faster |
| median | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 7.205831 | 1.755439 | 1.110739 | 1.581x (1.525..1.673) | 6.392x (6.222..6.914) | improved | planned-faster |
| average | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.235469 | 1.380752 | 0.375020 | 3.601x (3.415..3.809) | 0.630x (0.569..0.683) | improved | numpy-faster |
| mean | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 0.060320 | 1.261221 | 0.095667 | 13.183x (12.626..13.772) | 0.629x (0.542..0.690) | improved | numpy-faster |
| var | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 0.685108 | 1.389260 | 0.211053 | 6.642x (6.327..7.063) | 3.243x (3.036..3.589) | improved | planned-faster |
| std | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 0.680922 | 1.340107 | 0.208083 | 6.429x (6.065..6.824) | 3.263x (2.908..3.491) | improved | planned-faster |
| median | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 5.490414 | 1.625972 | 0.880249 | 1.859x (1.810..1.910) | 6.264x (6.153..6.461) | improved | planned-faster |
| average | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.539539 | 1.240647 | 0.102654 | 12.136x (11.298..12.493) | 5.276x (4.832..5.627) | improved | planned-faster |
| mean | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.071074 | 1.274419 | 0.097249 | 13.139x (12.798..13.484) | 0.731x (0.626..0.799) | improved | numpy-faster |
| var | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.333964 | 1.365456 | 0.202632 | 6.751x (6.404..6.820) | 1.648x (1.512..1.724) | improved | planned-faster |
| std | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.328789 | 1.335456 | 0.208724 | 6.451x (6.139..6.881) | 1.577x (1.465..1.812) | improved | planned-faster |
| median | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 5.516480 | 1.617897 | 0.887924 | 1.823x (1.729..1.874) | 6.137x (5.912..6.413) | improved | planned-faster |
| average | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.211213 | 1.249528 | 0.098633 | 12.704x (12.231..12.955) | 2.118x (1.825..2.266) | improved | planned-faster |
| mean | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.062914 | 1.258787 | 0.100037 | 12.563x (12.190..12.935) | 0.628x (0.548..0.691) | improved | numpy-faster |
| var | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.360244 | 1.380833 | 0.207992 | 6.699x (6.341..6.890) | 1.727x (1.668..1.970) | improved | planned-faster |
| std | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.359759 | 1.340413 | 0.207498 | 6.477x (6.268..6.685) | 1.717x (1.625..1.803) | improved | planned-faster |
| median | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 5.186597 | 1.617687 | 0.756017 | 2.140x (2.086..2.211) | 6.894x (6.632..7.137) | improved | planned-faster |
| average | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.262106 | 1.270227 | 0.107552 | 11.866x (10.912..12.127) | 2.371x (2.167..2.755) | improved | planned-faster |

### reduction-full

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| mean | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.039395 | 0.094597 | 0.093558 | 1.002x (0.977..1.057) | 0.417x (0.389..0.451) | inconclusive | numpy-faster |
| var | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.206644 | 1.030193 | 0.202018 | 5.154x (4.819..5.462) | 1.022x (0.961..1.193) | improved | inconclusive |
| std | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.220912 | 1.039649 | 0.200948 | 5.171x (4.803..5.353) | 1.072x (0.965..1.291) | improved | inconclusive |
| median | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 4.103332 | 1.319451 | 0.665053 | 1.998x (1.871..2.059) | 6.228x (5.911..6.331) | improved | planned-faster |
| average | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.201218 | 3.470124 | 0.194585 | 17.877x (16.414..18.450) | 1.025x (0.947..1.167) | improved | inconclusive |
| mean | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.039491 | 0.094441 | 0.095179 | 0.998x (0.936..1.095) | 0.418x (0.398..0.443) | inconclusive | numpy-faster |
| var | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.223583 | 1.068258 | 0.205083 | 5.231x (4.903..5.567) | 1.048x (0.983..1.305) | improved | inconclusive |
| std | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.213053 | 1.076715 | 0.200262 | 5.550x (5.191..5.673) | 1.074x (0.945..1.219) | improved | inconclusive |
| median | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 4.450365 | 1.373968 | 0.901737 | 1.532x (1.454..1.556) | 4.958x (4.743..5.030) | improved | planned-faster |
| average | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.553045 | 3.512737 | 0.260173 | 13.593x (12.724..14.475) | 2.141x (2.040..2.259) | improved | planned-faster |
| mean | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.039018 | 0.093149 | 0.093847 | 0.991x (0.958..1.067) | 0.409x (0.382..0.451) | inconclusive | numpy-faster |
| var | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.223297 | 1.037070 | 0.200787 | 5.179x (5.072..5.575) | 1.120x (1.039..1.291) | improved | inconclusive |
| std | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.230034 | 1.037814 | 0.201112 | 5.172x (4.846..5.487) | 1.164x (1.087..1.500) | improved | planned-faster |
| median | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 5.681821 | 1.285430 | 0.762410 | 1.710x (1.611..1.737) | 7.457x (7.148..7.719) | improved | planned-faster |
| average | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.196941 | 3.434448 | 0.190015 | 17.999x (16.178..18.311) | 1.038x (0.873..1.200) | improved | inconclusive |
| mean | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.051689 | 0.094039 | 0.095868 | 1.001x (0.886..1.063) | 0.526x (0.512..0.578) | inconclusive | numpy-faster |
| var | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.254544 | 1.038933 | 0.197317 | 5.277x (4.663..5.497) | 1.287x (1.227..1.349) | improved | planned-faster |
| std | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.265705 | 1.046352 | 0.199002 | 5.288x (5.077..5.572) | 1.363x (1.223..1.441) | improved | planned-faster |
| median | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 4.375871 | 1.322828 | 0.788195 | 1.674x (1.587..1.741) | 5.479x (5.375..5.727) | improved | planned-faster |
| average | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.257590 | 3.531525 | 0.197111 | 18.245x (15.140..19.137) | 1.301x (1.073..1.507) | improved | planned-faster |

### matmul

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matmul | matrix-matrix-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 24.716821 | 6.720031 | 0.546944 | 12.416x (11.847..13.347) | 45.185x (43.994..46.769) | improved | planned-faster |
| matmul | matrix-matrix-f | lhs: shape=(256,256), strides=(1,256), flags=F<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 24.746088 | 6.939062 | 0.510198 | 13.678x (12.783..14.594) | 48.453x (45.182..49.681) | improved | planned-faster |
| matmul | matrix-matrix-negative | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(256,256), strides=(-256,-1), flags=N | 7 | 24.676722 | 6.674730 | 0.599583 | 11.172x (10.514..11.396) | 41.498x (39.264..42.246) | improved | planned-faster |
| matmul | matrix-matrix-lhs-step2 | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 24.677515 | 6.698905 | 0.592291 | 11.259x (9.856..11.808) | 42.014x (36.315..43.969) | improved | planned-faster |
| matmul | matrix-matrix-rhs-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 26.224469 | 7.187791 | 0.576478 | 12.437x (11.828..13.032) | 45.432x (43.650..46.271) | improved | planned-faster |
| matmul | matrix-matrix-both-step2 | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 26.316813 | 7.253948 | 0.623199 | 11.786x (11.001..12.170) | 42.042x (40.159..43.980) | improved | planned-faster |
| matmul | matrix-matrix-mixed-c-f | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 23.975382 | 4.647705 | 0.547892 | 8.361x (7.696..8.837) | 44.535x (40.776..45.291) | improved | planned-faster |
| matmul | matrix-matrix-rectangular | lhs: shape=(128,256), strides=(256,1), flags=C<br>rhs: shape=(256,64), strides=(64,1), flags=C | 40 | 3.039231 | 0.819598 | 0.096705 | 8.429x (8.153..8.714) | 31.450x (30.009..32.162) | improved | planned-faster |
| matmul | matrix-matrix-small-direct | lhs: shape=(8,16), strides=(16,1), flags=C<br>rhs: shape=(16,8), strides=(8,1), flags=C | 2000 | 0.002713 | 0.000693 | 0.000644 | 1.078x (1.010..1.308) | 4.233x (4.095..4.366) | inconclusive | planned-faster |
| matmul | vector-vector | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256), strides=(1), flags=CF | 2000 | 0.001283 | 0.000416 | 0.000424 | 0.981x (0.947..0.997) | 3.024x (2.969..3.095) | inconclusive | planned-faster |
| matmul | vector-matrix | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(256,1), flags=C | 200 | 0.099467 | 0.026934 | 0.004124 | 6.621x (5.902..6.881) | 24.212x (21.422..24.891) | improved | planned-faster |
| matmul | matrix-vector | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.094447 | 0.018184 | 0.005079 | 3.670x (3.244..3.751) | 18.674x (16.826..19.765) | improved | planned-faster |

### matmul-batch

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matmul | batch-same-shape-c | lhs: shape=(8,32,32), strides=(1024,32,1), flags=C<br>rhs: shape=(8,32,32), strides=(1024,32,1), flags=C | 3 | 0.411749 | n/a | 0.020837 | n/a | 19.003x (15.231..21.494) | new-only | planned-faster |
| matmul | batch-broadcast-c | lhs: shape=(8,1,32,32), strides=(1024,1024,32,1), flags=C<br>rhs: shape=(1,8,32,32), strides=(8192,1024,32,1), flags=C | 3 | 3.162264 | n/a | 0.167984 | n/a | 19.230x (17.659..21.564) | new-only | planned-faster |
| matmul | batch-broadcast-negative-matrix | lhs: shape=(8,1,32,32), strides=(1024,1024,-32,-1), flags=N<br>rhs: shape=(1,8,32,32), strides=(8192,1024,-32,-1), flags=N | 3 | 3.049367 | n/a | 0.545922 | n/a | 5.588x (5.522..5.682) | new-only | planned-faster |
| matmul | batch-broadcast-step2-inner | lhs: shape=(8,1,32,32), strides=(2048,2048,64,2), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,64,2), flags=S | 3 | 3.078624 | n/a | 0.549166 | n/a | 5.609x (5.401..5.753) | new-only | planned-faster |
| matmul | batch-broadcast-step2-batch | lhs: shape=(8,1,32,32), strides=(2048,1024,32,1), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,32,1), flags=S | 3 | 3.124702 | n/a | 0.166739 | n/a | 18.239x (17.261..22.266) | new-only | planned-faster |

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
