# macOS detailed execution benchmark

## Recorded environment

- Code revision: `0db69f45b0184fc6eb7b64df14b5f026905c2447`.
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
    --output /tmp/solvcon-execution-0db69f45.json
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
| inplace-array | 12 | 0 | 0 | 0 | 8 | 0 | 4 |
| inplace-broadcast | 8 | 0 | 0 | 0 | 0 | 8 | 0 |
| inplace-scalar | 12 | 0 | 1 | 0 | 4 | 0 | 7 |
| reduction-axis | 25 | 25 | 0 | 0 | 0 | 0 | 0 |
| reduction-full | 20 | 16 | 4 | 0 | 0 | 0 | 0 |
| matmul | 12 | 10 | 2 | 0 | 0 | 0 | 0 |
| matmul-batch | 5 | 0 | 0 | 0 | 0 | 5 | 0 |

## Complete results

### elementwise-array

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 0.471975 | 0.951040 | 0.467329 | 2.027x (1.915..2.406) | 1.006x (0.927..1.044) | improved | inconclusive |
| sub | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 0.470771 | 0.943985 | 0.467787 | 2.016x (1.927..2.065) | 1.008x (0.897..1.050) | improved | inconclusive |
| mul | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 0.474035 | 0.974754 | 0.468988 | 2.010x (1.878..2.116) | 1.007x (1.000..1.045) | improved | parity |
| div | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 0.589558 | 1.096627 | 0.597096 | 1.855x (1.705..2.004) | 0.981x (0.910..1.057) | improved | inconclusive |
| add | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 0.529146 | 1.011612 | 0.529271 | 1.921x (1.838..1.996) | 0.999x (0.971..1.011) | improved | parity |
| sub | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 0.556129 | 1.067027 | 0.529906 | 1.934x (1.869..2.173) | 1.000x (0.972..1.088) | improved | inconclusive |
| mul | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 0.597229 | 1.157648 | 0.604096 | 1.903x (1.739..2.138) | 0.987x (0.882..1.013) | improved | inconclusive |
| div | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 0.590323 | 1.103990 | 0.589231 | 1.872x (1.824..1.954) | 1.003x (0.945..1.041) | improved | inconclusive |
| add | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 0.796035 | 1.105210 | 0.582627 | 1.925x (1.787..2.199) | 1.524x (1.463..1.610) | improved | planned-faster |
| sub | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 0.791902 | 1.004304 | 0.517288 | 1.936x (1.821..2.109) | 1.533x (1.501..1.540) | improved | planned-faster |
| mul | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 0.907329 | 1.096125 | 0.584233 | 1.919x (1.762..1.985) | 1.541x (1.431..1.722) | improved | planned-faster |
| div | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 0.812581 | 1.017383 | 0.532162 | 1.926x (1.748..2.084) | 1.534x (1.444..1.560) | improved | planned-faster |
| add | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 0.751746 | n/a | 6.140833 | n/a | 0.123x (0.122..0.124) | legacy-incorrect | numpy-faster |
| sub | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 0.764306 | n/a | 6.152567 | n/a | 0.124x (0.122..0.130) | legacy-incorrect | numpy-faster |
| mul | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 0.759154 | n/a | 6.131079 | n/a | 0.123x (0.121..0.129) | legacy-incorrect | numpy-faster |
| div | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 0.778698 | n/a | 6.091721 | n/a | 0.127x (0.122..0.153) | legacy-incorrect | numpy-faster |
| add | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 0.702629 | n/a | 6.117815 | n/a | 0.115x (0.113..0.138) | legacy-incorrect | numpy-faster |
| sub | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 0.698444 | n/a | 6.105742 | n/a | 0.115x (0.114..0.119) | legacy-incorrect | numpy-faster |
| mul | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 0.701419 | n/a | 6.124858 | n/a | 0.115x (0.114..0.131) | legacy-incorrect | numpy-faster |
| div | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 0.716623 | n/a | 6.093315 | n/a | 0.118x (0.115..0.124) | legacy-incorrect | numpy-faster |

### elementwise-scalar

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.299640 | 0.685840 | 0.462196 | 1.477x (1.441..1.535) | 0.649x (0.618..0.676) | improved | numpy-faster |
| sub | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.296442 | 0.678954 | 0.470081 | 1.443x (1.433..1.506) | 0.633x (0.608..0.645) | improved | numpy-faster |
| mul | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.299208 | 0.693898 | 0.464169 | 1.493x (1.453..1.542) | 0.644x (0.615..0.655) | improved | numpy-faster |
| div | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.297781 | 0.713710 | 0.500381 | 1.421x (1.381..1.450) | 0.596x (0.580..0.632) | improved | numpy-faster |
| add | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.301815 | 0.680638 | 0.464358 | 1.472x (1.448..1.524) | 0.651x (0.625..0.657) | improved | numpy-faster |
| sub | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.304394 | 0.707140 | 0.472210 | 1.479x (1.449..1.684) | 0.646x (0.617..0.732) | improved | numpy-faster |
| mul | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.302748 | 0.684321 | 0.464217 | 1.471x (1.426..1.495) | 0.652x (0.628..0.680) | improved | numpy-faster |
| div | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.304683 | 0.733192 | 0.503750 | 1.447x (1.405..1.560) | 0.605x (0.579..0.634) | improved | numpy-faster |
| add | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.682335 | 0.684975 | 0.461808 | 1.484x (1.437..1.724) | 1.475x (1.462..1.497) | improved | planned-faster |
| sub | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.681354 | 0.678725 | 0.468450 | 1.447x (1.435..1.477) | 1.455x (1.450..1.527) | improved | planned-faster |
| mul | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.682271 | 0.688121 | 0.461746 | 1.488x (1.469..1.584) | 1.478x (1.471..1.484) | improved | planned-faster |
| div | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.682356 | 0.711483 | 0.498758 | 1.432x (1.395..1.650) | 1.369x (1.359..1.422) | improved | planned-faster |
| add | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 0.694004 | n/a | 5.711481 | n/a | 0.122x (0.120..0.123) | legacy-incorrect | numpy-faster |
| sub | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 0.695477 | n/a | 5.691548 | n/a | 0.122x (0.120..0.122) | legacy-incorrect | numpy-faster |
| mul | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 0.700748 | n/a | 5.725244 | n/a | 0.122x (0.108..0.133) | legacy-incorrect | numpy-faster |
| div | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 0.763252 | n/a | 5.532388 | n/a | 0.138x (0.125..0.143) | legacy-incorrect | numpy-faster |

### elementwise-broadcast

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.036604 | n/a | 0.024829 | n/a | 1.475x (1.438..1.499) | new-only | planned-faster |
| sub | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.036567 | n/a | 0.024842 | n/a | 1.471x (1.421..1.477) | new-only | planned-faster |
| mul | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.036571 | n/a | 0.024708 | n/a | 1.479x (1.453..1.487) | new-only | planned-faster |
| div | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.036558 | n/a | 0.024983 | n/a | 1.464x (1.432..1.474) | new-only | planned-faster |
| add | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.035425 | n/a | 0.043600 | n/a | 0.810x (0.770..0.818) | new-only | numpy-faster |
| sub | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.035425 | n/a | 0.043625 | n/a | 0.811x (0.789..0.827) | new-only | numpy-faster |
| mul | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.035433 | n/a | 0.043767 | n/a | 0.810x (0.802..0.821) | new-only | numpy-faster |
| div | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.035392 | n/a | 0.043750 | n/a | 0.808x (0.797..0.815) | new-only | numpy-faster |
| add | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.042717 | n/a | 0.043121 | n/a | 0.991x (0.985..0.998) | new-only | parity |
| sub | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.042725 | n/a | 0.043196 | n/a | 0.988x (0.987..0.996) | new-only | parity |
| mul | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.042733 | n/a | 0.043271 | n/a | 0.991x (0.978..0.995) | new-only | parity |
| div | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.045221 | n/a | 0.043321 | n/a | 1.043x (1.022..1.059) | new-only | inconclusive |
| add | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.057521 | n/a | 0.382725 | n/a | 0.150x (0.150..0.152) | new-only | numpy-faster |
| sub | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.057492 | n/a | 0.382525 | n/a | 0.151x (0.150..0.152) | new-only | numpy-faster |
| mul | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.057525 | n/a | 0.382808 | n/a | 0.150x (0.150..0.151) | new-only | numpy-faster |
| div | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.057850 | n/a | 0.380575 | n/a | 0.152x (0.150..0.156) | new-only | numpy-faster |
| add | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.085333 | n/a | 0.380517 | n/a | 0.224x (0.223..0.231) | new-only | numpy-faster |
| sub | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.085367 | n/a | 0.380154 | n/a | 0.224x (0.223..0.226) | new-only | numpy-faster |
| mul | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.085367 | n/a | 0.380354 | n/a | 0.224x (0.216..0.226) | new-only | numpy-faster |
| div | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.085446 | n/a | 0.377967 | n/a | 0.226x (0.225..0.229) | new-only | numpy-faster |
| add | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.083183 | n/a | 0.084775 | n/a | 0.978x (0.964..0.987) | new-only | parity |
| sub | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.083308 | n/a | 0.085008 | n/a | 0.979x (0.973..0.986) | new-only | parity |
| mul | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.083117 | n/a | 0.084871 | n/a | 0.978x (0.941..0.982) | new-only | inconclusive |
| div | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.088125 | n/a | 0.085500 | n/a | 1.031x (1.018..1.035) | new-only | parity |

### inplace-array

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.102701 | 0.098812 | 0.098296 | 1.002x (0.999..1.093) | 1.045x (1.011..1.059) | inconclusive | inconclusive |
| sub | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.148074 | 0.132539 | 0.128828 | 1.006x (0.913..1.812) | 1.072x (0.932..1.174) | inconclusive | inconclusive |
| mul | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.116972 | 0.111486 | 0.112482 | 0.992x (0.903..1.036) | 1.044x (0.893..1.104) | inconclusive | inconclusive |
| div | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.122053 | 0.120158 | 0.109499 | 1.003x (0.408..1.450) | 1.079x (0.503..1.341) | inconclusive | inconclusive |
| add | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.281062 | n/a | 1.897466 | n/a | 0.136x (0.119..0.152) | legacy-incorrect | numpy-faster |
| sub | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.214697 | n/a | 1.552017 | n/a | 0.137x (0.134..0.143) | legacy-incorrect | numpy-faster |
| mul | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.216505 | n/a | 1.554894 | n/a | 0.138x (0.135..0.146) | legacy-incorrect | numpy-faster |
| div | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.217210 | n/a | 1.496273 | n/a | 0.145x (0.143..0.154) | legacy-incorrect | numpy-faster |
| add | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.205911 | n/a | 1.530378 | n/a | 0.134x (0.133..0.136) | legacy-incorrect | numpy-faster |
| sub | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.203233 | n/a | 1.532177 | n/a | 0.133x (0.131..0.133) | legacy-incorrect | numpy-faster |
| mul | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.203987 | n/a | 1.538448 | n/a | 0.133x (0.129..0.134) | legacy-incorrect | numpy-faster |
| div | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.206204 | n/a | 1.531575 | n/a | 0.135x (0.133..0.295) | legacy-incorrect | numpy-faster |

### inplace-broadcast

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.156195 | n/a | 0.113704 | n/a | 1.371x (1.326..1.416) | new-only | planned-faster |
| sub | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.154772 | n/a | 0.115036 | n/a | 1.349x (1.291..1.417) | new-only | planned-faster |
| mul | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.158256 | n/a | 0.117896 | n/a | 1.350x (1.152..1.452) | new-only | planned-faster |
| div | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.162647 | n/a | 0.120583 | n/a | 1.323x (1.198..1.405) | new-only | planned-faster |
| add | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.263430 | n/a | 1.560645 | n/a | 0.169x (0.163..0.175) | new-only | numpy-faster |
| sub | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.257304 | n/a | 1.570261 | n/a | 0.165x (0.160..0.168) | new-only | numpy-faster |
| mul | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.282643 | n/a | 1.695771 | n/a | 0.170x (0.159..0.246) | new-only | numpy-faster |
| div | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.286031 | n/a | 1.680417 | n/a | 0.170x (0.148..0.214) | new-only | numpy-faster |

### inplace-scalar

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.067223 | 0.068308 | 0.069628 | 1.002x (0.886..1.167) | 1.008x (0.861..1.104) | inconclusive | inconclusive |
| sub | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.073825 | 0.074519 | 0.073065 | 1.008x (0.950..1.273) | 1.000x (0.951..1.161) | inconclusive | inconclusive |
| mul | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.133093 | 0.125332 | 0.123623 | 1.052x (0.968..1.202) | 1.017x (0.780..1.254) | inconclusive | inconclusive |
| div | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.072872 | 0.077627 | 0.077869 | 1.000x (0.967..1.021) | 1.008x (0.972..1.033) | parity | parity |
| add | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.065125 | 0.063867 | 0.063585 | 1.001x (0.984..1.103) | 1.027x (1.015..1.081) | inconclusive | inconclusive |
| sub | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.065181 | 0.068347 | 0.064777 | 1.017x (0.945..1.166) | 1.007x (0.901..1.075) | inconclusive | inconclusive |
| mul | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.071611 | 0.064175 | 0.066863 | 0.965x (0.808..1.105) | 1.007x (0.873..1.265) | inconclusive | inconclusive |
| div | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.076268 | 0.071926 | 0.072427 | 0.997x (0.866..1.139) | 1.064x (0.893..1.166) | inconclusive | inconclusive |
| add | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.210274 | n/a | 1.562562 | n/a | 0.141x (0.125..0.194) | legacy-incorrect | numpy-faster |
| sub | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.181617 | n/a | 1.332817 | n/a | 0.136x (0.135..0.137) | legacy-incorrect | numpy-faster |
| mul | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.182311 | n/a | 1.344987 | n/a | 0.136x (0.127..0.233) | legacy-incorrect | numpy-faster |
| div | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.187058 | n/a | 1.311330 | n/a | 0.143x (0.137..0.146) | legacy-incorrect | numpy-faster |

### reduction-axis

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| mean | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.094796 | 4.005258 | 1.025750 | 3.898x (3.829..6.756) | 0.093x (0.090..0.103) | improved | numpy-faster |
| var | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.409408 | 4.354442 | 2.014346 | 2.151x (2.044..2.217) | 0.202x (0.194..0.213) | improved | numpy-faster |
| std | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.406563 | 4.438600 | 2.017817 | 2.202x (2.164..2.282) | 0.202x (0.196..0.243) | improved | numpy-faster |
| median | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 4.324263 | 4.507629 | 1.988275 | 2.262x (2.207..2.318) | 2.160x (2.135..2.220) | improved | planned-faster |
| average | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.247750 | 3.847046 | 1.263258 | 3.046x (3.007..3.065) | 0.195x (0.191..0.197) | improved | numpy-faster |
| mean | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.086379 | 3.989200 | 1.153037 | 3.501x (3.290..3.636) | 0.075x (0.070..0.084) | improved | numpy-faster |
| var | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.392517 | 4.275950 | 2.138333 | 1.999x (1.987..2.004) | 0.183x (0.182..0.186) | improved | numpy-faster |
| std | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.405567 | 4.391546 | 2.166883 | 2.015x (1.947..3.778) | 0.186x (0.182..0.230) | improved | numpy-faster |
| median | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 6.727567 | 4.532971 | 2.232183 | 2.021x (2.005..2.092) | 3.014x (2.950..3.155) | improved | planned-faster |
| average | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.233037 | 3.855362 | 1.319446 | 2.917x (2.818..2.935) | 0.173x (0.166..0.177) | improved | numpy-faster |
| mean | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 0.099754 | 3.926187 | 1.043108 | 3.763x (3.643..3.888) | 0.096x (0.092..0.103) | improved | numpy-faster |
| var | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 0.595646 | 4.460129 | 2.096050 | 2.120x (2.050..2.185) | 0.286x (0.279..0.344) | improved | numpy-faster |
| std | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 0.653833 | 4.719750 | 2.114954 | 2.148x (1.964..2.290) | 0.287x (0.245..0.317) | improved | numpy-faster |
| median | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 4.265217 | 4.385604 | 2.060613 | 2.125x (1.991..2.216) | 2.074x (1.937..2.100) | improved | planned-faster |
| average | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.414775 | 3.832842 | 1.245375 | 3.077x (2.890..3.088) | 0.332x (0.328..0.419) | improved | numpy-faster |
| mean | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.106908 | 3.848492 | 1.043075 | 3.687x (3.620..3.759) | 0.102x (0.101..0.107) | improved | numpy-faster |
| var | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.504921 | 4.265121 | 2.004758 | 2.131x (2.020..2.182) | 0.252x (0.239..0.255) | improved | numpy-faster |
| std | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.506412 | 4.281900 | 2.003338 | 2.141x (2.113..2.191) | 0.253x (0.249..0.255) | improved | numpy-faster |
| median | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 4.273817 | 4.369329 | 2.072825 | 2.114x (1.970..2.147) | 2.056x (1.841..2.117) | improved | planned-faster |
| average | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.341254 | 3.845013 | 1.265038 | 3.032x (3.015..3.096) | 0.270x (0.265..0.280) | improved | numpy-faster |
| mean | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.096675 | 3.849262 | 0.989929 | 3.896x (3.871..3.954) | 0.097x (0.096..0.102) | improved | numpy-faster |
| var | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.472000 | 4.266617 | 1.958971 | 2.178x (2.175..2.185) | 0.241x (0.239..0.242) | improved | numpy-faster |
| std | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.470454 | 4.280604 | 1.958746 | 2.186x (2.182..2.193) | 0.240x (0.239..0.243) | improved | numpy-faster |
| median | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 4.182783 | 4.366188 | 1.934371 | 2.261x (2.244..2.276) | 2.168x (2.143..2.184) | improved | planned-faster |
| average | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.332808 | 3.834967 | 1.256271 | 3.049x (3.037..3.058) | 0.265x (0.254..0.287) | improved | numpy-faster |

### reduction-full

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| mean | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.083437 | 0.402892 | 0.402954 | 1.000x (0.997..1.003) | 0.207x (0.206..0.209) | parity | numpy-faster |
| var | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.329621 | 3.087062 | 1.948592 | 1.586x (1.581..1.588) | 0.169x (0.168..0.174) | improved | numpy-faster |
| std | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.329029 | 3.086854 | 1.947421 | 1.585x (1.579..1.589) | 0.169x (0.168..0.171) | improved | numpy-faster |
| median | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 3.365496 | 3.622558 | 1.827058 | 1.982x (1.975..1.992) | 1.843x (1.836..1.852) | improved | planned-faster |
| average | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.269475 | 7.684646 | 1.220687 | 6.299x (6.267..6.308) | 0.221x (0.217..0.226) | improved | numpy-faster |
| mean | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.083746 | 0.403388 | 0.403187 | 1.000x (0.982..1.002) | 0.207x (0.205..0.216) | parity | numpy-faster |
| var | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.322092 | 3.091204 | 2.101304 | 1.470x (1.460..1.475) | 0.153x (0.144..0.155) | improved | numpy-faster |
| std | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.323883 | 3.087604 | 2.100437 | 1.470x (1.466..1.476) | 0.154x (0.154..0.157) | improved | numpy-faster |
| median | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 3.674017 | 3.629887 | 1.966962 | 1.845x (1.839..1.852) | 1.867x (1.864..1.872) | improved | planned-faster |
| average | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.628708 | 7.745925 | 1.279417 | 6.044x (6.003..6.365) | 0.488x (0.476..0.505) | improved | numpy-faster |
| mean | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.084517 | 0.402867 | 0.402883 | 1.000x (0.996..1.002) | 0.209x (0.209..0.211) | parity | numpy-faster |
| var | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.426654 | 3.089242 | 1.959408 | 1.577x (1.571..1.582) | 0.218x (0.217..0.222) | improved | numpy-faster |
| std | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.428346 | 3.087975 | 1.959008 | 1.577x (1.574..1.582) | 0.219x (0.217..0.221) | improved | numpy-faster |
| median | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 4.399442 | 3.598729 | 1.853346 | 1.942x (1.924..1.962) | 2.374x (2.344..2.390) | improved | planned-faster |
| average | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.339333 | 7.687288 | 1.207533 | 6.362x (6.331..6.392) | 0.281x (0.279..0.283) | improved | numpy-faster |
| mean | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.101775 | 0.403779 | 0.403042 | 1.001x (0.999..1.006) | 0.252x (0.249..0.259) | parity | numpy-faster |
| var | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.446321 | 3.091358 | 1.989075 | 1.554x (1.550..1.559) | 0.224x (0.223..0.229) | improved | numpy-faster |
| std | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.454492 | 3.090663 | 1.995717 | 1.552x (1.536..1.594) | 0.228x (0.224..0.230) | improved | numpy-faster |
| median | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 3.456158 | 3.635950 | 1.912100 | 1.901x (1.883..1.910) | 1.809x (1.797..1.828) | improved | planned-faster |
| average | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.353025 | 7.906504 | 1.219600 | 6.344x (6.311..7.007) | 0.285x (0.279..0.292) | improved | numpy-faster |

### matmul

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matmul | matrix-matrix-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 0.154804 | 24.918452 | 0.151673 | 164.792x (160.765..166.664) | 1.021x (0.992..1.037) | improved | parity |
| matmul | matrix-matrix-f | lhs: shape=(256,256), strides=(1,256), flags=F<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 0.153387 | 25.660929 | 0.238845 | 106.517x (103.164..110.302) | 0.649x (0.631..0.659) | improved | numpy-faster |
| matmul | matrix-matrix-negative | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(256,256), strides=(-256,-1), flags=N | 7 | 29.577756 | 24.769863 | 0.235899 | 104.685x (102.502..108.535) | 126.212x (122.792..131.234) | improved | planned-faster |
| matmul | matrix-matrix-lhs-step2 | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 29.643952 | 24.713107 | 0.196649 | 126.476x (124.215..130.111) | 150.858x (148.999..155.232) | improved | planned-faster |
| matmul | matrix-matrix-rhs-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 29.914690 | 26.839530 | 0.195268 | 136.687x (133.312..141.249) | 153.514x (149.925..158.103) | improved | planned-faster |
| matmul | matrix-matrix-both-step2 | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 29.907899 | 27.000911 | 0.236696 | 113.462x (110.380..117.552) | 126.578x (121.690..129.302) | improved | planned-faster |
| matmul | matrix-matrix-mixed-c-f | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 0.175512 | 14.782042 | 0.198208 | 74.983x (73.823..98.616) | 0.893x (0.868..0.945) | improved | numpy-faster |
| matmul | matrix-matrix-rectangular | lhs: shape=(128,256), strides=(256,1), flags=C<br>rhs: shape=(256,64), strides=(64,1), flags=C | 40 | 0.025217 | 3.045059 | 0.024330 | 124.715x (122.617..125.865) | 1.017x (1.000..1.046) | improved | parity |
| matmul | matrix-matrix-small-direct | lhs: shape=(8,16), strides=(16,1), flags=C<br>rhs: shape=(16,8), strides=(8,1), flags=C | 2000 | 0.001158 | 0.001529 | 0.001521 | 1.002x (0.980..1.046) | 0.760x (0.745..0.834) | parity | numpy-faster |
| matmul | vector-vector | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256), strides=(1), flags=CF | 2000 | 0.000886 | 0.001159 | 0.001161 | 0.999x (0.992..1.018) | 0.765x (0.758..0.784) | parity | numpy-faster |
| matmul | vector-matrix | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(256,1), flags=C | 200 | 0.018906 | 0.141006 | 0.014657 | 6.863x (6.652..8.075) | 1.006x (0.923..1.142) | improved | inconclusive |
| matmul | matrix-vector | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.007076 | 0.061048 | 0.006914 | 8.844x (7.813..8.958) | 1.024x (0.897..1.047) | improved | inconclusive |

### matmul-batch

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matmul | batch-same-shape-c | lhs: shape=(8,32,32), strides=(1024,32,1), flags=C<br>rhs: shape=(8,32,32), strides=(1024,32,1), flags=C | 3 | 0.007569 | n/a | 0.177236 | n/a | 0.042x (0.040..0.055) | new-only | numpy-faster |
| matmul | batch-broadcast-c | lhs: shape=(8,1,32,32), strides=(1024,1024,32,1), flags=C<br>rhs: shape=(1,8,32,32), strides=(8192,1024,32,1), flags=C | 3 | 0.044250 | n/a | 1.400750 | n/a | 0.031x (0.029..0.037) | new-only | numpy-faster |
| matmul | batch-broadcast-negative-matrix | lhs: shape=(8,1,32,32), strides=(1024,1024,-32,-1), flags=N<br>rhs: shape=(1,8,32,32), strides=(8192,1024,-32,-1), flags=N | 3 | 1.641625 | n/a | 1.405806 | n/a | 1.160x (1.086..1.250) | new-only | planned-faster |
| matmul | batch-broadcast-step2-inner | lhs: shape=(8,1,32,32), strides=(2048,2048,64,2), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,64,2), flags=S | 3 | 1.635958 | n/a | 1.441472 | n/a | 1.128x (1.005..1.224) | new-only | inconclusive |
| matmul | batch-broadcast-step2-batch | lhs: shape=(8,1,32,32), strides=(2048,1024,32,1), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,32,1), flags=S | 3 | 0.043139 | n/a | 1.417486 | n/a | 0.031x (0.029..0.034) | new-only | numpy-faster |

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
