# Linux detailed execution benchmark

## Recorded environment

- Code revision: `953a9623615506202621df7a87a486765ef2cc02`.
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
    --output /tmp/solvcon-execution-953a9623.json
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
| inplace-scalar | 12 | 0 | 0 | 0 | 4 | 0 | 8 |
| reduction-axis | 25 | 24 | 0 | 0 | 0 | 0 | 1 |
| reduction-full | 20 | 15 | 0 | 0 | 0 | 0 | 5 |
| matmul | 12 | 11 | 0 | 0 | 0 | 0 | 1 |
| matmul-batch | 5 | 0 | 0 | 0 | 0 | 5 | 0 |

## Complete results

### elementwise-array

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 1.587103 | 3.159854 | 1.476529 | 2.120x (1.380..3.366) | 0.953x (0.749..1.429) | improved | inconclusive |
| sub | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 1.395950 | 3.363348 | 1.468573 | 2.158x (1.435..4.480) | 1.110x (0.600..1.936) | improved | inconclusive |
| mul | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 1.326142 | 2.848145 | 1.448171 | 2.084x (1.238..2.561) | 1.036x (0.708..1.324) | improved | inconclusive |
| div | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(1024,1), flags=C | 20 | 1.342658 | 3.262919 | 1.425207 | 2.215x (1.605..3.668) | 0.941x (0.707..1.112) | improved | inconclusive |
| add | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 1.451002 | 3.015641 | 1.501233 | 1.890x (1.370..2.942) | 1.020x (0.525..1.933) | improved | inconclusive |
| sub | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 1.388253 | 3.894792 | 1.520031 | 2.197x (1.346..3.047) | 0.925x (0.612..1.332) | improved | inconclusive |
| mul | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 1.066308 | 2.623144 | 1.343920 | 2.332x (1.685..2.707) | 0.944x (0.695..1.317) | improved | inconclusive |
| div | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs: shape=(1024,1024), strides=(1,1024), flags=F | 20 | 0.542074 | 1.405022 | 0.591053 | 2.333x (2.194..2.681) | 0.882x (0.828..1.093) | improved | inconclusive |
| add | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 1.808055 | 2.812762 | 1.452878 | 1.986x (1.509..3.384) | 1.239x (0.848..1.577) | improved | inconclusive |
| sub | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 1.710789 | 3.113287 | 1.422031 | 2.084x (1.570..4.109) | 1.203x (0.758..2.095) | improved | inconclusive |
| mul | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 2.053579 | 3.566246 | 1.727342 | 2.065x (1.430..3.341) | 1.098x (0.723..1.749) | improved | inconclusive |
| div | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs: shape=(1024,1024), strides=(-1024,-1), flags=N | 20 | 1.842279 | 2.993854 | 1.590139 | 2.001x (1.604..3.316) | 1.181x (0.782..1.776) | improved | inconclusive |
| add | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 3.439521 | n/a | 3.539456 | n/a | 1.002x (0.647..1.363) | legacy-incorrect | inconclusive |
| sub | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 3.395400 | n/a | 3.598978 | n/a | 0.909x (0.631..1.209) | legacy-incorrect | inconclusive |
| mul | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 3.139358 | n/a | 3.691825 | n/a | 0.947x (0.623..1.175) | legacy-incorrect | inconclusive |
| div | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 3.338800 | n/a | 2.999132 | n/a | 1.094x (0.916..1.430) | legacy-incorrect | inconclusive |
| add | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 2.566862 | n/a | 2.456637 | n/a | 0.962x (0.637..1.405) | legacy-incorrect | inconclusive |
| sub | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 2.685038 | n/a | 2.539419 | n/a | 1.114x (0.809..1.417) | legacy-incorrect | inconclusive |
| mul | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 2.362934 | n/a | 2.459874 | n/a | 0.964x (0.824..1.251) | legacy-incorrect | inconclusive |
| div | mixed-c-step2 | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs: shape=(1024,1024), strides=(2048,2), flags=S | 20 | 2.315748 | n/a | 2.231420 | n/a | 1.018x (0.800..1.698) | legacy-incorrect | inconclusive |

### elementwise-scalar

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.581105 | 1.961304 | 0.608414 | 3.083x (1.727..4.985) | 0.919x (0.677..2.665) | improved | inconclusive |
| sub | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 1.016533 | 2.156759 | 0.648248 | 2.873x (2.265..5.813) | 0.943x (0.714..3.087) | improved | inconclusive |
| mul | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.673807 | 2.075557 | 0.659514 | 3.147x (2.106..4.036) | 0.963x (0.738..2.682) | improved | inconclusive |
| div | c-contiguous | lhs: shape=(1024,1024), strides=(1024,1), flags=C<br>rhs=scalar(1.0001) | 20 | 0.845952 | 2.376922 | 0.780634 | 3.544x (2.401..5.229) | 0.991x (0.679..1.925) | improved | inconclusive |
| add | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.541513 | 2.026302 | 0.613296 | 3.154x (2.577..7.884) | 0.881x (0.718..1.145) | improved | inconclusive |
| sub | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.494625 | 1.812205 | 0.543307 | 3.403x (1.803..4.260) | 0.972x (0.791..1.433) | improved | inconclusive |
| mul | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.546691 | 2.742876 | 0.618889 | 3.351x (1.901..6.163) | 0.939x (0.441..1.433) | improved | inconclusive |
| div | f-contiguous | lhs: shape=(1024,1024), strides=(1,1024), flags=F<br>rhs=scalar(1.0001) | 20 | 0.521869 | 2.425755 | 0.622859 | 3.689x (2.503..5.272) | 0.910x (0.594..1.030) | improved | inconclusive |
| add | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.995433 | 2.465489 | 1.131754 | 2.608x (1.378..3.576) | 1.202x (0.626..1.461) | improved | inconclusive |
| sub | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 0.910835 | 2.833887 | 0.646388 | 5.272x (2.559..7.121) | 1.464x (0.925..2.047) | improved | inconclusive |
| mul | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 1.023954 | 2.065221 | 0.757125 | 4.047x (2.346..5.038) | 1.097x (0.640..1.780) | improved | inconclusive |
| div | negative-dense | lhs: shape=(1024,1024), strides=(-1024,-1), flags=N<br>rhs=scalar(1.0001) | 20 | 1.067030 | 2.905064 | 0.690973 | 3.584x (2.244..5.330) | 1.375x (1.036..2.005) | improved | inconclusive |
| add | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 2.275827 | n/a | 2.161165 | n/a | 1.071x (0.663..1.590) | legacy-incorrect | inconclusive |
| sub | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 2.156845 | n/a | 1.803239 | n/a | 1.023x (0.784..2.466) | legacy-incorrect | inconclusive |
| mul | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 1.845661 | n/a | 1.881784 | n/a | 1.178x (0.847..1.562) | legacy-incorrect | inconclusive |
| div | step2-inner | lhs: shape=(1024,1024), strides=(2048,2), flags=S<br>rhs=scalar(1.0001) | 20 | 1.725655 | n/a | 1.479161 | n/a | 0.985x (0.818..1.216) | legacy-incorrect | inconclusive |

### elementwise-broadcast

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.019556 | n/a | 0.010764 | n/a | 1.818x (1.711..2.732) | new-only | planned-faster |
| sub | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.028148 | n/a | 0.012062 | n/a | 2.342x (2.091..2.573) | new-only | planned-faster |
| mul | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.028404 | n/a | 0.013917 | n/a | 2.048x (1.385..2.439) | new-only | planned-faster |
| div | rhs-row-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.032415 | n/a | 0.025768 | n/a | 1.264x (1.196..1.315) | new-only | planned-faster |
| add | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.027320 | n/a | 0.012194 | n/a | 2.265x (1.591..2.661) | new-only | planned-faster |
| sub | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.033503 | n/a | 0.012323 | n/a | 2.748x (2.514..3.148) | new-only | planned-faster |
| mul | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.041088 | n/a | 0.013903 | n/a | 3.273x (2.134..7.682) | new-only | planned-faster |
| div | rhs-column-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,1), strides=(2,2), flags=S | 10 | 0.046767 | n/a | 0.032836 | n/a | 1.463x (1.297..2.161) | new-only | planned-faster |
| add | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.036561 | n/a | 0.029827 | n/a | 1.335x (1.006..1.957) | new-only | inconclusive |
| sub | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.041979 | n/a | 0.021628 | n/a | 1.924x (1.167..2.099) | new-only | planned-faster |
| mul | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.041682 | n/a | 0.024707 | n/a | 1.687x (1.530..2.099) | new-only | planned-faster |
| div | outer-step2-row | lhs: shape=(256,1), strides=(2,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.052737 | n/a | 0.035820 | n/a | 1.780x (0.819..1.955) | new-only | inconclusive |
| add | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.051294 | n/a | 0.040410 | n/a | 1.370x (1.116..2.067) | new-only | planned-faster |
| sub | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.037866 | n/a | 0.026245 | n/a | 1.333x (0.939..2.118) | new-only | inconclusive |
| mul | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.050187 | n/a | 0.041708 | n/a | 1.243x (0.656..1.903) | new-only | inconclusive |
| div | lhs-step2-rhs-row | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(1,256), strides=(256,1), flags=CF | 10 | 0.056274 | n/a | 0.043712 | n/a | 1.167x (0.952..1.657) | new-only | inconclusive |
| add | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.135212 | n/a | 0.027857 | n/a | 4.591x (3.399..5.229) | new-only | planned-faster |
| sub | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.121522 | n/a | 0.024411 | n/a | 4.946x (3.860..5.620) | new-only | planned-faster |
| mul | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.125983 | n/a | 0.026579 | n/a | 4.269x (3.623..5.381) | new-only | planned-faster |
| div | negative-lhs-negative-row | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(1,256), strides=(256,-1), flags=N | 10 | 0.133572 | n/a | 0.030699 | n/a | 4.368x (3.174..4.718) | new-only | planned-faster |
| add | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.085221 | n/a | 0.047921 | n/a | 1.775x (1.548..1.964) | new-only | planned-faster |
| sub | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.063162 | n/a | 0.043044 | n/a | 1.467x (1.311..1.664) | new-only | planned-faster |
| mul | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.081758 | n/a | 0.046805 | n/a | 1.756x (1.607..1.851) | new-only | planned-faster |
| div | rank-aligned | lhs: shape=(2,256,1), strides=(256,1,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 10 | 0.091454 | n/a | 0.051121 | n/a | 1.786x (1.703..1.849) | new-only | planned-faster |

### inplace-array

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.100135 | 0.103137 | 0.104627 | 1.030x (0.792..1.413) | 1.027x (0.752..1.498) | inconclusive | inconclusive |
| sub | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.072185 | 0.069570 | 0.069136 | 0.996x (0.731..1.121) | 1.008x (0.771..1.114) | inconclusive | inconclusive |
| mul | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.100753 | 0.095749 | 0.097186 | 1.011x (0.799..1.458) | 1.041x (0.855..1.362) | inconclusive | inconclusive |
| div | c-destination-c-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.122639 | 0.129593 | 0.120507 | 0.992x (0.791..1.094) | 0.981x (0.850..1.093) | inconclusive | inconclusive |
| add | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.101785 | n/a | 0.098406 | n/a | 0.990x (0.777..1.216) | legacy-incorrect | inconclusive |
| sub | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.124246 | n/a | 0.144806 | n/a | 1.005x (0.613..1.304) | legacy-incorrect | inconclusive |
| mul | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.091065 | n/a | 0.092775 | n/a | 0.963x (0.866..1.040) | legacy-incorrect | inconclusive |
| div | negative-destination-c-rhs | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.229030 | n/a | 0.227172 | n/a | 1.001x (0.865..1.185) | legacy-incorrect | inconclusive |
| add | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.180846 | n/a | 0.176191 | n/a | 1.000x (0.761..1.654) | legacy-incorrect | inconclusive |
| sub | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.157140 | n/a | 0.150277 | n/a | 1.000x (0.670..1.307) | legacy-incorrect | inconclusive |
| mul | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.119481 | n/a | 0.119795 | n/a | 0.985x (0.907..1.098) | legacy-incorrect | inconclusive |
| div | step2-destination-c-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(512,512), strides=(512,1), flags=C | 50 | 0.258702 | n/a | 0.246322 | n/a | 1.050x (0.811..1.266) | legacy-incorrect | inconclusive |

### inplace-broadcast

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.099774 | n/a | 0.056509 | n/a | 1.833x (1.434..2.148) | new-only | planned-faster |
| sub | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.089530 | n/a | 0.056049 | n/a | 1.535x (1.245..1.935) | new-only | planned-faster |
| mul | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.080839 | n/a | 0.046288 | n/a | 1.848x (1.566..2.247) | new-only | planned-faster |
| div | c-destination-row-rhs | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.161680 | n/a | 0.135701 | n/a | 1.216x (0.955..1.495) | new-only | inconclusive |
| add | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.142552 | n/a | 0.107497 | n/a | 1.347x (1.067..1.588) | new-only | planned-faster |
| sub | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.175571 | n/a | 0.109161 | n/a | 1.410x (1.265..1.935) | new-only | planned-faster |
| mul | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.193043 | n/a | 0.134102 | n/a | 1.467x (1.122..1.879) | new-only | planned-faster |
| div | step2-destination-row-rhs | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs: shape=(1,512), strides=(512,1), flags=CF | 50 | 0.257093 | n/a | 0.216822 | n/a | 1.181x (1.072..1.422) | new-only | planned-faster |

### inplace-scalar

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| add | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.042586 | 0.036137 | 0.041863 | 0.879x (0.733..0.957) | 1.001x (0.864..1.138) | inconclusive | inconclusive |
| sub | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.047655 | 0.036868 | 0.039843 | 0.967x (0.755..1.180) | 1.191x (0.832..1.520) | inconclusive | inconclusive |
| mul | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.054707 | 0.062695 | 0.051784 | 1.142x (0.808..1.455) | 1.033x (0.877..1.755) | inconclusive | inconclusive |
| div | c-destination | destination: shape=(512,512), strides=(512,1), flags=C<br>rhs=scalar(1.0001) | 50 | 0.107054 | 0.109993 | 0.109837 | 1.019x (0.912..1.407) | 0.994x (0.911..1.010) | inconclusive | inconclusive |
| add | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.051327 | 0.043383 | 0.048191 | 0.941x (0.737..1.326) | 1.078x (0.766..1.548) | inconclusive | inconclusive |
| sub | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.042456 | 0.034136 | 0.040111 | 0.853x (0.792..1.110) | 1.081x (0.950..1.386) | inconclusive | inconclusive |
| mul | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.061542 | 0.044771 | 0.055477 | 0.870x (0.684..1.666) | 1.232x (0.923..2.053) | inconclusive | inconclusive |
| div | negative-destination | destination: shape=(512,512), strides=(-512,-1), flags=N<br>rhs=scalar(1.0001) | 50 | 0.109805 | 0.109557 | 0.108043 | 0.984x (0.876..1.134) | 1.011x (0.872..1.070) | inconclusive | inconclusive |
| add | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.117242 | n/a | 0.116400 | n/a | 1.009x (0.710..1.225) | legacy-incorrect | inconclusive |
| sub | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.102852 | n/a | 0.100178 | n/a | 1.076x (0.803..2.010) | legacy-incorrect | inconclusive |
| mul | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.102606 | n/a | 0.099725 | n/a | 1.079x (0.945..1.188) | legacy-incorrect | inconclusive |
| div | step2-destination | destination: shape=(512,512), strides=(1024,2), flags=S<br>rhs=scalar(1.0001) | 50 | 0.234916 | n/a | 0.233437 | n/a | 1.015x (0.866..1.374) | legacy-incorrect | inconclusive |

### reduction-axis

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| mean | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.081642 | 1.510437 | 0.114193 | 13.417x (10.336..18.606) | 0.616x (0.517..0.850) | improved | numpy-faster |
| var | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.421909 | 1.741258 | 0.247169 | 6.894x (6.259..8.840) | 1.821x (1.359..2.466) | improved | planned-faster |
| std | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.401106 | 1.914150 | 0.266129 | 6.594x (5.584..9.144) | 1.506x (1.359..2.277) | improved | planned-faster |
| median | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 6.890020 | 2.304881 | 0.972100 | 2.207x (1.891..2.636) | 7.542x (5.119..8.688) | improved | planned-faster |
| average | axis1-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.290555 | 1.537457 | 0.127941 | 11.695x (8.601..16.770) | 2.092x (1.349..4.264) | improved | planned-faster |
| mean | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.076781 | 2.396950 | 0.057542 | 39.156x (26.664..64.279) | 1.334x (0.932..1.556) | improved | inconclusive |
| var | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.381929 | 2.008014 | 0.114377 | 16.095x (13.497..27.314) | 2.774x (2.172..4.409) | improved | planned-faster |
| std | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.324580 | 2.181746 | 0.123228 | 17.970x (11.977..28.035) | 2.623x (2.136..3.609) | improved | planned-faster |
| median | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 10.730659 | 2.426557 | 1.657154 | 1.600x (0.903..2.600) | 6.242x (3.360..9.265) | inconclusive | planned-faster |
| average | axis1-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.293721 | 2.025644 | 0.062504 | 34.215x (22.043..53.191) | 4.793x (3.878..5.894) | improved | planned-faster |
| mean | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 0.090397 | 1.557414 | 0.118389 | 13.769x (11.230..16.931) | 0.820x (0.537..1.119) | improved | inconclusive |
| var | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 1.122482 | 1.882551 | 0.269339 | 6.760x (5.859..8.042) | 3.843x (3.146..5.950) | improved | planned-faster |
| std | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 0.918097 | 1.770165 | 0.273438 | 6.534x (4.377..7.945) | 3.299x (2.447..4.807) | improved | planned-faster |
| median | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N | 10 | 7.258937 | 2.206024 | 1.180831 | 1.940x (1.384..2.449) | 6.476x (4.212..7.733) | improved | planned-faster |
| average | axis1-negative-inner | values: shape=(512,512), strides=(512,-1), flags=N<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.726917 | 1.504545 | 0.142330 | 10.751x (8.207..12.643) | 5.532x (4.333..7.777) | improved | planned-faster |
| mean | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.121383 | 1.762235 | 0.140004 | 10.614x (8.566..14.500) | 0.780x (0.583..0.984) | improved | inconclusive |
| var | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.529186 | 1.614594 | 0.266508 | 6.415x (4.942..7.861) | 1.879x (1.618..3.762) | improved | planned-faster |
| std | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.485456 | 1.625623 | 0.282387 | 6.142x (5.097..6.976) | 1.951x (1.495..3.014) | improved | planned-faster |
| median | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 6.459923 | 1.793609 | 0.999472 | 1.796x (1.312..1.974) | 6.216x (4.358..6.582) | improved | planned-faster |
| average | axis1-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.429004 | 1.593420 | 0.152213 | 11.028x (6.258..12.358) | 2.902x (1.952..5.574) | improved | planned-faster |
| mean | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.141307 | 2.023391 | 0.141063 | 12.593x (9.182..16.732) | 0.868x (0.405..1.277) | improved | inconclusive |
| var | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.482659 | 2.226489 | 0.270811 | 6.880x (4.529..9.384) | 1.834x (1.584..2.707) | improved | planned-faster |
| std | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 0.521273 | 1.670951 | 0.257173 | 6.803x (6.276..7.363) | 2.038x (1.684..2.870) | improved | planned-faster |
| median | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S | 10 | 6.986909 | 2.038982 | 0.965219 | 2.112x (1.547..2.514) | 7.066x (5.710..8.018) | improved | planned-faster |
| average | axis1-step2-outer | values: shape=(512,512), strides=(1024,1), flags=S<br>weights: shape=(512), strides=(1), flags=CF | 10 | 0.470537 | 1.742468 | 0.154642 | 10.534x (7.142..13.051) | 2.633x (1.871..4.042) | improved | planned-faster |

### reduction-full

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| mean | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.048791 | 0.104247 | 0.104566 | 1.008x (0.908..1.067) | 0.474x (0.412..0.724) | inconclusive | numpy-faster |
| var | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.224711 | 1.096479 | 0.215563 | 5.054x (4.811..5.468) | 1.084x (0.973..1.131) | improved | inconclusive |
| std | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 0.217521 | 1.104260 | 0.213270 | 5.178x (4.983..5.641) | 1.019x (0.920..1.628) | improved | inconclusive |
| median | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C | 10 | 4.898324 | 1.676428 | 0.911468 | 1.778x (1.402..2.429) | 5.578x (4.157..6.608) | improved | planned-faster |
| average | full-c-contiguous | values: shape=(512,512), strides=(512,1), flags=C<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.432998 | 4.676054 | 0.261750 | 17.895x (12.968..22.281) | 1.552x (1.019..2.190) | improved | inconclusive |
| mean | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.074940 | 0.117150 | 0.113894 | 1.041x (0.904..1.721) | 0.656x (0.444..0.913) | inconclusive | numpy-faster |
| var | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.316191 | 1.884916 | 0.232329 | 7.898x (5.697..12.502) | 1.219x (1.085..1.783) | improved | planned-faster |
| std | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 0.256513 | 1.349029 | 0.227965 | 5.962x (5.405..7.286) | 1.175x (1.048..1.346) | improved | inconclusive |
| median | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F | 10 | 6.862891 | 3.656332 | 2.470439 | 1.183x (1.033..1.988) | 3.120x (1.603..3.870) | inconclusive | planned-faster |
| average | full-f-contiguous | values: shape=(512,512), strides=(1,512), flags=F<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 1.257880 | 6.216866 | 0.620743 | 9.245x (6.619..12.081) | 2.074x (1.496..3.566) | improved | planned-faster |
| mean | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.048116 | 0.104001 | 0.100979 | 1.010x (0.926..1.054) | 0.452x (0.407..0.506) | inconclusive | numpy-faster |
| var | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.367634 | 1.299419 | 0.256032 | 5.215x (4.693..6.065) | 1.336x (1.192..1.962) | improved | planned-faster |
| std | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 0.367315 | 1.357524 | 0.238915 | 5.187x (4.756..6.460) | 1.502x (1.198..2.864) | improved | planned-faster |
| median | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N | 10 | 7.651821 | 1.521883 | 1.063361 | 1.572x (1.264..1.902) | 6.674x (5.061..7.642) | improved | planned-faster |
| average | full-negative-dense | values: shape=(512,512), strides=(-512,-1), flags=N<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.339669 | 4.976316 | 0.258863 | 18.554x (12.134..21.865) | 1.469x (0.771..2.630) | improved | inconclusive |
| mean | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.070530 | 0.104989 | 0.106222 | 0.974x (0.914..1.104) | 0.673x (0.566..0.745) | inconclusive | numpy-faster |
| var | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.452453 | 1.474973 | 0.255930 | 5.460x (5.020..6.874) | 1.739x (1.332..3.303) | improved | planned-faster |
| std | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 0.486368 | 1.528853 | 0.265669 | 6.748x (4.248..9.793) | 1.824x (1.265..3.688) | improved | planned-faster |
| median | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S | 10 | 5.756348 | 1.595661 | 1.192073 | 1.541x (1.054..1.747) | 4.809x (3.305..6.227) | improved | planned-faster |
| average | full-step2-inner | values: shape=(512,512), strides=(1024,2), flags=S<br>weights: shape=(512,512), strides=(512,1), flags=C | 10 | 0.430800 | 4.552128 | 0.257333 | 17.690x (6.590..19.293) | 1.437x (1.222..2.882) | improved | planned-faster |

### matmul

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matmul | matrix-matrix-c | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 33.129881 | 8.744199 | 0.733392 | 11.665x (6.275..16.243) | 47.728x (21.482..57.380) | improved | planned-faster |
| matmul | matrix-matrix-f | lhs: shape=(256,256), strides=(1,256), flags=F<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 31.236469 | 8.704799 | 0.672884 | 13.146x (11.993..16.170) | 47.962x (36.203..57.119) | improved | planned-faster |
| matmul | matrix-matrix-negative | lhs: shape=(256,256), strides=(-256,-1), flags=N<br>rhs: shape=(256,256), strides=(-256,-1), flags=N | 7 | 33.871958 | 9.471939 | 0.840802 | 11.198x (7.065..14.893) | 38.638x (26.521..48.946) | improved | planned-faster |
| matmul | matrix-matrix-lhs-step2 | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(256,1), flags=C | 7 | 26.585751 | 7.151003 | 0.628917 | 11.443x (10.913..11.854) | 42.422x (41.866..43.988) | improved | planned-faster |
| matmul | matrix-matrix-rhs-step2 | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 28.016386 | 7.600718 | 0.615898 | 12.157x (11.437..12.782) | 45.429x (42.616..46.912) | improved | planned-faster |
| matmul | matrix-matrix-both-step2 | lhs: shape=(256,256), strides=(512,2), flags=S<br>rhs: shape=(256,256), strides=(512,2), flags=S | 7 | 28.007323 | 7.651649 | 0.659333 | 11.497x (10.340..12.073) | 42.524x (37.801..44.127) | improved | planned-faster |
| matmul | matrix-matrix-mixed-c-f | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256,256), strides=(1,256), flags=F | 7 | 25.439057 | 4.948992 | 0.579760 | 8.511x (7.663..8.918) | 43.792x (39.820..44.905) | improved | planned-faster |
| matmul | matrix-matrix-rectangular | lhs: shape=(128,256), strides=(256,1), flags=C<br>rhs: shape=(256,64), strides=(64,1), flags=C | 40 | 3.235843 | 0.870178 | 0.105049 | 8.243x (7.604..8.580) | 30.672x (28.217..31.714) | improved | planned-faster |
| matmul | matrix-matrix-small-direct | lhs: shape=(8,16), strides=(16,1), flags=C<br>rhs: shape=(16,8), strides=(8,1), flags=C | 2000 | 0.002938 | 0.000738 | 0.000676 | 1.107x (1.052..1.120) | 4.361x (4.266..4.450) | improved | planned-faster |
| matmul | vector-vector | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256), strides=(1), flags=CF | 2000 | 0.001379 | 0.000431 | 0.000466 | 0.952x (0.872..1.011) | 3.042x (2.768..3.407) | inconclusive | planned-faster |
| matmul | vector-matrix | lhs: shape=(256), strides=(1), flags=CF<br>rhs: shape=(256,256), strides=(256,1), flags=C | 200 | 0.105068 | 0.028883 | 0.005957 | 4.885x (4.382..5.169) | 17.590x (16.149..18.531) | improved | planned-faster |
| matmul | matrix-vector | lhs: shape=(256,256), strides=(256,1), flags=C<br>rhs: shape=(256), strides=(1), flags=CF | 200 | 0.101258 | 0.019921 | 0.004520 | 4.359x (4.069..4.717) | 22.874x (20.852..23.988) | improved | planned-faster |

### matmul-batch

| Operation | Scenario | Operands | Calls/sample | NumPy ms | Legacy ms | Planned ms | Legacy/planned (q10..q90) | NumPy/planned (q10..q90) | Legacy status | Planned vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| matmul | batch-same-shape-c | lhs: shape=(8,32,32), strides=(1024,32,1), flags=C<br>rhs: shape=(8,32,32), strides=(1024,32,1), flags=C | 3 | 0.438090 | n/a | 0.021867 | n/a | 20.287x (14.030..21.744) | new-only | planned-faster |
| matmul | batch-broadcast-c | lhs: shape=(8,1,32,32), strides=(1024,1024,32,1), flags=C<br>rhs: shape=(1,8,32,32), strides=(8192,1024,32,1), flags=C | 3 | 3.392318 | n/a | 0.181773 | n/a | 18.912x (15.005..22.142) | new-only | planned-faster |
| matmul | batch-broadcast-negative-matrix | lhs: shape=(8,1,32,32), strides=(1024,1024,-32,-1), flags=N<br>rhs: shape=(1,8,32,32), strides=(8192,1024,-32,-1), flags=N | 3 | 3.276226 | n/a | 0.585427 | n/a | 5.625x (5.433..5.759) | new-only | planned-faster |
| matmul | batch-broadcast-step2-inner | lhs: shape=(8,1,32,32), strides=(2048,2048,64,2), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,64,2), flags=S | 3 | 3.288337 | n/a | 0.582195 | n/a | 5.673x (5.400..5.830) | new-only | planned-faster |
| matmul | batch-broadcast-step2-batch | lhs: shape=(8,1,32,32), strides=(2048,1024,32,1), flags=S<br>rhs: shape=(1,8,32,32), strides=(16384,2048,32,1), flags=S | 3 | 3.353431 | n/a | 0.167448 | n/a | 20.074x (16.257..22.220) | new-only | planned-faster |

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
