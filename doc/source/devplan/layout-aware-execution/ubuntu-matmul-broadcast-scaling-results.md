# Linux matmul broadcast scaling

## Recorded environment

- Code revision: `c17db00400ef3024becbddf5fe0bdea8c4aac331`.
- Dirty tree: `false`.
- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`.
- Machine: `x86_64`.
- Python: `3.12.7`.
- NumPy: `2.3.0`.
- Matrix size: `256 by 256`.
- Batch sizes: `1, 2, 4, 8, 16, 32, 64, 128`.
- Samples per route: `7`.
- Warmups per route: `2`.
- Threads: `1`.
- CPU affinity: `0`.
- NumPy configured BLAS: `auto`.
- NumPy linked math libraries: `none visible`.
- `_solvcon` linked math libraries: `libopenblas.so.0`.

The JSON also retains every timing sample, NumPy build
configuration, extension linkage, thread-control variables,
source layouts, and planned batch mappings.

A NumPy/planned ratio is an external speed comparison only when
the two linkage records identify comparable math backends. The
planned/control ratios remain valid for testing dispatch within
the `_solvcon` build.

## Reproduction

The timing build compiles every probe out. The trace build is
separate so probe overhead never enters the timing samples.

```console
$ source /path/to/devenv/scripts/init
$ devenv use prime
$ export CMAKE_PREFIX_PATH="$DEVENVPREFIX"
$ export CMAKE_ARGS="-Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)"
$ make BUILD_PATH_EXT=_benchmark BUILD_QT=OFF \
    SOLVCON_PROFILE=OFF
$ python3 profiling/profile_matmul_broadcast_scaling.py \
    --batches 1,2,4,8,16,32,64,128 \
    --side 256 \
    --repeat 7 \
    --warmup 2 --cpu 0 \
    --output /tmp/matmul-broadcast-c17db004.json
$ make BUILD_PATH_EXT=_profile BUILD_QT=OFF \
    SOLVCON_PROFILE=ON
$ python3 profiling/profile_matmul_broadcast_scaling.py \
    --batches 1,2,4,8,16,32,64,128 \
    --side 256 \
    --repeat 7 \
    --warmup 2 --cpu 0 \
    --trace-only \
    --output /tmp/matmul-broadcast-trace-c17db004.json
$ python3 profiling/render_matmul_broadcast_scaling.py \
    /tmp/matmul-broadcast-c17db004.json \
    /tmp/matmul-broadcast-c17db004.md \
    --trace /tmp/matmul-broadcast-trace-c17db004.json
```

On macOS, omit `--cpu` because process affinity is unavailable.
The benchmark checks every planned result against a contiguous
NumPy reference before either timing or tracing.

## Question under test

The source lhs has shape `(1, 256, 256)`. Its stored batch
stride is 65,536 because consecutive physical
matrices would be that many elements apart. For `B > 1`,
`MatmulPlan` does not advance by that stride. Broadcast alignment
replaces it with zero:

```text
source lhs shape:          (1, 256, 256)
source lhs strides:        (65536, 256, 1)
result batch shape:        (B)
planned lhs batch stride:  (0)
planned rhs batch stride:  (65536)

for b in [0, B):
    gemm(lhs + 0, rhs + b * 65536, out + b * 65536)
```

The large source stride is therefore metadata for an extent-one
axis, not a per-element memory jump. The controls below test
whether this explanation matches both timing and actual dispatch.

## Route design

| Route | Purpose | Lhs shape | Lhs source strides | Expected dispatch |
| --- | --- | --- | --- | --- |
| broadcast-dense | Reuse one dense lhs through a zero batch stride. | `(1, 256, 256)` | `(65536, 256, 1)` | direct batched GEMM |
| materialized-dense | Use one independently stored dense lhs per batch item. | `(128, 256, 256)` | `(65536, 256, 1)` | direct batched GEMM |
| broadcast-negative | Pack one reversed physical lhs, then reuse it. | `(1, 256, 256)` | `(65536, 256, -1)` | pack lhs once, then batched GEMM |
| prepacked-negative | Control for the reversed lhs with packing outside timing. | `(1, 256, 256)` | `(65536, 256, 1)` | direct batched GEMM |
| broadcast-step2 | Pack one step-two physical lhs, then reuse it. | `(1, 256, 256)` | `(131072, 512, 2)` | pack lhs once, then batched GEMM |
| prepacked-step2 | Control for the step-two lhs with packing outside timing. | `(1, 256, 256)` | `(65536, 256, 1)` | direct batched GEMM |

`prepacked-negative` and `prepacked-step2` preserve the same
logical values as their views, but move the one-time copy outside
the timed call. `materialized-dense` physically stores `B` lhs
matrices and isolates the cost of zero-stride reuse.

## Complete timing results

A NumPy/planned ratio greater than one means planned is faster.
Intervals are q10 to q90 from paired samples.

| B | Route | Lhs shape | Source batch stride | Planned batch stride | NumPy ms | Planned ms | NumPy/planned | Pack lhs | GEMM calls |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | broadcast-dense | `(1, 256, 256)` | `(65536,)` | `(65536,)` | 26.1794 | 0.8120 | 35.034x (32.234..37.359) | 0 | 1 |
| 1 | materialized-dense | `(1, 256, 256)` | `(65536,)` | `(65536,)` | 27.0974 | 0.6868 | 37.603x (31.962..49.400) | 0 | 1 |
| 1 | broadcast-negative | `(1, 256, 256)` | `(65536,)` | `(65536,)` | 30.7937 | 1.1019 | 27.619x (17.310..31.603) | 1 | 1 |
| 1 | prepacked-negative | `(1, 256, 256)` | `(65536,)` | `(65536,)` | 29.3825 | 0.7792 | 37.464x (27.778..38.732) | 0 | 1 |
| 1 | broadcast-step2 | `(1, 256, 256)` | `(131072,)` | `(131072,)` | 28.6696 | 1.2552 | 23.587x (17.027..26.729) | 1 | 1 |
| 1 | prepacked-step2 | `(1, 256, 256)` | `(65536,)` | `(65536,)` | 26.8335 | 0.7898 | 34.899x (27.685..40.224) | 0 | 1 |
| 2 | broadcast-dense | `(1, 256, 256)` | `(65536,)` | `(0,)` | 54.2891 | 1.4745 | 36.048x (30.107..38.077) | 0 | 2 |
| 2 | materialized-dense | `(2, 256, 256)` | `(65536,)` | `(65536,)` | 56.1311 | 1.5230 | 37.009x (36.261..45.936) | 0 | 2 |
| 2 | broadcast-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 68.4583 | 1.7382 | 38.249x (25.922..39.598) | 1 | 2 |
| 2 | prepacked-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 59.2368 | 1.6868 | 35.885x (27.015..42.742) | 0 | 2 |
| 2 | broadcast-step2 | `(1, 256, 256)` | `(131072,)` | `(0,)` | 65.5796 | 1.7437 | 36.571x (30.673..39.560) | 1 | 2 |
| 2 | prepacked-step2 | `(1, 256, 256)` | `(65536,)` | `(0,)` | 53.7386 | 1.4930 | 36.865x (35.313..39.946) | 0 | 2 |
| 4 | broadcast-dense | `(1, 256, 256)` | `(65536,)` | `(0,)` | 115.7983 | 2.3885 | 41.498x (32.801..46.414) | 0 | 4 |
| 4 | materialized-dense | `(4, 256, 256)` | `(65536,)` | `(65536,)` | 118.0795 | 3.1040 | 42.472x (32.376..49.216) | 0 | 4 |
| 4 | broadcast-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 106.1472 | 2.8730 | 36.990x (32.382..41.926) | 1 | 4 |
| 4 | prepacked-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 109.2628 | 3.1686 | 34.443x (31.865..45.083) | 0 | 4 |
| 4 | broadcast-step2 | `(1, 256, 256)` | `(131072,)` | `(0,)` | 122.5208 | 3.5251 | 33.600x (27.086..40.949) | 1 | 4 |
| 4 | prepacked-step2 | `(1, 256, 256)` | `(65536,)` | `(0,)` | 117.8457 | 2.5433 | 44.386x (41.900..46.990) | 0 | 4 |
| 8 | broadcast-dense | `(1, 256, 256)` | `(65536,)` | `(0,)` | 218.4957 | 5.9892 | 35.498x (31.094..46.521) | 0 | 8 |
| 8 | materialized-dense | `(8, 256, 256)` | `(65536,)` | `(65536,)` | 240.0795 | 5.1484 | 46.144x (32.664..48.994) | 0 | 8 |
| 8 | broadcast-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 234.6454 | 5.3280 | 43.603x (34.437..48.778) | 1 | 8 |
| 8 | prepacked-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 240.0439 | 5.7820 | 38.684x (31.740..49.542) | 0 | 8 |
| 8 | broadcast-step2 | `(1, 256, 256)` | `(131072,)` | `(0,)` | 227.1584 | 5.0521 | 44.964x (38.347..46.958) | 1 | 8 |
| 8 | prepacked-step2 | `(1, 256, 256)` | `(65536,)` | `(0,)` | 239.8623 | 4.9852 | 42.349x (33.112..52.861) | 0 | 8 |
| 16 | broadcast-dense | `(1, 256, 256)` | `(65536,)` | `(0,)` | 425.7132 | 9.7331 | 43.789x (42.686..44.086) | 0 | 16 |
| 16 | materialized-dense | `(16, 256, 256)` | `(65536,)` | `(65536,)` | 431.2555 | 9.0874 | 43.158x (42.255..46.609) | 0 | 16 |
| 16 | broadcast-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 424.5378 | 9.7844 | 43.358x (41.640..44.118) | 1 | 16 |
| 16 | prepacked-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 426.4847 | 9.7307 | 43.820x (43.345..44.380) | 0 | 16 |
| 16 | broadcast-step2 | `(1, 256, 256)` | `(131072,)` | `(0,)` | 426.9011 | 10.1512 | 42.573x (41.671..43.185) | 1 | 16 |
| 16 | prepacked-step2 | `(1, 256, 256)` | `(65536,)` | `(0,)` | 428.8190 | 9.6832 | 43.082x (41.345..44.883) | 0 | 16 |
| 32 | broadcast-dense | `(1, 256, 256)` | `(65536,)` | `(0,)` | 860.5008 | 19.5916 | 43.532x (43.097..44.477) | 0 | 32 |
| 32 | materialized-dense | `(32, 256, 256)` | `(65536,)` | `(65536,)` | 855.5251 | 19.6139 | 43.421x (42.622..44.084) | 0 | 32 |
| 32 | broadcast-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 856.9283 | 19.7415 | 43.464x (40.607..43.723) | 1 | 32 |
| 32 | prepacked-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 863.5422 | 19.6402 | 44.245x (41.684..44.751) | 0 | 32 |
| 32 | broadcast-step2 | `(1, 256, 256)` | `(131072,)` | `(0,)` | 864.6740 | 20.0594 | 43.085x (42.694..43.849) | 1 | 32 |
| 32 | prepacked-step2 | `(1, 256, 256)` | `(65536,)` | `(0,)` | 864.4173 | 19.5486 | 44.270x (43.644..44.563) | 0 | 32 |
| 64 | broadcast-dense | `(1, 256, 256)` | `(65536,)` | `(0,)` | 1588.1512 | 50.8520 | 31.279x (27.858..34.999) | 0 | 64 |
| 64 | materialized-dense | `(64, 256, 256)` | `(65536,)` | `(65536,)` | 1602.2518 | 46.2180 | 34.675x (28.550..35.360) | 0 | 64 |
| 64 | broadcast-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 1604.0943 | 46.5362 | 34.683x (31.692..38.103) | 1 | 64 |
| 64 | prepacked-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 1597.2581 | 46.7124 | 34.931x (33.731..37.934) | 0 | 64 |
| 64 | broadcast-step2 | `(1, 256, 256)` | `(131072,)` | `(0,)` | 1593.8412 | 46.8748 | 34.002x (31.599..34.861) | 1 | 64 |
| 64 | prepacked-step2 | `(1, 256, 256)` | `(65536,)` | `(0,)` | 1585.2640 | 45.2460 | 34.913x (33.212..35.191) | 0 | 64 |
| 128 | broadcast-dense | `(1, 256, 256)` | `(65536,)` | `(0,)` | 3189.5163 | 90.6715 | 35.149x (31.915..35.784) | 0 | 128 |
| 128 | materialized-dense | `(128, 256, 256)` | `(65536,)` | `(65536,)` | 3213.1205 | 91.8093 | 35.051x (34.015..37.731) | 0 | 128 |
| 128 | broadcast-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 3253.1808 | 92.9275 | 34.575x (33.565..36.725) | 1 | 128 |
| 128 | prepacked-negative | `(1, 256, 256)` | `(65536,)` | `(0,)` | 3195.4094 | 92.5685 | 35.000x (34.658..35.424) | 0 | 128 |
| 128 | broadcast-step2 | `(1, 256, 256)` | `(131072,)` | `(0,)` | 3188.6127 | 91.5504 | 34.908x (32.965..35.795) | 1 | 128 |
| 128 | prepacked-step2 | `(1, 256, 256)` | `(65536,)` | `(0,)` | 3181.7662 | 91.5067 | 35.154x (34.675..36.721) | 0 | 128 |

## Paired controls

The ratio is tested route divided by matching control. One means
parity. Greater than one means the view or broadcast route costs
more than its control.

| B | Comparison | NumPy tested/control | Planned tested/control |
| ---: | --- | ---: | ---: |
| 1 | broadcast reuse / materialized batch | 0.997x (0.881..1.051) | 1.104x (0.805..1.407) |
| 1 | negative view / prepacked negative | 1.003x (0.840..1.216) | 1.399x (0.863..2.416) |
| 1 | step-two view / prepacked step-two | 1.011x (0.920..1.135) | 1.456x (1.239..1.846) |
| 2 | broadcast reuse / materialized batch | 0.999x (0.819..1.043) | 1.112x (0.834..1.316) |
| 2 | negative view / prepacked negative | 1.077x (0.806..1.225) | 0.954x (0.658..1.861) |
| 2 | step-two view / prepacked step-two | 1.079x (0.992..1.300) | 1.142x (0.995..1.454) |
| 4 | broadcast reuse / materialized batch | 0.986x (0.822..1.177) | 1.035x (0.651..1.596) |
| 4 | negative view / prepacked negative | 0.962x (0.797..1.108) | 0.967x (0.812..1.123) |
| 4 | step-two view / prepacked step-two | 1.096x (1.024..1.167) | 1.482x (1.157..1.857) |
| 8 | broadcast reuse / materialized batch | 0.922x (0.851..1.055) | 0.956x (0.703..1.472) |
| 8 | negative view / prepacked negative | 0.978x (0.825..1.220) | 0.934x (0.687..1.310) |
| 8 | step-two view / prepacked step-two | 0.931x (0.841..1.142) | 0.977x (0.690..1.184) |
| 16 | broadcast reuse / materialized batch | 0.993x (0.988..1.004) | 0.982x (0.969..1.068) |
| 16 | negative view / prepacked negative | 1.006x (0.997..1.018) | 1.023x (1.004..1.059) |
| 16 | step-two view / prepacked step-two | 1.001x (0.996..1.010) | 1.019x (0.968..1.072) |
| 32 | broadcast reuse / materialized batch | 1.005x (1.000..1.044) | 0.994x (0.986..1.048) |
| 32 | negative view / prepacked negative | 0.992x (0.953..1.004) | 1.006x (0.941..1.051) |
| 32 | step-two view / prepacked step-two | 0.999x (0.990..1.010) | 1.024x (1.009..1.031) |
| 64 | broadcast reuse / materialized batch | 0.985x (0.912..0.995) | 1.045x (0.868..1.107) |
| 64 | negative view / prepacked negative | 1.005x (0.998..1.134) | 1.016x (0.978..1.242) |
| 64 | step-two view / prepacked step-two | 1.004x (0.945..1.008) | 1.020x (0.967..1.052) |
| 128 | broadcast reuse / materialized batch | 0.997x (0.969..1.003) | 0.980x (0.941..1.177) |
| 128 | negative view / prepacked negative | 1.001x (0.982..1.051) | 1.011x (1.000..1.032) |
| 128 | step-two view / prepacked step-two | 1.002x (0.995..1.011) | 1.010x (0.984..1.119) |

## Dispatch validation

These counts come from a separate profile-enabled build. The
timing build contains no probes.

| B | Route | Expected pack lhs | Actual pack lhs | Expected GEMM | Actual GEMM | Generic calls | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | broadcast-dense | 0 | 0 | 1 | 1 | 0 | pass |
| 1 | materialized-dense | 0 | 0 | 1 | 1 | 0 | pass |
| 1 | broadcast-negative | 1 | 1 | 1 | 1 | 0 | pass |
| 1 | prepacked-negative | 0 | 0 | 1 | 1 | 0 | pass |
| 1 | broadcast-step2 | 1 | 1 | 1 | 1 | 0 | pass |
| 1 | prepacked-step2 | 0 | 0 | 1 | 1 | 0 | pass |
| 2 | broadcast-dense | 0 | 0 | 2 | 2 | 0 | pass |
| 2 | materialized-dense | 0 | 0 | 2 | 2 | 0 | pass |
| 2 | broadcast-negative | 1 | 1 | 2 | 2 | 0 | pass |
| 2 | prepacked-negative | 0 | 0 | 2 | 2 | 0 | pass |
| 2 | broadcast-step2 | 1 | 1 | 2 | 2 | 0 | pass |
| 2 | prepacked-step2 | 0 | 0 | 2 | 2 | 0 | pass |
| 4 | broadcast-dense | 0 | 0 | 4 | 4 | 0 | pass |
| 4 | materialized-dense | 0 | 0 | 4 | 4 | 0 | pass |
| 4 | broadcast-negative | 1 | 1 | 4 | 4 | 0 | pass |
| 4 | prepacked-negative | 0 | 0 | 4 | 4 | 0 | pass |
| 4 | broadcast-step2 | 1 | 1 | 4 | 4 | 0 | pass |
| 4 | prepacked-step2 | 0 | 0 | 4 | 4 | 0 | pass |
| 8 | broadcast-dense | 0 | 0 | 8 | 8 | 0 | pass |
| 8 | materialized-dense | 0 | 0 | 8 | 8 | 0 | pass |
| 8 | broadcast-negative | 1 | 1 | 8 | 8 | 0 | pass |
| 8 | prepacked-negative | 0 | 0 | 8 | 8 | 0 | pass |
| 8 | broadcast-step2 | 1 | 1 | 8 | 8 | 0 | pass |
| 8 | prepacked-step2 | 0 | 0 | 8 | 8 | 0 | pass |
| 16 | broadcast-dense | 0 | 0 | 16 | 16 | 0 | pass |
| 16 | materialized-dense | 0 | 0 | 16 | 16 | 0 | pass |
| 16 | broadcast-negative | 1 | 1 | 16 | 16 | 0 | pass |
| 16 | prepacked-negative | 0 | 0 | 16 | 16 | 0 | pass |
| 16 | broadcast-step2 | 1 | 1 | 16 | 16 | 0 | pass |
| 16 | prepacked-step2 | 0 | 0 | 16 | 16 | 0 | pass |
| 32 | broadcast-dense | 0 | 0 | 32 | 32 | 0 | pass |
| 32 | materialized-dense | 0 | 0 | 32 | 32 | 0 | pass |
| 32 | broadcast-negative | 1 | 1 | 32 | 32 | 0 | pass |
| 32 | prepacked-negative | 0 | 0 | 32 | 32 | 0 | pass |
| 32 | broadcast-step2 | 1 | 1 | 32 | 32 | 0 | pass |
| 32 | prepacked-step2 | 0 | 0 | 32 | 32 | 0 | pass |
| 64 | broadcast-dense | 0 | 0 | 64 | 64 | 0 | pass |
| 64 | materialized-dense | 0 | 0 | 64 | 64 | 0 | pass |
| 64 | broadcast-negative | 1 | 1 | 64 | 64 | 0 | pass |
| 64 | prepacked-negative | 0 | 0 | 64 | 64 | 0 | pass |
| 64 | broadcast-step2 | 1 | 1 | 64 | 64 | 0 | pass |
| 64 | prepacked-step2 | 0 | 0 | 64 | 64 | 0 | pass |
| 128 | broadcast-dense | 0 | 0 | 128 | 128 | 0 | pass |
| 128 | materialized-dense | 0 | 0 | 128 | 128 | 0 | pass |
| 128 | broadcast-negative | 1 | 1 | 128 | 128 | 0 | pass |
| 128 | prepacked-negative | 0 | 0 | 128 | 128 | 0 | pass |
| 128 | broadcast-step2 | 1 | 1 | 128 | 128 | 0 | pass |
| 128 | prepacked-step2 | 0 | 0 | 128 | 128 | 0 | pass |

## Raw timing samples

All values are milliseconds per complete matmul call.

| B | Route | NumPy samples | Planned samples |
| ---: | --- | --- | --- |
| 1 | broadcast-dense | 24.8150, 25.0076, 26.1794, 36.8903, 27.2819, 25.8836, 32.8756 | 0.6683, 0.7138, 0.7738, 0.9785, 0.8120, 0.8575, 0.8959 |
| 1 | materialized-dense | 24.5083, 33.2384, 27.0974, 35.1166, 25.9205, 25.9707, 33.2234 | 0.6868, 0.8839, 0.6604, 0.5669, 0.6797, 0.7764, 1.1175 |
| 1 | broadcast-negative | 25.3297, 31.1323, 25.9029, 25.4408, 34.2415, 32.6306, 30.7937 | 1.0134, 1.1019, 1.1238, 0.9211, 1.1808, 0.9189, 3.5397 |
| 1 | prepacked-negative | 35.1604, 33.8402, 26.9655, 25.0307, 29.3825, 25.2416, 30.7034 | 1.2657, 1.2183, 0.6883, 0.6681, 0.7792, 0.6567, 0.9855 |
| 1 | broadcast-step2 | 36.7516, 28.8861, 26.2171, 28.6696, 27.1196, 26.0546, 32.8103 | 2.2648, 1.2552, 1.0168, 1.0186, 1.1498, 1.4837, 1.2931 |
| 1 | prepacked-step2 | 32.1947, 25.5367, 26.5216, 35.0453, 26.8335, 25.7747, 31.1141 | 1.6923, 0.7226, 0.7927, 0.8684, 0.7898, 0.7386, 0.7752 |
| 2 | broadcast-dense | 49.2222, 54.2891, 54.3663, 55.1870, 63.3776, 49.5010, 49.9554 | 1.4166, 1.5060, 1.4745, 2.2284, 1.8824, 1.3061, 1.3028 |
| 2 | materialized-dense | 64.6952, 54.3283, 49.0535, 55.1725, 63.4366, 57.7312, 56.1311 | 1.3231, 1.2358, 1.3254, 1.5230, 1.6348, 1.5911, 1.5461 |
| 2 | broadcast-negative | 70.9414, 47.9557, 71.9243, 68.4583, 48.5941, 69.2006, 47.6533 | 1.8528, 1.2538, 1.7382, 2.2509, 1.2651, 3.6070, 1.2867 |
| 2 | prepacked-negative | 63.1777, 51.7571, 59.2368, 63.5496, 49.9809, 55.7037, 76.2523 | 1.7606, 1.6868, 1.2449, 2.9543, 1.3260, 1.4097, 2.4235 |
| 2 | broadcast-step2 | 52.3785, 67.7556, 50.9888, 70.1194, 65.5796, 54.9122, 69.1723 | 1.5263, 1.7437, 1.3942, 2.0231, 1.7610, 2.1787, 1.7032 |
| 2 | prepacked-step2 | 53.6465, 59.7266, 50.8407, 54.6469, 60.7512, 53.7386, 52.2177 | 1.4930, 1.6201, 1.4616, 1.5322, 1.5424, 1.3177, 1.3948 |
| 4 | broadcast-dense | 115.7983, 120.4049, 98.4425, 137.6442, 101.7822, 132.2001, 101.1098 | 5.0891, 2.3885, 2.3723, 3.4848, 2.3265, 3.2095, 2.3781 |
| 4 | materialized-dense | 133.7395, 100.8736, 130.0187, 118.0795, 103.1910, 121.6797, 111.7859 | 3.1489, 2.3086, 3.7250, 2.2013, 3.1040, 2.6300, 3.5975 |
| 4 | broadcast-negative | 105.1181, 109.6241, 106.1472, 129.6353, 98.2656, 102.9715, 122.1254 | 2.7143, 3.3829, 2.5555, 3.5046, 2.7178, 3.1834, 2.8730 |
| 4 | prepacked-negative | 109.2628, 130.5931, 101.3348, 124.8474, 105.9934, 140.2815, 101.8966 | 2.3599, 3.0628, 3.3199, 3.6247, 3.2356, 3.1686, 3.0492 |
| 4 | broadcast-step2 | 131.9278, 119.6996, 128.5099, 117.5617, 122.5208, 110.2466, 132.2939 | 6.3418, 2.8419, 4.0154, 2.9267, 3.1274, 3.5251, 3.9373 |
| 4 | prepacked-step2 | 117.8457, 100.9352, 121.4094, 101.8970, 123.4723, 105.5560, 120.6584 | 2.7334, 2.3510, 2.6004, 2.5253, 2.7112, 2.3781, 2.5433 |
| 8 | broadcast-dense | 233.0522, 244.4274, 202.4562, 205.6909, 218.4957, 227.4589, 214.1186 | 6.5652, 7.6193, 5.9892, 4.9094, 7.3778, 4.9898, 4.4678 |
| 8 | materialized-dense | 216.4654, 240.0795, 219.6897, 250.9733, 245.8272, 218.4512, 245.4805 | 4.6911, 4.9803, 6.2656, 6.7416, 5.1484, 7.5156, 4.8924 |
| 8 | broadcast-negative | 250.5505, 232.3147, 234.6967, 276.5006, 196.8686, 234.6454, 209.4963 | 5.0102, 5.3280, 4.8938, 5.8517, 7.6195, 5.3904, 5.2153 |
| 8 | prepacked-negative | 207.4366, 226.1767, 243.6091, 223.1067, 251.7870, 240.0439, 245.3817 | 5.3623, 4.8122, 5.7820, 6.1196, 4.7191, 9.4075, 6.8372 |
| 8 | broadcast-step2 | 227.1584, 253.3595, 210.8646, 232.3738, 204.7689, 240.5142, 218.7891 | 5.0521, 6.7313, 4.8709, 4.8287, 5.2750, 5.2081, 4.8027 |
| 8 | prepacked-step2 | 210.4886, 204.8696, 239.8623, 249.5439, 261.2161, 225.7647, 246.7689 | 10.0428, 4.9709, 4.9852, 4.7289, 4.9288, 5.3310, 5.9007 |
| 16 | broadcast-dense | 390.0387, 386.4940, 386.6082, 425.7132, 428.3901, 431.0385, 465.2934 | 8.8684, 8.7454, 8.8289, 9.7426, 9.7331, 10.0930, 10.9085 |
| 16 | materialized-dense | 388.2554, 389.0927, 387.0061, 432.1434, 431.2555, 435.5280, 463.6356 | 8.9960, 8.9250, 9.0874, 8.7144, 10.0912, 9.7603, 11.1030 |
| 16 | broadcast-negative | 391.6625, 389.5325, 384.8009, 424.5378, 433.3788, 433.4438, 464.6858 | 8.9858, 9.0891, 9.5005, 10.0133, 9.7844, 9.9969, 10.5606 |
| 16 | prepacked-negative | 387.8168, 387.1970, 385.4008, 426.4847, 429.9374, 432.9311, 451.5555 | 8.8784, 8.7020, 8.7962, 9.7327, 9.7307, 9.7719, 10.5404 |
| 16 | broadcast-step2 | 388.7850, 385.5109, 390.0131, 433.6750, 426.9011, 437.4219, 463.6904 | 9.1525, 8.8573, 9.0788, 10.1512, 10.2642, 10.4837, 10.8916 |
| 16 | prepacked-step2 | 386.6604, 385.2303, 386.7367, 428.8190, 430.1880, 437.9097, 463.7338 | 8.9856, 9.6832, 8.9767, 10.1169, 9.6151, 9.7105, 10.3817 |
| 32 | broadcast-dense | 867.4046, 854.6311, 867.6124, 860.5008, 868.8134, 852.8550, 788.4950 | 20.0734, 19.8425, 19.5730, 19.2501, 19.8881, 19.5916, 18.2886 |
| 32 | materialized-dense | 862.5143, 855.5251, 855.2682, 859.4700, 864.2689, 783.7639, 786.5240 | 20.3072, 19.3716, 19.6972, 19.6139, 19.6288, 18.0785, 18.4104 |
| 32 | broadcast-negative | 861.5066, 855.5439, 861.0717, 856.9283, 866.2982, 787.0161, 793.6850 | 19.8816, 19.6153, 19.8040, 19.7159, 19.7415, 19.7607, 19.2984 |
| 32 | prepacked-negative | 860.1047, 861.3351, 870.9832, 863.5422, 879.2678, 868.9801, 788.0786 | 19.9458, 21.0720, 19.5833, 19.3417, 20.8242, 19.6402, 17.5489 |
| 32 | broadcast-step2 | 858.0234, 853.1617, 871.0213, 864.6740, 869.6486, 867.5829, 788.5667 | 20.2240, 19.8740, 20.2163, 20.1680, 20.0594, 19.4565, 18.2020 |
| 32 | prepacked-step2 | 864.4173, 862.0011, 865.4100, 865.3317, 877.7058, 855.5693, 789.7233 | 19.7327, 19.3275, 19.5486, 19.9383, 19.7069, 19.3334, 17.7766 |
| 64 | broadcast-dense | 1578.7855, 1588.1512, 1590.5828, 1581.6607, 1580.7047, 1666.8173, 2108.2847 | 44.7513, 51.9369, 50.8520, 45.4343, 47.2748, 59.4069, 76.4997 |
| 64 | materialized-dense | 1602.2518, 1625.4873, 1591.6518, 1594.8169, 1594.4617, 2048.8298, 2149.5662 | 46.1754, 46.8783, 46.2180, 44.8720, 45.2462, 82.8895, 69.1067 |
| 64 | broadcast-negative | 1618.5765, 1604.0943, 1588.3239, 1600.7524, 1600.0384, 1988.6454, 2107.2552 | 46.4117, 46.6949, 46.9383, 45.7138, 46.1336, 46.5362, 74.0137 |
| 64 | prepacked-negative | 1631.7344, 1600.2201, 1580.1842, 1587.2014, 1597.2581, 1582.5776, 2001.5865 | 46.7124, 46.7787, 45.3013, 44.9929, 48.3789, 45.2106, 47.7491 |
| 64 | broadcast-step2 | 1639.2768, 1593.8412, 1591.6518, 1583.2279, 1598.8996, 1588.9019, 1990.3952 | 49.9201, 46.8748, 46.1568, 45.1772, 47.7018, 45.7392, 66.9265 |
| 64 | prepacked-step2 | 1733.4226, 1581.0624, 1579.6645, 1585.2640, 1588.1056, 1582.7239, 2109.6403 | 51.0678, 45.2368, 45.2460, 47.5388, 45.0479, 45.0279, 63.9107 |
| 128 | broadcast-dense | 3896.3580, 3186.9836, 3178.5514, 3249.0115, 3172.0543, 3189.5163, 3191.2616 | 134.9558, 90.6715, 93.6392, 90.3486, 88.9369, 90.9967, 89.8586 |
| 128 | materialized-dense | 3928.7042, 3390.4965, 3218.2899, 3213.1205, 3180.2568, 3194.7474, 3198.4551 | 96.3961, 98.8970, 91.0345, 89.9655, 90.7322, 95.0464, 91.8093 |
| 128 | broadcast-negative | 3494.4973, 3435.4397, 3199.8954, 3200.6577, 3253.1808, 3221.7361, 3255.9241 | 89.8167, 99.3615, 92.7504, 96.8790, 95.9159, 91.3451, 92.9275 |
| 128 | prepacked-negative | 3187.0718, 3436.1162, 3195.4094, 3248.5418, 3328.2232, 3193.0334, 3190.0000 | 90.2708, 96.6516, 91.2982, 93.3679, 95.6701, 90.3529, 92.5685 |
| 128 | broadcast-step2 | 3184.8516, 3426.7177, 3182.0640, 3195.8460, 3367.8931, 3186.5409, 3188.6127 | 87.9746, 100.1910, 89.5779, 91.5504, 105.3299, 91.2388, 94.8296 |
| 128 | prepacked-step2 | 3172.7914, 3355.1276, 3179.4692, 3225.7864, 3373.9009, 3174.9691, 3181.7662 | 88.6567, 90.6103, 91.5067, 92.6467, 92.3948, 90.3167, 92.0438 |

## Interpretation boundary

At the largest measured batch, `B=128`:

- broadcast reuse / materialized batch: NumPy 0.997x, planned 0.980x.
- negative view / prepacked negative: NumPy 1.001x, planned 1.011x.
- step-two view / prepacked step-two: NumPy 1.002x, planned 1.010x.

Broadcast and materialized routes have equal arithmetic: each
performs `B` GEMMs. Broadcast saves lhs storage and may improve
cache reuse, but it does not reduce the contraction count.

The negative and step-two comparisons isolate in-call packing.
A planned ratio near one is consistent with one physical lhs copy
amortized across `B` GEMMs. A large NumPy ratio says only that the
specific NumPy backend handles that view differently. It is not
evidence that broadcasting itself makes matrix multiplication
hundreds of times faster.

The trace table is the stronger implementation check: the
strided routes must report one lhs pack and exactly `B` GEMM
calls, while dense and prepacked routes report zero packs.

Results remain backend-specific. Compare Apple Silicon numbers
only after NumPy and `_solvcon` linkage confirm the intended
Accelerate configuration.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
