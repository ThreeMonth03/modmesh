# Matmul pack-once crossover

## Environment

- Revision: `d1ebc1cc3c3b5b1d22bc1e7d46d1e1f07266addd`.
- Dirty tree: `false`.
- Platform: `macOS-26.5.1-arm64-arm-64bit`.
- Machine: `arm64`.
- Python: `3.11.6`.
- NumPy: `2.2.4`.
- Threads: `1`.
- Samples per route: `15`.
- Warmups per route: `5`.
- Matrix sides: `24, 32, 40`.
- Batch sizes: `2, 4, 8, 16`.

## Routes

- `current` uses automatic prototype dispatch.
- `generic` forces signed-stride contraction.
- `pack_once` packs only unsupported supplied operands, rebuilds
  the plan, and must enter BLAS.
- `prepacked` moves the same copy outside the timed call.
- `direct` exists only when both operands already have direct
  BLAS descriptors.
- NumPy view and prepacked routes locate NumPy layout overhead.

Every route includes output allocation.  `pack_once` includes
validation, packing, plan reconstruction, and BLAS calls.

## Pack-once versus current dispatch

| Result | Cases |
| --- | ---: |
| Pack faster | 36 |
| Parity | 36 |
| Inconclusive | 0 |
| Current faster | 0 |

This aggregate is a screening result.  Dispatch thresholds must
come from the detailed side and batch crossover below.

## Detailed crossover

| Topology | Layout | S | B | Supplied operands | Number | Current us | Generic us | Pack us | Prepacked us | NumPy view us | Generic/current | Pack/current | Pack/prepacked | NumPy view/prepacked | Result |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1d-nd` | `negative-vector` | 24 | 2 | lhs: shape=(24,), strides=(-1,)<br>rhs: shape=(2, 24, 24), strides=(576, 24, 1) | 173 | 1.686 | 1.687 | 1.564 | 1.314 | 1.726 | 0.998x (0.994..1.005) | 0.929x (0.924..0.934) | 1.197x (1.181..1.209) | 1.535x (1.522..1.556) | pack-faster |
| `1d-nd` | `negative-vector` | 24 | 4 | lhs: shape=(24,), strides=(-1,)<br>rhs: shape=(4, 24, 24), strides=(576, 24, 1) | 86 | 2.406 | 2.402 | 1.801 | 1.550 | 2.527 | 0.999x (0.996..1.013) | 0.747x (0.744..0.753) | 1.162x (1.144..1.168) | 1.929x (1.911..1.937) | pack-faster |
| `1d-nd` | `negative-vector` | 24 | 8 | lhs: shape=(24,), strides=(-1,)<br>rhs: shape=(8, 24, 24), strides=(576, 24, 1) | 43 | 3.825 | 3.809 | 2.214 | 1.953 | 4.184 | 0.996x (0.983..0.997) | 0.578x (0.572..0.581) | 1.135x (1.121..1.141) | 2.408x (2.383..2.426) | pack-faster |
| `1d-nd` | `negative-vector` | 24 | 16 | lhs: shape=(24,), strides=(-1,)<br>rhs: shape=(16, 24, 24), strides=(576, 24, 1) | 21 | 6.696 | 6.679 | 3.169 | 2.940 | 7.419 | 0.998x (0.995..1.002) | 0.473x (0.472..0.485) | 1.081x (1.044..1.106) | 2.865x (2.846..2.886) | pack-faster |
| `1d-nd` | `negative-vector` | 32 | 2 | lhs: shape=(32,), strides=(-1,)<br>rhs: shape=(2, 32, 32), strides=(1024, 32, 1) | 97 | 2.372 | 2.364 | 1.729 | 1.402 | 2.503 | 0.996x (0.992..1.016) | 0.728x (0.722..0.740) | 1.234x (1.225..1.247) | 2.047x (2.027..2.060) | pack-faster |
| `1d-nd` | `negative-vector` | 32 | 4 | lhs: shape=(32,), strides=(-1,)<br>rhs: shape=(4, 32, 32), strides=(1024, 32, 1) | 48 | 2.082 | 3.795 | 2.067 | 1.753 | 4.163 | 1.823x (1.804..1.833) | 0.991x (0.985..1.000) | 1.179x (1.168..1.188) | 2.627x (2.607..2.655) | parity |
| `1d-nd` | `negative-vector` | 32 | 8 | lhs: shape=(32,), strides=(-1,)<br>rhs: shape=(8, 32, 32), strides=(1024, 32, 1) | 24 | 2.733 | 6.608 | 2.700 | 2.394 | 7.307 | 2.423x (2.408..2.460) | 0.990x (0.980..0.994) | 1.126x (1.117..1.141) | 3.362x (3.332..3.458) | parity |
| `1d-nd` | `negative-vector` | 32 | 16 | lhs: shape=(32,), strides=(-1,)<br>rhs: shape=(16, 32, 32), strides=(1024, 32, 1) | 12 | 5.073 | 12.465 | 5.024 | 4.726 | 14.063 | 2.457x (2.451..2.473) | 0.991x (0.984..0.996) | 1.063x (1.056..1.072) | 3.115x (3.104..3.164) | parity |
| `1d-nd` | `negative-vector` | 40 | 2 | lhs: shape=(40,), strides=(-1,)<br>rhs: shape=(2, 40, 40), strides=(1600, 40, 1) | 62 | 3.441 | 3.426 | 1.916 | 1.565 | 3.601 | 0.995x (0.993..1.002) | 0.557x (0.555..0.560) | 1.227x (1.212..1.245) | 2.586x (2.556..2.634) | pack-faster |
| `1d-nd` | `negative-vector` | 40 | 4 | lhs: shape=(40,), strides=(-1,)<br>rhs: shape=(4, 40, 40), strides=(1600, 40, 1) | 31 | 2.452 | 5.948 | 2.434 | 2.079 | 6.305 | 2.425x (2.412..2.480) | 0.992x (0.982..1.002) | 1.169x (1.154..1.177) | 3.311x (3.282..3.365) | parity |
| `1d-nd` | `negative-vector` | 40 | 8 | lhs: shape=(40,), strides=(-1,)<br>rhs: shape=(8, 40, 40), strides=(1600, 40, 1) | 15 | 3.839 | 11.014 | 3.783 | 3.453 | 11.811 | 2.879x (2.841..3.270) | 0.986x (0.975..0.994) | 1.099x (1.088..1.109) | 3.707x (3.667..3.778) | parity |
| `1d-nd` | `negative-vector` | 40 | 16 | lhs: shape=(40,), strides=(-1,)<br>rhs: shape=(16, 40, 40), strides=(1600, 40, 1) | 7 | 7.649 | 21.113 | 7.595 | 7.250 | 22.875 | 2.761x (2.747..3.007) | 0.991x (0.988..0.996) | 1.048x (1.042..1.050) | 3.365x (3.354..3.386) | parity |
| `1d-nd` | `negative-step2-vector` | 24 | 2 | lhs: shape=(24,), strides=(-2,)<br>rhs: shape=(2, 24, 24), strides=(576, 24, 1) | 173 | 1.684 | 1.685 | 1.572 | 1.312 | 1.723 | 0.996x (0.989..1.007) | 0.930x (0.924..0.947) | 1.198x (1.171..1.225) | 1.549x (1.535..1.576) | pack-faster |
| `1d-nd` | `negative-step2-vector` | 24 | 4 | lhs: shape=(24,), strides=(-2,)<br>rhs: shape=(4, 24, 24), strides=(576, 24, 1) | 86 | 2.418 | 2.411 | 1.818 | 1.563 | 2.521 | 0.995x (0.990..1.002) | 0.752x (0.747..0.755) | 1.164x (1.153..1.171) | 1.933x (1.916..1.956) | pack-faster |
| `1d-nd` | `negative-step2-vector` | 24 | 8 | lhs: shape=(24,), strides=(-2,)<br>rhs: shape=(8, 24, 24), strides=(576, 24, 1) | 43 | 3.837 | 3.828 | 2.223 | 1.968 | 4.174 | 0.995x (0.986..1.007) | 0.579x (0.573..0.589) | 1.131x (1.092..1.155) | 2.402x (2.387..2.412) | pack-faster |
| `1d-nd` | `negative-step2-vector` | 24 | 16 | lhs: shape=(24,), strides=(-2,)<br>rhs: shape=(16, 24, 24), strides=(576, 24, 1) | 21 | 6.722 | 6.687 | 3.186 | 2.939 | 7.435 | 0.996x (0.978..1.009) | 0.474x (0.465..0.493) | 1.087x (1.074..1.129) | 2.867x (2.844..2.874) | pack-faster |
| `1d-nd` | `negative-step2-vector` | 32 | 2 | lhs: shape=(32,), strides=(-2,)<br>rhs: shape=(2, 32, 32), strides=(1024, 32, 1) | 97 | 2.381 | 2.372 | 1.725 | 1.406 | 2.497 | 0.995x (0.987..1.002) | 0.724x (0.717..0.728) | 1.227x (1.216..1.234) | 2.044x (2.028..2.064) | pack-faster |
| `1d-nd` | `negative-step2-vector` | 32 | 4 | lhs: shape=(32,), strides=(-2,)<br>rhs: shape=(4, 32, 32), strides=(1024, 32, 1) | 48 | 2.087 | 3.795 | 2.064 | 1.747 | 4.148 | 1.816x (1.809..1.830) | 0.989x (0.980..0.997) | 1.180x (1.171..1.187) | 2.636x (2.619..2.654) | parity |
| `1d-nd` | `negative-step2-vector` | 32 | 8 | lhs: shape=(32,), strides=(-2,)<br>rhs: shape=(8, 32, 32), strides=(1024, 32, 1) | 24 | 2.740 | 6.606 | 2.707 | 2.396 | 7.300 | 2.411x (2.396..2.453) | 0.988x (0.981..0.994) | 1.130x (1.091..1.141) | 3.391x (3.358..3.447) | parity |
| `1d-nd` | `negative-step2-vector` | 32 | 16 | lhs: shape=(32,), strides=(-2,)<br>rhs: shape=(16, 32, 32), strides=(1024, 32, 1) | 12 | 5.056 | 12.472 | 5.038 | 4.750 | 14.014 | 2.467x (2.456..2.477) | 0.997x (0.990..1.003) | 1.064x (1.056..1.069) | 3.157x (3.130..3.171) | parity |
| `1d-nd` | `negative-step2-vector` | 40 | 2 | lhs: shape=(40,), strides=(-2,)<br>rhs: shape=(2, 40, 40), strides=(1600, 40, 1) | 62 | 3.444 | 3.431 | 1.917 | 1.567 | 3.583 | 0.996x (0.991..1.000) | 0.557x (0.553..0.560) | 1.222x (1.201..1.240) | 2.588x (2.560..2.601) | pack-faster |
| `1d-nd` | `negative-step2-vector` | 40 | 4 | lhs: shape=(40,), strides=(-2,)<br>rhs: shape=(4, 40, 40), strides=(1600, 40, 1) | 31 | 2.456 | 6.019 | 2.430 | 2.085 | 6.321 | 2.462x (2.421..2.668) | 0.990x (0.981..1.003) | 1.168x (1.153..1.181) | 3.319x (3.284..3.363) | parity |
| `1d-nd` | `negative-step2-vector` | 40 | 8 | lhs: shape=(40,), strides=(-2,)<br>rhs: shape=(8, 40, 40), strides=(1600, 40, 1) | 15 | 3.850 | 11.081 | 3.794 | 3.478 | 11.875 | 2.876x (2.849..2.958) | 0.986x (0.974..0.995) | 1.093x (1.085..1.102) | 3.691x (3.648..3.735) | parity |
| `1d-nd` | `negative-step2-vector` | 40 | 16 | lhs: shape=(40,), strides=(-2,)<br>rhs: shape=(16, 40, 40), strides=(1600, 40, 1) | 7 | 7.655 | 23.500 | 7.607 | 7.292 | 22.923 | 3.075x (2.755..3.183) | 0.995x (0.986..0.999) | 1.042x (1.036..1.051) | 3.367x (3.344..3.384) | parity |
| `1d-nd` | `zero-vector` | 24 | 2 | lhs: shape=(24,), strides=(0,)<br>rhs: shape=(2, 24, 24), strides=(576, 24, 1) | 173 | 1.692 | 1.690 | 1.575 | 1.319 | 1.736 | 0.997x (0.990..1.004) | 0.932x (0.920..0.939) | 1.194x (1.187..1.206) | 1.560x (1.547..1.565) | pack-faster |
| `1d-nd` | `zero-vector` | 24 | 4 | lhs: shape=(24,), strides=(0,)<br>rhs: shape=(4, 24, 24), strides=(576, 24, 1) | 86 | 2.425 | 2.415 | 1.815 | 1.566 | 2.547 | 0.994x (0.989..1.003) | 0.749x (0.743..0.754) | 1.156x (1.152..1.171) | 1.931x (1.907..1.959) | pack-faster |
| `1d-nd` | `zero-vector` | 24 | 8 | lhs: shape=(24,), strides=(0,)<br>rhs: shape=(8, 24, 24), strides=(576, 24, 1) | 43 | 3.832 | 3.816 | 2.215 | 1.954 | 4.193 | 0.995x (0.960..1.009) | 0.578x (0.570..0.588) | 1.137x (1.126..1.234) | 2.419x (2.378..2.434) | pack-faster |
| `1d-nd` | `zero-vector` | 24 | 16 | lhs: shape=(24,), strides=(0,)<br>rhs: shape=(16, 24, 24), strides=(576, 24, 1) | 21 | 6.692 | 6.673 | 3.175 | 2.942 | 7.446 | 0.997x (0.994..1.031) | 0.474x (0.470..0.475) | 1.079x (1.072..1.089) | 2.887x (2.854..2.974) | pack-faster |
| `1d-nd` | `zero-vector` | 32 | 2 | lhs: shape=(32,), strides=(0,)<br>rhs: shape=(2, 32, 32), strides=(1024, 32, 1) | 97 | 2.371 | 2.364 | 1.713 | 1.395 | 2.509 | 0.995x (0.991..1.003) | 0.723x (0.715..0.726) | 1.225x (1.216..1.237) | 2.049x (2.033..2.072) | pack-faster |
| `1d-nd` | `zero-vector` | 32 | 4 | lhs: shape=(32,), strides=(0,)<br>rhs: shape=(4, 32, 32), strides=(1024, 32, 1) | 48 | 2.067 | 3.779 | 2.058 | 1.749 | 4.152 | 1.832x (1.818..1.861) | 0.990x (0.983..1.005) | 1.173x (1.164..1.183) | 2.639x (2.608..2.668) | parity |
| `1d-nd` | `zero-vector` | 32 | 8 | lhs: shape=(32,), strides=(0,)<br>rhs: shape=(8, 32, 32), strides=(1024, 32, 1) | 24 | 2.715 | 6.583 | 2.689 | 2.394 | 7.314 | 2.425x (2.420..2.461) | 0.990x (0.988..0.994) | 1.125x (1.088..1.129) | 3.382x (3.272..3.427) | parity |
| `1d-nd` | `zero-vector` | 32 | 16 | lhs: shape=(32,), strides=(0,)<br>rhs: shape=(16, 32, 32), strides=(1024, 32, 1) | 12 | 5.010 | 12.424 | 4.993 | 4.691 | 14.038 | 2.481x (2.469..2.491) | 0.995x (0.989..1.002) | 1.063x (1.056..1.071) | 3.135x (3.115..3.156) | parity |
| `1d-nd` | `zero-vector` | 40 | 2 | lhs: shape=(40,), strides=(0,)<br>rhs: shape=(2, 40, 40), strides=(1600, 40, 1) | 62 | 3.444 | 3.430 | 1.918 | 1.579 | 3.592 | 0.996x (0.991..1.001) | 0.557x (0.551..0.560) | 1.215x (1.200..1.231) | 2.552x (2.531..2.594) | pack-faster |
| `1d-nd` | `zero-vector` | 40 | 4 | lhs: shape=(40,), strides=(0,)<br>rhs: shape=(4, 40, 40), strides=(1600, 40, 1) | 31 | 2.430 | 6.169 | 2.413 | 2.077 | 6.302 | 2.542x (2.433..2.775) | 0.993x (0.988..1.001) | 1.164x (1.156..1.176) | 3.296x (3.065..3.315) | parity |
| `1d-nd` | `zero-vector` | 40 | 8 | lhs: shape=(40,), strides=(0,)<br>rhs: shape=(8, 40, 40), strides=(1600, 40, 1) | 15 | 3.814 | 10.958 | 3.775 | 3.442 | 11.792 | 2.872x (2.856..2.923) | 0.989x (0.979..0.997) | 1.095x (1.089..1.105) | 3.629x (3.599..3.705) | parity |
| `1d-nd` | `zero-vector` | 40 | 16 | lhs: shape=(40,), strides=(0,)<br>rhs: shape=(16, 40, 40), strides=(1600, 40, 1) | 7 | 7.613 | 21.059 | 7.542 | 7.226 | 22.863 | 2.768x (2.762..2.845) | 0.991x (0.988..0.995) | 1.044x (1.041..1.047) | 3.384x (3.366..3.398) | parity |
| `nd-1d` | `negative-vector` | 24 | 2 | lhs: shape=(2, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-1,) | 173 | 1.804 | 1.800 | 1.651 | 1.412 | 1.748 | 0.996x (0.988..1.005) | 0.915x (0.903..0.922) | 1.171x (1.130..1.183) | 1.461x (1.442..1.480) | pack-faster |
| `nd-1d` | `negative-vector` | 24 | 4 | lhs: shape=(4, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-1,) | 86 | 2.644 | 2.636 | 1.939 | 1.702 | 2.581 | 0.997x (0.992..1.000) | 0.733x (0.729..0.737) | 1.140x (1.133..1.143) | 1.771x (1.737..1.791) | pack-faster |
| `nd-1d` | `negative-vector` | 24 | 8 | lhs: shape=(8, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-1,) | 43 | 4.287 | 4.282 | 2.516 | 2.265 | 4.294 | 1.000x (0.994..1.014) | 0.587x (0.582..0.595) | 1.109x (1.105..1.125) | 2.122x (2.100..2.132) | pack-faster |
| `nd-1d` | `negative-vector` | 24 | 16 | lhs: shape=(16, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-1,) | 21 | 7.687 | 7.677 | 3.726 | 3.492 | 7.716 | 0.999x (0.988..1.001) | 0.485x (0.479..0.496) | 1.069x (1.027..1.094) | 2.454x (2.433..2.473) | pack-faster |
| `nd-1d` | `negative-vector` | 32 | 2 | lhs: shape=(2, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-1,) | 97 | 2.503 | 2.496 | 1.825 | 1.531 | 2.527 | 0.995x (0.992..1.001) | 0.728x (0.725..0.733) | 1.193x (1.186..1.199) | 1.902x (1.875..1.914) | pack-faster |
| `nd-1d` | `negative-vector` | 32 | 4 | lhs: shape=(4, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-1,) | 48 | 2.292 | 4.053 | 2.261 | 1.965 | 4.200 | 1.767x (1.755..1.780) | 0.988x (0.984..0.997) | 1.155x (1.138..1.163) | 2.382x (2.345..2.453) | parity |
| `nd-1d` | `negative-vector` | 32 | 8 | lhs: shape=(8, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-1,) | 24 | 3.132 | 7.135 | 3.101 | 2.806 | 7.404 | 2.277x (2.264..2.321) | 0.990x (0.983..0.999) | 1.105x (1.095..1.111) | 2.900x (2.794..2.927) | parity |
| `nd-1d` | `negative-vector` | 32 | 16 | lhs: shape=(16, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-1,) | 12 | 5.809 | 13.712 | 5.757 | 5.517 | 14.337 | 2.361x (2.348..2.377) | 0.991x (0.987..0.994) | 1.043x (1.039..1.050) | 2.773x (2.750..2.789) | parity |
| `nd-1d` | `negative-vector` | 40 | 2 | lhs: shape=(2, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-1,) | 62 | 3.698 | 3.677 | 2.011 | 1.676 | 3.637 | 0.995x (0.983..1.000) | 0.544x (0.534..0.546) | 1.198x (1.185..1.210) | 2.456x (2.420..2.478) | pack-faster |
| `nd-1d` | `negative-vector` | 40 | 4 | lhs: shape=(4, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-1,) | 31 | 2.593 | 6.355 | 2.570 | 2.242 | 6.395 | 2.452x (2.439..2.460) | 0.994x (0.983..0.996) | 1.146x (1.141..1.152) | 3.096x (3.031..3.111) | parity |
| `nd-1d` | `negative-vector` | 40 | 8 | lhs: shape=(8, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-1,) | 15 | 4.072 | 11.906 | 4.019 | 3.719 | 12.025 | 2.924x (2.915..2.944) | 0.988x (0.982..0.990) | 1.080x (1.032..1.084) | 3.502x (3.455..3.529) | parity |
| `nd-1d` | `negative-vector` | 40 | 16 | lhs: shape=(16, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-1,) | 7 | 7.934 | 22.804 | 7.869 | 7.577 | 22.988 | 2.875x (2.859..2.890) | 0.992x (0.984..0.998) | 1.039x (1.032..1.044) | 3.203x (3.182..3.217) | parity |
| `nd-1d` | `negative-step2-vector` | 24 | 2 | lhs: shape=(2, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-2,) | 173 | 1.805 | 1.800 | 1.649 | 1.410 | 1.747 | 0.997x (0.993..1.010) | 0.917x (0.909..0.924) | 1.173x (1.163..1.182) | 1.466x (1.460..1.480) | pack-faster |
| `nd-1d` | `negative-step2-vector` | 24 | 4 | lhs: shape=(4, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-2,) | 86 | 2.643 | 2.637 | 1.937 | 1.702 | 2.572 | 0.999x (0.994..1.002) | 0.733x (0.729..0.735) | 1.137x (1.133..1.146) | 1.772x (1.766..1.790) | pack-faster |
| `nd-1d` | `negative-step2-vector` | 24 | 8 | lhs: shape=(8, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-2,) | 43 | 4.316 | 4.306 | 2.527 | 2.273 | 4.318 | 0.996x (0.980..1.004) | 0.584x (0.580..0.589) | 1.112x (1.107..1.122) | 2.126x (2.076..2.172) | pack-faster |
| `nd-1d` | `negative-step2-vector` | 24 | 16 | lhs: shape=(16, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-2,) | 21 | 7.790 | 7.764 | 3.810 | 3.546 | 7.812 | 0.997x (0.925..1.141) | 0.494x (0.484..0.877) | 1.074x (0.993..1.988) | 2.468x (2.223..3.819) | pack-faster |
| `nd-1d` | `negative-step2-vector` | 32 | 2 | lhs: shape=(2, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-2,) | 97 | 2.507 | 2.501 | 1.839 | 1.530 | 2.521 | 0.996x (0.991..1.003) | 0.732x (0.727..0.737) | 1.202x (1.193..1.209) | 1.911x (1.896..1.924) | pack-faster |
| `nd-1d` | `negative-step2-vector` | 32 | 4 | lhs: shape=(4, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-2,) | 48 | 2.293 | 4.053 | 2.267 | 1.968 | 4.201 | 1.768x (1.754..1.777) | 0.989x (0.981..0.997) | 1.152x (1.148..1.161) | 2.367x (2.359..2.378) | parity |
| `nd-1d` | `negative-step2-vector` | 32 | 8 | lhs: shape=(8, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-2,) | 24 | 3.167 | 7.132 | 3.122 | 2.814 | 7.424 | 2.253x (2.237..2.271) | 0.985x (0.979..0.989) | 1.106x (1.102..1.115) | 2.924x (2.898..2.950) | parity |
| `nd-1d` | `negative-step2-vector` | 32 | 16 | lhs: shape=(16, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-2,) | 12 | 5.854 | 13.760 | 5.802 | 5.528 | 14.455 | 2.351x (2.342..2.368) | 0.992x (0.988..0.996) | 1.049x (1.042..1.057) | 2.755x (2.734..2.776) | parity |
| `nd-1d` | `negative-step2-vector` | 40 | 2 | lhs: shape=(2, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-2,) | 62 | 3.689 | 3.671 | 2.015 | 1.674 | 3.633 | 0.995x (0.992..1.000) | 0.547x (0.540..0.550) | 1.202x (1.192..1.214) | 2.460x (2.429..2.481) | pack-faster |
| `nd-1d` | `negative-step2-vector` | 40 | 4 | lhs: shape=(4, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-2,) | 31 | 2.609 | 6.384 | 2.579 | 2.247 | 6.394 | 2.456x (2.435..2.497) | 0.987x (0.984..0.996) | 1.148x (1.137..1.156) | 3.116x (3.082..3.144) | parity |
| `nd-1d` | `negative-step2-vector` | 40 | 8 | lhs: shape=(8, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-2,) | 15 | 4.153 | 11.953 | 4.097 | 3.733 | 12.044 | 2.879x (2.862..2.923) | 0.985x (0.979..0.990) | 1.093x (1.089..1.105) | 3.447x (3.426..3.504) | parity |
| `nd-1d` | `negative-step2-vector` | 40 | 16 | lhs: shape=(16, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-2,) | 7 | 7.964 | 22.815 | 7.905 | 7.613 | 23.000 | 2.866x (2.849..2.872) | 0.992x (0.986..0.995) | 1.039x (1.030..1.042) | 3.200x (3.187..3.223) | parity |
| `nd-1d` | `zero-vector` | 24 | 2 | lhs: shape=(2, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(0,) | 173 | 1.809 | 1.811 | 1.653 | 1.414 | 1.760 | 0.998x (0.995..1.014) | 0.913x (0.905..0.926) | 1.171x (1.158..1.175) | 1.445x (1.438..1.465) | pack-faster |
| `nd-1d` | `zero-vector` | 24 | 4 | lhs: shape=(4, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(0,) | 86 | 2.660 | 2.647 | 1.948 | 1.718 | 2.588 | 0.996x (0.992..0.999) | 0.732x (0.728..0.739) | 1.135x (1.124..1.145) | 1.773x (1.762..1.792) | pack-faster |
| `nd-1d` | `zero-vector` | 24 | 8 | lhs: shape=(8, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(0,) | 43 | 4.299 | 4.282 | 2.513 | 2.263 | 4.295 | 0.995x (0.970..1.005) | 0.585x (0.568..0.594) | 1.111x (1.098..1.127) | 2.120x (2.109..2.135) | pack-faster |
| `nd-1d` | `zero-vector` | 24 | 16 | lhs: shape=(16, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(0,) | 21 | 7.677 | 7.659 | 3.730 | 3.498 | 7.700 | 0.997x (0.994..1.014) | 0.486x (0.480..0.489) | 1.067x (1.060..1.073) | 2.450x (2.422..2.457) | pack-faster |
| `nd-1d` | `zero-vector` | 32 | 2 | lhs: shape=(2, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(0,) | 97 | 2.519 | 2.502 | 1.835 | 1.527 | 2.534 | 0.993x (0.989..1.000) | 0.727x (0.722..0.731) | 1.202x (1.193..1.208) | 1.900x (1.897..1.915) | pack-faster |
| `nd-1d` | `zero-vector` | 32 | 4 | lhs: shape=(4, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(0,) | 48 | 2.301 | 4.065 | 2.277 | 1.977 | 4.213 | 1.766x (1.751..1.778) | 0.992x (0.986..1.000) | 1.154x (1.145..1.227) | 2.370x (2.350..2.383) | parity |
| `nd-1d` | `zero-vector` | 32 | 8 | lhs: shape=(8, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(0,) | 24 | 3.160 | 7.139 | 3.127 | 2.830 | 7.422 | 2.263x (2.254..2.298) | 0.991x (0.983..0.994) | 1.105x (1.078..1.109) | 2.913x (2.885..2.932) | parity |
| `nd-1d` | `zero-vector` | 32 | 16 | lhs: shape=(16, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(0,) | 12 | 5.844 | 13.767 | 5.788 | 5.542 | 14.399 | 2.358x (2.347..2.365) | 0.992x (0.987..0.996) | 1.046x (1.038..1.054) | 2.784x (2.706..2.834) | parity |
| `nd-1d` | `zero-vector` | 40 | 2 | lhs: shape=(2, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(0,) | 62 | 3.694 | 3.675 | 2.012 | 1.675 | 3.644 | 0.995x (0.990..1.005) | 0.545x (0.541..0.548) | 1.201x (1.195..1.206) | 2.448x (2.419..2.487) | pack-faster |
| `nd-1d` | `zero-vector` | 40 | 4 | lhs: shape=(4, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(0,) | 31 | 2.593 | 6.367 | 2.577 | 2.237 | 6.422 | 2.456x (2.448..2.494) | 0.992x (0.986..0.997) | 1.147x (1.139..1.158) | 3.100x (2.968..3.162) | parity |
| `nd-1d` | `zero-vector` | 40 | 8 | lhs: shape=(8, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(0,) | 15 | 4.089 | 11.931 | 4.039 | 3.731 | 12.072 | 2.916x (2.908..2.937) | 0.988x (0.981..0.993) | 1.082x (1.053..1.093) | 3.398x (3.370..3.474) | parity |
| `nd-1d` | `zero-vector` | 40 | 16 | lhs: shape=(16, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(0,) | 7 | 7.970 | 22.833 | 7.881 | 7.595 | 22.994 | 2.865x (2.858..2.873) | 0.989x (0.986..0.994) | 1.040x (1.034..1.044) | 3.197x (3.183..3.215) | parity |

## Reproduce

```console
$ source /path/to/devenv/scripts/init
$ devenv use prime
$ make BUILD_QT=OFF
$ PYTHONPATH=.:profiling python3 \
    profiling/profile_matmul_pack_crossover.py \
    --sides 24,32,40 \
    --batches 2,4,8,16 \
    --repeat 15 \
    --warmup 5 --filter negative-vector --filter negative-step2-vector --filter zero-vector \
    --output /tmp/matmul-pack-crossover.json
$ PYTHONPATH=.:profiling python3 \
    profiling/render_matmul_pack_crossover.py \
    /tmp/matmul-pack-crossover.json \
    /tmp/matmul-pack-crossover.md
```

On macOS, omit `--cpu`.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
