# Matmul pack-once crossover

## Environment

- Revision: `478a96e28cae9599d71b302727f6c3f814c7c0be`.
- Dirty tree: `false`.
- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`.
- Machine: `x86_64`.
- Python: `3.12.7`.
- NumPy: `2.3.0`.
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
| Pack faster | 25 |
| Parity | 0 |
| Inconclusive | 47 |
| Current faster | 0 |

This aggregate is a screening result.  Dispatch thresholds must
come from the detailed side and batch crossover below.

## Detailed crossover

| Topology | Layout | S | B | Supplied operands | Number | Current us | Generic us | Pack us | Prepacked us | NumPy view us | Generic/current | Pack/current | Pack/prepacked | NumPy view/prepacked | Result |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1d-nd` | `negative-vector` | 24 | 2 | lhs: shape=(24,), strides=(-1,)<br>rhs: shape=(2, 24, 24), strides=(576, 24, 1) | 173 | 0.855 | 0.768 | 0.766 | 0.636 | 3.162 | 0.916x (0.768..1.062) | 0.938x (0.819..1.059) | 1.209x (1.026..1.546) | 1.022x (0.979..1.113) | inconclusive |
| `1d-nd` | `negative-vector` | 24 | 4 | lhs: shape=(24,), strides=(-1,)<br>rhs: shape=(4, 24, 24), strides=(576, 24, 1) | 86 | 1.226 | 1.117 | 0.941 | 0.803 | 5.007 | 0.865x (0.813..1.031) | 0.791x (0.682..1.094) | 1.151x (1.104..1.733) | 1.023x (0.947..1.096) | inconclusive |
| `1d-nd` | `negative-vector` | 24 | 8 | lhs: shape=(24,), strides=(-1,)<br>rhs: shape=(8, 24, 24), strides=(576, 24, 1) | 43 | 2.245 | 1.866 | 1.338 | 1.190 | 9.270 | 0.842x (0.714..1.007) | 0.576x (0.511..0.782) | 1.112x (0.789..1.272) | 1.014x (0.850..1.092) | pack-faster |
| `1d-nd` | `negative-vector` | 24 | 16 | lhs: shape=(24,), strides=(-1,)<br>rhs: shape=(16, 24, 24), strides=(576, 24, 1) | 21 | 3.592 | 3.259 | 2.116 | 1.968 | 16.510 | 0.925x (0.665..0.961) | 0.588x (0.518..0.780) | 1.079x (0.936..1.641) | 0.998x (0.892..1.045) | pack-faster |
| `1d-nd` | `negative-vector` | 32 | 2 | lhs: shape=(32,), strides=(-1,)<br>rhs: shape=(2, 32, 32), strides=(1024, 32, 1) | 97 | 1.082 | 1.000 | 0.841 | 0.709 | 4.642 | 0.872x (0.730..0.977) | 0.753x (0.573..0.862) | 1.142x (1.035..1.276) | 1.022x (0.954..1.062) | pack-faster |
| `1d-nd` | `negative-vector` | 32 | 4 | lhs: shape=(32,), strides=(-1,)<br>rhs: shape=(4, 32, 32), strides=(1024, 32, 1) | 48 | 1.062 | 1.593 | 1.020 | 0.894 | 7.692 | 1.521x (1.171..1.738) | 0.983x (0.817..1.035) | 1.139x (1.004..1.190) | 1.003x (0.950..1.046) | inconclusive |
| `1d-nd` | `negative-vector` | 32 | 8 | lhs: shape=(32,), strides=(-1,)<br>rhs: shape=(8, 32, 32), strides=(1024, 32, 1) | 24 | 1.651 | 2.810 | 1.617 | 1.499 | 14.298 | 1.720x (1.658..1.877) | 0.973x (0.929..1.024) | 1.084x (1.006..1.124) | 1.008x (0.972..1.075) | inconclusive |
| `1d-nd` | `negative-vector` | 32 | 16 | lhs: shape=(32,), strides=(-1,)<br>rhs: shape=(16, 32, 32), strides=(1024, 32, 1) | 12 | 2.822 | 5.409 | 2.746 | 2.540 | 27.310 | 1.864x (1.528..2.046) | 0.964x (0.756..1.035) | 1.050x (0.843..1.137) | 0.984x (0.914..1.019) | inconclusive |
| `1d-nd` | `negative-vector` | 40 | 2 | lhs: shape=(40,), strides=(-1,)<br>rhs: shape=(2, 40, 40), strides=(1600, 40, 1) | 62 | 1.489 | 1.342 | 0.898 | 0.769 | 6.554 | 0.906x (0.751..1.032) | 0.613x (0.542..0.667) | 1.174x (0.950..1.301) | 1.034x (0.936..1.162) | pack-faster |
| `1d-nd` | `negative-vector` | 40 | 4 | lhs: shape=(40,), strides=(-1,)<br>rhs: shape=(4, 40, 40), strides=(1600, 40, 1) | 31 | 1.427 | 2.322 | 1.355 | 1.220 | 11.582 | 1.655x (1.460..1.891) | 0.962x (0.807..1.037) | 1.106x (1.040..1.123) | 1.022x (0.979..1.072) | inconclusive |
| `1d-nd` | `negative-vector` | 40 | 8 | lhs: shape=(40,), strides=(-1,)<br>rhs: shape=(8, 40, 40), strides=(1600, 40, 1) | 15 | 2.449 | 4.192 | 2.075 | 1.946 | 22.335 | 1.919x (1.520..2.522) | 0.893x (0.715..0.997) | 1.064x (1.029..1.121) | 1.007x (0.918..1.097) | inconclusive |
| `1d-nd` | `negative-vector` | 40 | 16 | lhs: shape=(40,), strides=(-1,)<br>rhs: shape=(16, 40, 40), strides=(1600, 40, 1) | 7 | 3.918 | 8.293 | 3.650 | 3.386 | 42.890 | 2.051x (1.709..2.289) | 0.948x (0.771..1.002) | 1.063x (0.959..1.147) | 0.996x (0.945..1.191) | inconclusive |
| `1d-nd` | `negative-step2-vector` | 24 | 2 | lhs: shape=(24,), strides=(-2,)<br>rhs: shape=(2, 24, 24), strides=(576, 24, 1) | 173 | 0.850 | 0.757 | 0.756 | 0.629 | 3.131 | 0.874x (0.788..0.965) | 0.935x (0.757..1.070) | 1.206x (1.065..1.371) | 1.021x (0.971..1.063) | inconclusive |
| `1d-nd` | `negative-step2-vector` | 24 | 4 | lhs: shape=(24,), strides=(-2,)<br>rhs: shape=(4, 24, 24), strides=(576, 24, 1) | 86 | 1.173 | 1.114 | 0.983 | 0.841 | 5.081 | 0.933x (0.695..1.023) | 0.838x (0.602..0.945) | 1.188x (1.059..1.291) | 1.011x (0.964..1.100) | pack-faster |
| `1d-nd` | `negative-step2-vector` | 24 | 8 | lhs: shape=(24,), strides=(-2,)<br>rhs: shape=(8, 24, 24), strides=(576, 24, 1) | 43 | 2.385 | 1.824 | 1.360 | 1.232 | 9.154 | 0.834x (0.714..0.938) | 0.617x (0.513..0.783) | 1.087x (0.875..1.290) | 1.057x (0.965..1.259) | pack-faster |
| `1d-nd` | `negative-step2-vector` | 24 | 16 | lhs: shape=(24,), strides=(-2,)<br>rhs: shape=(16, 24, 24), strides=(576, 24, 1) | 21 | 4.032 | 3.266 | 2.104 | 1.954 | 16.558 | 0.844x (0.698..0.970) | 0.532x (0.475..0.722) | 1.099x (0.793..1.385) | 1.034x (0.967..1.109) | pack-faster |
| `1d-nd` | `negative-step2-vector` | 32 | 2 | lhs: shape=(32,), strides=(-2,)<br>rhs: shape=(2, 32, 32), strides=(1024, 32, 1) | 97 | 1.037 | 1.000 | 0.840 | 0.722 | 4.552 | 0.939x (0.837..0.998) | 0.782x (0.682..1.137) | 1.169x (1.137..1.489) | 1.019x (0.979..1.103) | inconclusive |
| `1d-nd` | `negative-step2-vector` | 32 | 4 | lhs: shape=(32,), strides=(-2,)<br>rhs: shape=(4, 32, 32), strides=(1024, 32, 1) | 48 | 1.132 | 1.675 | 1.118 | 0.975 | 7.915 | 1.435x (1.226..1.572) | 0.953x (0.804..1.321) | 1.136x (1.035..1.418) | 1.000x (0.907..1.071) | inconclusive |
| `1d-nd` | `negative-step2-vector` | 32 | 8 | lhs: shape=(32,), strides=(-2,)<br>rhs: shape=(8, 32, 32), strides=(1024, 32, 1) | 24 | 1.705 | 2.950 | 1.665 | 1.536 | 14.823 | 1.714x (1.347..2.046) | 0.959x (0.714..1.024) | 1.072x (1.012..1.167) | 1.022x (0.957..1.082) | inconclusive |
| `1d-nd` | `negative-step2-vector` | 32 | 16 | lhs: shape=(32,), strides=(-2,)<br>rhs: shape=(16, 32, 32), strides=(1024, 32, 1) | 12 | 2.787 | 5.318 | 2.733 | 2.600 | 27.399 | 1.916x (1.683..2.267) | 0.949x (0.914..1.043) | 1.058x (0.750..1.093) | 1.013x (0.949..1.074) | inconclusive |
| `1d-nd` | `negative-step2-vector` | 40 | 2 | lhs: shape=(40,), strides=(-2,)<br>rhs: shape=(2, 40, 40), strides=(1600, 40, 1) | 62 | 1.442 | 1.320 | 0.922 | 0.826 | 6.372 | 0.914x (0.805..1.000) | 0.629x (0.549..0.794) | 1.169x (0.962..1.304) | 1.002x (0.982..1.102) | pack-faster |
| `1d-nd` | `negative-step2-vector` | 40 | 4 | lhs: shape=(40,), strides=(-2,)<br>rhs: shape=(4, 40, 40), strides=(1600, 40, 1) | 31 | 1.414 | 2.381 | 1.412 | 1.254 | 11.773 | 1.682x (1.466..1.883) | 0.988x (0.859..1.118) | 1.116x (1.054..1.340) | 0.995x (0.959..1.050) | inconclusive |
| `1d-nd` | `negative-step2-vector` | 40 | 8 | lhs: shape=(40,), strides=(-2,)<br>rhs: shape=(8, 40, 40), strides=(1600, 40, 1) | 15 | 2.194 | 4.222 | 2.098 | 1.917 | 22.098 | 1.908x (1.707..2.141) | 0.959x (0.793..0.978) | 1.089x (1.035..1.117) | 1.019x (0.930..1.073) | inconclusive |
| `1d-nd` | `negative-step2-vector` | 40 | 16 | lhs: shape=(40,), strides=(-2,)<br>rhs: shape=(16, 40, 40), strides=(1600, 40, 1) | 7 | 3.811 | 7.988 | 3.673 | 3.543 | 43.878 | 2.078x (1.303..2.318) | 0.951x (0.575..1.189) | 1.024x (0.671..1.142) | 1.029x (0.960..1.087) | inconclusive |
| `1d-nd` | `zero-vector` | 24 | 2 | lhs: shape=(24,), strides=(0,)<br>rhs: shape=(2, 24, 24), strides=(576, 24, 1) | 173 | 0.797 | 0.770 | 0.745 | 0.633 | 3.098 | 0.960x (0.751..1.137) | 0.930x (0.797..1.094) | 1.192x (1.132..1.449) | 0.990x (0.929..1.083) | inconclusive |
| `1d-nd` | `zero-vector` | 24 | 4 | lhs: shape=(24,), strides=(0,)<br>rhs: shape=(4, 24, 24), strides=(576, 24, 1) | 86 | 1.211 | 1.106 | 0.945 | 0.815 | 5.013 | 0.899x (0.756..0.967) | 0.794x (0.635..0.964) | 1.140x (1.035..1.349) | 1.030x (0.962..1.165) | inconclusive |
| `1d-nd` | `zero-vector` | 24 | 8 | lhs: shape=(24,), strides=(0,)<br>rhs: shape=(8, 24, 24), strides=(576, 24, 1) | 43 | 2.139 | 1.897 | 1.384 | 1.204 | 9.075 | 0.890x (0.721..0.978) | 0.663x (0.538..0.825) | 1.124x (1.094..1.344) | 1.009x (0.941..1.076) | pack-faster |
| `1d-nd` | `zero-vector` | 24 | 16 | lhs: shape=(24,), strides=(0,)<br>rhs: shape=(16, 24, 24), strides=(576, 24, 1) | 21 | 3.970 | 3.246 | 2.138 | 1.932 | 16.470 | 0.791x (0.683..1.057) | 0.548x (0.479..0.870) | 1.093x (0.837..1.276) | 0.989x (0.732..1.056) | pack-faster |
| `1d-nd` | `zero-vector` | 32 | 2 | lhs: shape=(32,), strides=(0,)<br>rhs: shape=(2, 32, 32), strides=(1024, 32, 1) | 97 | 1.097 | 1.001 | 0.831 | 0.689 | 4.634 | 0.928x (0.776..1.056) | 0.809x (0.654..0.935) | 1.188x (1.126..1.406) | 1.018x (0.956..1.089) | pack-faster |
| `1d-nd` | `zero-vector` | 32 | 4 | lhs: shape=(32,), strides=(0,)<br>rhs: shape=(4, 32, 32), strides=(1024, 32, 1) | 48 | 1.160 | 1.624 | 1.128 | 0.993 | 7.890 | 1.419x (1.168..1.680) | 0.971x (0.784..1.067) | 1.131x (1.075..1.239) | 0.989x (0.910..1.092) | inconclusive |
| `1d-nd` | `zero-vector` | 32 | 8 | lhs: shape=(32,), strides=(0,)<br>rhs: shape=(8, 32, 32), strides=(1024, 32, 1) | 24 | 1.845 | 3.047 | 1.713 | 1.571 | 14.760 | 1.666x (1.376..2.086) | 0.924x (0.797..1.122) | 1.088x (1.033..1.283) | 1.033x (0.973..1.116) | inconclusive |
| `1d-nd` | `zero-vector` | 32 | 16 | lhs: shape=(32,), strides=(0,)<br>rhs: shape=(16, 32, 32), strides=(1024, 32, 1) | 12 | 3.463 | 5.457 | 2.946 | 2.736 | 29.188 | 1.811x (1.147..1.981) | 0.900x (0.585..1.036) | 1.060x (0.979..1.179) | 1.022x (0.849..1.056) | inconclusive |
| `1d-nd` | `zero-vector` | 40 | 2 | lhs: shape=(40,), strides=(0,)<br>rhs: shape=(2, 40, 40), strides=(1600, 40, 1) | 62 | 1.437 | 1.335 | 0.935 | 0.791 | 6.406 | 0.928x (0.815..1.031) | 0.645x (0.612..0.811) | 1.182x (1.127..1.281) | 1.008x (0.962..1.139) | pack-faster |
| `1d-nd` | `zero-vector` | 40 | 4 | lhs: shape=(40,), strides=(0,)<br>rhs: shape=(4, 40, 40), strides=(1600, 40, 1) | 31 | 1.452 | 2.408 | 1.407 | 1.272 | 11.487 | 1.653x (1.369..1.738) | 0.974x (0.779..1.026) | 1.113x (1.062..1.167) | 1.004x (0.946..1.088) | inconclusive |
| `1d-nd` | `zero-vector` | 40 | 8 | lhs: shape=(40,), strides=(0,)<br>rhs: shape=(8, 40, 40), strides=(1600, 40, 1) | 15 | 2.403 | 4.217 | 2.127 | 1.939 | 21.873 | 1.871x (1.424..2.025) | 0.920x (0.712..0.993) | 1.067x (0.740..1.131) | 1.016x (0.930..1.066) | inconclusive |
| `1d-nd` | `zero-vector` | 40 | 16 | lhs: shape=(40,), strides=(0,)<br>rhs: shape=(16, 40, 40), strides=(1600, 40, 1) | 7 | 4.197 | 7.975 | 3.573 | 3.446 | 43.545 | 1.875x (1.336..2.273) | 0.849x (0.509..0.958) | 1.041x (0.963..1.119) | 0.987x (0.947..1.144) | inconclusive |
| `nd-1d` | `negative-vector` | 24 | 2 | lhs: shape=(2, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-1,) | 173 | 0.842 | 0.791 | 0.756 | 0.629 | 3.185 | 0.944x (0.879..1.077) | 0.943x (0.836..1.111) | 1.193x (1.090..1.473) | 1.006x (0.959..1.066) | inconclusive |
| `nd-1d` | `negative-vector` | 24 | 4 | lhs: shape=(4, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-1,) | 86 | 1.214 | 1.144 | 0.945 | 0.833 | 5.172 | 0.944x (0.843..0.990) | 0.778x (0.686..0.935) | 1.129x (0.980..1.299) | 1.042x (0.981..1.084) | pack-faster |
| `nd-1d` | `negative-vector` | 24 | 8 | lhs: shape=(8, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-1,) | 43 | 2.143 | 2.061 | 1.356 | 1.302 | 9.200 | 0.931x (0.858..1.102) | 0.627x (0.567..0.744) | 1.068x (0.925..1.230) | 1.005x (0.953..1.098) | pack-faster |
| `nd-1d` | `negative-vector` | 24 | 16 | lhs: shape=(16, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-1,) | 21 | 3.644 | 3.378 | 2.033 | 1.932 | 16.754 | 0.939x (0.850..1.098) | 0.557x (0.506..0.621) | 1.061x (0.899..1.105) | 1.006x (0.972..1.090) | pack-faster |
| `nd-1d` | `negative-vector` | 32 | 2 | lhs: shape=(2, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-1,) | 97 | 1.091 | 1.162 | 1.022 | 0.721 | 4.719 | 0.959x (0.859..1.155) | 0.799x (0.693..1.064) | 1.240x (1.139..1.591) | 1.038x (0.962..1.113) | inconclusive |
| `nd-1d` | `negative-vector` | 32 | 4 | lhs: shape=(4, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-1,) | 48 | 1.493 | 2.283 | 1.312 | 1.072 | 9.321 | 1.413x (1.152..1.756) | 0.861x (0.667..1.023) | 1.122x (0.984..1.299) | 1.006x (0.900..1.132) | inconclusive |
| `nd-1d` | `negative-vector` | 32 | 8 | lhs: shape=(8, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-1,) | 24 | 1.821 | 3.373 | 1.680 | 1.634 | 15.084 | 1.901x (1.556..2.308) | 0.932x (0.770..1.107) | 1.077x (0.813..1.115) | 0.990x (0.924..1.057) | inconclusive |
| `nd-1d` | `negative-vector` | 32 | 16 | lhs: shape=(16, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-1,) | 12 | 2.952 | 5.599 | 2.675 | 2.590 | 28.724 | 1.918x (1.783..2.244) | 0.914x (0.773..0.974) | 1.048x (0.917..1.098) | 1.020x (0.990..1.125) | inconclusive |
| `nd-1d` | `negative-vector` | 40 | 2 | lhs: shape=(2, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-1,) | 62 | 1.388 | 1.348 | 0.885 | 0.766 | 6.364 | 0.968x (0.913..1.066) | 0.638x (0.608..0.750) | 1.156x (1.094..1.245) | 1.005x (0.966..1.122) | pack-faster |
| `nd-1d` | `negative-vector` | 40 | 4 | lhs: shape=(4, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-1,) | 31 | 1.408 | 2.455 | 1.347 | 1.206 | 11.622 | 1.758x (1.696..2.560) | 0.952x (0.883..1.188) | 1.119x (1.059..1.170) | 1.012x (0.930..1.050) | inconclusive |
| `nd-1d` | `negative-vector` | 40 | 8 | lhs: shape=(8, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-1,) | 15 | 2.073 | 4.279 | 2.042 | 1.896 | 22.100 | 2.054x (1.663..2.168) | 0.998x (0.732..1.302) | 1.063x (0.975..1.335) | 1.009x (0.904..1.040) | inconclusive |
| `nd-1d` | `negative-vector` | 40 | 16 | lhs: shape=(16, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-1,) | 7 | 3.634 | 7.785 | 3.293 | 3.204 | 42.391 | 2.133x (1.826..2.335) | 0.903x (0.775..0.939) | 1.039x (0.723..1.062) | 1.010x (0.979..1.113) | pack-faster |
| `nd-1d` | `negative-step2-vector` | 24 | 2 | lhs: shape=(2, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-2,) | 173 | 0.737 | 0.743 | 0.697 | 0.589 | 2.951 | 0.975x (0.938..1.048) | 0.949x (0.879..1.020) | 1.175x (1.095..1.308) | 1.005x (0.972..1.061) | inconclusive |
| `nd-1d` | `negative-step2-vector` | 24 | 4 | lhs: shape=(4, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-2,) | 86 | 1.184 | 1.224 | 0.922 | 0.828 | 5.138 | 1.020x (0.842..1.136) | 0.771x (0.718..0.805) | 1.116x (0.986..1.163) | 1.021x (0.990..1.105) | pack-faster |
| `nd-1d` | `negative-step2-vector` | 24 | 8 | lhs: shape=(8, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-2,) | 43 | 2.263 | 2.204 | 1.355 | 1.232 | 9.536 | 0.992x (0.818..1.139) | 0.632x (0.507..0.675) | 1.115x (0.933..1.154) | 0.991x (0.940..1.251) | pack-faster |
| `nd-1d` | `negative-step2-vector` | 24 | 16 | lhs: shape=(16, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(-2,) | 21 | 3.841 | 3.918 | 2.140 | 1.961 | 17.602 | 0.976x (0.785..1.158) | 0.549x (0.461..0.831) | 1.105x (1.031..1.386) | 1.020x (0.899..1.192) | pack-faster |
| `nd-1d` | `negative-step2-vector` | 32 | 2 | lhs: shape=(2, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-2,) | 97 | 1.135 | 1.231 | 0.971 | 0.732 | 5.089 | 1.007x (0.804..1.162) | 0.768x (0.706..1.086) | 1.199x (0.717..1.486) | 1.000x (0.706..1.216) | inconclusive |
| `nd-1d` | `negative-step2-vector` | 32 | 4 | lhs: shape=(4, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-2,) | 48 | 1.222 | 1.818 | 1.100 | 0.977 | 8.082 | 1.494x (1.267..1.727) | 0.912x (0.705..1.206) | 1.117x (1.098..1.268) | 1.011x (0.974..1.164) | inconclusive |
| `nd-1d` | `negative-step2-vector` | 32 | 8 | lhs: shape=(8, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-2,) | 24 | 1.699 | 3.091 | 1.635 | 1.563 | 14.896 | 1.782x (1.483..2.067) | 0.962x (0.762..1.037) | 1.075x (0.933..1.128) | 1.011x (0.936..1.097) | inconclusive |
| `nd-1d` | `negative-step2-vector` | 32 | 16 | lhs: shape=(16, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(-2,) | 12 | 2.757 | 5.394 | 2.634 | 2.522 | 28.016 | 1.949x (1.867..2.218) | 0.930x (0.903..1.178) | 1.049x (1.001..1.252) | 1.008x (0.969..1.060) | inconclusive |
| `nd-1d` | `negative-step2-vector` | 40 | 2 | lhs: shape=(2, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-2,) | 62 | 1.385 | 1.356 | 0.905 | 0.762 | 6.306 | 0.979x (0.818..1.135) | 0.645x (0.608..0.774) | 1.175x (1.024..1.284) | 0.996x (0.927..1.088) | pack-faster |
| `nd-1d` | `negative-step2-vector` | 40 | 4 | lhs: shape=(4, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-2,) | 31 | 1.370 | 2.375 | 1.317 | 1.213 | 11.600 | 1.721x (1.648..1.832) | 0.961x (0.907..1.112) | 1.100x (0.950..1.246) | 1.029x (0.948..1.096) | inconclusive |
| `nd-1d` | `negative-step2-vector` | 40 | 8 | lhs: shape=(8, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-2,) | 15 | 2.342 | 4.368 | 2.058 | 1.934 | 22.024 | 1.875x (1.653..2.111) | 0.901x (0.734..0.987) | 1.067x (0.875..1.123) | 0.999x (0.919..1.120) | inconclusive |
| `nd-1d` | `negative-step2-vector` | 40 | 16 | lhs: shape=(16, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(-2,) | 7 | 4.207 | 8.798 | 3.607 | 3.369 | 43.503 | 2.061x (1.420..2.764) | 0.908x (0.664..1.236) | 1.074x (0.986..1.332) | 1.019x (0.936..1.183) | inconclusive |
| `nd-1d` | `zero-vector` | 24 | 2 | lhs: shape=(2, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(0,) | 173 | 0.806 | 0.810 | 0.755 | 0.645 | 3.172 | 1.021x (0.919..1.152) | 0.948x (0.872..1.173) | 1.189x (1.113..1.371) | 1.036x (0.955..1.149) | inconclusive |
| `nd-1d` | `zero-vector` | 24 | 4 | lhs: shape=(4, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(0,) | 86 | 1.222 | 1.184 | 0.947 | 0.839 | 5.221 | 0.981x (0.886..1.101) | 0.822x (0.731..0.991) | 1.149x (0.905..1.399) | 1.027x (0.934..1.120) | inconclusive |
| `nd-1d` | `zero-vector` | 24 | 8 | lhs: shape=(8, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(0,) | 43 | 2.097 | 1.976 | 1.315 | 1.241 | 9.147 | 0.950x (0.891..1.028) | 0.631x (0.584..0.683) | 1.067x (0.879..1.155) | 1.003x (0.900..1.070) | pack-faster |
| `nd-1d` | `zero-vector` | 24 | 16 | lhs: shape=(16, 24, 24), strides=(576, 24, 1)<br>rhs: shape=(24,), strides=(0,) | 21 | 3.705 | 3.664 | 2.012 | 1.866 | 16.769 | 0.900x (0.836..1.132) | 0.540x (0.486..0.574) | 1.051x (0.877..1.106) | 1.013x (0.926..1.058) | pack-faster |
| `nd-1d` | `zero-vector` | 32 | 2 | lhs: shape=(2, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(0,) | 97 | 1.065 | 1.115 | 0.839 | 0.700 | 4.692 | 1.018x (0.904..1.133) | 0.821x (0.703..0.948) | 1.216x (1.145..1.455) | 1.011x (0.982..1.165) | pack-faster |
| `nd-1d` | `zero-vector` | 32 | 4 | lhs: shape=(4, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(0,) | 48 | 1.148 | 1.861 | 1.107 | 0.976 | 8.111 | 1.546x (1.359..1.762) | 0.933x (0.878..1.001) | 1.113x (1.052..1.199) | 1.005x (0.916..1.052) | inconclusive |
| `nd-1d` | `zero-vector` | 32 | 8 | lhs: shape=(8, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(0,) | 24 | 1.813 | 3.086 | 1.678 | 1.548 | 15.306 | 1.607x (1.125..1.908) | 0.934x (0.572..1.007) | 1.074x (0.830..1.107) | 0.991x (0.920..1.089) | inconclusive |
| `nd-1d` | `zero-vector` | 32 | 16 | lhs: shape=(16, 32, 32), strides=(1024, 32, 1)<br>rhs: shape=(32,), strides=(0,) | 12 | 2.827 | 5.665 | 2.701 | 2.489 | 28.409 | 1.894x (1.685..2.203) | 0.937x (0.795..1.093) | 1.070x (1.020..1.165) | 0.994x (0.914..1.097) | inconclusive |
| `nd-1d` | `zero-vector` | 40 | 2 | lhs: shape=(2, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(0,) | 62 | 1.422 | 1.368 | 0.904 | 0.774 | 6.453 | 0.966x (0.870..1.093) | 0.641x (0.579..0.722) | 1.164x (0.878..1.243) | 1.007x (0.947..1.063) | pack-faster |
| `nd-1d` | `zero-vector` | 40 | 4 | lhs: shape=(4, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(0,) | 31 | 1.387 | 2.433 | 1.335 | 1.232 | 11.568 | 1.746x (1.489..1.813) | 0.963x (0.876..1.036) | 1.087x (1.050..1.229) | 1.053x (0.936..1.225) | inconclusive |
| `nd-1d` | `zero-vector` | 40 | 8 | lhs: shape=(8, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(0,) | 15 | 2.177 | 4.278 | 2.017 | 1.949 | 21.924 | 1.974x (1.712..2.098) | 0.960x (0.733..1.299) | 1.056x (1.011..1.388) | 1.014x (0.954..1.089) | inconclusive |
| `nd-1d` | `zero-vector` | 40 | 16 | lhs: shape=(16, 40, 40), strides=(1600, 40, 1)<br>rhs: shape=(40,), strides=(0,) | 7 | 3.500 | 7.961 | 3.400 | 3.242 | 42.475 | 2.241x (1.810..3.296) | 0.991x (0.705..1.459) | 1.068x (0.794..1.295) | 1.016x (0.914..1.175) | inconclusive |

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
    --warmup 5 --cpu 0 --filter negative-vector --filter negative-step2-vector --filter zero-vector \
    --output /tmp/matmul-pack-crossover.json
$ PYTHONPATH=.:profiling python3 \
    profiling/render_matmul_pack_crossover.py \
    /tmp/matmul-pack-crossover.json \
    /tmp/matmul-pack-crossover.md
```

On macOS, omit `--cpu`.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
