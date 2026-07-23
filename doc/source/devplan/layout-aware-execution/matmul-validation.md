# Matmul dispatch and broadcasting validation

## Question

`SimpleArray::matmul()` already has direct rank-1 and rank-2 implementations,
including an explicit `matmul_blas()` API.  The proposed extension adds NumPy
batch broadcasting and support for signed-stride matrix views.

The extension is useful only if it satisfies both requirements:

1. Existing dot, matrix-vector, vector-matrix, and matrix-matrix calls retain
   their short direct routes.
2. Batched and strided calls gain broadcasting, pack-once reuse, and a generic
   fallback without expanding broadcast operands.

The automatic route is called "planned" in the prototype.  This name does not
mean that every call constructs a runtime batch plan.  It means that one
semantic API selects an execution topology internally.

## Dispatch boundary

```text
SimpleArray::matmul(rhs)
|
+-- no leading batch axes
|   |
|   +-- small work ----------------------> direct typed kernel
|   +-- C-contiguous float/complex ------> existing DOT/GEMV/GEMM route
|   +-- BLAS-describable 2D strides -----> transpose/leading-dimension BLAS
|   +-- unsupported profitable strides --> pack once, then BLAS
|   `-- remaining case ------------------> generic contraction
|
`-- one or more leading batch axes
    |
    +-- MatmulPlan
    |   +-- vector and matrix roles
    |   +-- M, N, and contracted K
    |   +-- common broadcast batch shape
    |   `-- signed or zero batch strides
    |
    +-- BLAS-describable matrix strides --> one BLAS call per output matrix
    +-- repeated unsupported operand ----> pack physical operand once
    `-- remaining case ------------------> generic signed-stride contraction
```

The unbatched branch does not construct a batch cursor.  Dense unbatched
inputs call the same helper and BLAS wrapper as `matmul_blas()`.  The batch
plan exists only where leading-axis broadcasting needs it.

## What is compared

The profiler distinguishes four routes:

| Route | Meaning |
| --- | --- |
| NumPy | `numpy.matmul()` correctness reference and external baseline |
| Legacy | Current `SimpleArray::matmul()` |
| Existing BLAS | Current `SimpleArray::matmul_blas()` |
| Planned | Prototype automatic dispatch |

`matmul_blas()` is measured only where the current API accepts an unbatched
rank-1 or rank-2 operation.  Batched calls have no existing BLAS API baseline.

For repeated negative and step-two matrices, the profiler also measures a
dense planned control in the same rotating sample.  This comparison answers a
different question from NumPy/planned: whether pack-once adds unexpected cost
after both routes reach the same BLAS backend.

Every route is checked against NumPy before timing.  A mismatch prevents that
route from entering the performance sample.

## Reproduce the direct rank-1 and rank-2 comparison

```console
$ source /path/to/devenv/scripts/init
$ devenv use prime
$ make
$ python3 profiling/profile_execution_prototype.py \
    --benchmark-only \
    --filter matrix-matrix-c \
    --filter matrix-matrix-small-direct \
    --filter vector-vector \
    --filter vector-matrix \
    --filter matrix-vector \
    --repeat 31 \
    --warmup 3 \
    --output /tmp/solvcon-matmul-direct.json
$ python3 profiling/render_execution_profile.py \
    /tmp/solvcon-matmul-direct.json \
    /tmp/solvcon-matmul-direct.md
```

The Ubuntu run used revision `c26a2bbc`, Python 3.12.7, NumPy 2.3.0,
OpenBLAS for `_solvcon`, one thread, 31 paired samples, and three warmups.
NumPy on this machine does not use the same BLAS backend, so the existing
SimpleArray BLAS route is the primary regression baseline.

| Operation | Shape | Legacy | Existing BLAS | Planned | BLAS/planned interval |
| --- | --- | ---: | ---: | ---: | --- |
| Matrix-matrix | `(256,256) @ (256,256)` | 7.0845 ms | 0.5689 ms | 0.5742 ms | `0.965x (0.845..1.126)`, inconclusive |
| Small matrix-matrix | `(8,16) @ (16,8)` | 0.0007 ms | 0.0007 ms | 0.0007 ms | `0.931x (0.737..1.099)`, inconclusive |
| Vector-vector | `(256) @ (256)` | 0.0005 ms | 0.0004 ms | 0.0005 ms | `0.948x (0.848..0.992)`, inconclusive |
| Vector-matrix | `(256) @ (256,256)` | 0.0289 ms | 0.0061 ms | 0.0064 ms | `0.939x (0.892..1.065)`, inconclusive |
| Matrix-vector | `(256,256) @ (256)` | 0.0207 ms | 0.0045 ms | 0.0045 ms | `0.991x (0.861..1.158)`, inconclusive |

None of the five topologies shows a conclusive planned regression against the
existing BLAS API.  The sub-microsecond rows should be interpreted with their
absolute times as well as their ratios.

## Reproduce the large and pack-once comparison

```console
$ source /path/to/devenv/scripts/init
$ devenv use prime
$ make
$ python3 profiling/profile_execution_prototype.py \
    --hpc-matmul \
    --benchmark-only \
    --filter large-square-1024-c \
    --filter broadcast-dense-lhs-256-b64 \
    --filter broadcast-negative-lhs-256-b64 \
    --filter broadcast-step2-lhs-256-b64 \
    --repeat 7 \
    --warmup 2 \
    --output /tmp/solvcon-matmul-routes.json
$ python3 profiling/render_execution_profile.py \
    /tmp/solvcon-matmul-routes.json \
    /tmp/solvcon-matmul-routes.md
```

The Ubuntu route run used clean revision `a8dacc01`, one thread, seven paired
samples, and two warmups.

| Operation | NumPy | Legacy | Existing BLAS or dense control | Planned | Paired comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| `(1024,1024) @ (1024,1024)` | 5272.865 ms | 3673.318 ms | 33.556 ms BLAS | 32.026 ms | `1.062x (0.995..1.128)`, inconclusive |
| Dense `(1,256,256) @ (64,256,256)` | 1734.433 ms | n/a | n/a | 47.086 ms | n/a |
| Negative `(1,256,256) @ (64,256,256)` | 1755.809 ms | n/a | 47.487 ms dense control | 47.856 ms | `0.996x (0.964..1.071)`, inconclusive |
| Step-two `(1,256,256) @ (64,256,256)` | 1728.480 ms | n/a | 48.533 ms dense control | 51.361 ms | `0.934x (0.882..0.966)`, inconclusive |

The 1024-square automatic route remains in the same paired interval as
`matmul_blas()`.  The repeated negative and step-two routes are also
inconclusive against their dense planned controls.  Packing one 0.5 MiB
physical lhs does not create the earlier apparent 30 percent speedup.

Ubuntu is a portability and same-binary check, not the primary NumPy
comparison.  The NumPy extension and `_solvcon` use different matrix
backends on this machine.

## Existing Apple Silicon evidence and required rerun

The earlier Apple Silicon run linked both NumPy and `_solvcon` to Accelerate.
It showed:

| Operation | NumPy | Planned | Result |
| --- | ---: | ---: | --- |
| Dense `(1,256,256) @ (64,256,256)` | 26.409 ms | 26.760 ms | parity |
| Negative `(1,256,256) @ (64,256,256)` | 2961.193 ms | 17.748 ms | planned faster |
| Step-two `(1,256,256) @ (64,256,256)` | 3233.026 ms | 18.894 ms | planned faster |

Those NumPy/planned ratios remain valid within each layout, but the run did
not measure `matmul_blas()` and did not pair the two strided planned routes
with a dense planned control.  It therefore cannot support a claim that
packing is faster than dense input.

Rerun both commands above on Apple Silicon.  The new profiler records
`matmul_blas()` for unbatched inputs and the dense planned control for both
strided broadcast cases.  Use at least 20 samples for the four focused HPC
rows when time permits.

## Interpretation

The evidence supports the following architecture:

- Preserve a direct unbatched executor for rank-1 and rank-2 operations.
- Reuse the existing DOT, GEMV, and GEMM wrappers instead of routing dense
  calls through a batch cursor.
- Construct `MatmulPlan` only when leading batch axes exist or a signed-stride
  2D descriptor is needed.
- Keep layout classification and packing shared between unbatched and batched
  execution.
- Pack each unsupported physical broadcast operand once, then reuse it across
  every output matrix.
- Keep backend thresholds internal and revise them only with same-backend
  measurements.

The design unifies the semantic API and the low-level layout vocabulary.  It
does not force every matrix topology through one loop nest.

## Acceptance boundary

- Rank-1 and rank-2 results retain the existing `SimpleArray` shape contract.
- Eligible unbatched operations remain within the paired interval of
  `matmul_blas()`.
- Small operations retain a direct typed route when BLAS call overhead is not
  profitable.
- Leading batch axes follow NumPy broadcasting without materializing expanded
  operands.
- Negative and step-two repeated matrices are packed at most once.
- Packing, planning, allocation, and dispatch remain inside the timed call.
- Same-backend Apple Silicon data is required before publishing a performance
  claim upstream.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
