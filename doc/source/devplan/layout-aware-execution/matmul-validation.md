# Matmul dispatch and broadcasting validation

## Question

`SimpleArray::matmul()` already has rank-1 and rank-2 implementations,
including an explicit `matmul_blas()` API.  The prototype adds NumPy-style
batch broadcasting and signed-stride matrix views.  It must not make the
existing vector and matrix topologies slower or less correct.

The validation therefore covers all of these roles:

| Lhs | Rhs | Result | Meaning |
| --- | --- | --- | --- |
| `(K)` | `(K)` | scalar | DOT |
| `(K)` | `(K,N)` | `(N)` | vector-matrix |
| `(M,K)` | `(K)` | `(M)` | matrix-vector |
| `(M,K)` | `(K,N)` | `(M,N)` | matrix-matrix |
| `(K)` | `(...,K,N)` | `(...,N)` | 1D by ND broadcast |
| `(...,M,K)` | `(K)` | `(...,M)` | ND by 1D broadcast |
| `(...,M,K)` | `(...,K,N)` | `(...,M,N)` | batch broadcast |

For every role, the profiler checks dense and applicable non-contiguous
layouts before timing.  The non-contiguous inventory includes negative
strides, step-two strides, F-contiguous matrices, and padded leading
dimensions.

## Dispatch boundary

```text
SimpleArray::matmul(rhs)
|
+-- validate vector/matrix roles and contracted K
+-- derive common leading batch shape
|
+-- unbatched
|   +-- small C-contiguous work ----------> direct typed kernel
|   +-- positive-stride vectors ----------> DOT/GEMV with increment
|   +-- BLAS-describable matrix ----------> GEMV/GEMM with descriptor
|   +-- profitable unsupported operand ---> pack once, then BLAS
|   `-- remaining signed strides ---------> generic contraction
|
`-- batched
    +-- dense or BLAS-describable matrix --> one BLAS call per output
    +-- reusable negative vector ---------> pack vector once
    +-- positive-stride vector -----------> GEMV with increment
    `-- unsupported batched matrix -------> generic contraction
```

The public operation has one semantic entry point.  Internally, it keeps
short DOT, GEMV, and GEMM routes rather than forcing all matrix topologies
through one universal loop.

Whole-batch packing is deliberately absent.  The profiler showed that
packing every step-two matrix in a batch costs more than signed-stride
execution.  A repeated vector is different: packing one physical vector once
can be amortized across all output matrices.

## Correctness finding

The expanded tests found an existing bug before measuring performance.
Legacy rank-1 matmul treated negative and step-two vectors as contiguous:

```python
lhs = np.arange(8.0)[::-1]
rhs = np.arange(8.0)[::-1]

expected = np.matmul(lhs, rhs)  # 140
actual = to_simple(lhs).matmul(to_simple(rhs))  # 56
```

The planned route now uses signed-stride traversal for negative DOT and a
positive BLAS increment for positive step-two DOT.  An incorrect legacy or
existing-BLAS result is recorded as `legacy-incorrect` and is not timed.

Every planned result in both benchmark files passed the NumPy comparison.

## Reproduce the standard topology suite

```console
$ source /path/to/devenv/scripts/init
$ devenv use prime
$ make
$ python3 profiling/profile_execution_prototype.py \
    --matmul-only \
    --benchmark-only \
    --repeat 15 \
    --warmup 5 \
    --cpu 0 \
    --output /tmp/ubuntu-matmul-topology-results.json
$ python3 profiling/render_execution_profile.py \
    /tmp/ubuntu-matmul-topology-results.json \
    /tmp/ubuntu-matmul-topology-results.md
```

The complete 47-row report is
[Ubuntu matmul topology results](ubuntu-matmul-topology-results.md).
The JSON file beside it retains the raw samples and environment metadata for
machine processing.

Selected results:

| Topology and layout | NumPy | Control | Planned | NumPy/planned |
| --- | ---: | ---: | ---: | ---: |
| `(65536) @ (65536)`, negative vectors | 0.0972 ms | 0.0104 ms dense | 0.0254 ms | 3.834x |
| `(65536) @ (65536)`, step-two vectors | 0.0980 ms | 0.0110 ms dense | 0.0538 ms | 1.835x |
| `(256) @ (256,256)`, F matrix | 0.1136 ms | 0.0068 ms dense | 0.0063 ms | 17.624x |
| `(256) @ (256,256)`, padded matrix | 0.1061 ms | 0.0054 ms dense | 0.0061 ms | 17.413x |
| `(256,256) @ (256)`, F matrix | 0.1036 ms | 0.0052 ms dense | 0.0060 ms | 17.082x |
| `(256,256) @ (256)`, padded matrix | 0.1028 ms | 0.0054 ms dense | 0.0059 ms | 18.203x |

The standard suite contains:

- 30 unbatched rows, including DOT, GEMV, GEMM, and non-contiguous views.
- 17 small batch rows, including 1D by ND and ND by 1D broadcasts.
- 22 conclusive improvements over a correct legacy equivalent.
- Four legacy-incorrect rows.
- No measured planned regression.
- All 47 planned rows faster than NumPy on this Ubuntu environment.

## Reproduce the large 1D by ND and ND by 1D suite

```console
$ source /path/to/devenv/scripts/init
$ devenv use prime
$ make
$ python3 profiling/profile_execution_prototype.py \
    --hpc-matmul \
    --benchmark-only \
    --filter vector-batch \
    --filter batch-matrix-vector-256 \
    --filter batch-matrix-negative-vector \
    --filter batch-matrix-step2-vector \
    --filter batch-step2-matrix-vector \
    --repeat 15 \
    --warmup 5 \
    --cpu 0 \
    --output /tmp/ubuntu-matmul-vector-broadcast-results.json
$ python3 profiling/render_execution_profile.py \
    /tmp/ubuntu-matmul-vector-broadcast-results.json \
    /tmp/ubuntu-matmul-vector-broadcast-results.md
```

The complete report is
[Ubuntu vector-broadcast results](ubuntu-matmul-vector-broadcast-results.md).

Each row performs 64 contractions with `K=256`:

| Topology | Layout | NumPy | Control | Planned | NumPy/planned |
| --- | --- | ---: | ---: | ---: | ---: |
| `(256) @ (64,256,256)` | dense | 8.2461 ms | n/a | 1.4244 ms | 6.228x |
| `(256) @ (64,256,256)` | negative vector | 8.2974 ms | 1.3342 ms dense | 1.5735 ms | 5.444x |
| `(256) @ (64,256,256)` | step-two vector | 8.1699 ms | 1.3467 ms dense | 1.3556 ms | 6.157x |
| `(256) @ (64,256,256)` | step-two matrix | 15.6960 ms | 1.2442 ms dense | 12.2780 ms | 1.282x |
| `(64,256,256) @ (256)` | dense | 6.5715 ms | n/a | 1.3436 ms | 5.023x |
| `(64,256,256) @ (256)` | negative vector | 9.4817 ms | 3.5481 ms dense | 3.4017 ms | 2.718x |
| `(64,256,256) @ (256)` | step-two vector | 6.4834 ms | 1.4521 ms dense | 1.5477 ms | 4.183x |
| `(64,256,256) @ (256)` | step-two matrix | 7.4244 ms | 1.5550 ms dense | 3.8818 ms | 1.893x |

The vector-stride routes are close to their dense controls because positive
increments are passed to GEMV and a repeated negative vector is packed once.
The step-two matrix batches remain 2.5 to 9.9 times slower than their dense
controls.  They still beat NumPy on this Ubuntu run, but they identify the
remaining non-contiguous bottleneck.

## Reading the Ubuntu comparison

The clean run used revision `ea09ea2a`, Linux 6.6 on WSL2 x86-64, Python
3.12.7, NumPy 2.3.0, one thread, 15 paired samples, and five warmups.

NumPy and `_solvcon` use different matrix backends on this machine.  The
Ubuntu result is useful for correctness, topology coverage, and regression
testing against legacy `SimpleArray` and `matmul_blas()`.  It is not the
final same-backend performance claim.

The final performance statement requires rerunning both commands on Apple
Silicon, where NumPy and `_solvcon` can both use Accelerate.  A ratio greater
than one means planned is faster.  Every report includes q10 and q90 paired
ratio quantiles, so isolated medians are not treated as conclusive.

## Acceptance boundary

- Rank-1 and rank-2 results retain the existing `SimpleArray` shape contract.
- Eligible unbatched operations remain in the paired interval of
  `matmul_blas()` or improve on it.
- Small C-contiguous operations keep their direct typed route.
- Leading batch axes follow NumPy broadcasting without expanded operands.
- `1D @ ND` and `ND @ 1D` cover contiguous and non-contiguous inputs.
- Positive vector strides are represented as BLAS increments.
- A repeated negative vector is packed at most once.
- Unsupported batched matrix strides remain correct without whole-batch
  packing.
- Planning, packing, allocation, and dispatch remain inside the timed call.
- Same-backend Apple Silicon data is required before publishing an upstream
  speed claim.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
