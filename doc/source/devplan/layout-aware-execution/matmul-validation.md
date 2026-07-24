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
    +-- repeated unsupported matrix ------> pack physical matrix once
    +-- other unsupported matrices ------> pack supplied operand once
    +-- reusable negative vector ---------> pack vector once
    +-- positive-stride vector -----------> GEMV with increment
    `-- unsupported vector topology ------> generic contraction
```

The public operation has one semantic entry point.  Internally, it keeps
short DOT, GEMV, and GEMM routes rather than forcing all matrix topologies
through one universal loop.

Packing follows the supplied physical operand, not the broadcast result.  An
extent-one matrix or vector is packed once and reused through a zero batch
stride.  A non-broadcast strided batch currently packs the supplied batch
once.  The profiler measures these cases separately because their useful
packing granularity differs.

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

The clean Apple Silicon rerun used revision `0d906a8f`, 15 samples, five
warmups, one thread, and no CPU affinity.  NumPy and `_solvcon` both link to
Accelerate.  The complete data is in
[macOS matmul topology results](macos-matmul-topology-results.md), with raw
samples in the JSON file beside it.

| Comparison | Faster | Parity | Slower | Inconclusive | Not comparable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy versus planned | 24 planned | 2 | 0 | 0 | 4 legacy incorrect and 17 new only |
| NumPy versus planned | 15 planned | 17 | 9 NumPy | 6 | 0 |

The same-backend classification is more balanced than Ubuntu.  It preserves
the important regression result: every correct legacy route improves or
reaches parity, and none regresses.  The nine NumPy-faster rows are small
direct or small-batch cases, so the Ubuntu all-faster classification was a
backend comparison rather than a portable speed result.

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

The Apple Silicon report is
[macOS vector-broadcast results](macos-matmul-vector-broadcast-results.md).
All eight planned rows pass the NumPy check.  Three are conclusively faster
than NumPy and the other five are inconclusive.

| Topology | Layout | NumPy | Control | Planned | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `(256) @ (64,256,256)` | dense | 1.9126 ms | n/a | 1.9203 ms | inconclusive |
| `(256) @ (64,256,256)` | negative vector | 8.1480 ms | 1.7875 ms | 1.7790 ms | planned-faster |
| `(256) @ (64,256,256)` | step-two vector | 1.8876 ms | 1.9058 ms | 1.9016 ms | inconclusive |
| `(256) @ (64,256,256)` | step-two matrix | 10.0241 ms | 2.1242 ms | 9.1829 ms | inconclusive |
| `(64,256,256) @ (256)` | dense | 0.8406 ms | n/a | 0.8305 ms | inconclusive |
| `(64,256,256) @ (256)` | negative vector | 7.3350 ms | 0.8458 ms | 1.1041 ms | planned-faster |
| `(64,256,256) @ (256)` | step-two vector | 0.8564 ms | 0.8434 ms | 0.9382 ms | inconclusive |
| `(64,256,256) @ (256)` | step-two matrix | 7.3936 ms | 1.1719 ms | 6.1599 ms | planned-faster |

Positive and reusable negative vector layouts remain close to their dense
controls.  Step-two matrix batches remain 4.3 and 5.3 times slower than their
prepacked controls.  The same topology is therefore the remaining
non-contiguous bottleneck on both OpenBLAS and Accelerate.

## Reproduce matrix broadcast scaling

The focused benchmark answers why a broadcast matrix can remain close to a
dense batch even though its stored batch stride is large.  It compares:

- one dense lhs reused through a zero batch stride with `B` materialized lhs
  matrices;
- one negative-stride lhs with the same logical matrix prepacked outside the
  timed call;
- one step-two lhs with the same logical matrix prepacked outside the timed
  call.

All routes compute `(1, 256, 256) @ (B, 256, 256)`.  The sweep uses
`B=1,2,4,8,16,32,64,128`.

```console
$ source /path/to/devenv/scripts/init
$ devenv use prime
$ export CMAKE_PREFIX_PATH="$DEVENVPREFIX"
$ export CMAKE_ARGS="-Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)"
$ make BUILD_PATH_EXT=_benchmark BUILD_QT=OFF SOLVCON_PROFILE=OFF
$ python3 profiling/profile_matmul_broadcast_scaling.py \
    --batches 1,2,4,8,16,32,64,128 \
    --side 256 \
    --repeat 7 \
    --warmup 2 \
    --cpu 0 \
    --output /tmp/matmul-broadcast.json
$ make BUILD_PATH_EXT=_profile BUILD_QT=OFF SOLVCON_PROFILE=ON
$ python3 profiling/profile_matmul_broadcast_scaling.py \
    --batches 1,2,4,8,16,32,64,128 \
    --side 256 \
    --repeat 7 \
    --warmup 2 \
    --cpu 0 \
    --trace-only \
    --output /tmp/matmul-broadcast-trace.json
$ python3 profiling/render_matmul_broadcast_scaling.py \
    /tmp/matmul-broadcast.json \
    /tmp/matmul-broadcast.md \
    --trace /tmp/matmul-broadcast-trace.json
```

The complete timing samples, source strides, planned strides, paired
quantiles, backend linkage, and dispatch counts are in
[Ubuntu matmul broadcast scaling](ubuntu-matmul-broadcast-scaling-results.md).

The ratio below is tested route divided by its matching control.  One means
parity.

| B | Broadcast/materialized | Negative/prepacked | Step-two/prepacked |
| ---: | ---: | ---: | ---: |
| 1 | 1.104x | 1.399x | 1.456x |
| 8 | 0.956x | 0.934x | 0.977x |
| 64 | 1.045x | 1.016x | 1.020x |
| 128 | 0.980x | 1.011x | 1.010x |

For `B > 1`, the lhs batch mapping is zero.  The stored stride of 65,536
elements is not used to advance between outputs.  The separate
profile-enabled run confirms that dense and prepacked routes perform zero
packs, negative and step-two routes pack lhs once, and every route performs
exactly `B` GEMM calls.

The one-time copy is visible at `B=1` and amortized by larger batches.  The
experiment does not show that broadcasting reduces arithmetic.  Broadcast
and materialized routes both perform `B` contractions.

## Apple Silicon same-backend broadcast scaling

The clean Apple Silicon sweep used revision `0d906a8f`, 20 samples, five
warmups, and one thread.  Both linkage records identify Accelerate.  The
[complete macOS scaling report](macos-matmul-broadcast-scaling-results.md)
retains all 48 timing rows and dispatch counts.  The timing JSON and the
separate profile-build trace JSON are stored beside it.

| B | Planned broadcast/materialized | Planned negative/prepacked | Planned step-two/prepacked | NumPy negative/prepacked | NumPy step-two/prepacked |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000x | 1.260x | 1.283x | 164.650x | 165.379x |
| 8 | 0.888x | 1.220x | 1.221x | 202.920x | 200.458x |
| 64 | 0.879x | 1.017x | 1.061x | 167.699x | 167.195x |
| 128 | 0.888x | 1.026x | 1.032x | 161.427x | 161.309x |

The previous roughly 160-times result reproduces across the full sweep.
The paired controls locate it in NumPy's negative and step-two view routes,
not in broadcast planning.  NumPy's view-to-prepacked ratio remains 161
through 203 at the selected batch sizes.  Planned execution pays a 26 to
28 percent one-time packing cost at `B=1`; the ratio reaches an interval
containing parity at the larger batch sizes.

Broadcast reuse is at parity with or faster than the materialized planned
control.  It still performs the same `B` contractions.  Its advantage is
storing and reading one lhs matrix instead of `B` matrices, not reducing
arithmetic.

The separate profile build validates all 48 routes.  Negative and step-two
routes pack lhs exactly once, dense and prepacked routes pack zero times,
every route performs exactly `B` GEMM calls, and no route enters the generic
kernel.  The timing build contains no probes.

## Reading the backend comparisons

The topology and vector-broadcast reports used revision `ea09ea2a`.  The
broadcast-scaling report used clean revision `c17db004`, Linux 6.6 on WSL2
x86-64, Python 3.12.7, NumPy 2.3.0, one thread, seven paired samples, and two
warmups.

NumPy and `_solvcon` use different matrix backends on this machine.  The
broadcast-scaling linkage records show no BLAS library attached to NumPy's
core extension and OpenBLAS attached to `_solvcon`.  The Ubuntu result is
useful for correctness, topology coverage, dispatch validation, and
regression testing against legacy `SimpleArray` and `matmul_blas()`.  It is
not the final same-backend performance claim.

The Apple Silicon data supplies the same-backend comparison.  It confirms
the execution topology and the one-pack policy, while rejecting the broad
Ubuntu claim that every planned row is faster than NumPy.  It also explains
the large strided-input ratio: NumPy's view route is the outlier relative to
its own contiguous control.  It is not evidence that zero-stride
broadcasting avoids matrix arithmetic.

## Acceptance boundary

- Rank-1 and rank-2 results retain the existing `SimpleArray` shape contract.
- Eligible unbatched operations remain in the paired interval of
  `matmul_blas()` or improve on it.
- Small C-contiguous operations keep their direct typed route.
- Leading batch axes follow NumPy broadcasting without expanded operands.
- `1D @ ND` and `ND @ 1D` cover contiguous and non-contiguous inputs.
- Positive vector strides are represented as BLAS increments.
- A repeated negative vector is packed at most once.
- Repeated strided matrix operands are packed once rather than once per output
  matrix.
- Planning, packing, allocation, and dispatch remain inside the timed call.
- Same-backend Apple Silicon data must accompany any upstream speed claim and
  identify the NumPy strided-view route as the source of the large ratio.

The Apple decision gate is complete.  The prototype should now freeze rather
than widen the common layer.  The measured implementation can be split into
upstream-sized work for shared coordinate primitives, `MatmulPlan`, correct
batched execution, and automatic fast paths.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
