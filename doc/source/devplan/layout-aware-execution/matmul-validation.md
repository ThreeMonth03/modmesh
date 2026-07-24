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

## Exhaustive layout-pair validation

One-sided layout samples are not enough to validate matmul dispatch.  The
lhs and rhs descriptors jointly determine whether NumPy or planned execution
uses a direct kernel, BLAS descriptor, packing, or generic traversal.  In
particular, a `2D @ ND` call may follow a different NumPy path when either
operand stops being C-contiguous.

The Cartesian profiler therefore declares operand layout catalogs and pairs
every legal lhs entry with every legal rhs entry for each topology:

- Vectors use strides `1`, `-1`, `2`, `-2`, and `0`.
- Matrix cores use C, per-matrix F, step-two and negative choices on either
  matrix axis, matching changes on both matrix axes, and zero strides on
  either matrix axis.
- Every leading batch axis independently uses C, negative, step-two, or zero
  stride traversal.

A rank-two matrix has 10 core layouts.  A rank-three matrix has 40 layouts,
and a rank-four matrix has 160 layouts.  The final lhs by rhs inventory is:

| Topology | Lhs role | Rhs role | Layout pairs |
| --- | --- | --- | ---: |
| `1d-1d` | vector | vector | 25 |
| `1d-2d` | vector | matrix | 50 |
| `2d-1d` | matrix | vector | 50 |
| `2d-2d` | matrix | matrix | 100 |
| `1d-nd` | vector | one-axis matrix batch | 200 |
| `nd-1d` | one-axis matrix batch | vector | 200 |
| `2d-nd` | matrix | one-axis matrix batch | 400 |
| `nd-2d` | one-axis matrix batch | matrix | 400 |
| `nd-nd-same-batch` | one-axis matrix batch | same batch | 1,600 |
| `nd-nd-lhs-broadcast` | extent-one lhs batch | rhs batch | 1,600 |
| `nd-nd-rhs-broadcast` | lhs batch | extent-one rhs batch | 1,600 |
| `nd-nd-cross-broadcast` | rank-four lhs | rank-four rhs | 25,600 |
| **Total** | | | **31,825** |

Run the small complete differential check before timing:

```console
$ PYTHONPATH=.:profiling python3 \
    profiling/profile_matmul_cartesian.py \
    --side 4 --batch 3 --check-only --cpu 0
```

Run the complete performance matrix and render the exact shape, element
strides, flags, medians, raw ratio intervals, and status for every pair:

```console
$ PYTHONPATH=.:profiling python3 \
    profiling/profile_matmul_cartesian.py \
    --side 32 --batch 4 --number 1 \
    --repeat 7 --warmup 2 --cpu 0 \
    --output /tmp/matmul-cartesian.json
$ PYTHONPATH=.:profiling python3 \
    profiling/render_execution_profile.py \
    /tmp/matmul-cartesian.json \
    /tmp/matmul-cartesian.md
```

The clean Ubuntu run used revision `2f95e8bf`.  All 31,825 results matched
NumPy.  Planned execution was conclusively faster than NumPy in 31,814 rows
and inconclusive in 11.  NumPy and solvcon use different backends on this
machine, so this classification validates coverage and exposes dispatch
changes.  It is not a portable speed claim.

Microsecond dispatch thresholds need more calls per sample than the complete
matrix can afford.  The focused command retains all 400 vector-batch layout
pairs and executes each route 100 times in each of 15 paired samples:

```console
$ PYTHONPATH=.:profiling python3 \
    profiling/profile_matmul_cartesian.py \
    --side 32 --batch 4 --number 100 \
    --repeat 15 --warmup 5 --cpu 0 \
    --filter 1d-nd/ --filter nd-1d/ \
    --output /tmp/matmul-vector-cartesian.json
```

`Generic/current` above one means current dispatch is faster than forced
generic execution.  `BLAS/current` near one confirms that current selected
the intended BLAS route.

| Layout class | Pairs | Generic/current | BLAS/current | Current route |
| --- | ---: | ---: | ---: | --- |
| Positive vector stride and direct GEMV matrix | 48 | 1.518x (1.270..1.733) | 0.980x (0.949..0.999) | BLAS |
| Negative or zero vector and direct GEMV matrix | 72 | 0.970x (0.950..1.013) | 0.660x (0.623..0.715) | Generic |
| Matrix not directly describable by GEMV | 280 | 0.957x (0.931..0.977) | 0.998x (0.973..1.022) | Generic |

The small automatic fast path is deliberately limited to positive vector
strides and matrices that GEMV can describe without packing.  The first row
shows a 1.234 through 1.758 improvement across every individual layout pair.
Negative and zero vectors can benefit from pack-once execution on Ubuntu, but
their crossover depends on matrix size, batch size, and BLAS backend.  The
forced controls remain available for the Apple Silicon run before that
policy is widened.

### Apple Silicon Cartesian decision gate

Use the same revision and omit `--cpu` on macOS:

```console
$ PYTHONPATH=.:profiling python3 \
    profiling/profile_matmul_cartesian.py \
    --side 4 --batch 3 --check-only
$ PYTHONPATH=.:profiling python3 \
    profiling/profile_matmul_cartesian.py \
    --side 32 --batch 4 --number 1 \
    --repeat 7 --warmup 2 \
    --output /tmp/macos-matmul-cartesian.json
$ PYTHONPATH=.:profiling python3 \
    profiling/render_execution_profile.py \
    /tmp/macos-matmul-cartesian.json \
    /tmp/macos-matmul-cartesian.md
$ PYTHONPATH=.:profiling python3 \
    profiling/profile_matmul_cartesian.py \
    --side 32 --batch 4 --number 100 \
    --repeat 15 --warmup 5 \
    --filter 1d-nd/ --filter nd-1d/ \
    --output /tmp/macos-matmul-vector-cartesian.json
```

Confirm that NumPy and `_solvcon` both link to Accelerate in each timing
JSON.  The immediate gate is the 48-row positive-stride direct-GEMV class.
Forced generic execution should remain slower than current dispatch, while
forced BLAS should remain at parity.  The 72-row negative or zero vector
class decides whether the automatic pack-once policy can be widened on
Accelerate.

The clean Apple Silicon run used revision `6120aec8`, the timing parameters
above, and one thread.  Both linkage records identify Accelerate.  All 31,825
Cartesian layout pairs match NumPy.  The
[complete aggregate report](macos-matmul-cartesian-summary.md) retains the
per-topology result, and its
[machine-readable summary](macos-matmul-cartesian-summary.json) records the
exact counts.  The 96 MiB raw timing JSON is reproducible from the command
above rather than committed to the branch.

| Comparison with NumPy | Planned faster | Parity | Inconclusive | NumPy faster |
| --- | ---: | ---: | ---: | ---: |
| Complete Cartesian matrix | 29,939 | 378 | 1,508 | 0 |

The focused 400-row vector control is stored as a
[complete notebook](macos-matmul-vector-cartesian-results.md) and
[raw JSON](macos-matmul-vector-cartesian-results.json).

| Layout class | Pairs | Generic/current | BLAS/current | Current route |
| --- | ---: | ---: | ---: | --- |
| Positive vector stride and direct GEMV matrix | 48 | 2.093x (1.470..2.352) | 0.996x (0.992..0.999) | BLAS |
| Negative or zero vector and direct GEMV matrix | 72 | 0.997x (0.995..0.999) | 0.535x (0.490..0.582) | Generic |
| Matrix not directly describable by GEMV | 280 | 0.997x (0.995..0.999) | 0.999x (0.997..1.001) | Generic |

All 48 positive-stride rows make forced generic execution conclusively
slower.  Forced BLAS reaches parity in 45 rows and is inconclusive in three,
so the current small direct-GEMV route passes the gate.

All 72 negative or zero vector rows make pack-once forced BLAS conclusively
faster.  Its median time is 0.535 of current, or about 1.87 times faster.
This supports widening the automatic vector pack-once path on Accelerate at
the measured work and batch size.

The 280 non-describable matrix rows do not answer whether packing the matrix
would help.  The historical forced-BLAS helper returned to generic execution
when GEMV could not describe the matrix.  Its parity only shows that both
controls followed the same route.  The explicit pack-once control below
removes this ambiguity.

## Explicit pack-once crossover

The follow-up adds two benchmark-only contracts.  Direct BLAS rejects any
operand without a valid descriptor.  Pack-once BLAS copies only unsupported
supplied operands, rebuilds the plan, and must enter BLAS.  Neither control
may silently return to generic execution.

The [complete Ubuntu crossover](ubuntu-matmul-pack-crossover-results.md) and
its [raw JSON](ubuntu-matmul-pack-crossover-results.json) contain 1,080
rows.  They cover both vector-batch directions, 15 vector and matrix layout
pairs, sides 8 through 256, and batches 1 through 64.  Seven paired samples
follow two warmups.

The matrix result rejects a broad pack policy for vector-batch operations.
Among 504 rows requiring only a matrix pack, current generic execution is
conclusively faster in 339, pack-once is faster in 19, and 146 are
inconclusive.  When both operands require packing, current is faster in 47
of 72 rows, pack-once is faster in three, and 22 are inconclusive.  A dense
control can be much faster while copying the supplied matrix inside the call
is still too expensive.

The [stable vector boundary](ubuntu-matmul-vector-pack-boundary-results.md)
and its
[raw JSON](ubuntu-matmul-vector-pack-boundary-results.json) retain 72 rows
with 15 samples and five warmups.  At side 32 and batch 4, the six
negative, negative-step-two, and zero-vector direction pairs have a median
generic/current ratio of 1.465.  The individual medians range from 1.413 to
1.546, and all six are conclusively faster.

The prototype therefore adds one conservative automatic rule:

```text
matrix has a direct GEMV descriptor
vector stride is negative or zero
core work is at least 32 * 32
output batch contains at least four contractions
```

The rule does not enable matrix packing for vector-batch work.  It also leaves
smaller reusable vectors generic even when an isolated Ubuntu row improves.
The focused Accelerate rerun below confirms this exact portable boundary.

## Reproduce the pack-once crossover

The build command names the devenv pybind11 CMake directory explicitly, so a
fresh worktree does not depend on an existing CMake cache.

```console
$ source /path/to/devenv/scripts/init
$ devenv use prime
$ make BUILD_QT=OFF \
    CMAKE_PREFIX_PATH="$(python3 -m pybind11 --cmakedir)"
$ export OPENBLAS_NUM_THREADS=1
$ export OMP_NUM_THREADS=1
$ export MKL_NUM_THREADS=1
$ export VECLIB_MAXIMUM_THREADS=1
$ export BLIS_NUM_THREADS=1
$ PYTHONPATH=.:profiling python3 \
    profiling/profile_matmul_pack_crossover.py \
    --sides 8,16,32,64,128,256 \
    --batches 1,2,4,8,16,64 \
    --repeat 7 \
    --warmup 2 \
    --cpu 0 \
    --output /tmp/matmul-pack-crossover.json
$ python3 profiling/render_matmul_pack_crossover.py \
    /tmp/matmul-pack-crossover.json \
    /tmp/matmul-pack-crossover.md
```

The focused Accelerate decision gate uses the same script and changes only
the case set and sample count:

```console
$ PYTHONPATH=.:profiling python3 \
    profiling/profile_matmul_pack_crossover.py \
    --sides 24,32,40 \
    --batches 2,4,8,16 \
    --repeat 15 \
    --warmup 5 \
    --filter negative-vector \
    --filter negative-step2-vector \
    --filter zero-vector \
    --output /tmp/matmul-vector-pack-boundary.json
$ python3 profiling/render_matmul_pack_crossover.py \
    /tmp/matmul-vector-pack-boundary.json \
    /tmp/matmul-vector-pack-boundary.md
```

The clean Apple Silicon rerun used revision `d1ebc1cc`, 15 samples, five
warmups, and one thread.  NumPy and `_solvcon` both link to Accelerate.  All
72 cases and every explicit route match NumPy.  The
[complete boundary report](macos-matmul-vector-pack-boundary-results.md) and
[raw JSON](macos-matmul-vector-pack-boundary-results.json) retain every
sample and linkage record.

| Current automatic route | Rows | Generic/current | Pack/current | Classification |
| --- | ---: | ---: | ---: | --- |
| Generic outside the portable boundary | 36 | 0.996x (0.995..0.998) | 0.587x (0.485..0.915) | Generic parity; pack faster in 36 |
| Pack once at the portable boundary | 36 | 2.456x (1.816..2.879) | 0.990x (0.986..0.993) | Current faster in 36; pack parity in 36 |
| Exact side 32, batch 4 boundary | 6 | 1.768x (1.766..1.823) | 0.989x (0.988..0.991) | Current faster in 6; pack parity in 6 |

At side 32 and batch 4, the six individual generic/current medians range
from 1.766 to 1.832.  The automatic route therefore reproduces the Ubuntu
decision on Accelerate and selects the same execution as the explicit
pack-once control.

The threshold is conservative for this backend.  Every one of the 36
threshold-off rows makes explicit pack-once execution conclusively faster
than current generic dispatch, including side 24 and batch 2.  This does not
invalidate the portable rule, which is bounded by the Ubuntu crossover.  It
shows that a lower Accelerate-specific threshold could be measured later
without changing the common plan or the current cross-platform policy.

## Rectangular reuse gate

The square sweep ties vector packing volume, output extent, and contraction
work to one side length.  It cannot distinguish equal-work cases such as:

```text
K = 8,   output extent = 128, core work = 1024, packed values = 8
K = 32,  output extent = 32,  core work = 1024, packed values = 32
K = 128, output extent = 8,   core work = 1024, packed values = 128
```

The [Ubuntu rectangular report](ubuntu-matmul-vector-pack-rectangular-results.md)
and its
[raw JSON](ubuntu-matmul-vector-pack-rectangular-results.json) separate
these quantities.  All 270 rows pass NumPy correctness before timing.  They
cover both vector directions, three unsupported vector strides, three equal
work levels, three factorizations per work level, and batches 1 through 16.
The matrix core remains C layout so no matrix packing enters the comparison.

The current portable predicate remains unchanged:

```text
core work >= 1024
batch >= 4
```

The report evaluates this additional reuse-aware condition:

```text
core work >= 576
reuse intensity = batch * output extent
reuse intensity >= 128
```

The extension is combined with the current predicate using logical OR.  It
never disables a current pack-once selection.  Because core work is
`K * output extent`, reuse intensity compares the total contracted work with
the `K` values copied once from the vector.

| Ubuntu selection | Rows | Pack faster | Inconclusive | Generic faster |
| --- | ---: | ---: | ---: | ---: |
| Current portable predicate | 108 | 97 | 11 | 0 |
| Reuse-aware extension only | 72 | 41 | 31 | 0 |
| Combined predicate | 180 | 138 | 42 | 0 |
| All measured rows | 270 | 157 | 106 | 7 |

All seven conclusive generic wins remain outside the combined predicate.
This supports testing the extension on Accelerate, but it does not change
automatic dispatch yet.  The common plan and execution routes remain frozen;
only the private vector packing predicate is under review.

The clean Apple Silicon rerun used revision `aee484b6`, 15 samples, five
warmups, and one thread.  NumPy and `_solvcon` both link to Accelerate.  All
270 cases and every explicit route match NumPy.  The
[complete report](macos-matmul-vector-pack-rectangular-results.md) and
[raw JSON](macos-matmul-vector-pack-rectangular-results.json) retain every
sample, operand shape, predicate result, and linkage record.

| Apple selection | Rows | Pack faster | Parity | Inconclusive | Generic faster |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current portable predicate | 108 | 108 | 0 | 0 | 0 |
| Reuse-aware extension only | 72 | 72 | 0 | 0 | 0 |
| Combined predicate | 180 | 180 | 0 | 0 | 0 |
| All measured rows | 270 | 249 | 6 | 4 | 11 |

The 72 extension-only rows have a median pack/generic ratio of 0.514, with
q10 and q90 values of 0.389 and 0.811.  The worst individual q90 is 0.849,
still below the 0.95 pack-faster boundary.  Both vector directions pass in
36 of 36 rows, and each negative, negative-step-two, and zero-stride class
passes in 24 of 24 rows.

All 11 conclusive generic wins remain outside the combined predicate.  The
strict two-backend gate therefore passes.  The reuse-aware extension can be
implemented as a private logical-OR predicate without modifying
`MatmulPlan`, the execution routes, or the common layer.

Reproduce the Apple gate without CPU affinity:

```console
$ PYTHONPATH=.:profiling python3 \
    profiling/profile_matmul_vector_pack_rectangular.py \
    --dimension-pairs \
    8x72,24x24,72x8,8x128,32x32,128x8,16x256,64x64,256x16 \
    --batches 1,2,4,8,16 \
    --repeat 15 \
    --warmup 5 \
    --output /tmp/matmul-vector-pack-rectangular.json
$ PYTHONPATH=.:profiling python3 \
    profiling/render_matmul_vector_pack_rectangular.py \
    /tmp/matmul-vector-pack-rectangular.json \
    /tmp/matmul-vector-pack-rectangular.md
```

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

The focused Accelerate decision gate is complete.  Keep the positive-stride
direct-GEMV route and the bounded negative or zero vector pack-once policy.
Leave non-describable matrices generic for vector-batch work because the
explicit in-call packing control rejects a broad matrix policy.  The current
portable threshold is safe on both measured backends.  The rectangular
OpenBLAS and Accelerate gates both accept the reuse-aware extension.  Add it
only as a private predicate combined with the current rule using logical OR.
Freeze `MatmulPlan`, the execution routes, and the common layer.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
