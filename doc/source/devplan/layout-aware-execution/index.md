# Layout-aware SimpleArray execution prototype

## Status

This development plan accompanies the
`codex/prototype-layout-aware-execution` branch.  The prototype keeps the
existing public operations unchanged and exposes `_planned_*` methods for
side-by-side correctness and performance measurements.

The implementation is based on master commit `6424cfad`.  The committed
profiling script records the actual revision, dirty-tree state, platform,
Python and NumPy versions, NumPy build configuration, extension linkage,
random seed, timing parameters, and thread-related environment variables in
its JSON output.

## Problem

SimpleArray has several useful HPC implementations, but operation families
describe traversal differently.  Out-of-place elementwise arithmetic copies
an input before applying an in-place loop.  Axis statistics collect a checked
slice for every output coordinate.  Matrix multiplication makes the caller
choose between direct, tiled, and BLAS entry points and supports only rank one
or two.

A universal executor would erase important differences between these
operations.  The hypothesis tested here is smaller:

> Elementwise, reduction, and batched matrix operations can share loop-domain
> and operand-mapping facts while retaining family-specific plans, dispatch,
> and kernels.

## Architecture

.. pstake:: schematic/execution_domains.tex

The existing SimpleArray shape, logical origin, and signed strides remain the
only storage model.  The prototype adds shared iteration vocabulary and three
family plans:

```cpp
struct LoopDomain
{
    Shape shape;
};

struct OperandMapping
{
    ssize_t base_offset;
    Strides strides;
};

class MappedOffsetCursor;
class InnerLoopPlan;
class InnerLoopCursor;
class OperationDefinition;
class ElementwisePlan;
class ReductionPlan;
class ReductionSchedule;
class MatmulPlan;
```

`LoopDomain` defines a runtime-rank coordinate space.  `OperandMapping`
projects an operand onto that space.  A zero stride expresses broadcasting
without an expanded buffer.  A negative stride retains the existing logical
origin and reverse traversal.

`InnerLoopPlan` lowers a positive-rank domain into an outer mapped domain plus
one fixed-size inner loop and one inner stride per operand.  Elementwise and
reduction executors both use it.  The outer cursor advances once per row,
while each typed executor retains an ordinary, compiler-inlineable inner
loop.  The abstraction therefore removes repeated rank and offset setup
without introducing a virtual call, callback, or runtime operation object.

`OperationDefinition` composes four compile-time policies:

```cpp
template <typename BroadcastingRule,
          typename ReductionRule,
          typename IterationRule,
          typename ComputeRule>
class OperationDefinition;
```

Every operation therefore names how operands align, whether dimensions are
reduced, how a runtime plan is constructed, and what computation runs over the
plan.  `make_operation_plan<Operation>()` delegates to the selected iteration
rule.  C++ concepts reject a mismatched combination before an executor is
instantiated.  The policy types are empty or stateless and do not add a
runtime operation object.

The rules construct family-specific plan types rather than erasing them behind
one base class.  `CommonBroadcastingRule` creates NumPy-style result domains
and zero-stride operand mappings.  `AxisReductionRule` partitions exact input
mappings into kept and reduced domains.  `BatchBroadcastingRule` aligns only
leading matrix axes, while `ContractionReductionRule` validates the shared
matrix dimension.

The shared code calculates common shapes, maps axes, walks runtime-rank
offsets, detects dense layouts, and preserves zero-sized domains.  It does not
own dtype policy, Python objects, reducers, matrix packing, or backend cost
policy.

The implementation is split by responsibility under
`cpp/solvcon/buffer/execution/`:

| File | Responsibility |
| --- | --- |
| `loop.hpp` and `loop.cpp` | Runtime-rank domains, operand mappings, shape alignment, mapped cursors, and fixed-stride inner loops |
| `operation.hpp` and `operation.cpp` | Compile-time operation definitions, broadcasting/reduction/iteration rule contracts, and axis partitioning |
| `elementwise_kernel.hpp` | Compile-time scalar, in-place, and contiguous kernel contracts |
| `elementwise.hpp` | Elementwise planning, alias handling, output allocation, and dense or fixed-stride inner execution |
| `reduction_plan.hpp` and `reduction_plan.cpp` | Kept/reduced semantic domains plus slice and reduced-offset cursors |
| `reduction_schedule.hpp` and `reduction_schedule.cpp` | Layout-based selection between sliced and outer-contiguous traversal |
| `reduction_outer.hpp` | Typed tiled backend for contiguous kept axes |
| `reduction.hpp` | Reduction facade plus dense, fixed-stride row, and collecting algorithms |
| `matmul_plan.hpp` and `matmul_plan.cpp` | Matrix roles, contraction strides, and batch broadcasting |
| `matmul.hpp` | Direct, layout-described BLAS, packed BLAS, or mapped matrix execution |
| `SimpleArrayExecution.hpp` | The standalone facade used only by the prototype binding |

The facade supplies the object API.  An operation definition selects policies,
and each resulting plan remains a small value object with one
topology-specific responsibility.  Kernels and array types remain templates,
so dtype selection and elementwise kernel calls do not require a virtual
hierarchy, runtime operation enum, or type-erased callback.

The non-template shape and plan support is implemented in `.cpp` files.
Template kernels and hot cursor advancement remain in headers so the compiler
can specialize and inline the per-element loop.  Core traversal does not use
callback lambdas.  Consumers drive a cursor with an ordinary loop, which keeps
control flow and reducer state visible in the family executor.

## Family composition

| Family | Broadcasting rule | Reduction rule | Iteration rule | Compute rule |
| --- | --- | --- | --- | --- |
| Elementwise | `CommonBroadcastingRule` | `NoReductionRule` | `ElementwiseIterationRule` | typed arithmetic kernel |
| Reduction | `NoBroadcastingRule` | `AxisReductionRule` | `ReductionIterationRule` | mean, weighted mean, variance, or collection |
| Matmul | `BatchBroadcastingRule` | `ContractionReductionRule` | `MatmulIterationRule` | matrix computation semantics; executor owns direct, mapped, and BLAS backends |

For elementwise operations, the resulting plan has one common result domain.
For reductions, it has an outer kept domain and an inner reduced domain.
Matmul retains a batch domain plus explicit `M`, `N`, and `K`.  The plans
deliberately do not inherit from a common base because their runtime topology
is different.

CRTP remains local to code with a shared implementation.  For example,
`BinaryKernelBase<Derived, T>` supplies scalar and in-place loops to arithmetic
kernels.  The four independent operation policies use type composition and
concepts instead.  A single CRTP operation base would couple unrelated
families and eventually accumulate family checks.

The end-to-end construction is explicit:

```cpp
using add_operation_type = elementwise_operation_type<AddKernel<T>>;
auto add_plan = make_operation_plan<add_operation_type>(
    output, lhs, rhs);

using mean_operation_type =
    reduction_operation_type<MeanComputeRule<T>>;
auto mean_plan = make_operation_plan<mean_operation_type>(
    input, axes, false);

using matrix_operation_type =
    matmul_operation_type<MatmulComputeRule>;
auto matrix_plan = make_operation_plan<matrix_operation_type>(
    lhs, rhs);
```

The same entry point chooses different iteration-rule factories at compile
time.  The returned values remain `ElementwisePlan`, `ReductionPlan`, and
`MatmulPlan`, so dispatch below the plan stays specialized.

## Operation routes

The private prototype bindings cover all operations selected for the roadmap:

```python
lhs._planned_add(rhs)
lhs._planned_sub(rhs)
lhs._planned_mul(rhs)
lhs._planned_div(rhs)

lhs._planned_iadd(rhs)
lhs._planned_isub(rhs)
lhs._planned_imul(rhs)
lhs._planned_idiv(rhs)

values._planned_mean(axes)
values._planned_average(axes, weights)
values._planned_var(axes, ddof=0)
values._planned_std(axes, ddof=0)
values._planned_median(axes)

lhs._planned_matmul(rhs)
```

The bindings are intentionally prefixed with an underscore.  This branch is a
measurement vehicle, not a proposal to publish the plan types or duplicate
the public operation surface.

## Extension contract

The elementwise plan stores an arity-independent list of input mappings.  A
new same-dtype binary operation supplies a stateless kernel with scalar,
in-place, and contiguous entry points, then calls the existing `transform`
facade.  The C++ concept checks that contract at compile time:

```cpp
template <typename T>
struct MaximumKernel
{
    T operator()(T lhs, T rhs) const;

    static void scalar(T * output, size_t count, T rhs);
    static void inplace(T * output, size_t count, T const * rhs);
    static void contiguous(
        T * output, size_t count, T const * lhs, T const * rhs);
    static void contiguous_scalar(
        T * output, size_t count, T const * lhs, T rhs);
};

using maximum_operation_type =
    execution::elementwise_operation_type<MaximumKernel<T>>;

return execution::ElementwiseExecutor<
    Array, T, maximum_operation_type>::transform(
        lhs, rhs, MaximumKernel<T>{});
```

Adding that operation does not change shape alignment, alias handling,
layout classification, or traversal.  Unary, ternary, mixed-dtype, and
mixed-output operations can add an executor adapter over the existing input
mapping list without adding fields to `ElementwisePlan`.

A new streaming reduction constructs `ReductionPlan`, lowers it to
`ReductionSchedule`, then chooses a backend.  Dense inner domains use the
SIMD accumulator.  General layouts iterate outer slices with
`ReductionSliceCursor` and run the reduced last axis with a fixed stride.  A
contiguous kept axis can instead enter the tiled outer executor.  A two-pass
reducer such as variance and a collector such as median keep their
algorithm-specific state, but neither recreates axis normalization or
signed-stride traversal.  Weighted reducers add operand mappings to the
inner domain rather than multiplying temporary arrays.

Contractions, gathers, searches, and sorting should not be forced through the
elementwise executor.  They may define a topology-specific plan while reusing
`LoopDomain`, `OperandMapping`, `InnerLoopPlan`, broadcast alignment, and
offset traversal.  This is the extension boundary: reuse coordinate facts and
fixed-stride loop lowering, but retain the algorithm that makes each family
fast.

## Implementation notes

### Elementwise

Out-of-place operations allocate their destination and write the result in one
pass.  A shared dense input layout, including Fortran and negative-stride
layouts, is preserved and enters a linear physical traversal.  Non-dense
inputs receive compact output so inherited storage holes do not slow the
write.  Compatible different shapes enter NumPy-style common shape alignment
and use zero strides on broadcast axes.

Every positive-rank non-dense plan maps only its outer axes, then advances
output and input pointers with fixed last-axis strides.  Unit-stride output
with a unit-stride input and a zero-stride scalar or broadcast operand uses
the SIMD facade.  Other signed and step-two combinations remain tight scalar
loops rather than updating a runtime-rank index for every element.

In-place operations treat the destination shape as fixed.  A source sharing
the same buffer through a different mapping is cloned before execution.  An
exact alias remains safe and does not allocate a temporary.  Scalar in-place
work keeps its direct compiler-vectorizable loop because a runtime SIMD
backend check caused a measurable fixed-cost regression on the generic x86
backend.

### Reduction

`ReductionPlan` splits axes into an outer output domain and an inner reduction
domain.  It contains semantic topology and mappings, but no inner-loop or
contiguity decision.  `ReductionSchedule` lowers those facts to sliced or
outer-contiguous traversal.  Mean, weighted average, variance, and standard
deviation stream directly from the source.  Median uses the same schedule
facts and offset traversal but retains a collector because its algorithm
needs an ordered value set.

Dense inner domains accumulate over one physical span.  The NEON backend has
typed sum, sum-product, and squared-difference kernels, while the generic
backend retains compiler-vectorizable loops.  A one-dimensional inner domain
and a higher-rank row both use `InnerLoopPlan`; the runtime-rank cursor moves
once per row and stays out of the last-axis loop.  `ReductionSchedule` prepares
the single-input inner loop once for all output slices.  Weighted average
prepares its two-input loop once and computes the weight total once per
operation rather than once per output slice.

For a rank-two F-layout reduction over axis 1, the kept axis is physically
contiguous while each logical output row is strided.  The outer-contiguous
schedule interchanges those loops and processes a 4096-byte output tile.  It
therefore reads each source column sequentially while retaining several
output accumulators.  The typed backend is separate from the semantic plan,
so unsupported shapes and signed strides keep the general sliced route.

Empty kept domains produce empty outputs without entering the loop.  Empty
reduced domains raise before reading input.  Full reductions use an empty
outer domain with one logical iteration and return a scalar through the
existing API convention.

### Matrix multiplication

`MatmulExecutor::multiply` selects only unbatched or planned topology.
Unbatched execution then applies the work threshold and layout policy.  Work
below 4096 scalar multiply-adds uses the existing direct kernel.  At or above
that threshold, C- and F-compatible matrix views pass transpose and
leading-dimension metadata directly to row-major BLAS.  Negative matrix
strides and step-two inner strides cannot be represented by CBLAS, so those
rank-one/two inputs still pack to row-major storage.  Packing cost remains
inside the measured public call.

Higher-rank operands construct `MatmulPlan`.  It normalizes vector roles,
validates `K`, broadcasts only leading batch axes, and retains explicit matrix
strides.  Each compatible matrix pair enters BLAS at the batch offsets
produced by the plan.  A zero batch stride reuses the same matrix for
broadcasting without materialization.  Unsupported matrix strides retain the
mapped contraction fallback.

## Reproduction

Build a release extension with the project dependency environment:

```console
$ source /path/to/devenv/scripts/init
$ devenv use prime
$ make BUILD_QT=OFF CMAKE_BUILD_TYPE=Release SOLVCON_PROFILE=OFF \
    CMAKE_PREFIX_PATH="$(python3 -m pybind11 --cmakedir)"
```

Run fixed and deterministic randomized correctness checks:

```console
$ python3 -m pytest -s tests/test_execution_prototype.py -q
$ python3 profiling/profile_execution_prototype.py \
    --quick --check-only --stress 1000
```

Run the complete benchmark and save machine-readable results.  The script
sets OpenBLAS, MKL, Accelerate, BLIS, and OpenMP to one thread before importing
NumPy or solvcon.  Set `SOLVCON_BENCHMARK_THREADS` to override that count.

```console
$ python3 profiling/profile_execution_prototype.py \
    --benchmark-only --repeat 15 --warmup 5 --cpu 0 \
    --output /tmp/solvcon-execution.json
$ python3 profiling/render_execution_profile.py \
    /tmp/solvcon-execution.json \
    doc/source/devplan/layout-aware-execution/ubuntu-results.md
```

Each committed timing records 15 paired samples.  Within every sample, the
profiler rotates the NumPy, planned, and legacy call order.  Changing machine
load does not consistently favor one route.  Mutable destinations are reset
before each route and sample.  Allocation, validation, planning, dispatch, and
execution are inside the measured public call.  The script reports the median
ratio and its q10 through q90 interval.  A result is called faster or slower
only when the entire interval clears the five-percent threshold; an interval
that crosses a threshold is `inconclusive`.  The script always checks every
result against NumPy before timing.  The
`--benchmark-only` option suppresses correctness output and randomized stress,
but retains this gate so an incorrect legacy result is never timed as useful
work.

Use `--filter` to repeat one family, operation, or layout without editing the
script.  On Linux, `--cpu` pins the process to one CPU and records the effective
affinity in the JSON:

```console
$ python3 profiling/profile_execution_prototype.py \
    --benchmark-only --repeat 15 --warmup 5 --cpu 0 \
    --filter inplace-scalar/sub/negative-destination
```

### macOS reproduction

Use the same revision and the same single-thread protocol on macOS.  A native
arm64 dependency environment is required so the build can find Accelerate or
the configured BLAS implementation.  Do not compare an arm64 planned build
against an x86 process running through Rosetta.

```console
$ git clone --branch codex/prototype-layout-aware-execution \
    https://github.com/ThreeMonth03/solvcon.git
$ cd solvcon
$ source /path/to/devenv/scripts/init
$ devenv use prime
$ uname -m
$ python3 -c 'import platform; print(platform.machine())'
$ NPROC=2 MAKE_PARALLEL=-j2 make BUILD_QT=OFF \
    CMAKE_BUILD_TYPE=Release SOLVCON_PROFILE=OFF \
    CMAKE_PREFIX_PATH="$(python3 -m pybind11 --cmakedir)"
$ TMPDIR=/tmp PYTHONPATH=. pytest -q \
    tests/test_buffer.py tests/test_execution_prototype.py \
    tests/test_matrix.py tests/test_blas_lapack.py
$ PYTHONPATH=. python3 profiling/profile_execution_prototype.py \
    --check-only --stress 1000
$ PYTHONPATH=. python3 profiling/profile_execution_prototype.py \
    --benchmark-only --repeat 15 --warmup 5 \
    --output /tmp/solvcon-execution-macos.json
$ PYTHONPATH=. python3 profiling/render_execution_profile.py \
    /tmp/solvcon-execution-macos.json \
    doc/source/devplan/layout-aware-execution/macos-results.md
$ cp /tmp/solvcon-execution-macos.json \
    doc/source/devplan/layout-aware-execution/macos-results.json
```

The JSON records `platform`, `machine`, Python, NumPy, NumPy's build
configuration, `_solvcon` linkage from `otool -L`, Git revision, dirty state,
seed, sample counts, raw samples, paired ratio quantiles, and all
thread-control variables.  The benchmark script sets the supported BLAS and
OpenMP thread controls to one before importing NumPy.  Do not pass `--cpu` on
macOS because the Linux affinity API is unavailable.  Commit both generated
files when reporting a new run.  In particular, compare the contiguous,
F-contiguous, negative, step-two, and batched matmul rows.  They determine
whether the 4096-work packing threshold is also appropriate for Accelerate.

## Ubuntu measurement

The complete 154-row Linux result is recorded in the
[Ubuntu benchmark notebook](ubuntu-results.md).  It lists every operation
and layout separately with operand shape and element strides, NumPy, legacy,
and planned medians, both ratios, and correctness status.  Transposed
F-contiguous inputs and genuine non-dense step-two views are separate rows.
No range-only family summary substitutes for those measurements.

The clean optimized run used revision `d320b553`, Linux on WSL2 x86-64,
Python 3.12.7, NumPy 2.3.0, 15 paired samples, five warmups, one thread, and
CPU 0.  It produced the following complete classification:

| Comparison | Faster | Parity | Slower | Inconclusive | Not comparable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy versus planned | 75 planned | 2 | 0 | 16 | 24 legacy incorrect and 37 new only |
| NumPy versus planned | 76 planned | 3 | 13 NumPy | 62 | 0 |

Every correct legacy route with a conclusive classification improved or
reached parity, and no route regressed.  The 13 conclusive NumPy-faster rows
are eight axis reductions, four full reductions, and one microsecond in-place
array case.  The 12 reduction rows motivate the tiled multi-output reduction
experiment below rather than another common plan layer.  The in-place case
remains subject to the paired microbenchmark caveat below.

The matrix includes out-of-place array and scalar arithmetic, shape
broadcasting, in-place array, scalar, and broadcast operations, axis and full
reductions, matrix and vector roles, and batched matmul.  The correctness gate
also exposes legacy routes that return wrong values for step-two or mapped
in-place views.  Those rows retain NumPy and planned timings but deliberately
do not report a legacy timing.

The temporary C++ CallProfiler scopes were removed after this diagnosis.
They compile out of normal builds, but retaining fine probes would still
distort micro-operation timings in a profiling build.  Future investigations
should insert scopes around the suspected dispatch temporarily and remove
them after recording the result.

The committed script prints every operation row and writes all raw times,
paired ratios, ratio quantiles, operand metadata, and separate legacy and
NumPy comparison statuses to JSON.  NumPy timings are platform and
linked-backend evidence, not a claim that the same ratio will hold on another
machine.

NumPy reports its BLAS dependency as `auto` in this environment, while the
solvcon extension links to the prime environment's OpenBLAS.  NumPy matmul is
unusually slow in this run, so the 17 planned-faster matrix rows prove that
the dispatch avoids the prototype's legacy bottlenecks on this machine.  They
do not prove a universal advantage over tuned NumPy builds.  The Apple
Accelerate rerun is required for that comparison.

### Optimization findings

The first macOS run exposed three costs hidden by the original Ubuntu family
summary: per-element mapped cursor updates, copy-before-transform scalar
operations, and scalar batched contraction.  The optimized branch moves
runtime-rank bookkeeping outside the inner loop, writes scalar transforms in
one pass, accumulates reducible spans or rows directly, and sends compatible
matrix batches to BLAS.

The result also established two negative rules.  In-place scalar operations
do not enter the runtime SIMD facade because its fixed dispatch cost is
visible on the generic x86 backend.  Non-dense scalar inputs do not force
their storage holes onto a new output.  The first rule restored the in-place
route to parity.  The second reduced the four 1024 by 1024 step-two scalar
cases from roughly 1.2 to 1.6 ms to 0.59 to 0.76 ms in a targeted 15-sample
run.

An earlier full run classified
`inplace-scalar/mul/negative-destination` as a regression at 0.0283 ms legacy
versus 0.0328 ms planned.  A 31-sample audit of the same binary produced
0.0270 ms versus 0.0271 ms and a paired ratio interval of 0.917 to 1.161.
Microsecond in-place rows should therefore be read with their paired interval,
not from one median.

## Optimized macOS measurement

The complete 154-row Apple Silicon result is recorded in the
[macOS benchmark notebook](macos-results.md), with raw samples in the
[machine-readable JSON](macos-results.json).  The clean run used revision
`e3a0ba93`, macOS 26.5.1 on an Apple M1 with 8 GB of memory, native arm64
Python 3.11.6, NumPy 2.2.4, 15 paired samples, five warmups, and one thread.
NumPy and the extension both link to Accelerate.  CPU affinity is not
available on macOS.

| Comparison | Faster | Parity | Slower | Inconclusive | Not comparable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy versus planned | 76 planned | 14 | 0 | 3 | 24 legacy incorrect and 37 new only |
| NumPy versus planned | 59 planned | 28 | 36 NumPy | 31 | 0 |

Every correct legacy route with a conclusive classification improved or
reached parity, and none regressed.  Compared with the diagnosis baseline,
planned-faster NumPy comparisons increased from 30 to 59 and NumPy-faster
comparisons fell from 91 to 36.

The following medians show where the optimized implementation changed the
Apple Silicon result.  Cross-run speedups compare two separately recorded
medians, so they diagnose the direction and size of the change rather than a
paired uncertainty interval.

| Case | Baseline ms | Optimized ms | Speedup | Optimized versus NumPy |
| --- | ---: | ---: | ---: | --- |
| Step-two array add | 6.1408 | 1.1722 | 5.24x | inconclusive |
| Step-two scalar add | 5.7115 | 0.6023 | 9.48x | planned-faster |
| Step-two broadcast add | 0.3827 | 0.0529 | 7.24x | planned-faster |
| Step-two in-place array add | 1.5304 | 0.2016 | 7.59x | parity |
| Step-two in-place broadcast add | 1.5606 | 0.1797 | 8.68x | planned-faster |
| C-layout axis mean | 1.0257 | 0.1955 | 5.25x | NumPy-faster |
| C-layout axis variance | 2.0143 | 0.4117 | 4.89x | parity |
| Same-shape C batch matmul | 0.1772 | 0.0072 | 24.68x | inconclusive |
| Broadcast C batch matmul | 1.4008 | 0.0424 | 33.03x | parity |

Fixed-stride inner loops remove most mapped cursor cost.  Step-two scalar,
broadcast, and in-place routes move from large NumPy losses to parity or
planned wins.  NEON accumulation cuts the C-layout axis mean and variance
costs by about five times; variance reaches parity, while mean and
F-contiguous axis reductions still favor NumPy.  This gap motivates the
outer-contiguous schedule experiment below.

Batch BLAS removes the scalar-contraction bottleneck.  Compatible contiguous
batches now reach parity or an interval crossing parity, while negative and
step-two matrix batches are planned-faster.  Direct F-layout and mixed C/F
matrix descriptors also move from NumPy-faster to inconclusive or parity.
For large negative and step-two 256 by 256 inputs, packing before Accelerate
remains roughly 104 to 138 times faster than legacy and 126 to 154 times
faster than NumPy.

## Outer-contiguous reduction experiment

The next Ubuntu experiment implemented the loop-interchange recommendation
without changing reduction semantics.  `ReductionPlan` now ends at the
kept/reduced topology.  `ReductionSchedule` selects the traversal, and
`OuterReductionExecutor` owns the typed 4096-byte tiled backend.  General
layouts retain the previous sliced execution.

The before run used the clean `a1643483` revision with 15 samples and five
warmups.  The after run used the same release environment and CPU with 30
samples and ten warmups.  These are cross-run medians rather than paired
samples, so they establish the size and direction of the change.  The
benchmark still checks each result against NumPy before timing.

| F-layout axis-1 operation | Before ms | After ms | Speedup | After versus NumPy |
| --- | ---: | ---: | ---: | --- |
| Mean | 0.1820 | 0.0553 | 3.29x | inconclusive |
| Variance | 0.3328 | 0.1044 | 3.19x | planned-faster |
| Standard deviation | 0.3286 | 0.1076 | 3.05x | planned-faster |
| Weighted average | 0.1929 | 0.0577 | 3.34x | planned-faster |

Median does not use this backend because it must collect values for ordering.
The optimized macOS notebook above predates this schedule.  It must be rerun
at the new branch revision before the Apple Silicon result can be updated.

## Recommendations

1. Keep `LoopDomain`, `OperandMapping`, and fixed-stride inner-loop lowering
   as the shared vocabulary, but keep the three family plans and executors
   separate.  Their measured fast paths depend on different topology and cost
   rules.
2. Reproduce the outer-contiguous reduction result on Apple Silicon.  Keep the
   schedule and backend split only if Accelerate-era measurements confirm the
   Ubuntu improvement without harming the other reduction layouts.
3. Add an x86 SIMD backend only after profiling the remaining elementwise and
   reduction rows.  This Ubuntu machine has AVX2, but the solvcon SIMD facade
   currently falls back to generic loops outside AArch64.
4. Keep the 4096 matmul threshold as a measured initial policy, not an ABI.
   Offline backend-specific benchmarks may justify separate direct, BLAS, and
   pack thresholds.  Do not benchmark alternatives inside each public call.
5. Add unary, ternary, mixed-dtype, and mixed-output executor adapters only
   when a concrete operation needs them.  The existing mapping list can
   support them without a virtual plan hierarchy.
6. Keep CallProfiler probes temporary.  Insert them around a suspected plan,
   dispatch, or kernel boundary for diagnosis, then remove them before final
   microbenchmarks so probe overhead does not enter the result.

## Verification

The prototype currently passes:

- 16 focused Python tests, including float and complex outer-contiguous routes.
- 154 fixed profiler cases across C-contiguous, F-contiguous, negative-stride,
  step-two, mixed-layout, broadcast, vector, matrix, and batch roles.
- 5000 deterministic randomized iterations covering ranks one through four,
  broadcasting, axis permutations, negative strides, reductions, and batch
  broadcasting.

The full Python suite passes with 1405 tests, 315 skips, and three expected
failures.  All 228 C++ tests, the standalone buffer build, and project linters
also pass.  The local prime environment does not contain `doxygen` or
`sphinx-build`, so the documentation source is included but could not be
rendered in that environment.

## Out of scope

- Replacing the existing public methods in this draft.
- A virtual plan hierarchy or universal executor.
- Runtime multithreading or an in-call backend benchmark.
- NumPy dtype promotion.
- Assignment, comparison, `where`, search, sort, or gather migration.
- A new storage layout or execution-time `nghost` policy.

Python selection or an operation facade must normalize ghost semantics before
constructing a mapping.  Execution plans consume only the supplied logical
layout.

## Delivery notes

The branch is intended for a draft pull request on the personal fork.  The
implementation, tests, profiler, and this plan are kept together so another
machine, including macOS with NEON and Accelerate, can check out one revision
and reproduce the same correctness matrix and timing protocol.

## Conversation history

1. The initial roadmap attempted to specify a broad universal array framework.
   Reviewer feedback required the proposal to start from measured HPC
   workloads instead.
2. The scope was narrowed to matrix contraction, elementwise broadcasting,
   and statistical reduction, with common code extracted only after concrete
   reuse.
3. The implementation request asked for one branch containing all three plan
   skeletons, side-by-side legacy and planned measurements, contiguous and
   non-contiguous correctness checks, reproducible Ubuntu data, and a draft
   pull request that can also be cloned for macOS profiling.
4. The optimized macOS result identified F-layout axis reduction as the main
   remaining topology bottleneck.  The follow-up split semantic planning from
   execution scheduling and validated a tiled kept-axis backend on Ubuntu.
5. The architecture review asked whether each operation should own explicit
   broadcasting, reduction, and iteration rules.  The follow-up added
   compile-time policy composition, kept CRTP inside kernels that actually
   share code, and left measured family backends unchanged.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
