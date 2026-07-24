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

The prototype deliberately has no universal operation definition.  An
operation such as addition is mathematical semantics, while elementwise
transform, reduction, and contraction are different execution topologies.
Binding one operation type to four mandatory broadcasting, reduction,
iteration, and compute slots would duplicate facts already owned by the
family plan and require placeholder rules such as "no reduction."

Each family therefore owns the rules required to construct its runtime plan:

- `ElementwisePlan` computes a common result shape and inserts zero strides
  for NumPy-style value broadcasting.
- `ReductionPlan` partitions an exact input mapping into kept and reduced
  domains.  `ReductionSchedule` then chooses a layout-specific traversal.
- `MatmulPlan` validates `K`, assigns matrix roles, and broadcasts only leading
  batch axes.

Mathematical semantics enter only where they execute.  `ElementwiseExecutor`
is templated on a typed kernel.  Reduction algorithms retain their accumulator
and finalization state in `ReductionExecutor`.  `MatmulExecutor` owns matrix
kernel and backend selection.  The compiler can still inline typed hot loops,
but an operation does not claim rules that do not apply to it.

The shared code calculates common shapes, maps axes, walks runtime-rank
offsets, detects dense layouts, and preserves zero-sized domains.  It does not
own dtype policy, Python objects, reducers, matrix packing, or backend cost
policy.

The implementation is split by responsibility under
`cpp/solvcon/buffer/execution/`:

| File | Responsibility |
| --- | --- |
| `loop.hpp` and `loop.cpp` | Runtime-rank domains, operand mappings, shape alignment, mapped cursors, and fixed-stride inner loops |
| `elementwise_kernel.hpp` | Compile-time scalar, in-place, and contiguous kernel contracts |
| `elementwise.hpp` | Common-shape broadcasting, elementwise planning, alias handling, output allocation, and dense or fixed-stride inner execution |
| `reduction_plan.hpp` and `reduction_plan.cpp` | Axis partitioning, kept/reduced semantic domains, and reduction cursors |
| `reduction_schedule.hpp` and `reduction_schedule.cpp` | Layout-based selection between sliced and outer-contiguous traversal |
| `reduction_outer.hpp` | Typed tiled backend for contiguous kept axes |
| `reduction.hpp` | Reduction facade plus dense, fixed-stride row, and collecting algorithms |
| `matmul_plan.hpp` and `matmul_plan.cpp` | Matrix roles, contraction validation, matrix strides, and batch broadcasting |
| `matmul.hpp` | Direct, layout-described BLAS, packed BLAS, or mapped matrix execution |
| `SimpleArrayExecution.hpp` | The standalone facade used only by the prototype binding |

## Evidence for the shared layer

The shared layer is not justified by a future operation list.  Every retained
primitive has at least two concrete consumers in this prototype:

| Shared primitive | Elementwise | Reduction | Matmul |
| --- | --- | --- | --- |
| `LoopDomain` | Result coordinates | Kept and reduced coordinates | Broadcast batch coordinates |
| `OperandMapping` | Output, exact input, and zero-stride input mappings | Kept, reduced, and weight mappings | Batch and matrix-role mappings |
| `MappedOffsetCursor` | General strided fallback | Slice and reduced-offset traversal | Broadcast batch traversal |
| `InnerLoopPlan` | Fixed-stride elementwise rows | Fixed-stride reduction rows | Not used |
| Common-shape and zero-stride mapping | NumPy-style value broadcasting | Weight alignment | Leading batch broadcasting |

The direct C++ contracts in `gtests/test_nopython_execution.cpp` verify the
shared behavior independently of Python bindings and numerical kernels.  They
cover common-shape alignment, zero-stride offsets, negative strides, fixed
inner-loop lowering, and construction of all three family plans over the same
layout vocabulary.

This is the bounded reason the common layer is necessary.  Removing
`LoopDomain`, `OperandMapping`, or mapped traversal would make two or three
families independently reimplement runtime-rank carry, signed offset
calculation, empty-domain handling, or broadcasting.  The table does not
justify a universal executor.  Reduction still owns kept/reduced traversal,
and matmul still owns contraction and backend policy.  A primitive with only
one consumer remains in its family.

The facade supplies the object API and calls the plan belonging to that
operation family.  Each plan remains a small value object with one
topology-specific responsibility.  Kernels and array types remain templates,
so dtype selection and elementwise kernel calls do not require a virtual
hierarchy, runtime operation enum, or type-erased callback.

The non-template shape and plan support is implemented in `.cpp` files.
Template kernels and hot cursor advancement remain in headers so the compiler
can specialize and inline the per-element loop.  Core traversal does not use
callback lambdas.  Consumers drive a cursor with an ordinary loop, which keeps
control flow and reducer state visible in the family executor.

## Family interfaces

The high-level facade uses three separate plan factories:

```cpp
using executor_type =
    ElementwiseExecutor<Array, T, AddKernel<T>>;
auto result = executor_type::transform(
    lhs, rhs, AddKernel<T>{});

auto reduction = ReductionPlan::make(input, axes, false);
auto mean = ReductionExecutor<Array, T>::mean(input, reduction);

auto product = MatmulExecutor<Array, T>::multiply(lhs, rhs);
```

The elementwise facade asks `ElementwisePlan::broadcast_shape()` for output
allocation, then builds the complete plan.  The reduction facade passes
explicit axes to `ReductionPlan`.  The matrix facade keeps plan construction
private because output roles and backend choice are one matrix-family
contract.

For elementwise operations, the resulting plan has one common result domain.
For reductions, it has an outer kept domain and an inner reduced domain.
Matmul retains a batch domain plus explicit `M`, `N`, and `K`.  The plans
deliberately do not inherit from a common base because their runtime topology
is different.

CRTP remains local to code with an actual shared implementation.  For
example, `BinaryKernelBase<Derived, T>` supplies scalar and in-place loops to
arithmetic kernels.  A CRTP base across the three plans would couple unrelated
topologies and eventually accumulate family checks.

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

return execution::ElementwiseExecutor<
    Array, T, MaximumKernel<T>>::transform(
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

Run the isolated large-matrix and large-batch suite separately.  Its five
paired samples and one warmup keep the single-thread run practical while
retaining raw samples and ratio quantiles:

```console
$ python3 profiling/profile_execution_prototype.py \
    --hpc-matmul --benchmark-only \
    --repeat 5 --warmup 1 --cpu 0 \
    --output /tmp/solvcon-matmul-hpc.json
$ python3 profiling/render_execution_profile.py \
    /tmp/solvcon-matmul-hpc.json \
    doc/source/devplan/layout-aware-execution/ubuntu-matmul-hpc-results.md
```

Pass `--hpc-matmul-slow` to include a 2048 by 2048 case.  It is excluded
from the standard suite because one NumPy call takes more than two minutes
in the recorded Ubuntu environment.  The standard suite still includes
1024 by 1024 matrices with a batch of eight and 512 by 512 matrices with a
cross-broadcast batch of 64.

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
script.  On Linux, `--cpu` pins the process to one CPU and records the
effective affinity in the JSON:

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
$ PYTHONPATH=. python3 profiling/profile_execution_prototype.py \
    --hpc-matmul --benchmark-only \
    --repeat 5 --warmup 1 \
    --output /tmp/solvcon-matmul-hpc-macos.json
$ PYTHONPATH=. python3 profiling/render_execution_profile.py \
    /tmp/solvcon-matmul-hpc-macos.json \
    doc/source/devplan/layout-aware-execution/macos-matmul-hpc-results.md
$ cp /tmp/solvcon-matmul-hpc-macos.json \
    doc/source/devplan/layout-aware-execution/macos-matmul-hpc-results.json
```

The JSON records `platform`, `machine`, Python, NumPy, NumPy's build
configuration, NumPy matmul extension linkage, `_solvcon` linkage from
`otool -L`, Git revision, dirty state, seed, sample counts, raw samples,
paired ratio quantiles, and all thread-control variables.  The benchmark
script sets the supported BLAS and OpenMP thread controls to one before
importing NumPy.  Do not pass `--cpu` on macOS because the Linux affinity API
is unavailable.  Commit both generated files when reporting a new run.  In
particular, compare the contiguous, F-contiguous, negative, step-two, and
batched matmul rows.  They determine whether the 4096-work packing threshold
is also appropriate for Accelerate.

## Ubuntu measurement

The complete 154-row Linux result is recorded in the
[Ubuntu benchmark notebook](ubuntu-results.md).  It lists every operation
and layout separately with operand shape and element strides, NumPy, legacy,
and planned medians, both ratios, and correctness status.  Transposed
F-contiguous inputs and genuine non-dense step-two views are separate rows.
No range-only family summary substitutes for those measurements.

The clean optimized run used revision `953a9623`, Linux on WSL2 x86-64,
Python 3.12.7, NumPy 2.3.0, 15 paired samples, five warmups, one thread, and
CPU 0.  It produced the following complete classification:

| Comparison | Faster | Parity | Slower | Inconclusive | Not comparable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy versus planned | 74 planned | 0 | 0 | 19 | 24 legacy incorrect and 37 new only |
| NumPy versus planned | 74 planned | 0 | 5 NumPy | 75 | 0 |

Every correct legacy route with a conclusive classification improved or
reached parity, and no route regressed.  The five conclusive NumPy-faster rows
are mean reductions: one axis reduction and four full-layout cases.  They
motivate specialized accumulation work rather than another common plan
layer.

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
Accelerate rerun below supplies the same-backend comparison.

### Large-scale matmul measurement

The [large-scale Ubuntu notebook](ubuntu-matmul-hpc-results.md) records all
14 rows, and the [machine-readable JSON](ubuntu-matmul-hpc-results.json)
retains the raw samples and backend linkage.  The clean run used revision
`3804672e`, five paired samples, one warmup, one thread, and CPU 0.  Every
planned result passed the NumPy differential check before timing.

The NumPy matmul extension has no BLAS dependency in `ldd`, while `_solvcon`
links to the prime environment's OpenBLAS.  The Ubuntu NumPy speedup ratios
therefore show a backend mismatch, not a universal advantage.  The planned
absolute times, within-run layout controls, and memory dimensions are the
useful architecture evidence:

| Case | B | R | MxKxN | Logical / expanded input MiB | Output MiB | Planned ms | Effective GMAC/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Large square | 1 | 0 | 1024x1024x1024 | 16 / 16 | 8 | 33.972 | 31.61 |
| One-sided broadcast | 8 | 1 | 1024x1024x1024 | 72 / 128 | 64 | 271.236 | 31.67 |
| Transposed lhs broadcast | 8 | 1 | 1024x1024x1024 | 72 / 128 | 64 | 261.330 | 32.87 |
| Cross broadcast | 64 | 2 | 512x512x512 | 32 / 256 | 128 | 314.942 | 27.27 |
| Same-shape batch | 4096 | 1 | 32x32x32 | 64 / 64 | 32 | 27.337 | 4.91 |
| One-sided batch | 16384 | 1 | 32x32x32 | 128 / 256 | 128 | 87.218 | 6.16 |
| Cross batch | 16384 | 2 | 32x32x32 | 2 / 256 | 128 | 81.414 | 6.59 |
| High-rank batch | 16384 | 4 | 32x32x32 | 2 / 256 | 128 | 86.080 | 6.24 |
| Dense lhs broadcast | 64 | 1 | 256x256x256 | 32.5 / 64 | 32 | 48.879 | 21.97 |
| Materialized dense lhs | 64 | 1 | 256x256x256 | 64 / 64 | 32 | 48.936 | 21.94 |
| Negative-stride lhs | 64 | 1 | 256x256x256 | 32.5 / 64 | 32 | 460.702 | 2.33 |
| Step-two lhs | 64 | 1 | 256x256x256 | 32.5 / 64 | 32 | 459.865 | 2.33 |
| Dense batch axis | 128 | 1 | 256x256x256 | 64.5 / 128 | 64 | 92.880 | 23.12 |
| Step-two batch axis | 128 | 1 | 256x256x256 | 64.5 / 128 | 64 | 94.389 | 22.75 |

The dense broadcast and pre-expanded controls have the same throughput, but
the broadcast form stores one lhs matrix instead of 64.  A step-two batch
axis is within two percent of the dense batch-axis control, so mapped batch
offsets remain outside the matrix kernel.  Raising batch rank from two to
four at the same 16384 outputs changes the median from 81.414 to 86.080 ms;
runtime-rank carry does not become the dominant cost.

Negative and step-two matrix strides are about 9.4 times slower than the
dense broadcast control because higher-rank unsupported matrix layouts still
use mapped scalar contraction.  This isolates matrix packing as an executor
policy problem.  It does not justify changing `LoopDomain` or
`OperandMapping`.

### Batched pack-once measurement

The [pack-once Ubuntu notebook](ubuntu-matmul-pack-once-results.md) records
the four equal-work controls, and its
[machine-readable JSON](ubuntu-matmul-pack-once-results.json) retains the raw
samples.  The clean run used revision `b66516c8`, five samples, one warmup,
one thread, and CPU 0.

The executor now keeps every directly describable operand and converts only
an unsupported matrix layout to row-major storage.  It performs that
conversion once before the existing batch loop.  A broadcast operand retains
its extent-one batch dimensions, so the rebuilt plan supplies the same
zero-stride reuse instead of expanding or repacking it for every output.

| Lhs layout | Baseline ms | Pack-once ms | Cross-run speedup | Difference from pack-once dense |
| --- | ---: | ---: | ---: | ---: |
| Dense broadcast | 48.879 | 49.167 | 0.99x | control |
| Materialized dense | 48.936 | 51.299 | 0.95x | +4.3% |
| Negative-stride broadcast | 460.702 | 51.149 | 9.01x | +4.0% |
| Step-two broadcast | 459.865 | 51.019 | 9.01x | +3.8% |

The before and after values are separate clean-revision medians, not paired
samples.  The two dense controls bound run-to-run movement.  Negative and
step-two inputs move from about 9.4 times slower than dense to within four
percent of the same-run dense control.  This supports one executor-local
pack decision while leaving broadcast planning and the BLAS batch loop
unchanged.  The Apple Silicon measurement below reproduces the pack-once
result and adds the 16384-call controls.

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
`72283a19`, macOS 26.5.1 on an Apple M1 with 8 GB of memory, native arm64
Python 3.11.6, NumPy 2.2.4, 15 paired samples, five warmups, and one thread.
NumPy and the extension both link to Accelerate.  CPU affinity is not
available on macOS.

| Comparison | Faster | Parity | Slower | Inconclusive | Not comparable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy versus planned | 78 planned | 3 | 0 | 12 | 24 legacy incorrect and 37 new only |
| NumPy versus planned | 60 planned | 2 | 22 NumPy | 70 | 0 |

Every correct legacy route with a conclusive classification improved or
reached parity, and none regressed.  The wider inconclusive count reflects
paired sample intervals on sub-millisecond routes rather than a measured
slowdown.

This run includes the outer-contiguous reduction schedule.  Cross-run
speedups compare the previous clean `e3a0ba93` Apple result with the current
clean result, so they establish direction and scale rather than paired
uncertainty.

| F-layout axis-1 operation | Before ms | Current ms | Speedup | Current versus NumPy |
| --- | ---: | ---: | ---: | --- |
| Mean | 0.4504 | 0.2049 | 2.20x | NumPy-faster |
| Variance | 0.9800 | 0.3257 | 3.01x | planned-faster |
| Standard deviation | 0.9751 | 0.4865 | 2.00x | planned-faster |
| Weighted average | 0.5346 | 0.1898 | 2.82x | planned-faster |

The schedule removes the previous Apple Silicon loss for variance, standard
deviation, and weighted average.  Mean improves by 2.20 times but remains
conclusively slower than NumPy at 0.2049 versus 0.1554 ms.  Median retains its
collecting backend and is not part of this schedule comparison.

The reproduction also passed 262 focused Python tests, all 232 C++ tests, and
5000 deterministic stress iterations.  The standalone buffer copied and
compiled its sources but failed at the final macOS link because its link
command omitted Accelerate, leaving the ILP64 `cblas_*` symbols unresolved.
The regular extension links to Accelerate and passed the BLAS tests.

### Apple Silicon large-scale matmul measurement

The [Apple Silicon matmul notebook](macos-matmul-hpc-results.md) records all
14 rows, and the
[machine-readable JSON](macos-matmul-hpc-results.json) retains raw samples
and linkage.  This clean `72283a19` run used five paired samples, one warmup,
one thread, and the same Accelerate backend for NumPy and solvcon.

| Case | B | R | MxKxN | Logical / expanded input MiB | Output MiB | Planned ms | Effective GMAC/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Large square | 1 | 0 | 1024x1024x1024 | 16 / 16 | 8 | 18.778 | 57.18 |
| One-sided broadcast | 8 | 1 | 1024x1024x1024 | 72 / 128 | 64 | 201.156 | 42.70 |
| Transposed lhs broadcast | 8 | 1 | 1024x1024x1024 | 72 / 128 | 64 | 196.971 | 43.61 |
| Cross broadcast | 64 | 2 | 512x512x512 | 32 / 256 | 128 | 282.221 | 30.44 |
| Same-shape batch | 4096 | 1 | 32x32x32 | 64 / 64 | 32 | 7.512 | 17.87 |
| One-sided batch | 16384 | 1 | 32x32x32 | 128 / 256 | 128 | 36.575 | 14.68 |
| Cross batch | 16384 | 2 | 32x32x32 | 2 / 256 | 128 | 38.397 | 13.98 |
| High-rank batch | 16384 | 4 | 32x32x32 | 2 / 256 | 128 | 32.128 | 16.71 |
| Dense lhs broadcast | 64 | 1 | 256x256x256 | 32.5 / 64 | 32 | 26.760 | 40.12 |
| Materialized dense lhs | 64 | 1 | 256x256x256 | 64 / 64 | 32 | 24.534 | 43.77 |
| Negative-stride lhs | 64 | 1 | 256x256x256 | 32.5 / 64 | 32 | 17.748 | 60.50 |
| Step-two lhs | 64 | 1 | 256x256x256 | 32.5 / 64 | 32 | 18.894 | 56.83 |
| Dense batch axis | 128 | 1 | 256x256x256 | 64.5 / 128 | 64 | 49.141 | 43.70 |
| Step-two batch axis | 128 | 1 | 256x256x256 | 64.5 / 128 | 64 | 61.493 | 34.92 |

The equal-work pack-once controls are stronger than the Ubuntu result on this
backend.  Negative and step-two broadcast inputs complete in 17.748 and
18.894 ms, compared with 26.760 ms for the dense broadcast control.  NumPy
takes 2961.193 and 3233.026 ms for those two layouts, so planned execution is
conclusively 160 and 170 times faster.  Packing once therefore removes the
strided matrix penalty on Accelerate without expanding the broadcast operand.

All four 32 by 32 large-batch cases are inconclusive against NumPy, and none
is NumPy-faster.  Their 13.98 through 17.87 effective GMAC/s does not provide
evidence that this prototype needs an Accelerate-specific batched call.  The
step-two batch-axis control is 25 percent slower than its dense control, but
its paired comparison with NumPy is also inconclusive.

The current `0d906a8f` follow-up expands this into 47 topology rows, eight
large vector-broadcast rows, and a 48-route batch sweep.  The
[matmul validation page](matmul-validation.md) records the complete
same-backend interpretation.  Both libraries link to Accelerate.  The earlier
roughly 160-times result reproduces, but paired controls locate it in NumPy's
negative and step-two view routes rather than zero-stride broadcasting.
Planned execution packs the physical lhs once and reaches its prepacked
control as the batch grows.

The clean `6120aec8` follow-up validates every declared lhs by rhs layout
pair.  All 31,825 Cartesian cases match NumPy.  The
[Apple Cartesian summary](macos-matmul-cartesian-summary.md) records 29,939
planned-faster, 378 parity, 1,508 inconclusive, and no NumPy-faster rows.  Its
[machine-readable summary](macos-matmul-cartesian-summary.json) retains the
per-topology counts.

The same run measures all 400 stable vector layout pairs with 100 calls per
sample.  The
[complete vector notebook](macos-matmul-vector-cartesian-results.md) and
[raw JSON](macos-matmul-vector-cartesian-results.json) show that all 48
positive-stride direct-GEMV rows retain the intended fast path.  Forced
pack-once BLAS is faster in all 72 negative or zero vector rows, with a
median time of 0.535 of current generic dispatch.  The historical forced-BLAS
helper did not test matrix packing in the remaining 280 rows because it
returned to generic execution when GEMV could not describe the matrix.

The follow-up
[Ubuntu pack crossover](ubuntu-matmul-pack-crossover-results.md) adds
explicit direct and pack-once contracts that cannot silently fall back.
Among 504 rows requiring a matrix pack, current generic execution is faster
in 339, pack-once is faster in 19, and 146 are inconclusive.  The
[stable vector boundary](ubuntu-matmul-vector-pack-boundary-results.md)
supports a conservative automatic rule at side 32, batch 4.  Its six
negative, negative-step-two, and zero-vector direction pairs have a median
generic/current ratio of 1.465, and all six show a conclusive improvement.
This adds the reusable-vector policy without
widening vector-batch matrix packing.

The clean `d1ebc1cc` Apple rerun completes that boundary check.  The
[complete Accelerate report](macos-matmul-vector-pack-boundary-results.md)
and [raw JSON](macos-matmul-vector-pack-boundary-results.json) retain all
72 rows.  At side 32 and batch 4, all six direction and layout pairs improve,
with a median generic/current ratio of 1.768.  Across all 36 rows selected
by automatic pack-once dispatch, generic/current has a median of 2.456 and
the explicit pack/current control remains at parity.

Accelerate also makes pack-once faster in all 36 rows below or outside the
portable threshold.  The implemented cross-platform boundary is therefore
safe but conservative on Apple.  A lower Apple-specific threshold is a
backend tuning option, not evidence for widening matrix packing or changing
the common plan.

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
The current macOS notebook above reproduces this schedule.  Apple variance,
standard deviation, and weighted average improve by 2.00 through 3.01 times
and change from NumPy-faster to planned-faster.  Mean improves by 2.20 times
but remains NumPy-faster, so specialized mean accumulation is the remaining
reduction target.

## Recommendations

1. Keep `LoopDomain`, `OperandMapping`, and fixed-stride inner-loop lowering
   as the shared vocabulary, but keep the three family plans and executors
   separate.  Their measured fast paths depend on different topology and cost
   rules.
2. Retain the outer-contiguous reduction schedule.  Ubuntu and Apple Silicon
   both confirm the improvement.  Investigate the remaining Apple axis mean
   gap without changing the semantic reduction plan.
3. Add an x86 SIMD backend only after profiling the remaining elementwise and
   reduction rows.  This Ubuntu machine has AVX2, but the solvcon SIMD facade
   currently falls back to generic loops outside AArch64.
4. Keep the 4096 matmul threshold as a measured initial policy, not an ABI.
   Offline backend-specific benchmarks may justify separate direct, BLAS, and
   pack thresholds.  Do not benchmark alternatives inside each public call.
5. Retain the matrix-family pack-once route for unsupported matrix strides.
   Both OpenBLAS and Accelerate equal-work controls remove the repeated
   strided-matrix cost.  Treat its 4096-work threshold as backend-tunable.
6. Freeze the matrix common layer at the current boundary.  Retain the
   positive-stride direct-GEMV route and the measured negative or zero vector
   pack-once policy for a directly describable matrix.  The explicit Ubuntu
   control rejects wider vector-batch matrix packing, and the focused
   Accelerate run confirms the portable vector boundary.  Do not add a
   platform-specific batched call or another common abstraction.  Treat a
   lower Apple vector threshold only as later backend tuning.
7. Add unary, ternary, mixed-dtype, and mixed-output executor adapters only
   when a concrete operation needs them.  The existing mapping list can
   support them without a virtual plan hierarchy.
8. Keep CallProfiler probes temporary.  Insert them around a suspected plan,
   dispatch, or kernel boundary for diagnosis, then remove them before final
   microbenchmarks so probe overhead does not enter the result.

## Verification

The prototype currently passes:

- 72 focused Python tests, including partial overlap, invalid axes and matrix
  shapes, zero contraction, and float and complex outer-contiguous routes.
- 154 fixed profiler cases across C-contiguous, F-contiguous, negative-stride,
  step-two, mixed-layout, broadcast, vector, matrix, and batch roles.
- 14 large-scale matmul cases covering 1024-square matrices, 16384 outputs,
  batch ranks zero through four, zero-stride broadcasting, and strided
  matrix and batch axes.
- 47 focused matmul topology rows, eight large vector-broadcast rows, and 48
  batch-scaling routes with matching pack and GEMM dispatch counts.
- 31,825 Cartesian lhs by rhs matmul layout pairs and a 400-row stable vector
  control covering forced generic and forced BLAS dispatch.
- 72 Apple vector pack-boundary rows covering both directions, three
  unsupported vector strides, three matrix sides, and four batch sizes.
- 5000 deterministic randomized iterations covering ranks one through four,
  broadcasting, axis permutations, negative strides, reductions, and batch
  broadcasting.

The full Python suite passes with 1409 tests, 315 skips, and three expected
failures.  All 232 C++ tests, the standalone buffer build, and project linters
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
   broadcasting, reduction, and iteration rules.  A comparison with mature
   array systems showed that they share layout iteration below separate
   elementwise, reduction, and contraction families.  The follow-up removed
   the universal four-slot operation definition, moved topology rules into
   their family plans, and kept CRTP only inside kernels that share code.
6. The current Apple Silicon rerun confirmed the outer-contiguous schedule,
   reproduced pack-once matmul against Accelerate, and found no conclusive
   need for an Apple-specific small-matrix batch kernel.
7. The focused same-backend batch sweep reproduced the large strided-input
   ratio and located it in NumPy's view route.  Planned broadcasting retains
   `B` contractions and amortizes one physical-operand pack.
8. The exhaustive Apple Cartesian rerun matched NumPy in all 31,825 cases.
   Its stable vector control retained the positive-stride fast path and
   justified a bounded negative or zero vector pack-once follow-up.
9. The explicit 72-row Accelerate boundary confirmed the portable pack-once
   rule and showed that its threshold is conservative for Apple without
   justifying wider matrix packing.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
