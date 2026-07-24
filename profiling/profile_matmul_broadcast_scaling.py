# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import contextlib
import dataclasses
import io
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import timeit

from profile_benchmark_environment import THREAD_COUNT
from profile_benchmark_environment import THREAD_VARIABLES

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import solvcon  # noqa: E402


SEED = 20260724
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
PROFILE_NAMES = (
    'execution.matmul',
    'execution.matmul.gemm',
    'execution.matmul.generic',
    'execution.matmul.pack_lhs',
    'execution.matmul.pack_rhs',
)


@dataclasses.dataclass
class BenchmarkCase:
    batch: int
    route: str
    purpose: str
    lhs: object
    rhs: object
    planned_call: object
    numpy_call: object
    plan: dict
    operands: tuple


def make_array(values):
    return solvcon.SimpleArrayFloat64(array=values)


def make_stepped(values):
    storage_shape = list(values.shape)
    storage_shape[-1] *= 2
    storage = np.empty(storage_shape, dtype='float64')
    view = storage[..., ::2]
    view[...] = values
    return view


def element_strides(values):
    itemsize = values.dtype.itemsize
    return tuple(stride // itemsize for stride in values.strides)


def root_storage(values):
    root = values
    while isinstance(root.base, np.ndarray):
        root = root.base
    return root


def describe_operand(name, values):
    strides = element_strides(values)
    return {
        'name': name,
        'shape': list(values.shape),
        'strides': list(strides),
        'c_contiguous': bool(values.flags.c_contiguous),
        'f_contiguous': bool(values.flags.f_contiguous),
        'backing_bytes': root_storage(values).nbytes,
        'logical_bytes': values.nbytes,
    }


def broadcast_mapping(values, batch_shape):
    operand_shape = values.shape[:-2]
    operand_strides = element_strides(values)[:-2]
    rank_delta = len(batch_shape) - len(operand_shape)
    aligned = [0] * rank_delta
    for axis, extent in enumerate(operand_shape):
        result_extent = batch_shape[rank_delta + axis]
        if extent == result_extent:
            aligned.append(operand_strides[axis])
        elif extent == 1:
            aligned.append(0)
        else:
            raise ValueError(
                f'cannot broadcast {extent} to {result_extent}')
    return aligned


def describe_plan(lhs, rhs):
    batch_shape = np.broadcast_shapes(lhs.shape[:-2], rhs.shape[:-2])
    lhs_strides = element_strides(lhs)
    rhs_strides = element_strides(rhs)
    lhs_matrix = lhs_strides[-2:]
    rhs_matrix = rhs_strides[-2:]
    lhs_blas = (
        (lhs_matrix[1] == 1 and lhs_matrix[0] >= lhs.shape[-1]) or
        (lhs_matrix[0] == 1 and lhs_matrix[1] >= lhs.shape[-2]))
    rhs_blas = (
        (rhs_matrix[1] == 1 and rhs_matrix[0] >= rhs.shape[-1]) or
        (rhs_matrix[0] == 1 and rhs_matrix[1] >= rhs.shape[-2]))
    return {
        'batch_shape': list(batch_shape),
        'lhs_batch_strides': broadcast_mapping(lhs, batch_shape),
        'rhs_batch_strides': broadcast_mapping(rhs, batch_shape),
        'lhs_matrix_strides': list(lhs_matrix),
        'rhs_matrix_strides': list(rhs_matrix),
        'lhs_blas_describable': lhs_blas,
        'rhs_blas_describable': rhs_blas,
        'expected_pack_lhs': not lhs_blas,
        'expected_pack_rhs': not rhs_blas,
        'expected_gemm_calls': int(np.prod(batch_shape, dtype='int64')),
    }


def make_case(batch, route, purpose, lhs, rhs):
    lhs_sa = make_array(lhs)
    rhs_sa = make_array(rhs)

    def planned_call():
        return lhs_sa._planned_matmul(rhs_sa)

    def numpy_call():
        return np.matmul(lhs, rhs)

    return BenchmarkCase(
        batch=batch,
        route=route,
        purpose=purpose,
        lhs=lhs,
        rhs=rhs,
        planned_call=planned_call,
        numpy_call=numpy_call,
        plan=describe_plan(lhs, rhs),
        operands=(
            describe_operand('lhs', lhs),
            describe_operand('rhs', rhs),
        ),
    )


def make_cases(batch, lhs_base, rhs_storage):
    rhs = rhs_storage[:batch]
    negative = lhs_base[..., ::-1]
    stepped = make_stepped(lhs_base)
    routes = (
        (
            'broadcast-dense',
            'Reuse one dense lhs through a zero batch stride.',
            lhs_base,
        ),
        (
            'materialized-dense',
            'Use one independently stored dense lhs per batch item.',
            np.repeat(lhs_base, batch, axis=0),
        ),
        (
            'broadcast-negative',
            'Pack one reversed physical lhs, then reuse it.',
            negative,
        ),
        (
            'prepacked-negative',
            'Control for the reversed lhs with packing outside timing.',
            np.ascontiguousarray(negative),
        ),
        (
            'broadcast-step2',
            'Pack one step-two physical lhs, then reuse it.',
            stepped,
        ),
        (
            'prepacked-step2',
            'Control for the step-two lhs with packing outside timing.',
            np.ascontiguousarray(stepped),
        ),
    )
    return [
        make_case(batch, route, purpose, lhs, rhs)
        for route, purpose, lhs in routes
    ]


def as_numpy(result):
    if hasattr(result, 'ndarray'):
        return np.asarray(result.ndarray, dtype='float64')
    return np.asarray(result, dtype='float64')


def check_case(case):
    dense_lhs = np.ascontiguousarray(case.lhs)
    dense_rhs = np.ascontiguousarray(case.rhs)
    expected = np.matmul(dense_lhs, dense_rhs)
    actual = as_numpy(case.planned_call())
    np.testing.assert_allclose(
        actual, expected, rtol=1e-11, atol=1e-12)


def measure_calls(cases, repeat, warmup):
    calls = {}
    for case in cases:
        calls[f'numpy/{case.route}'] = case.numpy_call
        calls[f'planned/{case.route}'] = case.planned_call
    names = tuple(calls)
    samples = {name: [] for name in names}
    for function in calls.values():
        for _ in range(warmup):
            function()
    for repetition in range(repeat):
        first = repetition % len(names)
        order = names[first:] + names[:first]
        for name in order:
            samples[name].append(timeit.timeit(calls[name], number=1))
    return samples


def quantile(values, fraction):
    return float(np.quantile(values, fraction))


def ratio_statistics(numerator, denominator):
    values = [
        left / right
        for left, right in zip(numerator, denominator)
    ]
    return {
        'median': statistics.median(values),
        'q10': quantile(values, 0.1),
        'q90': quantile(values, 0.9),
        'samples': values,
    }


def timing_rows(cases, samples):
    rows = []
    for case in cases:
        numpy_samples = samples[f'numpy/{case.route}']
        planned_samples = samples[f'planned/{case.route}']
        rows.append({
            'batch': case.batch,
            'route': case.route,
            'purpose': case.purpose,
            'operands': case.operands,
            'plan': case.plan,
            'numpy_seconds': statistics.median(numpy_samples),
            'numpy_samples_seconds': numpy_samples,
            'planned_seconds': statistics.median(planned_samples),
            'planned_samples_seconds': planned_samples,
            'numpy_over_planned': ratio_statistics(
                numpy_samples, planned_samples),
        })
    return rows


def comparison_rows(rows):
    pairs = (
        (
            'broadcast reuse / materialized batch',
            'broadcast-dense',
            'materialized-dense',
        ),
        (
            'negative view / prepacked negative',
            'broadcast-negative',
            'prepacked-negative',
        ),
        (
            'step-two view / prepacked step-two',
            'broadcast-step2',
            'prepacked-step2',
        ),
    )
    by_route = {row['route']: row for row in rows}
    comparisons = []
    for label, tested, control in pairs:
        tested_row = by_route[tested]
        control_row = by_route[control]
        comparisons.append({
            'batch': tested_row['batch'],
            'comparison': label,
            'tested_route': tested,
            'control_route': control,
            'numpy_tested_over_control': ratio_statistics(
                tested_row['numpy_samples_seconds'],
                control_row['numpy_samples_seconds'],
            ),
            'planned_tested_over_control': ratio_statistics(
                tested_row['planned_samples_seconds'],
                control_row['planned_samples_seconds'],
            ),
        })
    return comparisons


def profile_counts(node, counts):
    name = node.get('name')
    if name in counts:
        counts[name] += node.get('count', 0)
    for child in node.get('children', []):
        profile_counts(child, counts)


def trace_case(case):
    solvcon.call_profiler.reset()
    case.planned_call()
    tree = solvcon.call_profiler.result()
    counts = {name: 0 for name in PROFILE_NAMES}
    if tree:
        profile_counts(tree, counts)
    if counts['execution.matmul'] == 0:
        raise RuntimeError(
            'matmul probes are disabled; rebuild with '
            'SOLVCON_PROFILE=ON')
    expected_pack = int(case.plan['expected_pack_lhs'])
    expected_gemm = case.plan['expected_gemm_calls']
    if (
            counts['execution.matmul.pack_lhs'] != expected_pack or
            counts['execution.matmul.gemm'] != expected_gemm or
            counts['execution.matmul.generic'] != 0):
        raise RuntimeError(
            f'unexpected dispatch counts for B={case.batch} '
            f'{case.route}: {counts}')
    return {
        'batch': case.batch,
        'route': case.route,
        'plan': case.plan,
        'counts': counts,
        'profile_tree': tree,
    }


def git_revision():
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def numpy_build_configuration():
    try:
        configuration = np.show_config(mode='dicts')
        if configuration is not None:
            return configuration
    except TypeError:
        pass
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.show_config()
    return output.getvalue()


def shared_object_linkage(extension):
    if sys.platform == 'darwin':
        command = ['otool', '-L', str(extension)]
    elif sys.platform.startswith('linux'):
        command = ['ldd', str(extension)]
    else:
        return {'extension': str(extension), 'command': None}
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        'extension': str(extension),
        'command': command,
        'returncode': result.returncode,
        'output': result.stdout.strip(),
        'error': result.stderr.strip(),
    }


def metadata(args):
    dirty = subprocess.run(
        ['git', 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        'git_revision': git_revision(),
        'git_dirty': bool(dirty),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'numpy_build_configuration': numpy_build_configuration(),
        'numpy_extension_linkage': shared_object_linkage(
            np._core._multiarray_umath.__file__),
        'solvcon_extension_linkage': shared_object_linkage(
            solvcon.core._impl.__file__),
        'seed': SEED,
        'side': args.side,
        'batches': list(args.batches),
        'repeat': args.repeat,
        'warmup': args.warmup,
        'trace_only': args.trace_only,
        'thread_count': THREAD_COUNT,
        'thread_environment': {
            name: os.environ.get(name)
            for name in THREAD_VARIABLES
        },
        'requested_cpu': args.cpu,
        'cpu_affinity': sorted(os.sched_getaffinity(0))
        if hasattr(os, 'sched_getaffinity') else None,
    }


def parse_batches(value):
    try:
        batches = tuple(int(item) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            'batches must be comma-separated integers') from error
    if not batches or any(batch <= 0 for batch in batches):
        raise argparse.ArgumentTypeError(
            'batch sizes must be positive')
    return batches


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure zero-stride matmul broadcast scaling.')
    parser.add_argument(
        '--batches',
        type=parse_batches,
        default=DEFAULT_BATCHES,
        help='Comma-separated batch sizes.',
    )
    parser.add_argument(
        '--side',
        type=int,
        default=256,
        help='Square matrix side length.',
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=7,
        help='Timing samples per route.',
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=2,
        help='Warmup calls per route.',
    )
    parser.add_argument(
        '--trace-only',
        action='store_true',
        help='Record pack and BLAS probe counts without timing.',
    )
    parser.add_argument(
        '--cpu',
        type=int,
        help='Pin this process to one CPU where supported.',
    )
    parser.add_argument(
        '--output',
        type=pathlib.Path,
        default=(
            REPO_ROOT / 'profiling' / 'results' /
            'matmul_broadcast_scaling.json'
        ),
        help='JSON output path.',
    )
    args = parser.parse_args()
    if args.side <= 0:
        parser.error('--side must be positive')
    if args.repeat <= 0:
        parser.error('--repeat must be positive')
    if args.warmup < 0:
        parser.error('--warmup cannot be negative')
    return args


def set_cpu_affinity(cpu):
    if cpu is None:
        return
    if not hasattr(os, 'sched_setaffinity'):
        raise RuntimeError('--cpu is not supported on this platform')
    os.sched_setaffinity(0, {cpu})


def main():
    args = parse_args()
    set_cpu_affinity(args.cpu)
    rng = np.random.default_rng(SEED)
    lhs_base = rng.random((1, args.side, args.side), dtype='float64')
    rhs_storage = rng.random(
        (max(args.batches), args.side, args.side),
        dtype='float64',
    )
    results = []
    comparisons = []
    traces = []
    for batch in args.batches:
        cases = make_cases(batch, lhs_base, rhs_storage)
        print(f'Checking B={batch}...')
        for case in cases:
            check_case(case)
        if args.trace_only:
            for case in cases:
                traces.append(trace_case(case))
            continue
        print(f'Benchmarking B={batch}...')
        samples = measure_calls(
            cases, args.repeat, args.warmup)
        batch_rows = timing_rows(cases, samples)
        results.extend(batch_rows)
        comparisons.extend(comparison_rows(batch_rows))

    payload = {
        'metadata': metadata(args),
        'results': results,
        'comparisons': comparisons,
        'traces': traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + '\n',
        encoding='utf-8',
    )
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
