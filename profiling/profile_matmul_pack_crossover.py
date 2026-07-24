# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import json
import pathlib

import numpy as np

from profile_execution_prototype import describe_operands
from profile_execution_prototype import make_array
from profile_execution_prototype import measure_calls
from profile_execution_prototype import metadata
from profile_execution_prototype import ratio_statistics
from profile_execution_prototype import set_cpu_affinity
from profile_matmul_cartesian import matrix_axis_layouts
from profile_matmul_cartesian import vector_layouts


SEED = 20260724
LAYOUT_PAIRS = (
    ('dense', 'stride-1', 'c'),
    ('positive-step2-vector', 'stride-2', 'c'),
    ('fortran-matrix', 'stride-1', 'f-matrix'),
    ('negative-vector', 'stride-minus-1', 'c'),
    ('negative-step2-vector', 'stride-minus-2', 'c'),
    ('zero-vector', 'stride-0', 'c'),
    ('matrix-row-step2', 'stride-1', 'axis-minus-2-step-2'),
    ('matrix-column-step2', 'stride-1', 'axis-minus-1-step-2'),
    (
        'matrix-both-step2',
        'stride-1',
        'both-matrix-axes-step-2',
    ),
    ('matrix-row-negative', 'stride-1', 'axis-minus-2-negative'),
    ('matrix-column-negative', 'stride-1', 'axis-minus-1-negative'),
    (
        'matrix-both-negative',
        'stride-1',
        'both-matrix-axes-negative',
    ),
    ('matrix-row-zero', 'stride-1', 'axis-minus-2-zero'),
    ('matrix-column-zero', 'stride-1', 'axis-minus-1-zero'),
    (
        'both-operands-unsupported',
        'stride-minus-1',
        'both-matrix-axes-step-2',
    ),
)


def comma_separated_integers(value):
    return tuple(int(item) for item in value.split(','))


def check_result(actual, expected):
    if hasattr(actual, 'ndarray'):
        actual = actual.ndarray
    np.testing.assert_allclose(
        actual, expected, rtol=1e-11, atol=1e-12)


def make_operands(topology, side, batch, vector_layout, matrix_layout,
                  rng):
    vector_values = rng.random((side,), dtype='float64')
    matrix_values = rng.random(
        (batch, side, side), dtype='float64')
    vector = vector_layouts(vector_values)[vector_layout]
    matrices = matrix_axis_layouts(matrix_values)[matrix_layout]
    if topology == '1d-nd':
        return vector, matrices
    return matrices, vector


def vector_blas_describable(vector):
    return vector.strides[0] > 0


def matrix_blas_describable(matrix):
    itemsize = matrix.dtype.itemsize
    rows, columns = matrix.shape[-2:]
    row_stride = matrix.strides[-2] // itemsize
    column_stride = matrix.strides[-1] // itemsize
    return (
        (column_stride == 1 and row_stride >= columns) or
        (row_stride == 1 and column_stride >= rows)
    )


def operand_blas_describable(operand):
    if operand.ndim == 1:
        return vector_blas_describable(operand)
    return matrix_blas_describable(operand)


def pack_metadata(topology, lhs, rhs):
    lhs_pack = not operand_blas_describable(lhs)
    rhs_pack = not operand_blas_describable(rhs)
    if topology == '1d-nd':
        lhs_reuse, rhs_reuse = rhs.shape[0], 1
    else:
        lhs_reuse, rhs_reuse = 1, lhs.shape[0]
    return {
        'lhs_pack': lhs_pack,
        'rhs_pack': rhs_pack,
        'lhs_pack_elements': lhs.size if lhs_pack else 0,
        'rhs_pack_elements': rhs.size if rhs_pack else 0,
        'lhs_reuse': lhs_reuse,
        'rhs_reuse': rhs_reuse,
    }


def make_calls(lhs, rhs, direct_supported):
    expected = np.matmul(lhs, rhs)
    packed_lhs = np.ascontiguousarray(lhs, dtype='float64')
    packed_rhs = np.ascontiguousarray(rhs, dtype='float64')
    lhs_array = make_array(lhs)
    rhs_array = make_array(rhs)
    packed_lhs_array = make_array(packed_lhs)
    packed_rhs_array = make_array(packed_rhs)
    calls = {
        'numpy_view': lambda: np.matmul(lhs, rhs),
        'numpy_prepacked': lambda: np.matmul(
            packed_lhs, packed_rhs),
        'current': lambda: lhs_array._planned_matmul(rhs_array),
        'generic': (
            lambda: lhs_array._planned_matmul_force_generic(
                rhs_array)),
        'pack_once': (
            lambda: lhs_array._planned_matmul_force_pack_once_blas(
                rhs_array)),
        'prepacked': (
            lambda: packed_lhs_array._planned_matmul(
                packed_rhs_array)),
    }
    if direct_supported:
        calls['direct'] = (
            lambda: lhs_array._planned_matmul_force_direct_blas(
                rhs_array))

    for call in calls.values():
        check_result(call(), expected)
    return calls


def benchmark_case(topology, layout, vector_layout, matrix_layout,
                   side, batch, repeat, warmup, check_only, rng):
    lhs, rhs = make_operands(
        topology,
        side,
        batch,
        vector_layout,
        matrix_layout,
        rng,
    )
    direct_supported = (
        operand_blas_describable(lhs) and
        operand_blas_describable(rhs)
    )
    calls = make_calls(lhs, rhs, direct_supported)
    if check_only:
        return {
            'topology': topology,
            'layout': layout,
            'side': side,
            'batch': batch,
        }
    work = max(batch * side * side, 1)
    number = max(3, min(500, 200000 // work))
    medians, samples = measure_calls(
        calls, number, repeat, warmup)
    ratios = {
        'generic_over_current': ratio_statistics(
            samples['generic'], samples['current']),
        'pack_once_over_current': ratio_statistics(
            samples['pack_once'], samples['current']),
        'pack_once_over_prepacked': ratio_statistics(
            samples['pack_once'], samples['prepacked']),
        'numpy_view_over_prepacked': ratio_statistics(
            samples['numpy_view'], samples['numpy_prepacked']),
        'numpy_view_over_current': ratio_statistics(
            samples['numpy_view'], samples['current']),
    }
    if direct_supported:
        ratios['direct_over_current'] = ratio_statistics(
            samples['direct'], samples['current'])
    return {
        'topology': topology,
        'layout': layout,
        'vector_layout': vector_layout,
        'matrix_layout': matrix_layout,
        'side': side,
        'batch': batch,
        'number': number,
        'direct_supported': direct_supported,
        'pack': pack_metadata(
            topology,
            lhs,
            rhs,
        ),
        'operands': describe_operands(lhs=lhs, rhs=rhs),
        'medians_seconds': medians,
        'samples_seconds': samples,
        'ratios': ratios,
    }


def iter_cases(args, rng):
    filters = args.case_filters
    for topology in ('1d-nd', 'nd-1d'):
        for layout_data in LAYOUT_PAIRS:
            layout, vector_layout, matrix_layout = layout_data
            case_name = f'{topology}/{layout}'
            if filters and not any(
                    text in case_name for text in filters):
                continue
            for side in args.sides:
                for batch in args.batches:
                    yield benchmark_case(
                        topology,
                        layout,
                        vector_layout,
                        matrix_layout,
                        side,
                        batch,
                        args.repeat,
                        args.warmup,
                        args.check_only,
                        rng,
                    )


def print_rows(rows):
    print('| Topology | Layout | S | B | Current us | '
          'Pack us | Prepacked us | Pack/current |')
    print('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for row in rows:
        medians = row['medians_seconds']
        ratio = row['ratios']['pack_once_over_current']['median']
        print(
            f"| {row['topology']} | {row['layout']} | "
            f"{row['side']} | {row['batch']} | "
            f"{medians['current'] * 1e6:.3f} | "
            f"{medians['pack_once'] * 1e6:.3f} | "
            f"{medians['prepacked'] * 1e6:.3f} | "
            f"{ratio:.3f}x |"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure the matmul pack-once crossover.')
    parser.add_argument(
        '--sides',
        type=comma_separated_integers,
        default=(8, 16, 32, 64, 128, 256),
    )
    parser.add_argument(
        '--batches',
        type=comma_separated_integers,
        default=(1, 2, 4, 8, 16, 64),
    )
    parser.add_argument('--repeat', type=int, default=15)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--filter', dest='case_filters',
                        action='append', default=[])
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--print-results', action='store_true')
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--output', type=pathlib.Path)
    return parser.parse_args()


def main():
    args = parse_args()
    set_cpu_affinity(args.cpu)
    rng = np.random.default_rng(SEED)
    rows = []
    for row in iter_cases(args, rng):
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f'Validated pack crossover cases: {len(rows)}',
                  flush=True)
    if not rows:
        raise RuntimeError('no pack crossover case matches --filter')
    print(f'Validated pack crossover cases: {len(rows)}')
    if args.check_only:
        return
    if args.print_results:
        print_rows(rows)
    if args.output is None:
        return

    args.hpc_matmul = False
    args.hpc_matmul_slow = False
    args.matmul_only = True
    args.quick = False
    args.case_count = len(rows)
    args.benchmark_script = 'profile_matmul_pack_crossover.py'
    run_metadata = metadata(args)
    run_metadata['sides'] = list(args.sides)
    run_metadata['batches'] = list(args.batches)
    run_metadata['layout_pairs'] = [
        list(layout) for layout in LAYOUT_PAIRS]
    payload = {'metadata': run_metadata, 'results': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
