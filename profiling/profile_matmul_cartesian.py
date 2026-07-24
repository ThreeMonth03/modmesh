# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import itertools
import json
import pathlib

import numpy as np

from profile_execution_prototype import BenchmarkCase
from profile_execution_prototype import benchmark_case
from profile_execution_prototype import check_case
from profile_execution_prototype import describe_operands
from profile_execution_prototype import make_array
from profile_execution_prototype import make_stepped
from profile_execution_prototype import matmul_workload
from profile_execution_prototype import metadata
from profile_execution_prototype import print_results
from profile_execution_prototype import set_cpu_affinity


SEED = 20260724
VECTOR_LAYOUT_NAMES = (
    'stride-1',
    'stride-minus-1',
    'stride-2',
    'stride-minus-2',
    'stride-0',
)
BATCH_AXIS_LAYOUT_NAMES = ('c', 'negative', 'step-2', 'zero')
MATRIX_LAYOUT_SPECS = {
    'c': ('c', 'c', 'c'),
    'f-matrix': ('f', 'c', 'c'),
    'axis-minus-2-step-2': ('c', 'step-2', 'c'),
    'axis-minus-1-step-2': ('c', 'c', 'step-2'),
    'both-matrix-axes-step-2': ('c', 'step-2', 'step-2'),
    'axis-minus-2-negative': ('c', 'negative', 'c'),
    'axis-minus-1-negative': ('c', 'c', 'negative'),
    'both-matrix-axes-negative': ('c', 'negative', 'negative'),
    'axis-minus-2-zero': ('c', 'zero', 'c'),
    'axis-minus-1-zero': ('c', 'c', 'zero'),
}


def make_zero_stride(values, axis):
    strides = list(values.strides)
    strides[axis] = 0
    return np.lib.stride_tricks.as_strided(
        values,
        shape=values.shape,
        strides=tuple(strides),
        writeable=True,
    )


def make_combined_layout(values, modes, matrix_order='c'):
    shape = values.shape
    if matrix_order == 'f':
        fastest_axes = [
            values.ndim - 2,
            values.ndim - 1,
            *range(values.ndim - 3, -1, -1),
        ]
    else:
        fastest_axes = list(range(values.ndim - 1, -1, -1))

    strides = [0] * values.ndim
    storage_size = 1
    offset = 0
    for axis in fastest_axes:
        mode = modes[axis]
        if mode == 'zero':
            strides[axis] = 0
            continue
        step = 2 if mode == 'step-2' else 1
        stride = storage_size * step
        if mode == 'negative':
            strides[axis] = -stride
            offset += (shape[axis] - 1) * stride
        else:
            strides[axis] = stride
        storage_size *= max(shape[axis], 1) * step

    storage = np.empty(storage_size, dtype='float64')
    storage.fill(0)
    itemsize = storage.dtype.itemsize
    view = np.ndarray(
        shape=shape,
        dtype='float64',
        buffer=storage,
        offset=offset * itemsize,
        strides=tuple(stride * itemsize for stride in strides),
    )
    view[...] = values
    return view


def vector_layouts(values):
    contiguous = np.ascontiguousarray(values, dtype='float64')
    layouts = {
        'stride-1': contiguous,
        'stride-minus-1': contiguous[::-1],
        'stride-2': make_stepped(contiguous),
        'stride-minus-2': make_stepped(contiguous)[::-1],
        'stride-0': make_zero_stride(contiguous, 0),
    }
    return {name: layouts[name] for name in VECTOR_LAYOUT_NAMES}


def matrix_axis_layouts(values):
    contiguous = np.ascontiguousarray(values, dtype='float64')
    batch_modes = ('c',) * (values.ndim - 2)
    return {
        name: make_combined_layout(
            contiguous,
            batch_modes + layout_spec[1:],
            matrix_order=layout_spec[0],
        )
        for name, layout_spec in MATRIX_LAYOUT_SPECS.items()
    }


def matrix_layouts(values):
    if values.ndim == 2:
        return matrix_axis_layouts(values)

    result = {}
    batch_modes = BATCH_AXIS_LAYOUT_NAMES
    for axis_name, layout_spec in MATRIX_LAYOUT_SPECS.items():
        matrix_order = layout_spec[0]
        core_modes = layout_spec[1:]
        for batch_cases in itertools.product(
                batch_modes, repeat=values.ndim - 2):
            modes = batch_cases + core_modes
            batch_names = [
                f'axis-{axis}-{mode}'
                for axis, mode in enumerate(batch_cases)
            ]
            layout = f"{axis_name}__{'__'.join(batch_names)}"
            result[layout] = make_combined_layout(
                values, modes, matrix_order)
    return result


def topology_bases(side, batch):
    rng = np.random.default_rng(SEED)
    vector_lhs = rng.random((side,), dtype='float64')
    vector_rhs = rng.random((side,), dtype='float64')
    matrix_lhs = rng.random((side, side), dtype='float64')
    matrix_rhs = rng.random((side, side), dtype='float64')
    batch_lhs = rng.random((batch, side, side), dtype='float64')
    batch_rhs = rng.random((batch, side, side), dtype='float64')
    lhs_one = rng.random((1, side, side), dtype='float64')
    rhs_one = rng.random((1, side, side), dtype='float64')
    cross_lhs = rng.random(
        (batch, 1, side, side), dtype='float64')
    cross_rhs = rng.random(
        (1, batch, side, side), dtype='float64')
    return (
        ('1d-1d', vector_lhs, vector_rhs),
        ('1d-2d', vector_lhs, matrix_rhs),
        ('2d-1d', matrix_lhs, vector_rhs),
        ('2d-2d', matrix_lhs, matrix_rhs),
        ('1d-nd', vector_lhs, batch_rhs),
        ('nd-1d', batch_lhs, vector_rhs),
        ('2d-nd', matrix_lhs, batch_rhs),
        ('nd-2d', batch_lhs, matrix_rhs),
        ('nd-nd-same-batch', batch_lhs, batch_rhs),
        ('nd-nd-lhs-broadcast', lhs_one, batch_rhs),
        ('nd-nd-rhs-broadcast', batch_lhs, rhs_one),
        ('nd-nd-cross-broadcast', cross_lhs, cross_rhs),
    )


def operand_layouts(values):
    if values.ndim == 1:
        return vector_layouts(values)
    return matrix_layouts(values)


def make_case(topology, lhs, rhs, lhs_layout, rhs_layout, number):
    lhs_array = make_array(lhs)
    rhs_array = make_array(rhs)
    control_call = None
    control_label = ''
    blas_call = None
    if (lhs.ndim == 1) != (rhs.ndim == 1):
        def run_blas(lhs=lhs_array, rhs=rhs_array):
            return lhs._planned_matmul_force_blas(rhs)

        def run_generic(lhs=lhs_array, rhs=rhs_array):
            return lhs._planned_matmul_force_generic(rhs)

        blas_call = run_blas
        control_call = run_generic
        control_label = 'same plan with forced generic execution'

    return BenchmarkCase(
        family='matmul-cartesian',
        operation='matmul',
        layout=(
            f'{topology}/lhs={lhs_layout}/rhs={rhs_layout}'),
        numpy_call=lambda lhs=lhs, rhs=rhs: np.matmul(lhs, rhs),
        planned_call=lambda lhs=lhs_array,
        rhs=rhs_array: lhs._planned_matmul(rhs),
        blas_call=blas_call,
        control_call=control_call,
        control_label=control_label,
        number=number,
        operands=describe_operands(lhs=lhs, rhs=rhs),
        workload=matmul_workload(lhs, rhs),
    )


def iter_cases(side, batch, number, filters):
    for topology, lhs_base, rhs_base in topology_bases(side, batch):
        lhs_layouts = operand_layouts(lhs_base)
        rhs_layouts = operand_layouts(rhs_base)
        for lhs_case, rhs_case in itertools.product(
                lhs_layouts.items(), rhs_layouts.items()):
            lhs_layout, lhs = lhs_case
            rhs_layout, rhs = rhs_case
            case = make_case(
                topology, lhs, rhs, lhs_layout, rhs_layout, number)
            if filters and not any(
                    text in case.name for text in filters):
                continue
            yield case


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile every legal matmul layout pair.')
    parser.add_argument('--side', type=int, default=32)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--number', type=int, default=1)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--print-results', action='store_true')
    parser.add_argument('--filter', dest='case_filters', action='append',
                        default=[])
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--output', type=pathlib.Path)
    return parser.parse_args()


def main():
    args = parse_args()
    set_cpu_affinity(args.cpu)
    rows = []
    case_count = 0
    topology_counts = {}
    for case in iter_cases(
            args.side, args.batch, args.number, args.case_filters):
        check_case(case)
        case_count += 1
        topology = case.layout.split('/', 1)[0]
        topology_counts[topology] = topology_counts.get(topology, 0) + 1
        if not args.check_only:
            rows.append(benchmark_case(
                case, args.repeat, args.warmup))
        if case_count % 250 == 0:
            print(
                f'Validated {case_count} Cartesian cases...',
                flush=True)

    if case_count == 0:
        raise RuntimeError('no Cartesian case matches --filter')
    print(f'Validated Cartesian matmul cases: {case_count}')
    if args.check_only:
        return

    if args.print_results:
        print_results(rows)
    if args.output is None:
        return
    args.hpc_matmul = False
    args.hpc_matmul_slow = False
    args.matmul_only = True
    args.quick = False
    args.case_count = case_count
    args.benchmark_script = 'profile_matmul_cartesian.py'
    args.cartesian_side = args.side
    args.cartesian_batch = args.batch
    args.cartesian_number = args.number
    run_metadata = metadata(args)
    run_metadata['cartesian_catalog'] = {
        'vector_layouts': list(VECTOR_LAYOUT_NAMES),
        'matrix_core_layouts': list(MATRIX_LAYOUT_SPECS),
        'batch_axis_layouts': list(BATCH_AXIS_LAYOUT_NAMES),
        'topology_counts': topology_counts,
    }
    payload = {'metadata': run_metadata, 'results': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
