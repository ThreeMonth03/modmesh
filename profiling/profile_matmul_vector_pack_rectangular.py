# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import json
import pathlib

import numpy as np

from profile_execution_prototype import describe_operands
from profile_execution_prototype import measure_calls
from profile_execution_prototype import metadata
from profile_execution_prototype import ratio_statistics
from profile_execution_prototype import set_cpu_affinity
from profile_matmul_cartesian import matrix_axis_layouts
from profile_matmul_cartesian import vector_layouts
from profile_matmul_pack_crossover import make_calls
from profile_matmul_pack_crossover import pack_metadata


SEED = 20260725
DIMENSION_PAIRS = (
    (8, 72),
    (24, 24),
    (72, 8),
    (8, 128),
    (32, 32),
    (128, 8),
    (16, 256),
    (64, 64),
    (256, 16),
)
VECTOR_LAYOUTS = (
    ('negative-vector', 'stride-minus-1'),
    ('negative-step2-vector', 'stride-minus-2'),
    ('zero-vector', 'stride-0'),
)
CURRENT_MINIMUM_CORE_WORK = 1024
CURRENT_MINIMUM_BATCHES = 4
REUSE_MINIMUM_CORE_WORK = 576
REUSE_MINIMUM_INTENSITY = 128


def comma_separated_integers(value):
    return tuple(int(item) for item in value.split(','))


def comma_separated_pairs(value):
    result = []
    for item in value.split(','):
        inner_size, output_extent = item.lower().split('x', 1)
        result.append((int(inner_size), int(output_extent)))
    return tuple(result)


def make_operands(topology, inner_size, output_extent, batch,
                  vector_layout, rng):
    vector_values = rng.random((inner_size,), dtype='float64')
    vector = vector_layouts(vector_values)[vector_layout]
    if topology == '1d-nd':
        matrix_shape = (batch, inner_size, output_extent)
    else:
        matrix_shape = (batch, output_extent, inner_size)
    matrix_values = rng.random(matrix_shape, dtype='float64')
    matrix = matrix_axis_layouts(matrix_values)['c']
    if topology == '1d-nd':
        return vector, matrix
    return matrix, vector


def current_policy(core_work, batch):
    return (
        core_work >= CURRENT_MINIMUM_CORE_WORK and
        batch >= CURRENT_MINIMUM_BATCHES
    )


def reuse_policy(core_work, output_extent, batch):
    reuse_intensity = batch * output_extent
    return (
        core_work >= REUSE_MINIMUM_CORE_WORK and
        reuse_intensity >= REUSE_MINIMUM_INTENSITY
    )


def benchmark_case(topology, layout, vector_layout, inner_size,
                   output_extent, batch, repeat, warmup, check_only,
                   rng):
    lhs, rhs = make_operands(
        topology,
        inner_size,
        output_extent,
        batch,
        vector_layout,
        rng,
    )
    calls = make_calls(lhs, rhs, direct_supported=False)
    core_work = inner_size * output_extent
    reuse_intensity = batch * output_extent
    if check_only:
        return {
            'topology': topology,
            'layout': layout,
            'inner_size': inner_size,
            'output_extent': output_extent,
            'batch': batch,
        }

    work = max(batch * core_work, 1)
    number = max(3, min(500, 200000 // work))
    medians, samples = measure_calls(
        calls, number, repeat, warmup)
    ratios = {
        'pack_once_over_generic': ratio_statistics(
            samples['pack_once'], samples['generic']),
        'generic_over_current': ratio_statistics(
            samples['generic'], samples['current']),
        'pack_once_over_current': ratio_statistics(
            samples['pack_once'], samples['current']),
        'pack_once_over_prepacked': ratio_statistics(
            samples['pack_once'], samples['prepacked']),
        'numpy_view_over_current': ratio_statistics(
            samples['numpy_view'], samples['current']),
    }
    current_selected = current_policy(core_work, batch)
    reuse_selected = reuse_policy(
        core_work, output_extent, batch)
    return {
        'topology': topology,
        'layout': layout,
        'vector_layout': vector_layout,
        'matrix_layout': 'c',
        'inner_size': inner_size,
        'output_extent': output_extent,
        'core_work': core_work,
        'batch': batch,
        'reuse_intensity': reuse_intensity,
        'number': number,
        'current_policy_selected': current_selected,
        'reuse_policy_selected': reuse_selected,
        'combined_policy_selected': (
            current_selected or reuse_selected),
        'pack': pack_metadata(topology, lhs, rhs),
        'operands': describe_operands(lhs=lhs, rhs=rhs),
        'medians_seconds': medians,
        'samples_seconds': samples,
        'ratios': ratios,
    }


def iter_cases(args, rng):
    filters = args.case_filters
    for topology in ('1d-nd', 'nd-1d'):
        for layout, vector_layout in VECTOR_LAYOUTS:
            case_name = f'{topology}/{layout}'
            if filters and not any(
                    text in case_name for text in filters):
                continue
            for inner_size, output_extent in args.dimension_pairs:
                for batch in args.batches:
                    yield benchmark_case(
                        topology,
                        layout,
                        vector_layout,
                        inner_size,
                        output_extent,
                        batch,
                        args.repeat,
                        args.warmup,
                        args.check_only,
                        rng,
                    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Measure reusable vector packing for rectangular matmul. '
            'Each dimension pair is KxO, where O is N for 1D @ ND '
            'and M for ND @ 1D.'
        ),
    )
    parser.add_argument(
        '--dimension-pairs',
        type=comma_separated_pairs,
        default=DIMENSION_PAIRS,
    )
    parser.add_argument(
        '--batches',
        type=comma_separated_integers,
        default=(1, 2, 4, 8, 16),
    )
    parser.add_argument('--repeat', type=int, default=15)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--filter', dest='case_filters',
                        action='append', default=[])
    parser.add_argument('--check-only', action='store_true')
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
            print(
                f'Validated rectangular vector cases: {len(rows)}',
                flush=True,
            )
    if not rows:
        raise RuntimeError('no rectangular vector case matches --filter')
    print(f'Validated rectangular vector cases: {len(rows)}')
    if args.check_only or args.output is None:
        return

    args.hpc_matmul = False
    args.hpc_matmul_slow = False
    args.matmul_only = True
    args.quick = False
    args.case_count = len(rows)
    args.benchmark_script = (
        'profile_matmul_vector_pack_rectangular.py')
    run_metadata = metadata(args)
    run_metadata['seed'] = SEED
    run_metadata['dimension_pairs'] = [
        list(pair) for pair in args.dimension_pairs]
    run_metadata['batches'] = list(args.batches)
    run_metadata['vector_layouts'] = [
        list(layout) for layout in VECTOR_LAYOUTS]
    run_metadata['matrix_layout'] = 'c'
    run_metadata['current_policy'] = {
        'minimum_core_work': CURRENT_MINIMUM_CORE_WORK,
        'minimum_batches': CURRENT_MINIMUM_BATCHES,
    }
    run_metadata['reuse_extension'] = {
        'minimum_core_work': REUSE_MINIMUM_CORE_WORK,
        'minimum_reuse_intensity': REUSE_MINIMUM_INTENSITY,
    }
    payload = {'metadata': run_metadata, 'results': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
