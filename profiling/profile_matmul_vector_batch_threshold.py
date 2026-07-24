# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import argparse
import json
import pathlib

import profile_benchmark_environment  # noqa: F401

import numpy as np

from profile_execution_prototype import describe_operands
from profile_execution_prototype import make_array
from profile_execution_prototype import measure_calls
from profile_execution_prototype import metadata
from profile_execution_prototype import ratio_statistics
from profile_execution_prototype import set_cpu_affinity


SEED = 20260724


def comma_separated_integers(value):
    return tuple(int(item) for item in value.split(','))


def check_result(actual, expected):
    if hasattr(actual, 'ndarray'):
        actual = actual.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def make_route_data(topology, side, batch, rng):
    vector = rng.random((side,), dtype='float64')
    matrices = rng.random((batch, side, side), dtype='float64')
    if topology == '1d-nd':
        lhs, rhs = vector, matrices
    else:
        lhs, rhs = matrices, vector
    expected = np.matmul(lhs, rhs)
    lhs_array = make_array(lhs)
    rhs_array = make_array(rhs)
    output = make_array(np.empty_like(expected))
    calls = {
        'numpy': lambda: np.matmul(lhs, rhs),
        'planned': lambda: lhs_array._planned_matmul(rhs_array),
        'forced_blas': (
            lambda: lhs_array._planned_matmul_force_blas(rhs_array)),
        'forced_blas_into': (
            lambda: lhs_array._planned_matmul_force_blas_into(
                rhs_array, output)),
        'affine_blas_into': (
            lambda: lhs_array._planned_matmul_affine_blas_into(
                rhs_array, output)),
    }
    check_result(calls['planned'](), expected)
    check_result(calls['forced_blas'](), expected)
    calls['forced_blas_into']()
    check_result(output, expected)
    calls['affine_blas_into']()
    check_result(output, expected)
    return lhs, rhs, calls


def benchmark_case(topology, side, batch, repeat, warmup, rng):
    lhs, rhs, calls = make_route_data(topology, side, batch, rng)
    work = batch * side * side
    number = max(3, min(2000, 1000000 // work))
    medians, samples = measure_calls(
        calls, number, repeat, warmup)
    ratios = {}
    for route in calls:
        if route == 'planned':
            continue
        ratios[f'{route}_over_planned'] = ratio_statistics(
            samples[route], samples['planned'])
    return {
        'topology': topology,
        'side': side,
        'batch': batch,
        'number': number,
        'operands': describe_operands(lhs=lhs, rhs=rhs),
        'output_shape': list(np.matmul(lhs, rhs).shape),
        'medians_seconds': medians,
        'samples_seconds': samples,
        'ratios_to_planned': ratios,
    }


def print_rows(rows):
    print('| Topology | B | Shape | NumPy us | Planned us | '
          'Forced BLAS us | Preallocated us | Affine us |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for row in rows:
        medians = row['medians_seconds']
        print(
            f"| {row['topology']} | {row['batch']} | "
            f"{row['side']} | {medians['numpy'] * 1e6:.3f} | "
            f"{medians['planned'] * 1e6:.3f} | "
            f"{medians['forced_blas'] * 1e6:.3f} | "
            f"{medians['forced_blas_into'] * 1e6:.3f} | "
            f"{medians['affine_blas_into'] * 1e6:.3f} |")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Locate the batched vector BLAS crossover.')
    parser.add_argument(
        '--sides',
        type=comma_separated_integers,
        default=(2, 4, 8, 16, 24, 32, 48, 64, 96, 128),
    )
    parser.add_argument(
        '--batches',
        type=comma_separated_integers,
        default=(1, 4, 16, 64),
    )
    parser.add_argument('--repeat', type=int, default=15)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--output', type=pathlib.Path)
    return parser.parse_args()


def main():
    args = parse_args()
    set_cpu_affinity(args.cpu)
    rng = np.random.default_rng(SEED)
    rows = []
    for topology in ('1d-nd', 'nd-1d'):
        for side in args.sides:
            for batch in args.batches:
                rows.append(benchmark_case(
                    topology,
                    side,
                    batch,
                    args.repeat,
                    args.warmup,
                    rng,
                ))
    print_rows(rows)
    if args.output is None:
        return

    args.hpc_matmul = False
    args.hpc_matmul_slow = False
    args.matmul_only = True
    args.quick = False
    args.case_count = len(rows)
    args.case_filters = []
    payload = {'metadata': metadata(args), 'results': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
