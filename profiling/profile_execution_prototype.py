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


SEED = 20260722
OPERATIONS = {
    'add': np.add,
    'sub': np.subtract,
    'mul': np.multiply,
    'div': np.divide,
}


def make_array(data):
    return solvcon.SimpleArrayFloat64(array=data)


def as_numpy(value):
    if hasattr(value, 'ndarray'):
        return np.asarray(value.ndarray, dtype='float64')
    result = np.asarray(value, dtype='float64')
    return result.reshape(1) if result.ndim == 0 else result


@dataclasses.dataclass
class BenchmarkCase:
    family: str
    operation: str
    layout: str
    numpy_call: object
    planned_call: object
    legacy_call: object = None
    blas_call: object = None
    control_call: object = None
    control_label: str = ''
    number: int = 1
    note: str = ''
    operands: tuple = ()
    legacy_correct: object = None
    legacy_error_type: str = ''
    legacy_error: str = ''
    blas_correct: object = None
    blas_error_type: str = ''
    blas_error: str = ''
    reset_calls: object = None
    workload: object = None

    @property
    def name(self):
        return f'{self.family}/{self.operation}/{self.layout}'


def describe_operand(name, value):
    if np.isscalar(value):
        return {'name': name, 'scalar': float(value)}
    array = np.asarray(value, dtype='float64')
    itemsize = array.dtype.itemsize
    return {
        'name': name,
        'shape': list(array.shape),
        'strides': [stride // itemsize for stride in array.strides],
        'c_contiguous': bool(array.flags.c_contiguous),
        'f_contiguous': bool(array.flags.f_contiguous),
        'negative_stride': any(stride < 0 for stride in array.strides),
        'zero_stride': any(stride == 0 for stride in array.strides),
    }


def describe_operands(**operands):
    return tuple(
        describe_operand(name, value) for name, value in operands.items())


def root_storage(value):
    root = np.asarray(value, dtype='float64')
    while isinstance(root.base, np.ndarray):
        root = root.base
    return root


def matmul_workload(lhs, rhs):
    lhs_array = np.asarray(lhs, dtype='float64')
    rhs_array = np.asarray(rhs, dtype='float64')
    lhs_vector = lhs_array.ndim == 1
    rhs_vector = rhs_array.ndim == 1
    lhs_batch = () if lhs_vector else lhs_array.shape[:-2]
    rhs_batch = () if rhs_vector else rhs_array.shape[:-2]
    batch_shape = np.broadcast_shapes(lhs_batch, rhs_batch)
    batch_matrices = int(np.prod(batch_shape, dtype='int64'))
    rows = 1 if lhs_vector else lhs_array.shape[-2]
    inner_size = lhs_array.shape[-1]
    columns = 1 if rhs_vector else rhs_array.shape[-1]
    output_shape = list(batch_shape)
    if lhs_vector and rhs_vector:
        output_shape.append(1)
    elif lhs_vector:
        output_shape.append(columns)
    elif rhs_vector:
        output_shape.append(rows)
    else:
        output_shape.extend((rows, columns))
    itemsize = lhs_array.dtype.itemsize
    logical_input_bytes = (lhs_array.size + rhs_array.size) * itemsize
    lhs_storage = root_storage(lhs_array)
    rhs_storage = root_storage(rhs_array)
    backing_storages = {
        id(storage): storage for storage in (lhs_storage, rhs_storage)}
    backing_input_bytes = sum(
        storage.nbytes for storage in backing_storages.values())
    expanded_input_bytes = (
        batch_matrices * (rows * inner_size + inner_size * columns) *
        itemsize)
    output_bytes = batch_matrices * rows * columns * itemsize
    return {
        'batch_shape': list(batch_shape),
        'batch_rank': len(batch_shape),
        'batch_matrices': batch_matrices,
        'output_shape': output_shape,
        'rows': rows,
        'inner_size': inner_size,
        'columns': columns,
        'multiply_accumulates': (
            batch_matrices * rows * inner_size * columns),
        'logical_input_bytes': logical_input_bytes,
        'backing_input_bytes': backing_input_bytes,
        'expanded_input_bytes': expanded_input_bytes,
        'output_bytes': output_bytes,
    }


def format_operands(operands):
    descriptions = []
    for operand in operands:
        name = operand['name']
        if 'scalar' in operand:
            descriptions.append(f"{name}=scalar")
            continue
        flags = ''
        if operand['c_contiguous']:
            flags += 'C'
        if operand['f_contiguous']:
            flags += 'F'
        if operand['negative_stride']:
            flags += 'N'
        if not flags:
            flags = 'S'
        shape = 'x'.join(str(value) for value in operand['shape'])
        strides = ','.join(str(value) for value in operand['strides'])
        descriptions.append(
            f'{name}=({shape})/({strides})/{flags}')
    return '; '.join(descriptions)


def make_stepped(values, axis=-1):
    storage_shape = list(values.shape)
    storage_shape[axis] *= 2
    storage = np.empty(storage_shape, dtype='float64')
    selection = [slice(None)] * values.ndim
    selection[axis] = slice(None, None, 2)
    view = storage[tuple(selection)]
    view[...] = values
    return view


def append_matmul_case(
        cases, layout, lhs, rhs, number, note='', dense_control=False,
        force_blas_control=False, family='matmul', legacy=True, blas=True):
    lhs_sa = make_array(lhs)
    rhs_sa = make_array(rhs)
    legacy_call = None
    blas_call = None
    if legacy:
        def run_legacy(lhs=lhs_sa, rhs=rhs_sa):
            return lhs.matmul(rhs)

        legacy_call = run_legacy
    if blas:
        def run_blas(lhs=lhs_sa, rhs=rhs_sa):
            return lhs.matmul_blas(rhs)

        blas_call = run_blas
    control_call = None
    control_label = ''
    if force_blas_control:
        def run_control(lhs=lhs_sa, rhs=rhs_sa):
            return lhs._planned_matmul_force_blas(rhs)

        control_call = run_control
        control_label = 'same plan with forced BLAS'
    elif dense_control:
        dense_lhs = make_array(
            np.ascontiguousarray(lhs, dtype='float64'))
        dense_rhs = make_array(
            np.ascontiguousarray(rhs, dtype='float64'))

        def run_control(lhs=dense_lhs, rhs=dense_rhs):
            return lhs._planned_matmul(rhs)

        control_call = run_control
        control_label = 'dense planned'
    cases.append(BenchmarkCase(
        family=family,
        operation='matmul',
        layout=layout,
        numpy_call=lambda lhs=lhs, rhs=rhs: np.matmul(lhs, rhs),
        legacy_call=legacy_call,
        blas_call=blas_call,
        control_call=control_call,
        control_label=control_label,
        planned_call=lambda lhs=lhs_sa,
        rhs=rhs_sa: lhs._planned_matmul(rhs),
        number=number,
        note=note,
        operands=describe_operands(lhs=lhs, rhs=rhs),
        workload=matmul_workload(lhs, rhs),
    ))


def call_inplace(destination, operand, operation, planned=False):
    prefix = '_planned_' if planned else ''
    getattr(destination, f'{prefix}i{operation}')(operand)
    return destination


def call_numpy_inplace(destination, operand, operation):
    OPERATIONS[operation](destination, operand, out=destination)
    return destination


def reset_numpy(destination, values):
    destination[...] = values


def reset_simple_array(destination, values):
    destination.ndarray[...] = values


def elementwise_cases(rng, quick):
    side = 256 if quick else 1024
    lhs_base = rng.random((side, side), dtype='float64') + 0.5
    rhs_base = rng.random((side, side), dtype='float64') + 0.5
    layouts = {
        'c-contiguous': (lhs_base, rhs_base),
        'f-contiguous': (
            np.asfortranarray(lhs_base),
            np.asfortranarray(rhs_base)),
        'negative-dense': (
            lhs_base[::-1, ::-1],
            rhs_base[::-1, ::-1]),
        'step2-inner': (
            make_stepped(lhs_base),
            make_stepped(rhs_base)),
        'mixed-c-step2': (
            lhs_base,
            make_stepped(rhs_base)),
    }
    cases = []
    for layout_name, operands in layouts.items():
        lhs, rhs = operands
        for operation, numpy_operation in OPERATIONS.items():
            lhs_sa = make_array(lhs)
            rhs_sa = make_array(rhs)
            cases.append(BenchmarkCase(
                family='elementwise-array',
                operation=operation,
                layout=layout_name,
                numpy_call=lambda lhs=lhs, rhs=rhs,
                op=numpy_operation: op(lhs, rhs),
                legacy_call=lambda lhs=lhs_sa, rhs=rhs_sa,
                op=operation: getattr(lhs, op)(rhs),
                planned_call=lambda lhs=lhs_sa, rhs=rhs_sa,
                op=operation: getattr(lhs, f'_planned_{op}')(rhs),
                number=8 if quick else 20,
                operands=describe_operands(lhs=lhs, rhs=rhs),
            ))

    scalar = 1.0001
    scalar_layouts = {
        'c-contiguous': lhs_base,
        'f-contiguous': np.asfortranarray(lhs_base),
        'negative-dense': lhs_base[::-1, ::-1],
        'step2-inner': make_stepped(lhs_base),
    }
    for layout_name, lhs in scalar_layouts.items():
        for operation, numpy_operation in OPERATIONS.items():
            lhs_sa = make_array(lhs)
            cases.append(BenchmarkCase(
                family='elementwise-scalar',
                operation=operation,
                layout=layout_name,
                numpy_call=lambda lhs=lhs, scalar=scalar,
                op=numpy_operation: op(lhs, scalar),
                legacy_call=lambda lhs=lhs_sa, scalar=scalar,
                op=operation: getattr(lhs, op)(scalar),
                planned_call=lambda lhs=lhs_sa, scalar=scalar,
                op=operation: getattr(lhs, f'_planned_{op}')(scalar),
                number=8 if quick else 20,
                operands=describe_operands(lhs=lhs, rhs=scalar),
            ))

    broadcast_side = 128 if quick else 256
    full_lhs = rng.random(
        (broadcast_side, broadcast_side), dtype='float64') + 0.5
    row_rhs = rng.random((1, broadcast_side), dtype='float64') + 0.5
    column_rhs = make_stepped(
        rng.random((broadcast_side, 1), dtype='float64') + 0.5,
        axis=1)
    column_lhs = make_stepped(
        rng.random((broadcast_side, 1), dtype='float64') + 0.5,
        axis=1)
    broadcast_layouts = {
        'rhs-row-c': (full_lhs, row_rhs),
        'rhs-column-step2': (full_lhs, column_rhs),
        'outer-step2-row': (column_lhs, row_rhs),
        'lhs-step2-rhs-row': (make_stepped(full_lhs), row_rhs),
        'negative-lhs-negative-row': (
            full_lhs[::-1, ::-1], row_rhs[:, ::-1]),
        'rank-aligned': (
            rng.random(
                (2, broadcast_side, 1), dtype='float64') + 0.5,
            rng.random((broadcast_side,), dtype='float64') + 0.5),
    }
    for layout_name, operands in broadcast_layouts.items():
        lhs, rhs = operands
        for operation, numpy_operation in OPERATIONS.items():
            lhs_sa = make_array(lhs)
            rhs_sa = make_array(rhs)
            cases.append(BenchmarkCase(
                family='elementwise-broadcast',
                operation=operation,
                layout=layout_name,
                numpy_call=lambda lhs=lhs, rhs=rhs,
                op=numpy_operation: op(lhs, rhs),
                planned_call=lambda lhs=lhs_sa, rhs=rhs_sa,
                op=operation: getattr(lhs, f'_planned_{op}')(rhs),
                number=5 if quick else 10,
                note='Legacy rejects different shapes.',
                operands=describe_operands(lhs=lhs, rhs=rhs),
            ))
    return cases


def inplace_cases(rng, quick):
    side = 128 if quick else 512
    base = rng.random((side, side), dtype='float64') * 0.1 + 1.0
    rhs_base = rng.random((side, side), dtype='float64') * 1e-4 + 1.0
    layouts = {
        'c-destination-c-rhs': (
            lambda values: values.copy(), rhs_base),
        'negative-destination-c-rhs': (
            lambda values: values.copy()[::-1, ::-1], rhs_base),
        'step2-destination-c-rhs': (
            make_stepped, rhs_base),
    }
    cases = []
    for layout_name, layout_data in layouts.items():
        destination_factory, rhs = layout_data
        for operation in OPERATIONS:
            numpy_destination = destination_factory(base)
            legacy_destination = make_array(destination_factory(base))
            planned_destination = make_array(destination_factory(base))
            legacy_rhs = make_array(rhs)
            planned_rhs = make_array(rhs)
            initial_values = np.asarray(numpy_destination).copy()
            cases.append(BenchmarkCase(
                family='inplace-array',
                operation=operation,
                layout=layout_name,
                numpy_call=lambda destination=numpy_destination,
                rhs=rhs, op=operation: call_numpy_inplace(
                    destination, rhs, op),
                legacy_call=lambda destination=legacy_destination,
                rhs=legacy_rhs, op=operation: call_inplace(
                    destination, rhs, op),
                planned_call=lambda destination=planned_destination,
                rhs=planned_rhs, op=operation: call_inplace(
                    destination, rhs, op, planned=True),
                number=10 if quick else 50,
                operands=describe_operands(
                    destination=numpy_destination, rhs=rhs),
                reset_calls={
                    'numpy': lambda destination=numpy_destination,
                    values=initial_values: reset_numpy(
                        destination, values),
                    'legacy': lambda destination=legacy_destination,
                    values=initial_values: reset_simple_array(
                        destination, values),
                    'planned': lambda destination=planned_destination,
                    values=initial_values: reset_simple_array(
                        destination, values),
                },
            ))

    row_rhs = rng.random((1, side), dtype='float64') * 1e-4 + 1.0
    broadcast_layouts = {
        'c-destination-row-rhs': lambda values: values.copy(),
        'step2-destination-row-rhs': make_stepped,
    }
    for layout_name, destination_factory in broadcast_layouts.items():
        for operation in OPERATIONS:
            numpy_destination = destination_factory(base)
            planned_destination = make_array(destination_factory(base))
            planned_rhs = make_array(row_rhs)
            initial_values = np.asarray(numpy_destination).copy()
            cases.append(BenchmarkCase(
                family='inplace-broadcast',
                operation=operation,
                layout=layout_name,
                numpy_call=lambda destination=numpy_destination,
                rhs=row_rhs, op=operation: call_numpy_inplace(
                    destination, rhs, op),
                planned_call=lambda destination=planned_destination,
                rhs=planned_rhs, op=operation: call_inplace(
                    destination, rhs, op, planned=True),
                number=10 if quick else 50,
                note='Legacy rejects different shapes.',
                operands=describe_operands(
                    destination=numpy_destination, rhs=row_rhs),
                reset_calls={
                    'numpy': lambda destination=numpy_destination,
                    values=initial_values: reset_numpy(
                        destination, values),
                    'planned': lambda destination=planned_destination,
                    values=initial_values: reset_simple_array(
                        destination, values),
                },
            ))

    scalar = 1.0001
    scalar_layouts = {
        'c-destination': lambda values: values.copy(),
        'negative-destination': lambda values: values.copy()[::-1, ::-1],
        'step2-destination': make_stepped,
    }
    for layout_name, destination_factory in scalar_layouts.items():
        for operation in OPERATIONS:
            numpy_destination = destination_factory(base)
            legacy_destination = make_array(destination_factory(base))
            planned_destination = make_array(destination_factory(base))
            initial_values = np.asarray(numpy_destination).copy()
            cases.append(BenchmarkCase(
                family='inplace-scalar',
                operation=operation,
                layout=layout_name,
                numpy_call=lambda destination=numpy_destination,
                scalar=scalar, op=operation: call_numpy_inplace(
                    destination, scalar, op),
                legacy_call=lambda destination=legacy_destination,
                scalar=scalar, op=operation: call_inplace(
                    destination, scalar, op),
                planned_call=lambda destination=planned_destination,
                scalar=scalar, op=operation: call_inplace(
                    destination, scalar, op, planned=True),
                number=10 if quick else 50,
                operands=describe_operands(
                    destination=numpy_destination, rhs=scalar),
                reset_calls={
                    'numpy': lambda destination=numpy_destination,
                    values=initial_values: reset_numpy(
                        destination, values),
                    'legacy': lambda destination=legacy_destination,
                    values=initial_values: reset_simple_array(
                        destination, values),
                    'planned': lambda destination=planned_destination,
                    values=initial_values: reset_simple_array(
                        destination, values),
                },
            ))
    return cases


def reduction_cases(rng, quick):
    side = 128 if quick else 512
    base = rng.random((side, side), dtype='float64') + 0.5
    weights = rng.random(side, dtype='float64') + 0.5
    weights_sa = make_array(weights)
    layouts = {
        'axis1-c-contiguous': base,
        'axis1-f-contiguous': np.asfortranarray(base),
        'axis1-negative-inner': base[:, ::-1],
        'axis1-step2-inner': make_stepped(base),
        'axis1-step2-outer': make_stepped(base, axis=0),
    }
    cases = []
    for layout_name, values in layouts.items():
        values_sa = make_array(values)
        operations = {
            'mean': (
                lambda values=values: np.mean(values, axis=1),
                lambda values=values_sa: values.mean(axis=1),
                lambda values=values_sa: values._planned_mean((1,))),
            'var': (
                lambda values=values: np.var(values, axis=1),
                lambda values=values_sa: values.var(axis=1),
                lambda values=values_sa: values._planned_var((1,))),
            'std': (
                lambda values=values: np.std(values, axis=1),
                lambda values=values_sa: values.std(axis=1),
                lambda values=values_sa: values._planned_std((1,))),
            'median': (
                lambda values=values: np.median(values, axis=1),
                lambda values=values_sa: values.median(axis=1),
                lambda values=values_sa: values._planned_median((1,))),
            'average': (
                lambda values=values: np.average(
                    values, axis=1, weights=weights),
                lambda values=values_sa: values.average(
                    axis=1, weight=weights_sa),
                lambda values=values_sa: values._planned_average(
                    (1,), weights_sa)),
        }
        for operation, calls in operations.items():
            operand_values = {'values': values}
            if operation == 'average':
                operand_values['weights'] = weights
            cases.append(BenchmarkCase(
                family='reduction-axis',
                operation=operation,
                layout=layout_name,
                numpy_call=calls[0],
                legacy_call=calls[1],
                planned_call=calls[2],
                number=3 if quick else 10,
                operands=describe_operands(**operand_values),
            ))

    full_layouts = {
        'full-c-contiguous': base,
        'full-f-contiguous': np.asfortranarray(base),
        'full-negative-dense': base[::-1, ::-1],
        'full-step2-inner': make_stepped(base),
    }
    for layout_name, values in full_layouts.items():
        values_sa = make_array(values)
        full_weights = rng.random(values.shape, dtype='float64') + 0.5
        full_weights_sa = make_array(full_weights)
        operations = {
            'mean': (
                lambda values=values: np.mean(values),
                lambda values=values_sa: values.mean(),
                lambda values=values_sa: values._planned_mean()),
            'var': (
                lambda values=values: np.var(values),
                lambda values=values_sa: values.var(),
                lambda values=values_sa: values._planned_var()),
            'std': (
                lambda values=values: np.std(values),
                lambda values=values_sa: values.std(),
                lambda values=values_sa: values._planned_std()),
            'median': (
                lambda values=values: np.median(values),
                lambda values=values_sa: values.median(),
                lambda values=values_sa: values._planned_median()),
            'average': (
                lambda values=values,
                weights=full_weights: np.average(values, weights=weights),
                lambda values=values_sa,
                weights=full_weights_sa: values.average(weight=weights),
                lambda values=values_sa,
                weights=full_weights_sa: values._planned_average(weights)),
        }
        for operation, calls in operations.items():
            operand_values = {'values': values}
            if operation == 'average':
                operand_values['weights'] = full_weights
            cases.append(BenchmarkCase(
                family='reduction-full',
                operation=operation,
                layout=layout_name,
                numpy_call=calls[0],
                legacy_call=calls[1],
                planned_call=calls[2],
                number=3 if quick else 10,
                operands=describe_operands(**operand_values),
            ))
    return cases


def matmul_cases(rng, quick):
    side = 64 if quick else 256
    lhs_base = rng.random((side, side), dtype='float64')
    rhs_base = rng.random((side, side), dtype='float64')
    layouts = {
        'matrix-matrix-c': (
            lhs_base, rhs_base, 'Dense GEMM control.'),
        'matrix-matrix-f': (
            np.asfortranarray(lhs_base, dtype='float64'),
            np.asfortranarray(rhs_base, dtype='float64'),
            'Both matrices have BLAS-describable F layout.'),
        'matrix-matrix-negative': (
            lhs_base[::-1, ::-1],
            rhs_base[::-1, ::-1],
            'Both matrices require signed-stride handling.'),
        'matrix-matrix-lhs-step2': (
            make_stepped(lhs_base),
            rhs_base,
            'The lhs inner stride cannot be described by GEMM.'),
        'matrix-matrix-rhs-step2': (
            lhs_base,
            make_stepped(rhs_base),
            'The rhs column stride cannot be described by GEMM.'),
        'matrix-matrix-both-step2': (
            make_stepped(lhs_base),
            make_stepped(rhs_base),
            'Both matrices require packing or generic execution.'),
        'matrix-matrix-mixed-c-f': (
            lhs_base,
            np.asfortranarray(rhs_base, dtype='float64'),
            'C lhs and F rhs are both BLAS-describable.'),
        'matrix-matrix-padded-leading-dimension': (
            make_stepped(lhs_base, axis=0),
            make_stepped(rhs_base, axis=0),
            'Positive padded leading dimensions need no packing.'),
    }
    cases = []
    for layout_name, layout_data in layouts.items():
        lhs, rhs, note = layout_data
        append_matmul_case(
            cases,
            layout_name,
            lhs,
            rhs,
            3 if quick else 7,
            note,
            dense_control=not (
                lhs.flags.c_contiguous and rhs.flags.c_contiguous),
        )

    rectangular_lhs = rng.random(
        (side // 2, side), dtype='float64')
    rectangular_rhs = rng.random(
        (side, side // 4), dtype='float64')
    small_lhs = rng.random((8, 16), dtype='float64')
    small_rhs = rng.random((16, 8), dtype='float64')
    vector_lhs = rng.random((side,), dtype='float64')
    vector_rhs = rng.random((side,), dtype='float64')
    long_vector_size = 4096 if quick else 65536
    long_vector_lhs = rng.random(
        (long_vector_size,), dtype='float64')
    long_vector_rhs = rng.random(
        (long_vector_size,), dtype='float64')
    role_layouts = {
        'matrix-matrix-rectangular': (
            rectangular_lhs,
            rectangular_rhs,
            10 if quick else 40,
            'Rectangular dense GEMM control.'),
        'matrix-matrix-small-direct': (
            small_lhs,
            small_rhs,
            100 if quick else 2000,
            'Work stays below the BLAS dispatch threshold.'),
        'vector-vector': (
            vector_lhs,
            vector_rhs,
            100 if quick else 2000,
            'Dense DOT control.'),
        'vector-vector-negative': (
            vector_lhs[::-1],
            vector_rhs[::-1],
            100 if quick else 2000,
            'DOT operands both have negative increments.'),
        'vector-vector-step2': (
            make_stepped(vector_lhs),
            make_stepped(vector_rhs),
            100 if quick else 2000,
            'DOT operands both have increment two.'),
        'vector-vector-long-c': (
            long_vector_lhs,
            long_vector_rhs,
            20,
            'Long dense DOT exercises the BLAS route.'),
        'vector-vector-long-negative': (
            long_vector_lhs[::-1],
            long_vector_rhs[::-1],
            20,
            'Long negative DOT uses signed-stride generic execution.'),
        'vector-vector-long-step2': (
            make_stepped(long_vector_lhs),
            make_stepped(long_vector_rhs),
            20,
            'Long positive-stride DOT passes increments to BLAS.'),
        'vector-matrix': (
            vector_lhs,
            rhs_base,
            20 if quick else 200,
            'Dense vector-matrix control.'),
        'vector-matrix-negative-vector': (
            vector_lhs[::-1],
            rhs_base,
            20 if quick else 200,
            'The GEMV vector has a negative increment.'),
        'vector-matrix-step2-vector': (
            make_stepped(vector_lhs),
            rhs_base,
            20 if quick else 200,
            'The GEMV vector has increment two.'),
        'vector-matrix-f-matrix': (
            vector_lhs,
            np.asfortranarray(rhs_base, dtype='float64'),
            20 if quick else 200,
            'The GEMV matrix is F-contiguous.'),
        'vector-matrix-negative-matrix': (
            vector_lhs,
            rhs_base[::-1, :],
            20 if quick else 200,
            'The contracted matrix axis has a negative stride.'),
        'vector-matrix-step2-matrix': (
            vector_lhs,
            make_stepped(rhs_base),
            20 if quick else 200,
            'The matrix column stride cannot be described by GEMV.'),
        'vector-matrix-padded-matrix': (
            vector_lhs,
            make_stepped(rhs_base, axis=0),
            20 if quick else 200,
            'The matrix has a padded leading dimension.'),
        'matrix-vector': (
            lhs_base,
            vector_rhs,
            20 if quick else 200,
            'Dense matrix-vector control.'),
        'matrix-vector-negative-vector': (
            lhs_base,
            vector_rhs[::-1],
            20 if quick else 200,
            'The GEMV vector has a negative increment.'),
        'matrix-vector-step2-vector': (
            lhs_base,
            make_stepped(vector_rhs),
            20 if quick else 200,
            'The GEMV vector has increment two.'),
        'matrix-vector-f-matrix': (
            np.asfortranarray(lhs_base, dtype='float64'),
            vector_rhs,
            20 if quick else 200,
            'The GEMV matrix is F-contiguous.'),
        'matrix-vector-negative-matrix': (
            lhs_base[:, ::-1],
            vector_rhs,
            20 if quick else 200,
            'The contracted matrix axis has a negative stride.'),
        'matrix-vector-step2-matrix': (
            make_stepped(lhs_base),
            vector_rhs,
            20 if quick else 200,
            'The matrix inner stride cannot be described by GEMV.'),
        'matrix-vector-padded-matrix': (
            make_stepped(lhs_base, axis=0),
            vector_rhs,
            20 if quick else 200,
            'The matrix has a padded leading dimension.'),
    }
    for layout_name, layout_data in role_layouts.items():
        lhs, rhs, number, note = layout_data
        append_matmul_case(
            cases,
            layout_name,
            lhs,
            rhs,
            number,
            note,
            dense_control=not (
                lhs.flags.c_contiguous and rhs.flags.c_contiguous),
        )

    batch = 3 if quick else 8
    matrix = 16 if quick else 32
    batch_lhs = rng.random(
        (batch, 1, matrix, matrix), dtype='float64')
    batch_rhs = rng.random(
        (1, batch, matrix, matrix), dtype='float64')
    same_batch_lhs = rng.random(
        (batch, matrix, matrix), dtype='float64')
    same_batch_rhs = rng.random(
        (batch, matrix, matrix), dtype='float64')
    matrix_lhs = rng.random((matrix, matrix), dtype='float64')
    matrix_rhs = rng.random((matrix, matrix), dtype='float64')
    vector = rng.random((matrix,), dtype='float64')
    vector_batch_rhs = rng.random(
        (2, batch, matrix, matrix), dtype='float64')
    vector_batch_lhs = rng.random(
        (2, batch, matrix, matrix), dtype='float64')
    batch_layouts = {
        'batch-same-shape-c': (
            same_batch_lhs,
            same_batch_rhs,
            'Dense same-shape matrix batches.'),
        'matrix-batch-matrix-c': (
            matrix_lhs,
            same_batch_rhs,
            'A rank-two lhs is implicitly reused by the batch.'),
        'batch-matrix-matrix-c': (
            same_batch_lhs,
            matrix_rhs,
            'A rank-two rhs is implicitly reused by the batch.'),
        'batch-same-shape-f-matrices': (
            same_batch_lhs.swapaxes(-1, -2),
            same_batch_rhs.swapaxes(-1, -2),
            'Each matrix has a BLAS-describable F layout.'),
        'batch-same-shape-padded-matrices': (
            make_stepped(same_batch_lhs, axis=-2),
            make_stepped(same_batch_rhs, axis=-2),
            'Each matrix has a padded positive leading dimension.'),
        'batch-broadcast-c': (
            batch_lhs,
            batch_rhs,
            'Leading matrix batch axes broadcast.'),
        'batch-broadcast-negative-matrix': (
            batch_lhs[..., ::-1, ::-1],
            batch_rhs[..., ::-1, ::-1],
            'Broadcast matrices have negative inner strides.'),
        'batch-broadcast-step2-inner': (
            make_stepped(batch_lhs),
            make_stepped(batch_rhs),
            'Broadcast matrices have step-two inner strides.'),
        'batch-broadcast-step2-batch': (
            make_stepped(batch_lhs, axis=0),
            make_stepped(batch_rhs, axis=1),
            'Matrix blocks are dense and batch axes are strided.'),
        'vector-batch-matrix-c': (
            vector,
            vector_batch_rhs,
            'One dense vector is reused across two batch axes.'),
        'vector-batch-matrix-negative-vector': (
            vector[::-1],
            vector_batch_rhs,
            'A negative-stride vector is reused across two batch axes.'),
        'vector-batch-matrix-step2-vector': (
            make_stepped(vector),
            vector_batch_rhs,
            'A step-two vector is reused across two batch axes.'),
        'vector-batch-matrix-negative-matrix': (
            vector,
            vector_batch_rhs[..., ::-1, :],
            'The batched matrix contracted axis is negative.'),
        'vector-batch-matrix-step2-matrix': (
            vector,
            make_stepped(vector_batch_rhs, axis=-1),
            'The batched matrix column stride is two.'),
        'vector-batch-matrix-padded-matrix': (
            vector,
            make_stepped(vector_batch_rhs, axis=-2),
            'The batched matrix has a padded leading dimension.'),
        'batch-matrix-vector-c': (
            vector_batch_lhs,
            vector,
            'One dense vector is reused across two batch axes.'),
        'batch-matrix-vector-negative-vector': (
            vector_batch_lhs,
            vector[::-1],
            'A negative-stride vector is reused across two batch axes.'),
        'batch-matrix-vector-step2-vector': (
            vector_batch_lhs,
            make_stepped(vector),
            'A step-two vector is reused across two batch axes.'),
        'batch-matrix-vector-negative-matrix': (
            vector_batch_lhs[..., ::-1],
            vector,
            'The batched matrix contracted axis is negative.'),
        'batch-matrix-vector-step2-matrix': (
            make_stepped(vector_batch_lhs),
            vector,
            'The batched matrix inner stride is two.'),
        'batch-matrix-vector-padded-matrix': (
            make_stepped(vector_batch_lhs, axis=-2),
            vector,
            'The batched matrix has a padded leading dimension.'),
    }
    for layout_name, layout_data in batch_layouts.items():
        lhs, rhs, note = layout_data
        append_matmul_case(
            cases,
            layout_name,
            lhs,
            rhs,
            1 if quick else 3,
            note,
            dense_control=not (
                lhs.flags.c_contiguous and rhs.flags.c_contiguous),
            force_blas_control=layout_name in (
                'vector-batch-matrix-c',
                'batch-matrix-vector-c',
            ),
            family='matmul-batch',
            legacy=False,
            blas=False,
        )
    return cases


def append_hpc_matmul_case(
        cases, layout, lhs, rhs, legacy=False, blas=False,
        control_lhs=None, control_rhs=None, control_label='', note=''):
    lhs_sa = make_array(lhs)
    rhs_sa = make_array(rhs)
    legacy_call = None
    blas_call = None
    control_call = None
    if legacy:
        def run_legacy(lhs=lhs_sa, rhs=rhs_sa):
            return lhs.matmul(rhs)
        legacy_call = run_legacy
    if blas:
        def run_blas(lhs=lhs_sa, rhs=rhs_sa):
            return lhs.matmul_blas(rhs)
        blas_call = run_blas
    if control_lhs is not None or control_rhs is not None:
        control_lhs_sa = make_array(
            lhs if control_lhs is None else control_lhs)
        control_rhs_sa = make_array(
            rhs if control_rhs is None else control_rhs)

        def run_control(lhs=control_lhs_sa, rhs=control_rhs_sa):
            return lhs._planned_matmul(rhs)
        control_call = run_control
    cases.append(BenchmarkCase(
        family='matmul-hpc',
        operation='matmul',
        layout=layout,
        numpy_call=lambda lhs=lhs, rhs=rhs: np.matmul(lhs, rhs),
        legacy_call=legacy_call,
        blas_call=blas_call,
        control_call=control_call,
        control_label=control_label,
        planned_call=lambda lhs=lhs_sa,
        rhs=rhs_sa: lhs._planned_matmul(rhs),
        number=1,
        note=note,
        operands=describe_operands(lhs=lhs, rhs=rhs),
        workload=matmul_workload(lhs, rhs),
    ))


def append_hpc_matmul_rank_cases(cases, rng):
    side = 256
    batch = 64
    lhs_matrix = rng.random((side, side), dtype='float64')
    rhs_matrix = rng.random((side, side), dtype='float64')
    lhs_batch = rng.random((batch, side, side), dtype='float64')
    rhs_batch = rng.random((batch, side, side), dtype='float64')
    rank_cases = {
        'matrix-batch-matrix-256-b64': (
            lhs_matrix,
            rhs_batch,
            'A rank-two lhs is implicitly reused by 64 GEMMs.'),
        'batch-matrix-matrix-256-b64': (
            lhs_batch,
            rhs_matrix,
            'A rank-two rhs is implicitly reused by 64 GEMMs.'),
        'batch-paired-matrix-256-b64': (
            lhs_batch,
            rhs_batch,
            'Two batches advance independently without zero strides.'),
        'matrix-negative-batch-matrix-256-b64': (
            lhs_matrix[..., ::-1],
            rhs_batch,
            'An implicit lhs broadcast packs one negative matrix.'),
        'batch-matrix-negative-matrix-256-b64': (
            lhs_batch,
            rhs_matrix[..., ::-1],
            'An implicit rhs broadcast packs one negative matrix.'),
        'matrix-step2-batch-matrix-256-b64': (
            make_stepped(lhs_matrix),
            rhs_batch,
            'An implicit lhs broadcast packs one step-two matrix.'),
        'batch-matrix-step2-matrix-256-b64': (
            lhs_batch,
            make_stepped(rhs_matrix),
            'An implicit rhs broadcast packs one step-two matrix.'),
    }
    for layout, case in rank_cases.items():
        lhs, rhs, note = case
        control_lhs = None
        control_rhs = None
        control_label = ''
        if not lhs.flags.c_contiguous:
            control_lhs = np.ascontiguousarray(lhs, dtype='float64')
            control_label = 'contiguous lhs planned'
        if not rhs.flags.c_contiguous:
            control_rhs = np.ascontiguousarray(rhs, dtype='float64')
            control_label = 'contiguous rhs planned'
        append_hpc_matmul_case(
            cases,
            layout,
            lhs,
            rhs,
            control_lhs=control_lhs,
            control_rhs=control_rhs,
            control_label=control_label,
            note=note,
        )


def hpc_matmul_cases(rng, include_slow=False):
    cases = []

    lhs_1024 = rng.random((1024, 1024), dtype='float64')
    rhs_1024 = rng.random((1024, 1024), dtype='float64')
    append_hpc_matmul_case(
        cases,
        'large-square-1024-c',
        lhs_1024,
        rhs_1024,
        legacy=True,
        blas=True,
        note='Large unbatched legacy, BLAS, planned, and NumPy baseline.',
    )

    if include_slow:
        lhs_2048 = rng.random((2048, 2048), dtype='float64')
        rhs_2048 = rng.random((2048, 2048), dtype='float64')
        append_hpc_matmul_case(
            cases,
            'large-square-2048-c',
            lhs_2048,
            rhs_2048,
            note='Opt-in slow case; the 1024 case measures legacy.',
        )

    lhs_broadcast_1024 = rng.random(
        (1, 1024, 1024), dtype='float64')
    rhs_batch_1024 = rng.random(
        (8, 1024, 1024), dtype='float64')
    append_hpc_matmul_case(
        cases,
        'large-broadcast-one-sided-1024-b8',
        lhs_broadcast_1024,
        rhs_batch_1024,
        note='One 1024-square lhs is reused by eight BLAS calls.',
    )
    append_hpc_matmul_case(
        cases,
        'large-broadcast-transposed-lhs-1024-b8',
        lhs_broadcast_1024.swapaxes(-1, -2),
        rhs_batch_1024,
        note='The broadcast lhs remains directly BLAS-describable.',
    )

    lhs_cross_512 = rng.random(
        (8, 1, 512, 512), dtype='float64')
    rhs_cross_512 = rng.random(
        (1, 8, 512, 512), dtype='float64')
    append_hpc_matmul_case(
        cases,
        'large-broadcast-cross-512-b64',
        lhs_cross_512,
        rhs_cross_512,
        note='Two batch axes cross-broadcast into 64 large matrices.',
    )
    append_hpc_matmul_rank_cases(cases, rng)

    same_lhs_32 = rng.random(
        (4096, 32, 32), dtype='float64')
    same_rhs_32 = rng.random(
        (4096, 32, 32), dtype='float64')
    append_hpc_matmul_case(
        cases,
        'large-batch-same-32-b4096',
        same_lhs_32,
        same_rhs_32,
        note='Many small independent matrices expose dispatch cost.',
    )

    lhs_one_32 = rng.random((1, 32, 32), dtype='float64')
    rhs_many_32 = rng.random(
        (16384, 32, 32), dtype='float64')
    append_hpc_matmul_case(
        cases,
        'large-batch-one-sided-32-b16384',
        lhs_one_32,
        rhs_many_32,
        note='One physical lhs is reused by 16384 outputs.',
    )

    lhs_cross_32 = rng.random(
        (128, 1, 32, 32), dtype='float64')
    rhs_cross_32 = rng.random(
        (1, 128, 32, 32), dtype='float64')
    append_hpc_matmul_case(
        cases,
        'large-batch-cross-32-b16384',
        lhs_cross_32,
        rhs_cross_32,
        note='Only 256 input matrices produce 16384 outputs.',
    )

    lhs_rank4_32 = rng.random(
        (8, 1, 16, 1, 32, 32), dtype='float64')
    rhs_rank4_32 = rng.random(
        (1, 8, 1, 16, 32, 32), dtype='float64')
    append_hpc_matmul_case(
        cases,
        'large-batch-high-rank-32-b16384',
        lhs_rank4_32,
        rhs_rank4_32,
        note='The same output count as cross-b32 with batch rank four.',
    )

    lhs_strided_256 = rng.random(
        (1, 256, 256), dtype='float64')
    rhs_batch_256 = rng.random(
        (64, 256, 256), dtype='float64')
    append_hpc_matmul_case(
        cases,
        'broadcast-dense-lhs-256-b64',
        lhs_strided_256,
        rhs_batch_256,
        note='Dense zero-stride control for repeated strided lhs cases.',
    )
    append_hpc_matmul_case(
        cases,
        'materialized-dense-lhs-256-b64',
        np.repeat(lhs_strided_256, 64, axis=0),
        rhs_batch_256,
        note='Pre-expanded control; materialization is outside timing.',
    )
    append_hpc_matmul_case(
        cases,
        'broadcast-negative-lhs-256-b64',
        lhs_strided_256[..., ::-1],
        rhs_batch_256,
        control_lhs=lhs_strided_256,
        control_label='dense planned',
        note='A repeated negative-stride lhs must not be repacked 64 times.',
    )
    append_hpc_matmul_case(
        cases,
        'broadcast-step2-lhs-256-b64',
        make_stepped(lhs_strided_256),
        rhs_batch_256,
        control_lhs=lhs_strided_256,
        control_label='dense planned',
        note='A repeated step-two lhs must not be repacked 64 times.',
    )

    lhs_batch_256 = rng.random(
        (128, 256, 256), dtype='float64')
    rhs_one_256 = rng.random(
        (1, 256, 256), dtype='float64')
    append_hpc_matmul_case(
        cases,
        'large-batch-dense-axis-256-b128',
        lhs_batch_256,
        rhs_one_256,
        note='Dense batch-axis control for the step-two batch view.',
    )
    append_hpc_matmul_case(
        cases,
        'large-batch-step2-axis-256-b128',
        make_stepped(lhs_batch_256, axis=0),
        rhs_one_256,
        note='The matrix blocks are dense but the batch axis is strided.',
    )

    vector_256 = rng.random((256,), dtype='float64')
    vector_rhs_batch_256 = rng.random(
        (64, 256, 256), dtype='float64')
    append_hpc_matmul_case(
        cases,
        'vector-batch-matrix-256-b64',
        vector_256,
        vector_rhs_batch_256,
        note='One vector is reused by 64 matrix-vector contractions.',
    )
    negative_vector_256 = vector_256[::-1]
    append_hpc_matmul_case(
        cases,
        'negative-vector-batch-matrix-256-b64',
        negative_vector_256,
        vector_rhs_batch_256,
        control_lhs=np.ascontiguousarray(
            negative_vector_256, dtype='float64'),
        control_label='dense planned',
        note='A repeated negative vector may be packed at most once.',
    )
    stepped_vector_256 = make_stepped(vector_256)
    append_hpc_matmul_case(
        cases,
        'step2-vector-batch-matrix-256-b64',
        stepped_vector_256,
        vector_rhs_batch_256,
        control_lhs=np.ascontiguousarray(
            stepped_vector_256, dtype='float64'),
        control_label='dense planned',
        note='A repeated step-two vector retains one physical copy.',
    )
    stepped_vector_rhs_256 = make_stepped(
        vector_rhs_batch_256)
    append_hpc_matmul_case(
        cases,
        'vector-batch-step2-matrix-256-b64',
        vector_256,
        stepped_vector_rhs_256,
        control_rhs=np.ascontiguousarray(
            stepped_vector_rhs_256, dtype='float64'),
        control_label='dense planned',
        note='Every rhs matrix has a step-two column stride.',
    )

    vector_lhs_batch_256 = rng.random(
        (64, 256, 256), dtype='float64')
    append_hpc_matmul_case(
        cases,
        'batch-matrix-vector-256-b64',
        vector_lhs_batch_256,
        vector_256,
        note='One vector is reused by 64 matrix-vector contractions.',
    )
    append_hpc_matmul_case(
        cases,
        'batch-matrix-negative-vector-256-b64',
        vector_lhs_batch_256,
        negative_vector_256,
        control_rhs=np.ascontiguousarray(
            negative_vector_256, dtype='float64'),
        control_label='dense planned',
        note='A repeated negative vector may be packed at most once.',
    )
    append_hpc_matmul_case(
        cases,
        'batch-matrix-step2-vector-256-b64',
        vector_lhs_batch_256,
        stepped_vector_256,
        control_rhs=np.ascontiguousarray(
            stepped_vector_256, dtype='float64'),
        control_label='dense planned',
        note='A repeated step-two vector retains one physical copy.',
    )
    stepped_vector_lhs_256 = make_stepped(
        vector_lhs_batch_256)
    append_hpc_matmul_case(
        cases,
        'batch-step2-matrix-vector-256-b64',
        stepped_vector_lhs_256,
        vector_256,
        control_lhs=np.ascontiguousarray(
            stepped_vector_lhs_256, dtype='float64'),
        control_label='dense planned',
        note='Every lhs matrix has a step-two inner stride.',
    )
    return cases


def make_cases(
        quick, hpc_matmul=False, hpc_matmul_slow=False,
        matmul_only=False):
    rng = np.random.default_rng(SEED)
    if hpc_matmul:
        return hpc_matmul_cases(rng, hpc_matmul_slow)
    if matmul_only:
        return matmul_cases(rng, quick)
    cases = []
    cases.extend(elementwise_cases(rng, quick))
    cases.extend(inplace_cases(rng, quick))
    cases.extend(reduction_cases(rng, quick))
    cases.extend(matmul_cases(rng, quick))
    return cases


def check_case(case):
    expected = as_numpy(case.numpy_call())
    planned = as_numpy(case.planned_call())
    np.testing.assert_allclose(planned, expected, rtol=1e-11, atol=1e-12)
    check_optional_route(case, 'legacy', expected)
    check_optional_route(case, 'blas', expected)


def check_optional_route(case, route, expected):
    call = getattr(case, f'{route}_call')
    if call is not None:
        try:
            result = as_numpy(call())
            np.testing.assert_allclose(
                result, expected, rtol=1e-11, atol=1e-12)
            setattr(case, f'{route}_correct', True)
        except Exception as error:
            setattr(case, f'{route}_correct', False)
            setattr(case, f'{route}_error_type', type(error).__name__)
            setattr(case, f'{route}_error', str(error).strip())


def run_correctness(cases, verbose=True):
    if verbose:
        print('Checking correctness against NumPy...')
    for case in cases:
        check_case(case)
        if not verbose:
            continue
        if case.legacy_correct is False:
            print(f'  PASS planned, FAIL legacy {case.name}: '
                  f"{case.legacy_error.splitlines()[0]}")
        elif case.blas_correct is False:
            print(f'  PASS planned, FAIL BLAS {case.name}: '
                  f"{case.blas_error.splitlines()[0]}")
        else:
            print(f'  PASS {case.name}')


def random_view(rng, values):
    axes = list(range(values.ndim))
    rng.shuffle(axes)
    result = values.transpose(axes)
    steps = (1, 1, -1, 2)
    slices = tuple(
        slice(None, None, steps[int(rng.integers(0, len(steps)))])
        for _ in range(result.ndim))
    return result[slices]


def run_stress(iterations):
    rng = np.random.default_rng(SEED + 1)
    operations = {
        'add': np.add,
        'sub': np.subtract,
        'mul': np.multiply,
        'div': np.divide,
    }
    reductions = {
        'mean': np.mean,
        'var': np.var,
        'std': np.std,
        'median': np.median,
    }
    for _ in range(iterations):
        rank = int(rng.integers(1, 5))
        result_shape = tuple(int(v) for v in rng.integers(1, 6, size=rank))
        lhs_shape = tuple(
            extent if rng.integers(0, 2) else 1
            for extent in result_shape)
        rhs_shape = tuple(
            extent if lhs_extent == 1 else 1
            for extent, lhs_extent in zip(result_shape, lhs_shape))
        lhs = rng.random(lhs_shape, dtype='float64') + 0.5
        rhs = rng.random(rhs_shape, dtype='float64') + 0.5
        if rng.integers(0, 2):
            lhs = make_stepped(lhs)
        if rng.integers(0, 2):
            rhs = rhs[..., ::-1]
        lhs_sa = make_array(lhs)
        rhs_sa = make_array(rhs)
        for operation, reference in operations.items():
            planned = getattr(lhs_sa, f'_planned_{operation}')(rhs_sa)
            np.testing.assert_allclose(
                planned.ndarray, reference(lhs, rhs),
                rtol=1e-11, atol=1e-12)

        reduction_rank = int(rng.integers(2, 5))
        shape = tuple(
            int(v) for v in rng.integers(2, 7, size=reduction_rank))
        values = rng.random(shape, dtype='float64') + 0.5
        values = random_view(rng, values)
        axis = int(rng.integers(0, values.ndim))
        values_sa = make_array(values)
        for operation, reference in reductions.items():
            planned = getattr(values_sa, f'_planned_{operation}')((axis,))
            np.testing.assert_allclose(
                planned.ndarray, reference(values, axis=axis),
                rtol=1e-11, atol=1e-12)

        batch_lhs = int(rng.integers(1, 4))
        batch_rhs = int(rng.integers(1, 4))
        m, k, n = (int(v) for v in rng.integers(1, 7, size=3))
        lhs = rng.random((batch_lhs, 1, m, k), dtype='float64')
        rhs = rng.random((1, batch_rhs, k, n), dtype='float64')
        if rng.integers(0, 2):
            lhs = make_stepped(lhs)
        if rng.integers(0, 2):
            rhs = rhs[..., ::-1]
        planned = make_array(lhs)._planned_matmul(make_array(rhs))
        np.testing.assert_allclose(
            planned.ndarray, np.matmul(lhs, rhs),
            rtol=1e-11, atol=1e-12)
    print(f'Randomized stress checks passed: {iterations} iterations')


def measure_calls(calls, number, repeat, warmup, reset_calls=None):
    names = tuple(calls)
    samples = {name: [] for name in names}
    reset_calls = reset_calls or {}
    for name, function in calls.items():
        if name in reset_calls:
            reset_calls[name]()
        for _ in range(warmup):
            function()
    for repetition in range(repeat):
        first = repetition % len(names)
        order = names[first:] + names[:first]
        for name in order:
            if name in reset_calls:
                reset_calls[name]()
            elapsed = timeit.timeit(calls[name], number=number)
            samples[name].append(elapsed / number)
    medians = {
        name: statistics.median(values)
        for name, values in samples.items()
    }
    return medians, samples


def benchmark_case(case, repeat, warmup):
    calls = {
        'numpy': case.numpy_call,
        'planned': case.planned_call,
    }
    if case.blas_call is not None and case.blas_correct is not False:
        calls['blas'] = case.blas_call
    if case.legacy_call is not None and case.legacy_correct is not False:
        calls['legacy'] = case.legacy_call
    if case.control_call is not None:
        calls['control'] = case.control_call
    timings, samples = measure_calls(
        calls, case.number, repeat, warmup, case.reset_calls)
    numpy_seconds = timings['numpy']
    planned_seconds = timings['planned']
    numpy_ratio = ratio_statistics(
        samples['numpy'], samples['planned'])
    row = {
        'family': case.family,
        'operation': case.operation,
        'layout': case.layout,
        'number': case.number,
        'note': case.note,
        'operands': case.operands,
        'workload': case.workload,
        'numpy_seconds': numpy_seconds,
        'numpy_samples_seconds': samples['numpy'],
        'planned_seconds': planned_seconds,
        'planned_samples_seconds': samples['planned'],
        'numpy_over_planned': numpy_ratio['median'],
        'numpy_over_planned_q10': numpy_ratio['q10'],
        'numpy_over_planned_q90': numpy_ratio['q90'],
        'numpy_over_planned_samples': numpy_ratio['samples'],
        'planned_vs_numpy': comparison_status(
            numpy_ratio, 'planned-faster', 'numpy-faster'),
        'legacy_seconds': None,
        'legacy_samples_seconds': None,
        'legacy_over_planned': None,
        'legacy_correct': case.legacy_correct,
        'legacy_error_type': case.legacy_error_type,
        'legacy_error': case.legacy_error,
        'blas_seconds': None,
        'blas_samples_seconds': None,
        'blas_over_planned': None,
        'blas_correct': case.blas_correct,
        'blas_error_type': case.blas_error_type,
        'blas_error': case.blas_error,
        'planned_vs_blas': (
            'blas-incorrect'
            if case.blas_correct is False else 'not-measured'),
        'control_label': case.control_label,
        'control_seconds': None,
        'control_samples_seconds': None,
        'control_over_planned': None,
        'planned_vs_control': 'not-measured',
        'status': (
            'legacy-incorrect'
            if case.legacy_correct is False else 'new-only'),
    }
    if 'legacy' in timings:
        legacy = timings['legacy']
        row['legacy_seconds'] = legacy
        row['legacy_samples_seconds'] = samples['legacy']
        legacy_ratio = ratio_statistics(
            samples['legacy'], samples['planned'])
        row['legacy_over_planned'] = legacy_ratio['median']
        row['legacy_over_planned_q10'] = legacy_ratio['q10']
        row['legacy_over_planned_q90'] = legacy_ratio['q90']
        row['legacy_over_planned_samples'] = legacy_ratio['samples']
        row['status'] = comparison_status(
            legacy_ratio, 'improved', 'regression')
    if 'blas' in timings:
        blas = timings['blas']
        row['blas_seconds'] = blas
        row['blas_samples_seconds'] = samples['blas']
        blas_ratio = ratio_statistics(
            samples['blas'], samples['planned'])
        row['blas_over_planned'] = blas_ratio['median']
        row['blas_over_planned_q10'] = blas_ratio['q10']
        row['blas_over_planned_q90'] = blas_ratio['q90']
        row['blas_over_planned_samples'] = blas_ratio['samples']
        row['planned_vs_blas'] = comparison_status(
            blas_ratio, 'planned-faster', 'blas-faster')
    if 'control' in timings:
        control = timings['control']
        row['control_seconds'] = control
        row['control_samples_seconds'] = samples['control']
        control_ratio = ratio_statistics(
            samples['control'], samples['planned'])
        row['control_over_planned'] = control_ratio['median']
        row['control_over_planned_q10'] = control_ratio['q10']
        row['control_over_planned_q90'] = control_ratio['q90']
        row['control_over_planned_samples'] = control_ratio['samples']
        row['planned_vs_control'] = comparison_status(
            control_ratio, 'planned-faster', 'control-faster')
    return row


def ratio_statistics(numerator, denominator):
    samples = [
        numerator_value / denominator_value
        for numerator_value, denominator_value in zip(
            numerator, denominator)
    ]
    return {
        'median': statistics.median(samples),
        'q10': float(np.quantile(samples, 0.1)),
        'q90': float(np.quantile(samples, 0.9)),
        'samples': samples,
    }


def comparison_status(ratio, faster, slower):
    if ratio['q10'] >= 1.05:
        return faster
    if ratio['q90'] < 0.95:
        return slower
    if ratio['q10'] >= 0.95 and ratio['q90'] < 1.05:
        return 'parity'
    return 'inconclusive'


def print_results(rows):
    print('\n| family | operation | scenario | operands | NumPy ms | '
          'legacy ms | BLAS ms | control ms | planned ms | '
          'legacy / planned (q10..q90) | '
          'BLAS / planned (q10..q90) | '
          'control / planned (q10..q90) | '
          'NumPy / planned (q10..q90) | '
          'legacy status | planned vs BLAS | planned vs control | '
          'planned vs NumPy |')
    print('| --- | --- | --- | --- | ---: | ---: | ---: | ---: | '
          '---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |')
    for row in rows:
        legacy = row['legacy_seconds']
        blas = row['blas_seconds']
        control = row['control_seconds']
        legacy_speedup = row['legacy_over_planned']
        legacy_text = 'n/a' if legacy is None else f'{legacy * 1000:.4f}'
        blas_text = 'n/a' if blas is None else f'{blas * 1000:.4f}'
        control_text = (
            'n/a' if control is None else f'{control * 1000:.4f}')
        legacy_speedup_text = 'n/a'
        if legacy_speedup is not None:
            legacy_speedup_text = (
                f"{legacy_speedup:.3f}x "
                f"({row['legacy_over_planned_q10']:.3f}.."
                f"{row['legacy_over_planned_q90']:.3f})")
        blas_speedup_text = format_optional_ratio(
            row, 'blas_over_planned')
        control_speedup_text = format_optional_ratio(
            row, 'control_over_planned')
        numpy_speedup_text = (
            f"{row['numpy_over_planned']:.3f}x "
            f"({row['numpy_over_planned_q10']:.3f}.."
            f"{row['numpy_over_planned_q90']:.3f})")
        print(f"| {row['family']} | {row['operation']} | "
              f"{row['layout']} | {format_operands(row['operands'])} | "
              f"{row['numpy_seconds'] * 1000:.4f} | "
              f"{legacy_text} | "
              f"{blas_text} | "
              f"{control_text} | "
              f"{row['planned_seconds'] * 1000:.4f} | "
              f"{legacy_speedup_text} | "
              f"{blas_speedup_text} | "
              f"{control_speedup_text} | "
              f"{numpy_speedup_text} | "
              f"{row['status']} | {row['planned_vs_blas']} | "
              f"{row['planned_vs_control']} | "
              f"{row['planned_vs_numpy']} |")

    counts = {
        status: sum(row['status'] == status for row in rows)
        for status in (
            'improved', 'parity', 'regression', 'legacy-incorrect',
            'new-only', 'inconclusive')
    }
    print('\nStatus summary:', ', '.join(
        f'{status}={count}' for status, count in counts.items()))
    numpy_counts = {
        status: sum(row['planned_vs_numpy'] == status for row in rows)
        for status in (
            'planned-faster', 'parity', 'numpy-faster', 'inconclusive')
    }
    print('Planned vs NumPy:', ', '.join(
        f'{status}={count}' for status, count in numpy_counts.items()))


def format_optional_ratio(row, prefix):
    median = row[prefix]
    if median is None:
        return 'n/a'
    return (
        f"{median:.3f}x "
        f"({row[f'{prefix}_q10']:.3f}.."
        f"{row[f'{prefix}_q90']:.3f})")


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
    extension = pathlib.Path(extension)
    if sys.platform == 'darwin':
        command = ['otool', '-L', str(extension)]
    elif sys.platform.startswith('linux'):
        command = ['ldd', str(extension)]
    else:
        return {'extension': str(extension), 'command': None, 'output': None}

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


def extension_linkage():
    return shared_object_linkage(solvcon.core._impl.__file__)


def numpy_extension_linkage():
    return shared_object_linkage(np._core._multiarray_umath.__file__)


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
        'numpy_extension_linkage': numpy_extension_linkage(),
        'solvcon_extension_linkage': extension_linkage(),
        'seed': SEED,
        'repeat': args.repeat,
        'warmup': args.warmup,
        'quick': args.quick,
        'hpc_matmul': args.hpc_matmul,
        'hpc_matmul_slow': args.hpc_matmul_slow,
        'matmul_only': args.matmul_only,
        'case_count': args.case_count,
        'case_filters': args.case_filters,
        'benchmark_script': getattr(
            args, 'benchmark_script',
            'profile_execution_prototype.py'),
        'cartesian_side': getattr(args, 'cartesian_side', None),
        'cartesian_batch': getattr(args, 'cartesian_batch', None),
        'cartesian_number': getattr(args, 'cartesian_number', None),
        'requested_cpu': args.cpu,
        'cpu_affinity': sorted(os.sched_getaffinity(0))
        if hasattr(os, 'sched_getaffinity') else None,
        'thread_count': THREAD_COUNT,
        'thread_environment': {
            name: os.environ.get(name)
            for name in THREAD_VARIABLES
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare legacy and planned SimpleArray execution.')
    parser.add_argument('--quick', action='store_true',
                        help='Use smaller arrays and fewer calls.')
    parser.add_argument(
        '--hpc-matmul',
        action='store_true',
        help='Run the isolated large-matrix and large-batch matmul suite.',
    )
    parser.add_argument(
        '--hpc-matmul-slow',
        action='store_true',
        help='Include the optional 2048-square case in the HPC suite.',
    )
    parser.add_argument(
        '--matmul-only',
        action='store_true',
        help='Run only matmul topology cases.',
    )
    parser.add_argument('--check-only', action='store_true',
                        help='Run correctness checks without timing.')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Suppress correctness output and random stress.')
    parser.add_argument('--repeat', type=int, default=7,
                        help='Number of timing samples per route.')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Warmup calls before each timing route.')
    parser.add_argument('--output', type=pathlib.Path,
                        help='Optional JSON output path.')
    parser.add_argument('--stress', type=int, default=0,
                        help='Run this many deterministic randomized checks.')
    parser.add_argument('--filter', dest='case_filters', action='append',
                        default=[],
                        help='Run cases whose family/operation/layout name '
                             'contains this text. May be repeated.')
    parser.add_argument('--cpu', type=int,
                        help='Pin this process to one CPU on supported '
                             'platforms.')
    args = parser.parse_args()
    if args.hpc_matmul_slow:
        args.hpc_matmul = True
    if args.check_only and args.benchmark_only:
        parser.error(
            '--check-only and --benchmark-only are mutually exclusive')
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
    cases = make_cases(
        args.quick,
        args.hpc_matmul,
        args.hpc_matmul_slow,
        args.matmul_only,
    )
    if args.case_filters:
        cases = [
            case for case in cases
            if any(text in case.name for text in args.case_filters)
        ]
        if not cases:
            raise RuntimeError('no benchmark case matches --filter')
    args.case_count = len(cases)
    run_correctness(cases, verbose=not args.benchmark_only)
    if not args.benchmark_only and args.stress:
        run_stress(args.stress)
    if args.check_only:
        return

    rows = []
    for case in cases:
        print(f'Benchmarking {case.name}...')
        rows.append(benchmark_case(case, args.repeat, args.warmup))
    print_results(rows)

    if args.output is not None:
        payload = {'metadata': metadata(args), 'results': rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + '\n',
                               encoding='utf-8')
        print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
