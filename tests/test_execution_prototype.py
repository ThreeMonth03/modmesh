import unittest

import numpy as np

import solvcon


def make_array(data):
    return solvcon.SimpleArrayFloat64(array=data)


def make_stepped(data, axis=-1):
    storage_shape = list(data.shape)
    storage_shape[axis] *= 2
    storage = np.empty(storage_shape, dtype='float64')
    selection = [slice(None)] * data.ndim
    selection[axis] = slice(None, None, 2)
    view = storage[tuple(selection)]
    view[...] = data
    return view


class PlannedElementwiseTC(unittest.TestCase):

    def test_binary_layouts(self):
        base_lhs = np.arange(2 * 3 * 4, dtype='float64').reshape(2, 3, 4)
        base_rhs = np.arange(2 * 3 * 4, dtype='float64').reshape(2, 3, 4)
        base_rhs += 1.5
        layouts = {
            'contiguous': lambda values: values,
            'fortran': lambda values: np.asfortranarray(
                values, dtype='float64'),
            'transposed': lambda values: values.transpose(2, 0, 1),
            'reversed': lambda values: values[::-1, :, ::-1],
            'stepped': make_stepped,
        }
        operations = {
            'add': np.add,
            'sub': np.subtract,
            'mul': np.multiply,
            'div': np.divide,
        }

        for layout_name, layout in layouts.items():
            lhs = layout(base_lhs)
            rhs = layout(base_rhs)
            for operation, reference in operations.items():
                with self.subTest(layout=layout_name, operation=operation):
                    result = getattr(make_array(lhs),
                                     f'_planned_{operation}')(make_array(rhs))
                    np.testing.assert_allclose(
                        result.ndarray, reference(lhs, rhs))

    def test_numpy_broadcasting(self):
        lhs = np.arange(2 * 3, dtype='float64').reshape(2, 3, 1)
        lhs = make_stepped(lhs)
        rhs = np.arange(4, dtype='float64').reshape(1, 4) + 1

        result = make_array(lhs)._planned_mul(make_array(rhs))

        np.testing.assert_array_equal(result.ndarray, lhs * rhs)

    def test_scalar_and_inplace_broadcasting(self):
        source = np.arange(3 * 4, dtype='float64').reshape(3, 4)
        layouts = (source[:, ::-1], make_stepped(source))
        for destination in layouts:
            with self.subTest(strides=destination.strides):
                expected = destination + 2.5
                sarr = make_array(destination)

                scalar_result = sarr._planned_add(2.5)
                np.testing.assert_allclose(
                    scalar_result.ndarray, expected)

                rhs = np.arange(4, dtype='float64').reshape(1, 4)
                expected = destination + rhs
                sarr._planned_iadd(make_array(rhs))
                np.testing.assert_allclose(sarr.ndarray, expected)

    def test_invalid_broadcast_raises(self):
        lhs = make_array(np.zeros((2, 3), dtype='float64'))
        rhs = make_array(np.zeros((4,), dtype='float64'))

        with self.assertRaisesRegex(ValueError, 'broadcast'):
            lhs._planned_add(rhs)

    def test_zero_extent_broadcasting(self):
        lhs = solvcon.SimpleArrayFloat64(shape=(0, 1), value=1.0)
        rhs = solvcon.SimpleArrayFloat64(shape=(1, 3), value=2.0)

        result = lhs._planned_add(rhs)

        self.assertEqual((0, 3), result.shape)

    def test_inplace_partial_overlap_uses_snapshot(self):
        cases = (
            (slice(1, None), slice(None, -1)),
            (slice(None, -1), slice(1, None)),
        )
        for destination_slice, source_slice in cases:
            with self.subTest(destination=destination_slice):
                storage = np.arange(8, dtype='float64')
                destination_values = storage[destination_slice]
                source_values = storage[source_slice]
                expected = (
                    destination_values.copy() + source_values.copy())
                destination = make_array(destination_values)
                source = make_array(source_values)

                destination._planned_iadd(source)

                np.testing.assert_array_equal(
                    destination.ndarray, expected)


class PlannedReductionTC(unittest.TestCase):

    def test_value_reductions_across_layouts(self):
        base = np.arange(2 * 3 * 4, dtype='float64').reshape(2, 3, 4)
        base = base * 0.25 + 1.0
        layouts = {
            'contiguous': base,
            'transposed': base.transpose(2, 0, 1),
            'reversed': base[::-1, :, ::-1],
            'stepped': make_stepped(base),
        }
        operations = {
            'mean': np.mean,
            'var': np.var,
            'std': np.std,
            'median': np.median,
        }

        for layout_name, values in layouts.items():
            axes = (0, values.ndim - 1)
            for operation, reference in operations.items():
                with self.subTest(layout=layout_name, operation=operation):
                    result = getattr(make_array(values),
                                     f'_planned_{operation}')(axes)
                    np.testing.assert_allclose(
                        result.ndarray, reference(values, axis=axes))

    def test_full_reductions(self):
        values = np.arange(2 * 3 * 4, dtype='float64').reshape(2, 3, 4)
        values = values.transpose(2, 0, 1)
        sarr = make_array(values)

        self.assertAlmostEqual(sarr._planned_mean(), np.mean(values))
        self.assertAlmostEqual(sarr._planned_var(), np.var(values))
        self.assertAlmostEqual(sarr._planned_std(), np.std(values))
        self.assertAlmostEqual(sarr._planned_median(), np.median(values))

    def test_weighted_reduction(self):
        values = np.arange(2 * 3 * 4, dtype='float64').reshape(2, 3, 4)
        weights = np.arange(2 * 4, dtype='float64').reshape(2, 4) + 1
        expanded_weights = weights[:, None, :]
        expected = np.sum(values * expanded_weights, axis=(0, 2))
        expected /= np.sum(weights)

        result = make_array(values)._planned_average(
            (0, 2), make_array(weights))

        np.testing.assert_allclose(result.ndarray, expected)

    def test_fortran_outer_contiguous_reductions(self):
        values = np.arange(5 * 7, dtype='float64').reshape(5, 7)
        values = np.asfortranarray(
            values * 0.25 + 1.0, dtype='float64')
        weights = np.arange(7, dtype='float64') + 1.0
        sarr = make_array(values)

        operations = {
            'mean': np.mean,
            'var': np.var,
            'std': np.std,
        }
        for operation, reference in operations.items():
            with self.subTest(operation=operation):
                result = getattr(
                    sarr, f'_planned_{operation}')((1,))
                np.testing.assert_allclose(
                    result.ndarray, reference(values, axis=1))

        result = sarr._planned_average((1,), make_array(weights))
        np.testing.assert_allclose(
            result.ndarray,
            np.average(values, axis=1, weights=weights))

    def test_empty_kept_domain(self):
        values = solvcon.SimpleArrayFloat64(shape=(0, 3), value=0.0)
        result = values._planned_mean((1,))
        self.assertEqual((0,), result.shape)

    def test_empty_reduced_domain_raises(self):
        values = solvcon.SimpleArrayFloat64(shape=(2, 0), value=0.0)
        with self.assertRaisesRegex(RuntimeError, 'empty reduction'):
            values._planned_mean((1,))
        with self.assertRaisesRegex(RuntimeError, 'empty reduction'):
            values._planned_median((1,))

    def test_invalid_axes_raise(self):
        values = solvcon.SimpleArrayFloat64(shape=(2, 3), value=0.0)

        with self.assertRaisesRegex(ValueError, 'duplicate'):
            values._planned_mean((0, 0))
        with self.assertRaisesRegex(IndexError, 'out of range'):
            values._planned_mean((2,))
        with self.assertRaisesRegex(ValueError, 'proper axis'):
            values._planned_mean((0, 1))


class PlannedMatmulTC(unittest.TestCase):

    @staticmethod
    def assert_matmul_equal(lhs, rhs):
        result = make_array(lhs)._planned_matmul(make_array(rhs)).ndarray
        expected = np.matmul(lhs, rhs)
        if np.ndim(expected) == 0:
            expected = np.array([expected], dtype='float64')
        np.testing.assert_allclose(result, expected)

    def test_vector_and_matrix_roles(self):
        self.assert_matmul_equal(
            np.arange(3, dtype='float64'),
            np.arange(3, dtype='float64') + 1)
        self.assert_matmul_equal(
            np.arange(3, dtype='float64'),
            np.arange(3 * 4, dtype='float64').reshape(3, 4))
        self.assert_matmul_equal(
            np.arange(2 * 3, dtype='float64').reshape(2, 3),
            np.arange(3, dtype='float64'))

    def test_non_contiguous_matrix(self):
        lhs = np.arange(4 * 3, dtype='float64').reshape(4, 3).T
        rhs = np.arange(4 * 5, dtype='float64').reshape(4, 5)
        cases = (
            (lhs, rhs[::-1, :]),
            (make_stepped(lhs), rhs),
            (lhs, make_stepped(rhs)),
        )
        for case_lhs, case_rhs in cases:
            with self.subTest(
                    lhs_strides=case_lhs.strides,
                    rhs_strides=case_rhs.strides):
                self.assert_matmul_equal(case_lhs, case_rhs)

    def test_non_contiguous_vector_roles(self):
        vector = np.arange(8, dtype='float64')
        matrix = np.arange(8 * 6, dtype='float64').reshape(8, 6)
        cases = (
            (vector[::-1], vector[::-1]),
            (make_stepped(vector), make_stepped(vector)),
            (vector[::-1], matrix),
            (make_stepped(vector), matrix),
            (vector, np.asfortranarray(matrix, dtype='float64')),
            (vector, matrix[::-1, :]),
            (vector, make_stepped(matrix)),
            (matrix.T, vector[::-1]),
            (np.asfortranarray(matrix.T, dtype='float64'), vector),
            (make_stepped(matrix.T), vector),
        )
        for case_lhs, case_rhs in cases:
            with self.subTest(
                    lhs_strides=case_lhs.strides,
                    rhs_strides=case_rhs.strides):
                self.assert_matmul_equal(case_lhs, case_rhs)

    def test_strided_blas_vector_roles(self):
        long_vector = np.arange(4096, dtype='float64') / 4096
        matrix = np.arange(
            65 * 65, dtype='float64').reshape(65, 65) / (65 * 65)
        vector = np.arange(65, dtype='float64') / 65
        batch = np.arange(
            2 * 65 * 65, dtype='float64').reshape(2, 65, 65)
        batch /= 2 * 65 * 65
        cases = (
            (make_stepped(long_vector), make_stepped(long_vector)),
            (make_stepped(vector),
             np.asfortranarray(matrix, dtype='float64')),
            (np.asfortranarray(matrix, dtype='float64'),
             make_stepped(vector)),
            (make_stepped(vector), batch),
            (batch, make_stepped(vector)),
        )
        for case_lhs, case_rhs in cases:
            with self.subTest(
                    lhs_strides=case_lhs.strides,
                    rhs_strides=case_rhs.strides):
                self.assert_matmul_equal(case_lhs, case_rhs)

    def test_batched_broadcasting(self):
        lhs = np.arange(2 * 1 * 3 * 4, dtype='float64')
        lhs = lhs.reshape(2, 1, 3, 4)
        rhs = np.arange(1 * 5 * 4 * 2, dtype='float64')
        rhs = rhs.reshape(1, 5, 4, 2)
        cases = (
            (lhs, rhs),
            (make_stepped(lhs), make_stepped(rhs)),
        )
        for case_lhs, case_rhs in cases:
            with self.subTest(
                    lhs_strides=case_lhs.strides,
                    rhs_strides=case_rhs.strides):
                self.assert_matmul_equal(case_lhs, case_rhs)

    def test_batched_vector_roles(self):
        vector = np.arange(4, dtype='float64')
        rhs = np.arange(
            2 * 1 * 4 * 3, dtype='float64').reshape(2, 1, 4, 3)
        lhs = np.arange(
            2 * 1 * 3 * 4, dtype='float64').reshape(2, 1, 3, 4)
        cases = (
            (vector, rhs),
            (vector[::-1], rhs),
            (make_stepped(vector), rhs),
            (vector, rhs[..., ::-1, :]),
            (vector, make_stepped(rhs)),
            (lhs, vector),
            (lhs, vector[::-1]),
            (lhs, make_stepped(vector)),
            (lhs[..., ::-1], vector),
            (make_stepped(lhs), vector),
        )
        for case_lhs, case_rhs in cases:
            with self.subTest(
                    lhs_strides=case_lhs.strides,
                    rhs_strides=case_rhs.strides):
                self.assert_matmul_equal(case_lhs, case_rhs)

    def test_batched_packed_blas(self):
        rng = np.random.default_rng(20260723)
        lhs = rng.random((2, 1, 17, 17), dtype='float64')
        rhs = rng.random((1, 3, 17, 17), dtype='float64')
        cases = (
            (lhs[..., ::-1], rhs),
            (make_stepped(lhs), rhs),
            (lhs, make_stepped(rhs)),
            (make_stepped(lhs), make_stepped(rhs)),
        )
        for case_lhs, case_rhs in cases:
            with self.subTest(
                    lhs_strides=case_lhs.strides,
                    rhs_strides=case_rhs.strides):
                self.assert_matmul_equal(case_lhs, case_rhs)

    def test_zero_extent_batch_broadcasting(self):
        lhs = solvcon.SimpleArrayFloat64(
            shape=(1, 0, 2, 3), value=1.0)
        rhs = solvcon.SimpleArrayFloat64(
            shape=(4, 1, 3, 5), value=2.0)

        result = lhs._planned_matmul(rhs)

        self.assertEqual((4, 0, 2, 5), result.shape)

    def test_zero_contraction(self):
        lhs = solvcon.SimpleArrayFloat64(shape=(2, 0), value=1.0)
        rhs = solvcon.SimpleArrayFloat64(shape=(0, 3), value=2.0)

        result = lhs._planned_matmul(rhs)

        self.assertEqual((2, 3), result.shape)
        np.testing.assert_array_equal(
            result.ndarray, np.zeros((2, 3), dtype='float64'))

    def test_invalid_shapes_raise(self):
        lhs = solvcon.SimpleArrayFloat64(shape=(2, 3), value=1.0)
        wrong_inner = solvcon.SimpleArrayFloat64(
            shape=(4, 2), value=1.0)
        with self.assertRaisesRegex(IndexError, 'shape mismatch'):
            lhs._planned_matmul(wrong_inner)

        batch_lhs = solvcon.SimpleArrayFloat64(
            shape=(2, 3, 4), value=1.0)
        batch_rhs = solvcon.SimpleArrayFloat64(
            shape=(5, 4, 6), value=1.0)
        with self.assertRaisesRegex(ValueError, 'broadcast'):
            batch_lhs._planned_matmul(batch_rhs)


class PlannedTypedExecutionTC(unittest.TestCase):

    def test_float_and_complex_families(self):
        cases = (
            ('float32', solvcon.SimpleArrayFloat32),
            ('float64', solvcon.SimpleArrayFloat64),
            ('complex64', solvcon.SimpleArrayComplex64),
            ('complex128', solvcon.SimpleArrayComplex128),
        )
        for dtype, array_type in cases:
            with self.subTest(dtype=dtype):
                values = np.arange(24, dtype=dtype).reshape(2, 3, 4) + 1
                if np.issubdtype(np.dtype(dtype), np.complexfloating):
                    values = values + values[::-1] * 0.25j
                rhs = np.arange(4, dtype=dtype).reshape(1, 4) + 1
                result = array_type(array=values)._planned_mul(
                    array_type(array=rhs))
                np.testing.assert_allclose(
                    result.ndarray, values * rhs, rtol=1e-5)

                transposed = values.transpose(2, 0, 1)
                result = array_type(array=transposed)._planned_var((2,))
                np.testing.assert_allclose(
                    result.ndarray, np.var(transposed, axis=2), rtol=1e-5)

                fortran = np.asfortranarray(
                    values.reshape(6, 4), dtype=dtype)
                result = array_type(array=fortran)._planned_var((1,))
                np.testing.assert_allclose(
                    result.ndarray, np.var(fortran, axis=1), rtol=1e-5)

                matrix_rhs = np.arange(
                    20, dtype=dtype).reshape(1, 1, 4, 5) + 1
                result = array_type(
                    array=values.reshape(2, 1, 3, 4))._planned_matmul(
                        array_type(array=matrix_rhs))
                np.testing.assert_allclose(
                    result.ndarray,
                    np.matmul(values.reshape(2, 1, 3, 4), matrix_rhs),
                    rtol=1e-5)


if __name__ == '__main__':
    unittest.main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
