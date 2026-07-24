#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/matmul_plan.hpp>
#include <solvcon/buffer/matmul.hpp>

#include <optional>

namespace solvcon
{

namespace detail
{

namespace execution
{

template <typename Array, typename T>
class MatmulExecutor
{
public:
    using value_type = T;
    using helper_type = SimpleArrayMatmulHelper<Array, value_type>;

    static Array multiply(Array const & lhs, Array const & rhs);
    static Array multiply_force_generic(
        Array const & lhs, Array const & rhs);
    static Array multiply_force_blas(Array const & lhs, Array const & rhs);
    static void multiply_force_blas_into(Array const & lhs,
                                         Array const & rhs,
                                         Array & output);
    static void multiply_affine_blas_into(Array const & lhs,
                                          Array const & rhs,
                                          Array & output);

private:
    static constexpr size_t BLAS_MINIMUM_WORK = 4096;
    static constexpr size_t DIRECT_VECTOR_BLAS_MINIMUM_WORK = 512;

    class BlasMatrixLayout
    {
    public:
        BlasMatrixLayout(bool transpose, ssize_t leading_dimension)
            : m_transpose(transpose)
            , m_leading_dimension(leading_dimension)
        {
        }

        bool transpose() const noexcept { return m_transpose; }
        ssize_t leading_dimension() const noexcept
        {
            return m_leading_dimension;
        }

    private:
        bool m_transpose;
        ssize_t m_leading_dimension;
    }; /* end class BlasMatrixLayout */

    static Array execute_generic(MatmulPlan const & plan,
                                 Array const & lhs,
                                 Array const & rhs);
    static std::optional<BlasMatrixLayout> lhs_blas_layout(
        MatmulPlan const & plan);
    static std::optional<BlasMatrixLayout> rhs_blas_layout(
        MatmulPlan const & plan);
    static Array execute_matrix_blas(MatmulPlan const & plan,
                                     Array const & lhs,
                                     Array const & rhs,
                                     BlasMatrixLayout const & lhs_layout,
                                     BlasMatrixLayout const & rhs_layout);
    static Array execute_vector_blas(MatmulPlan const & plan,
                                     Array const & lhs,
                                     Array const & rhs);
    static Array execute_vector_batch_blas(MatmulPlan const & plan,
                                           Array const & lhs,
                                           Array const & rhs);
    static void execute_vector_batch_blas_into(MatmulPlan const & plan,
                                               Array const & lhs,
                                               Array const & rhs,
                                               Array & output);
    static void execute_gemv(MatmulPlan const & plan,
                             BlasMatrixLayout const & layout,
                             value_type const * lhs_data,
                             value_type const * rhs_data,
                             value_type * output_data);
    static Array execute_packed_blas(Array const & lhs,
                                     Array const & rhs);
    static Array execute_vector_blas_dispatch(
        MatmulPlan const & plan, Array const & lhs, Array const & rhs);
    static Array execute_packed_batch_blas(Array const & lhs,
                                           Array const & rhs,
                                           bool pack_lhs,
                                           bool pack_rhs);
    static bool large_enough_for_blas(Array const & lhs,
                                      Array const & rhs);
    static bool use_small_batched_vector_blas(
        MatmulPlan const & plan, size_t matrix_work);
    static Array execute_unbatched(Array const & lhs,
                                   Array const & rhs);
    static Array execute_unbatched_blas(Array const & lhs,
                                        Array const & rhs);
    static Array execute_planned(MatmulPlan const & plan,
                                 Array const & lhs,
                                 Array const & rhs);
}; /* end class MatmulExecutor */

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::execute_generic(
    MatmulPlan const & plan,
    Array const & lhs,
    Array const & rhs)
{
    SOLVCON_PROFILE_SCOPE("execution.matmul.generic");
    Array output(plan.output_shape());
    value_type * output_data = output.logical_data();
    value_type const * lhs_data = lhs.logical_data();
    value_type const * rhs_data = rhs.logical_data();
    small_vector<OperandMapping> const batch_mappings{
        plan.output_batch(), plan.lhs_batch(), plan.rhs_batch()};
    for (MappedOffsetCursor cursor(plan.batch(), batch_mappings); cursor;
         cursor.advance())
    {
        for (ssize_t row = 0; row < plan.rows(); ++row)
        {
            for (ssize_t column = 0; column < plan.columns(); ++column)
            {
                value_type total{};
                for (ssize_t inner = 0;
                     inner < plan.inner_size();
                     ++inner)
                {
                    ssize_t const lhs_offset =
                        cursor.offset(1) +
                        row * plan.lhs_row_stride() +
                        inner * plan.lhs_inner_stride();
                    ssize_t const rhs_offset =
                        cursor.offset(2) +
                        inner * plan.rhs_inner_stride() +
                        column * plan.rhs_column_stride();
                    total += lhs_data[lhs_offset] * rhs_data[rhs_offset];
                }

                ssize_t output_offset = cursor.offset(0);
                if (plan.lhs_vector() && plan.rhs_vector())
                {
                    output_offset = 0;
                }
                else if (plan.lhs_vector())
                {
                    output_offset += column;
                }
                else if (plan.rhs_vector())
                {
                    output_offset += row;
                }
                else
                {
                    output_offset += row * plan.columns() + column;
                }
                output_data[output_offset] = total;
            }
        }
    }
    return output;
}

template <typename Array, typename T>
std::optional<
    typename MatmulExecutor<Array, T>::BlasMatrixLayout>
MatmulExecutor<Array, T>::lhs_blas_layout(MatmulPlan const & plan)
{
    if (plan.lhs_inner_stride() == 1 &&
        plan.lhs_row_stride() >= plan.inner_size())
    {
        return BlasMatrixLayout(false, plan.lhs_row_stride());
    }
    if (plan.lhs_row_stride() == 1 &&
        plan.lhs_inner_stride() >= plan.rows())
    {
        return BlasMatrixLayout(true, plan.lhs_inner_stride());
    }
    return std::nullopt;
}

template <typename Array, typename T>
std::optional<
    typename MatmulExecutor<Array, T>::BlasMatrixLayout>
MatmulExecutor<Array, T>::rhs_blas_layout(MatmulPlan const & plan)
{
    if (plan.rhs_column_stride() == 1 &&
        plan.rhs_inner_stride() >= plan.columns())
    {
        return BlasMatrixLayout(false, plan.rhs_inner_stride());
    }
    if (plan.rhs_inner_stride() == 1 &&
        plan.rhs_column_stride() >= plan.inner_size())
    {
        return BlasMatrixLayout(true, plan.rhs_column_stride());
    }
    return std::nullopt;
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::execute_matrix_blas(
    MatmulPlan const & plan,
    Array const & lhs,
    Array const & rhs,
    BlasMatrixLayout const & lhs_layout,
    BlasMatrixLayout const & rhs_layout)
{
    Array output(plan.output_shape());
    value_type * output_data = output.logical_data();
    value_type const * lhs_data = lhs.logical_data();
    value_type const * rhs_data = rhs.logical_data();
    small_vector<OperandMapping> const batch_mappings{
        plan.output_batch(), plan.lhs_batch(), plan.rhs_batch()};
    for (MappedOffsetCursor cursor(plan.batch(), batch_mappings); cursor;
         cursor.advance())
    {
        SOLVCON_PROFILE_SCOPE("execution.matmul.gemm");
        gemm_blas(
            plan.rows(),
            plan.columns(),
            plan.inner_size(),
            lhs_data + cursor.offset(1),
            rhs_data + cursor.offset(2),
            output_data + cursor.offset(0),
            lhs_layout.transpose(),
            rhs_layout.transpose(),
            lhs_layout.leading_dimension(),
            rhs_layout.leading_dimension());
    }
    return output;
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::execute_vector_blas(
    MatmulPlan const & plan,
    Array const & lhs,
    Array const & rhs)
{
    Array output(plan.output_shape());
    value_type * output_data = output.logical_data();
    if (plan.lhs_vector() && plan.rhs_vector())
    {
        output_data[0] = dot_blas(
            plan.inner_size(),
            lhs.logical_data(),
            plan.lhs_inner_stride(),
            rhs.logical_data(),
            plan.rhs_inner_stride());
        return output;
    }

    std::optional<BlasMatrixLayout> const layout =
        plan.lhs_vector() ? rhs_blas_layout(plan)
                          : lhs_blas_layout(plan);
    if (!layout)
    {
        return execute_generic(plan, lhs, rhs);
    }
    execute_gemv(plan,
                 layout.value(),
                 lhs.logical_data(),
                 rhs.logical_data(),
                 output_data);
    return output;
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::execute_vector_batch_blas(
    MatmulPlan const & plan,
    Array const & lhs,
    Array const & rhs)
{
    std::optional<BlasMatrixLayout> const layout =
        plan.lhs_vector() ? rhs_blas_layout(plan)
                          : lhs_blas_layout(plan);
    if (!layout)
    {
        return execute_generic(plan, lhs, rhs);
    }
    Array output(plan.output_shape());
    execute_vector_batch_blas_into(plan, lhs, rhs, output);
    return output;
}

template <typename Array, typename T>
void MatmulExecutor<Array, T>::execute_vector_batch_blas_into(
    MatmulPlan const & plan,
    Array const & lhs,
    Array const & rhs,
    Array & output)
{
    std::optional<BlasMatrixLayout> const layout =
        plan.lhs_vector() ? rhs_blas_layout(plan)
                          : lhs_blas_layout(plan);
    if (!layout)
    {
        throw std::invalid_argument(
            "vector batch is not BLAS-describable");
    }
    value_type * output_data = output.logical_data();
    value_type const * lhs_data = lhs.logical_data();
    value_type const * rhs_data = rhs.logical_data();
    small_vector<OperandMapping> const batch_mappings{
        plan.output_batch(), plan.lhs_batch(), plan.rhs_batch()};
    for (MappedOffsetCursor cursor(plan.batch(), batch_mappings); cursor;
         cursor.advance())
    {
        execute_gemv(plan,
                     layout.value(),
                     lhs_data + cursor.offset(1),
                     rhs_data + cursor.offset(2),
                     output_data + cursor.offset(0));
    }
}

template <typename Array, typename T>
void MatmulExecutor<Array, T>::execute_gemv(
    MatmulPlan const & plan,
    BlasMatrixLayout const & layout,
    value_type const * lhs_data,
    value_type const * rhs_data,
    value_type * output_data)
{
    if (plan.lhs_vector())
    {
        bool const transpose = !layout.transpose();
        ssize_t const rows = layout.transpose()
                                 ? plan.columns()
                                 : plan.inner_size();
        ssize_t const columns = layout.transpose()
                                    ? plan.inner_size()
                                    : plan.columns();
        gemv_blas(rows,
                  columns,
                  rhs_data,
                  lhs_data,
                  output_data,
                  transpose,
                  layout.leading_dimension(),
                  plan.lhs_inner_stride());
        return;
    }
    ssize_t const rows = layout.transpose()
                             ? plan.inner_size()
                             : plan.rows();
    ssize_t const columns = layout.transpose()
                                ? plan.rows()
                                : plan.inner_size();
    gemv_blas(rows,
              columns,
              lhs_data,
              rhs_data,
              output_data,
              layout.transpose(),
              layout.leading_dimension(),
              plan.rhs_inner_stride());
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::execute_packed_blas(
    Array const & lhs, Array const & rhs)
{
    std::optional<Array> packed_lhs;
    std::optional<Array> packed_rhs;
    Array const * ready_lhs = &lhs;
    Array const * ready_rhs = &rhs;
    if (!lhs.is_c_contiguous())
    {
        SOLVCON_PROFILE_SCOPE("execution.matmul.pack_lhs");
        packed_lhs.emplace(lhs.to_row_major());
        ready_lhs = &packed_lhs.value();
    }
    if (!rhs.is_c_contiguous())
    {
        SOLVCON_PROFILE_SCOPE("execution.matmul.pack_rhs");
        packed_rhs.emplace(rhs.to_row_major());
        ready_rhs = &packed_rhs.value();
    }

    helper_type helper(*ready_lhs, *ready_rhs);
    return helper.matmul_blas();
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::execute_vector_blas_dispatch(
    MatmulPlan const & plan,
    Array const & lhs,
    Array const & rhs)
{
    if (plan.lhs_vector() && plan.rhs_vector() &&
        (plan.lhs_inner_stride() <= 0 ||
         plan.rhs_inner_stride() <= 0))
    {
        return execute_generic(plan, lhs, rhs);
    }
    if (plan.batch().rank() != 0)
    {
        std::optional<BlasMatrixLayout> const matrix_layout =
            plan.lhs_vector() ? rhs_blas_layout(plan)
                              : lhs_blas_layout(plan);
        if (!matrix_layout)
        {
            return execute_generic(plan, lhs, rhs);
        }
    }
    bool const pack_lhs =
        plan.lhs_vector()
            ? plan.lhs_inner_stride() <= 0
            : !lhs_blas_layout(plan);
    bool const pack_rhs =
        plan.rhs_vector()
            ? plan.rhs_inner_stride() <= 0
            : !rhs_blas_layout(plan);
    std::optional<Array> packed_lhs;
    std::optional<Array> packed_rhs;
    Array const * ready_lhs = &lhs;
    Array const * ready_rhs = &rhs;
    if (pack_lhs)
    {
        SOLVCON_PROFILE_SCOPE("execution.matmul.pack_lhs");
        packed_lhs.emplace(lhs.to_row_major());
        ready_lhs = &packed_lhs.value();
    }
    if (pack_rhs)
    {
        SOLVCON_PROFILE_SCOPE("execution.matmul.pack_rhs");
        packed_rhs.emplace(rhs.to_row_major());
        ready_rhs = &packed_rhs.value();
    }

    MatmulPlan const ready_plan =
        MatmulPlan::make(*ready_lhs, *ready_rhs);
    if (ready_plan.batch().rank() != 0)
    {
        return execute_vector_batch_blas(
            ready_plan, *ready_lhs, *ready_rhs);
    }
    return execute_vector_blas(ready_plan, *ready_lhs, *ready_rhs);
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::execute_packed_batch_blas(
    Array const & lhs,
    Array const & rhs,
    bool pack_lhs,
    bool pack_rhs)
{
    std::optional<Array> packed_lhs;
    std::optional<Array> packed_rhs;
    Array const * ready_lhs = &lhs;
    Array const * ready_rhs = &rhs;
    if (pack_lhs)
    {
        SOLVCON_PROFILE_SCOPE("execution.matmul.pack_lhs");
        packed_lhs.emplace(lhs.to_row_major());
        ready_lhs = &packed_lhs.value();
    }
    if (pack_rhs)
    {
        SOLVCON_PROFILE_SCOPE("execution.matmul.pack_rhs");
        packed_rhs.emplace(rhs.to_row_major());
        ready_rhs = &packed_rhs.value();
    }

    MatmulPlan const ready_plan =
        MatmulPlan::make(*ready_lhs, *ready_rhs);
    std::optional<BlasMatrixLayout> const lhs_layout =
        lhs_blas_layout(ready_plan);
    std::optional<BlasMatrixLayout> const rhs_layout =
        rhs_blas_layout(ready_plan);
    if (lhs_layout && rhs_layout)
    {
        return execute_matrix_blas(
            ready_plan,
            *ready_lhs,
            *ready_rhs,
            lhs_layout.value(),
            rhs_layout.value());
    }
    return execute_generic(ready_plan, *ready_lhs, *ready_rhs);
}

template <typename Array, typename T>
bool MatmulExecutor<Array, T>::large_enough_for_blas(
    Array const & lhs, Array const & rhs)
{
    ssize_t const rows = lhs.ndim() == 1
                             ? 1
                             : lhs.shape(lhs.ndim() - 2);
    ssize_t const columns = rhs.ndim() == 1
                                ? 1
                                : rhs.shape(rhs.ndim() - 1);
    ssize_t const inner_size = lhs.shape(lhs.ndim() - 1);
    size_t const work = static_cast<size_t>(rows) *
                        static_cast<size_t>(columns) *
                        static_cast<size_t>(inner_size);
    return work >= BLAS_MINIMUM_WORK;
}

template <typename Array, typename T>
bool MatmulExecutor<Array, T>::use_small_batched_vector_blas(
    MatmulPlan const & plan, size_t matrix_work)
{
    if (plan.batch().rank() == 0 ||
        matrix_work < DIRECT_VECTOR_BLAS_MINIMUM_WORK)
    {
        return false;
    }
    std::optional<BlasMatrixLayout> const matrix_layout =
        plan.lhs_vector() ? rhs_blas_layout(plan)
                          : lhs_blas_layout(plan);
    if (!matrix_layout)
    {
        return false;
    }
    ssize_t const vector_stride =
        plan.lhs_vector() ? plan.lhs_inner_stride()
                          : plan.rhs_inner_stride();
    return vector_stride > 0;
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::execute_unbatched_blas(
    Array const & lhs, Array const & rhs)
{
    if (lhs.is_c_contiguous() && rhs.is_c_contiguous())
    {
        helper_type helper(lhs, rhs);
        return helper.matmul_blas();
    }
    MatmulPlan const plan = MatmulPlan::make(lhs, rhs);
    if (plan.lhs_vector() || plan.rhs_vector())
    {
        return execute_vector_blas_dispatch(plan, lhs, rhs);
    }
    std::optional<BlasMatrixLayout> const lhs_layout =
        lhs_blas_layout(plan);
    std::optional<BlasMatrixLayout> const rhs_layout =
        rhs_blas_layout(plan);
    if (lhs_layout && rhs_layout)
    {
        return execute_matrix_blas(
            plan,
            lhs,
            rhs,
            lhs_layout.value(),
            rhs_layout.value());
    }
    return execute_packed_blas(lhs, rhs);
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::execute_unbatched(
    Array const & lhs, Array const & rhs)
{
    if constexpr (can_matmul_blas_v<value_type>)
    {
        if (large_enough_for_blas(lhs, rhs))
        {
            return execute_unbatched_blas(lhs, rhs);
        }
    }
    if (lhs.is_c_contiguous() && rhs.is_c_contiguous())
    {
        helper_type helper(lhs, rhs);
        return helper.matmul();
    }
    MatmulPlan const plan = MatmulPlan::make(lhs, rhs);
    return execute_generic(plan, lhs, rhs);
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::execute_planned(
    MatmulPlan const & plan,
    Array const & lhs,
    Array const & rhs)
{
    if constexpr (can_matmul_blas_v<value_type>)
    {
        size_t const matrix_work =
            static_cast<size_t>(plan.rows()) *
            static_cast<size_t>(plan.columns()) *
            static_cast<size_t>(plan.inner_size());
        bool const matrix_operands =
            !plan.lhs_vector() && !plan.rhs_vector();
        if (!matrix_operands &&
            (matrix_work >= BLAS_MINIMUM_WORK ||
             use_small_batched_vector_blas(plan, matrix_work)))
        {
            return execute_vector_blas_dispatch(plan, lhs, rhs);
        }
        if (matrix_operands && matrix_work >= BLAS_MINIMUM_WORK)
        {
            std::optional<BlasMatrixLayout> const lhs_layout =
                lhs_blas_layout(plan);
            std::optional<BlasMatrixLayout> const rhs_layout =
                rhs_blas_layout(plan);
            if (lhs_layout && rhs_layout)
            {
                return execute_matrix_blas(
                    plan,
                    lhs,
                    rhs,
                    lhs_layout.value(),
                    rhs_layout.value());
            }
            return execute_packed_batch_blas(
                lhs,
                rhs,
                !lhs_layout,
                !rhs_layout);
        }
    }
    return execute_generic(plan, lhs, rhs);
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::multiply(
    Array const & lhs, Array const & rhs)
{
    SOLVCON_PROFILE_SCOPE("execution.matmul");
    if (lhs.ndim() <= 2 && rhs.ndim() <= 2)
    {
        return execute_unbatched(lhs, rhs);
    }
    MatmulPlan const plan = MatmulPlan::make(lhs, rhs);
    return execute_planned(plan, lhs, rhs);
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::multiply_force_generic(
    Array const & lhs, Array const & rhs)
{
    MatmulPlan const plan = MatmulPlan::make(lhs, rhs);
    return execute_generic(plan, lhs, rhs);
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::multiply_force_blas(
    Array const & lhs, Array const & rhs)
{
    MatmulPlan const plan = MatmulPlan::make(lhs, rhs);
    if constexpr (can_matmul_blas_v<value_type>)
    {
        if (plan.lhs_vector() || plan.rhs_vector())
        {
            return execute_vector_blas_dispatch(plan, lhs, rhs);
        }
    }
    return execute_planned(plan, lhs, rhs);
}

template <typename Array, typename T>
void MatmulExecutor<Array, T>::multiply_force_blas_into(
    Array const & lhs, Array const & rhs, Array & output)
{
    MatmulPlan const plan = MatmulPlan::make(lhs, rhs);
    if (output.shape() != plan.output_shape())
    {
        throw std::invalid_argument(
            "preallocated matmul output shape mismatch");
    }
    if constexpr (can_matmul_blas_v<value_type>)
    {
        if (plan.batch().rank() != 0 &&
            (plan.lhs_vector() || plan.rhs_vector()))
        {
            execute_vector_batch_blas_into(plan, lhs, rhs, output);
            return;
        }
    }
    throw std::invalid_argument(
        "preallocated BLAS control requires a vector batch");
}

template <typename Array, typename T>
void MatmulExecutor<Array, T>::multiply_affine_blas_into(
    Array const & lhs, Array const & rhs, Array & output)
{
    if constexpr (!can_matmul_blas_v<value_type>)
    {
        throw std::invalid_argument(
            "affine BLAS control requires a BLAS value type");
    }
    if (!lhs.is_c_contiguous() || !rhs.is_c_contiguous() ||
        !output.is_c_contiguous())
    {
        throw std::invalid_argument(
            "affine BLAS control requires C-contiguous arrays");
    }

    bool const lhs_vector = lhs.ndim() == 1;
    bool const rhs_vector = rhs.ndim() == 1;
    if (lhs_vector == rhs_vector)
    {
        throw std::invalid_argument(
            "affine BLAS control requires one vector operand");
    }

    Array const & matrix = lhs_vector ? rhs : lhs;
    ssize_t const rows = matrix.shape(matrix.ndim() - 2);
    ssize_t const columns = matrix.shape(matrix.ndim() - 1);
    ssize_t const result_size = lhs_vector ? columns : rows;
    if (lhs.shape(lhs.ndim() - 1) !=
        rhs.shape(rhs_vector ? 0 : rhs.ndim() - 2))
    {
        throw std::invalid_argument(
            "affine BLAS control shape mismatch");
    }

    size_t const matrix_size =
        static_cast<size_t>(rows * columns);
    size_t const batch_size = matrix.size() / matrix_size;
    if (output.size() !=
        batch_size * static_cast<size_t>(result_size))
    {
        throw std::invalid_argument(
            "affine BLAS control output shape mismatch");
    }

    value_type const * lhs_data = lhs.logical_data();
    value_type const * rhs_data = rhs.logical_data();
    value_type * output_data = output.logical_data();
    for (size_t batch = 0; batch < batch_size; ++batch)
    {
        value_type const * matrix_data =
            (lhs_vector ? rhs_data : lhs_data) + batch * matrix_size;
        value_type const * vector_data =
            lhs_vector ? lhs_data : rhs_data;
        if constexpr (can_matmul_blas_v<value_type>)
        {
            gemv_blas(
                rows,
                columns,
                matrix_data,
                vector_data,
                output_data + batch * static_cast<size_t>(result_size),
                lhs_vector,
                columns,
                1);
        }
    }
}

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
