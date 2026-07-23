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

private:
    static constexpr size_t BLAS_MINIMUM_WORK = 4096;

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
    static Array execute_packed_blas(Array const & lhs,
                                     Array const & rhs);
    static bool large_enough_for_blas(Array const & lhs,
                                      Array const & rhs);
    static Array execute_unbatched(Array const & lhs,
                                   Array const & rhs);
    static Array execute_unbatched_blas(Array const & lhs,
                                        Array const & rhs,
                                        helper_type & helper);
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
Array MatmulExecutor<Array, T>::execute_packed_blas(
    Array const & lhs, Array const & rhs)
{
    std::optional<Array> packed_lhs;
    std::optional<Array> packed_rhs;
    Array const * ready_lhs = &lhs;
    Array const * ready_rhs = &rhs;
    if (!lhs.is_c_contiguous())
    {
        packed_lhs.emplace(lhs.to_row_major());
        ready_lhs = &packed_lhs.value();
    }
    if (!rhs.is_c_contiguous())
    {
        packed_rhs.emplace(rhs.to_row_major());
        ready_rhs = &packed_rhs.value();
    }

    helper_type helper(*ready_lhs, *ready_rhs);
    return helper.matmul_blas();
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
Array MatmulExecutor<Array, T>::execute_unbatched_blas(
    Array const & lhs, Array const & rhs, helper_type & helper)
{
    if (lhs.ndim() == 2 && rhs.ndim() == 2)
    {
        MatmulPlan const plan = MatmulPlan::make(lhs, rhs);
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
    }
    if (lhs.is_c_contiguous() && rhs.is_c_contiguous())
    {
        return helper.matmul_blas();
    }
    return execute_packed_blas(lhs, rhs);
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::execute_unbatched(
    Array const & lhs, Array const & rhs)
{
    helper_type helper(lhs, rhs);
    if constexpr (can_matmul_blas_v<value_type>)
    {
        if (large_enough_for_blas(lhs, rhs))
        {
            return execute_unbatched_blas(lhs, rhs, helper);
        }
    }
    return helper.matmul();
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
        }
    }
    return execute_generic(plan, lhs, rhs);
}

template <typename Array, typename T>
Array MatmulExecutor<Array, T>::multiply(
    Array const & lhs, Array const & rhs)
{
    if (lhs.ndim() <= 2 && rhs.ndim() <= 2)
    {
        return execute_unbatched(lhs, rhs);
    }
    MatmulPlan const plan = MatmulPlan::make(lhs, rhs);
    return execute_planned(plan, lhs, rhs);
}

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
