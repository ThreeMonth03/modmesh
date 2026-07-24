#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/loop.hpp>

#include <format>
#include <stdexcept>
#include <string>

namespace solvcon
{

namespace detail
{

namespace execution
{

class MatmulPlan
{
public:
    LoopDomain const & batch() const noexcept { return m_batch; }
    shape_type const & output_shape() const noexcept
    {
        return m_output_shape;
    }
    OperandMapping const & output_batch() const noexcept
    {
        return m_output_batch;
    }
    OperandMapping const & lhs_batch() const noexcept { return m_lhs_batch; }
    OperandMapping const & rhs_batch() const noexcept { return m_rhs_batch; }
    ssize_t rows() const noexcept { return m_rows; }
    ssize_t columns() const noexcept { return m_columns; }
    ssize_t inner_size() const noexcept { return m_inner_size; }
    ssize_t lhs_row_stride() const noexcept { return m_lhs_row_stride; }
    ssize_t lhs_inner_stride() const noexcept
    {
        return m_lhs_inner_stride;
    }
    ssize_t rhs_inner_stride() const noexcept
    {
        return m_rhs_inner_stride;
    }
    ssize_t rhs_column_stride() const noexcept
    {
        return m_rhs_column_stride;
    }
    bool lhs_vector() const noexcept { return m_lhs_vector; }
    bool rhs_vector() const noexcept { return m_rhs_vector; }

    template <typename Array>
    static MatmulPlan make(Array const & lhs, Array const & rhs);

private:
    static std::string shape_string(shape_type const & shape);

    LoopDomain m_batch;
    shape_type m_output_shape;
    OperandMapping m_output_batch;
    OperandMapping m_lhs_batch;
    OperandMapping m_rhs_batch;
    ssize_t m_rows = 1;
    ssize_t m_columns = 1;
    ssize_t m_inner_size = 0;
    ssize_t m_lhs_row_stride = 0;
    ssize_t m_lhs_inner_stride = 0;
    ssize_t m_rhs_inner_stride = 0;
    ssize_t m_rhs_column_stride = 0;
    bool m_lhs_vector = false;
    bool m_rhs_vector = false;
}; /* end class MatmulPlan */

template <typename Array>
MatmulPlan MatmulPlan::make(Array const & lhs, Array const & rhs)
{
    if (lhs.ndim() == 0 || rhs.ndim() == 0)
    {
        throw std::invalid_argument(
            "planned matmul requires non-scalar operands");
    }

    MatmulPlan plan;
    plan.m_lhs_vector = lhs.ndim() == 1;
    plan.m_rhs_vector = rhs.ndim() == 1;
    plan.m_rows = plan.m_lhs_vector
                      ? 1
                      : lhs.shape(lhs.ndim() - 2);
    plan.m_inner_size = lhs.shape(lhs.ndim() - 1);
    ssize_t const rhs_inner_size = plan.m_rhs_vector
                                       ? rhs.shape(0)
                                       : rhs.shape(rhs.ndim() - 2);
    plan.m_columns = plan.m_rhs_vector
                         ? 1
                         : rhs.shape(rhs.ndim() - 1);
    if (plan.m_inner_size != rhs_inner_size)
    {
        throw std::invalid_argument(std::format(
            "planned matmul shape mismatch: lhs={} rhs={}",
            shape_string(lhs.shape()),
            shape_string(rhs.shape())));
    }

    size_t const lhs_batch_rank = plan.m_lhs_vector
                                      ? 0
                                      : lhs.ndim() - 2;
    size_t const rhs_batch_rank = plan.m_rhs_vector
                                      ? 0
                                      : rhs.ndim() - 2;
    shape_type const lhs_batch_shape(
        lhs.shape().begin(), lhs.shape().begin() + lhs_batch_rank);
    shape_type const rhs_batch_shape(
        rhs.shape().begin(), rhs.shape().begin() + rhs_batch_rank);
    stride_type const lhs_batch_strides(
        lhs.stride().begin(), lhs.stride().begin() + lhs_batch_rank);
    stride_type const rhs_batch_strides(
        rhs.stride().begin(), rhs.stride().begin() + rhs_batch_rank);
    plan.m_batch = LoopDomain(LoopDomain::common_shape(
        lhs_batch_shape, rhs_batch_shape));
    plan.m_lhs_batch = OperandMapping::broadcast(
        lhs_batch_shape, lhs_batch_strides, plan.m_batch);
    plan.m_rhs_batch = OperandMapping::broadcast(
        rhs_batch_shape, rhs_batch_strides, plan.m_batch);

    plan.m_output_shape = plan.m_batch.shape();
    if (plan.m_lhs_vector && plan.m_rhs_vector)
    {
        plan.m_output_shape.push_back(1);
    }
    else if (plan.m_lhs_vector)
    {
        plan.m_output_shape.push_back(plan.m_columns);
    }
    else if (plan.m_rhs_vector)
    {
        plan.m_output_shape.push_back(plan.m_rows);
    }
    else
    {
        plan.m_output_shape.push_back(plan.m_rows);
        plan.m_output_shape.push_back(plan.m_columns);
    }

    stride_type const output_strides = LoopDomain::row_major_strides(
        plan.m_output_shape);
    plan.m_output_batch = OperandMapping(stride_type(
        output_strides.begin(),
        output_strides.begin() + plan.m_batch.rank()));
    plan.m_lhs_row_stride = plan.m_lhs_vector
                                ? 0
                                : lhs.stride(lhs.ndim() - 2);
    plan.m_lhs_inner_stride = lhs.stride(lhs.ndim() - 1);
    plan.m_rhs_inner_stride = plan.m_rhs_vector
                                  ? rhs.stride(0)
                                  : rhs.stride(rhs.ndim() - 2);
    plan.m_rhs_column_stride = plan.m_rhs_vector
                                   ? 0
                                   : rhs.stride(rhs.ndim() - 1);
    return plan;
}

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
