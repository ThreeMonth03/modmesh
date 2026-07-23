/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/SimpleArray.hpp>
#include <solvcon/buffer/execution/elementwise.hpp>
#include <solvcon/buffer/execution/matmul_plan.hpp>
#include <solvcon/buffer/execution/reduction_plan.hpp>

#include <array>

#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

namespace ex = solvcon::detail::execution;

TEST(ExecutionLoop, common_shape_and_broadcast_offsets)
{
    ex::shape_type const lhs_shape{2, 1, 4};
    ex::shape_type const rhs_shape{1, 3, 1};
    ex::LoopDomain const domain(
        ex::LoopDomain::common_shape(lhs_shape, rhs_shape));
    ex::OperandMapping const lhs = ex::OperandMapping::broadcast(
        lhs_shape, ex::stride_type{4, 4, 1}, domain);
    ex::OperandMapping const rhs = ex::OperandMapping::broadcast(
        rhs_shape, ex::stride_type{3, 1, 1}, domain);

    ASSERT_EQ(domain.rank(), 3);
    EXPECT_EQ(domain.shape()[0], 2);
    EXPECT_EQ(domain.shape()[1], 3);
    EXPECT_EQ(domain.shape()[2], 4);
    EXPECT_EQ(lhs.stride(0), 4);
    EXPECT_EQ(lhs.stride(1), 0);
    EXPECT_EQ(lhs.stride(2), 1);
    EXPECT_EQ(rhs.stride(0), 0);
    EXPECT_EQ(rhs.stride(1), 1);
    EXPECT_EQ(rhs.stride(2), 0);

    solvcon::small_vector<ex::OperandMapping> const mappings{lhs, rhs};
    size_t linear_index = 0;
    for (ex::MappedOffsetCursor cursor(domain, mappings); cursor;
         cursor.advance(), ++linear_index)
    {
        size_t const outer = linear_index / 12;
        size_t const middle = linear_index / 4 % 3;
        size_t const inner = linear_index % 4;
        EXPECT_EQ(
            cursor.offset(0),
            static_cast<ssize_t>(outer * 4 + inner));
        EXPECT_EQ(cursor.offset(1), static_cast<ssize_t>(middle));
    }
    EXPECT_EQ(linear_index, 24);
}

TEST(ExecutionLoop, signed_offsets_retain_logical_origin)
{
    ex::LoopDomain const domain(ex::shape_type{2, 3});
    ex::OperandMapping const mapping(ex::stride_type{-3, 1}, 3);
    solvcon::small_vector<ex::OperandMapping> const mappings{mapping};
    std::array<ssize_t, 6> const expected{3, 4, 5, 0, 1, 2};

    size_t index = 0;
    for (ex::MappedOffsetCursor cursor(domain, mappings); cursor;
         cursor.advance(), ++index)
    {
        EXPECT_EQ(cursor.offset(0), expected[index]);
    }
    EXPECT_EQ(index, expected.size());
    EXPECT_EQ(mapping.span(domain).minimum(), 0);
    EXPECT_EQ(mapping.span(domain).maximum(), 5);
    EXPECT_TRUE(mapping.is_dense(domain));
}

TEST(ExecutionLoop, inner_loop_lowers_runtime_rank_once_per_row)
{
    ex::LoopDomain const domain(ex::shape_type{2, 3, 4});
    solvcon::small_vector<ex::OperandMapping> const mappings{
        ex::OperandMapping(ex::stride_type{12, 4, 1}),
        ex::OperandMapping(ex::stride_type{0, 0, 1})};
    ex::InnerLoopPlan const plan(domain, mappings);

    ASSERT_EQ(plan.outer().rank(), 2);
    EXPECT_EQ(plan.outer().shape()[0], 2);
    EXPECT_EQ(plan.outer().shape()[1], 3);
    EXPECT_EQ(plan.size(), 4);
    EXPECT_EQ(plan.stride(0), 1);
    EXPECT_EQ(plan.stride(1), 1);

    size_t row = 0;
    for (ex::InnerLoopCursor cursor(plan); cursor;
         cursor.advance(), ++row)
    {
        EXPECT_EQ(cursor.offset(0), static_cast<ssize_t>(row * 4));
        EXPECT_EQ(cursor.offset(1), 0);
    }
    EXPECT_EQ(row, 6);
}

TEST(ExecutionPlans, families_own_topology_over_shared_layout)
{
    solvcon::SimpleArray<double> lhs(ex::shape_type{2, 1, 4});
    solvcon::SimpleArray<double> rhs(ex::shape_type{1, 3, 1});
    solvcon::SimpleArray<double> output(ex::shape_type{2, 3, 4});
    ex::ElementwisePlan const elementwise =
        ex::ElementwisePlan::make(output, lhs, rhs);
    EXPECT_EQ(elementwise.domain().shape()[0], 2);
    EXPECT_EQ(elementwise.domain().shape()[1], 3);
    EXPECT_EQ(elementwise.domain().shape()[2], 4);
    EXPECT_EQ(elementwise.input(0).stride(1), 0);
    EXPECT_EQ(elementwise.input(1).stride(0), 0);
    EXPECT_EQ(elementwise.input(1).stride(2), 0);

    solvcon::SimpleArray<double> values(ex::shape_type{2, 3, 4});
    ex::ReductionPlan const reduction = ex::ReductionPlan::make(
        values, ex::shape_type{0, 2}, false);
    ASSERT_EQ(reduction.output_shape().size(), 1);
    EXPECT_EQ(reduction.output_shape()[0], 3);
    EXPECT_EQ(reduction.outer().shape()[0], 3);
    EXPECT_EQ(reduction.inner().shape()[0], 2);
    EXPECT_EQ(reduction.inner().shape()[1], 4);
    EXPECT_EQ(reduction.outer_input().stride(0), 4);
    EXPECT_EQ(reduction.inner_input().stride(0), 12);
    EXPECT_EQ(reduction.inner_input().stride(1), 1);

    solvcon::SimpleArray<double> matrix_lhs(
        ex::shape_type{2, 1, 3, 4});
    solvcon::SimpleArray<double> matrix_rhs(
        ex::shape_type{1, 5, 4, 6});
    ex::MatmulPlan const matmul =
        ex::MatmulPlan::make(matrix_lhs, matrix_rhs);
    EXPECT_EQ(matmul.batch().shape()[0], 2);
    EXPECT_EQ(matmul.batch().shape()[1], 5);
    EXPECT_EQ(matmul.lhs_batch().stride(0), 12);
    EXPECT_EQ(matmul.lhs_batch().stride(1), 0);
    EXPECT_EQ(matmul.rhs_batch().stride(0), 0);
    EXPECT_EQ(matmul.rhs_batch().stride(1), 24);
    EXPECT_EQ(matmul.rows(), 3);
    EXPECT_EQ(matmul.columns(), 6);
    EXPECT_EQ(matmul.inner_size(), 4);
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
