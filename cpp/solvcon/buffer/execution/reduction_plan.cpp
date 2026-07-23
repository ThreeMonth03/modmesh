/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/reduction_plan.hpp>
#include <solvcon/buffer/execution/reduction_schedule.hpp>

namespace solvcon
{

namespace detail
{

namespace execution
{

ReductionPlan ReductionPlan::make(
    shape_type const & input_shape,
    stride_type const & input_strides,
    shape_type const & axes,
    bool allow_all_axes)
{
    size_t const rank = input_shape.size();
    small_vector<bool> reduced(rank, false);
    for (ssize_t const axis : axes)
    {
        if (axis < 0 || static_cast<size_t>(axis) >= rank)
        {
            throw std::out_of_range("reduction axis out of range");
        }
        if (reduced[axis])
        {
            throw std::invalid_argument("duplicate reduction axis");
        }
        reduced[axis] = true;
    }

    size_t const reduced_count = reduced.count(true);
    if (reduced_count == 0 ||
        (!allow_all_axes && reduced_count == rank))
    {
        throw std::invalid_argument(
            "reduction requires a non-empty proper axis set");
    }

    ReductionPlan plan;
    plan.m_output_shape = shape_type(rank - reduced_count);
    shape_type outer_shape(rank - reduced_count);
    shape_type inner_shape(reduced_count);
    stride_type outer_strides(rank - reduced_count);
    stride_type inner_strides(reduced_count);
    for (size_t axis = 0, outer_axis = 0, inner_axis = 0;
         axis < rank;
         ++axis)
    {
        if (reduced[axis])
        {
            inner_shape[inner_axis] = input_shape[axis];
            inner_strides[inner_axis++] = input_strides[axis];
        }
        else
        {
            outer_shape[outer_axis] = input_shape[axis];
            plan.m_output_shape[outer_axis] = input_shape[axis];
            outer_strides[outer_axis++] = input_strides[axis];
        }
    }

    plan.m_outer = LoopDomain(std::move(outer_shape));
    plan.m_inner = LoopDomain(std::move(inner_shape));
    plan.m_outer_input = OperandMapping(std::move(outer_strides));
    plan.m_inner_input = OperandMapping(std::move(inner_strides));
    return plan;
}

ReductionSliceCursor::ReductionSliceCursor(ReductionPlan const & plan)
    : m_mappings{plan.outer_input()}
    , m_cursor(plan.outer(), m_mappings)
{
}

ReducedOffsetCursor::ReducedOffsetCursor(
    ReductionPlan const & plan,
    ReductionSchedule const & schedule,
    ssize_t outer_offset)
    : m_plan(&plan)
    , m_outer_offset(outer_offset)
    , m_inner_contiguous(schedule.inner_contiguous())
    , m_mappings{plan.inner_input()}
    , m_cursor(plan.inner(), m_mappings)
{
}

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
