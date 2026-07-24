#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/loop.hpp>

namespace solvcon
{

namespace detail
{

namespace execution
{

class ReductionPlan
{
public:
    LoopDomain const & outer() const noexcept { return m_outer; }
    LoopDomain const & inner() const noexcept { return m_inner; }
    shape_type const & output_shape() const noexcept
    {
        return m_output_shape;
    }
    OperandMapping const & outer_input() const noexcept
    {
        return m_outer_input;
    }
    OperandMapping const & inner_input() const noexcept
    {
        return m_inner_input;
    }

    template <typename Array>
    static ReductionPlan make(Array const & input,
                              shape_type const & axes,
                              bool allow_all_axes);

    template <typename Array>
    static ReductionPlan make_all(Array const & input);

private:
    static ReductionPlan make(shape_type const & input_shape,
                              stride_type const & input_strides,
                              shape_type const & axes,
                              bool allow_all_axes);

    LoopDomain m_outer;
    LoopDomain m_inner;
    shape_type m_output_shape;
    OperandMapping m_outer_input;
    OperandMapping m_inner_input;
}; /* end class ReductionPlan */

template <typename Array>
ReductionPlan ReductionPlan::make(
    Array const & input,
    shape_type const & axes,
    bool allow_all_axes)
{
    return make(
        input.shape(), input.stride(), axes, allow_all_axes);
}

template <typename Array>
ReductionPlan ReductionPlan::make_all(Array const & input)
{
    shape_type axes(input.ndim());
    for (ssize_t axis = 0; axis < input.ndim(); ++axis)
    {
        axes[axis] = axis;
    }
    return make(input, axes, true);
}

class ReductionSliceCursor
{
public:
    explicit ReductionSliceCursor(ReductionPlan const & plan);

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(m_cursor);
    }
    size_t output_index() const noexcept { return m_output_index; }
    ssize_t input_offset() const noexcept { return m_cursor.offset(0); }
    void advance()
    {
        ++m_output_index;
        m_cursor.advance();
    }

private:
    small_vector<OperandMapping> m_mappings;
    MappedOffsetCursor m_cursor;
    size_t m_output_index = 0;
}; /* end class ReductionSliceCursor */

class ReductionSchedule;

class ReducedOffsetCursor
{
public:
    ReducedOffsetCursor(ReductionPlan const & plan,
                        ReductionSchedule const & schedule,
                        ssize_t outer_offset);

    explicit operator bool() const noexcept
    {
        if (m_inner_contiguous)
        {
            return m_index < m_plan->inner().size();
        }
        return static_cast<bool>(m_cursor);
    }
    ssize_t offset() const noexcept
    {
        if (m_inner_contiguous)
        {
            return m_outer_offset + static_cast<ssize_t>(m_index);
        }
        return m_outer_offset + m_cursor.offset(0);
    }
    void advance()
    {
        if (m_inner_contiguous)
        {
            ++m_index;
        }
        else
        {
            m_cursor.advance();
        }
    }

private:
    ReductionPlan const * m_plan;
    ssize_t m_outer_offset;
    bool m_inner_contiguous;
    size_t m_index = 0;
    small_vector<OperandMapping> m_mappings;
    MappedOffsetCursor m_cursor;
}; /* end class ReducedOffsetCursor */

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
