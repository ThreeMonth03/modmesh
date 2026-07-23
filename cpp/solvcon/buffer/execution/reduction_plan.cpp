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
