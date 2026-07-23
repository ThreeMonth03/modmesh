/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/reduction_schedule.hpp>

namespace solvcon
{

namespace detail
{

namespace execution
{

ReductionSchedule ReductionSchedule::make(ReductionPlan const & plan)
{
    ReductionSchedule schedule;
    schedule.m_inner_contiguous =
        plan.inner_input().is_row_major(plan.inner());
    bool const matrix_domains =
        plan.outer().rank() == 1 && plan.inner().rank() == 1;
    bool const contiguous_outputs =
        matrix_domains && !plan.outer().empty() &&
        plan.outer_input().stride(0) == 1;
    bool const disjoint_reduced_slices =
        contiguous_outputs &&
        plan.inner_input().stride(0) >=
            static_cast<ssize_t>(plan.outer().size());
    if (disjoint_reduced_slices)
    {
        schedule.m_traversal = ReductionTraversal::outer_contiguous;
    }
    else
    {
        small_vector<OperandMapping> const mappings{plan.inner_input()};
        schedule.m_inner_loop = InnerLoopPlan(plan.inner(), mappings);
    }
    return schedule;
}

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
