#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/reduction_plan.hpp>

#include <cstdint>

namespace solvcon
{

namespace detail
{

namespace execution
{

enum class ReductionTraversal : uint8_t
{
    sliced,
    outer_contiguous,
}; /* end enum class ReductionTraversal */

class ReductionSchedule
{
public:
    ReductionTraversal traversal() const noexcept { return m_traversal; }
    InnerLoopPlan const & inner_loop() const noexcept
    {
        return m_inner_loop;
    }
    bool inner_contiguous() const noexcept { return m_inner_contiguous; }

    static ReductionSchedule make(ReductionPlan const & plan);

private:
    ReductionTraversal m_traversal = ReductionTraversal::sliced;
    InnerLoopPlan m_inner_loop;
    bool m_inner_contiguous = false;
}; /* end class ReductionSchedule */

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
