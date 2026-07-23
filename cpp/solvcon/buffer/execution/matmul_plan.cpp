/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/matmul_plan.hpp>

namespace solvcon
{

namespace detail
{

namespace execution
{

std::string MatmulPlan::shape_string(shape_type const & shape)
{
    std::string result = "(";
    for (size_t axis = 0; axis < shape.size(); ++axis)
    {
        if (axis != 0)
        {
            result += ",";
        }
        result += std::to_string(shape[axis]);
    }
    return result + ")";
}

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
