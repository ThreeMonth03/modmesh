#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/reduction_plan.hpp>
#include <solvcon/math/math.hpp>
#include <solvcon/simd/simd.hpp>

#include <algorithm>
#include <stdexcept>

namespace solvcon
{

namespace detail
{

namespace execution
{

template <typename Array, typename Value, typename Real>
class OuterReductionExecutor
{
public:
    using value_type = Value;
    using real_type = Real;
    using real_array_type = typename Array::template rebind<real_type>;

    static Array mean(Array const & input, ReductionPlan const & plan);
    static Array average(Array const & input,
                         Array const & weight,
                         ReductionPlan const & plan,
                         OperandMapping const & weight_mapping,
                         value_type total_weight);
    static real_array_type variance(Array const & input,
                                    ReductionPlan const & plan,
                                    size_t ddof);

private:
    static constexpr size_t OUTER_TILE_BYTES = 4096;

    static size_t tile_size(ReductionPlan const & plan) noexcept;
}; /* end class OuterReductionExecutor */

template <typename Array, typename Value, typename Real>
size_t OuterReductionExecutor<Array, Value, Real>::tile_size(
    ReductionPlan const & plan) noexcept
{
    size_t constexpr minimum_tile = 1;
    size_t const capacity = std::max(
        minimum_tile, OUTER_TILE_BYTES / sizeof(value_type));
    return std::min(plan.outer().size(), capacity);
}

template <typename Array, typename Value, typename Real>
Array OuterReductionExecutor<Array, Value, Real>::mean(
    Array const & input, ReductionPlan const & plan)
{
    if (plan.inner().empty())
    {
        throw std::runtime_error(
            "planned mean of an empty reduction domain");
    }

    Array output(plan.output_shape());
    value_type * output_data = output.logical_data();
    value_type const * input_data = input.logical_data();
    size_t const output_tile_size = tile_size(plan);
    ssize_t const base_offset =
        plan.outer_input().base_offset() +
        plan.inner_input().base_offset();
    ssize_t const inner_stride = plan.inner_input().stride(0);
    for (size_t tile_begin = 0;
         tile_begin < plan.outer().size();
         tile_begin += output_tile_size)
    {
        size_t const tile_end = std::min(
            tile_begin + output_tile_size, plan.outer().size());
        value_type * output_begin = output_data + tile_begin;
        value_type * output_end = output_data + tile_end;
        std::fill(output_begin, output_end, value_type{});
        for (size_t reduced = 0;
             reduced < plan.inner().size();
             ++reduced)
        {
            value_type const * input_begin =
                input_data + base_offset +
                static_cast<ssize_t>(tile_begin) +
                static_cast<ssize_t>(reduced) * inner_stride;
            simd::add(
                output_begin,
                output_end,
                output_begin,
                input_begin);
        }
        simd::div_scalar(
            output_begin,
            output_end,
            output_begin,
            static_cast<value_type>(plan.inner().size()));
    }
    return output;
}

template <typename Array, typename Value, typename Real>
Array OuterReductionExecutor<Array, Value, Real>::average(
    Array const & input,
    Array const & weight,
    ReductionPlan const & plan,
    OperandMapping const & weight_mapping,
    value_type total_weight)
{
    if (plan.inner().empty())
    {
        throw std::runtime_error(
            "planned average of an empty reduction domain");
    }

    Array output(plan.output_shape());
    value_type * output_data = output.logical_data();
    value_type const * input_data = input.logical_data();
    value_type const * weight_data = weight.logical_data();
    size_t const output_tile_size = tile_size(plan);
    ssize_t const input_base =
        plan.outer_input().base_offset() +
        plan.inner_input().base_offset();
    ssize_t const input_inner_stride = plan.inner_input().stride(0);
    ssize_t const weight_stride = weight_mapping.stride(0);
    for (size_t tile_begin = 0;
         tile_begin < plan.outer().size();
         tile_begin += output_tile_size)
    {
        size_t const tile_end = std::min(
            tile_begin + output_tile_size, plan.outer().size());
        std::fill(
            output_data + tile_begin,
            output_data + tile_end,
            value_type{});
        for (size_t reduced = 0;
             reduced < plan.inner().size();
             ++reduced)
        {
            value_type const * input_begin =
                input_data + input_base +
                static_cast<ssize_t>(tile_begin) +
                static_cast<ssize_t>(reduced) * input_inner_stride;
            ssize_t const weight_offset =
                weight_mapping.base_offset() +
                static_cast<ssize_t>(reduced) * weight_stride;
            value_type const weight_value = weight_data[weight_offset];
            for (size_t output_index = tile_begin;
                 output_index < tile_end;
                 ++output_index)
            {
                output_data[output_index] +=
                    input_begin[output_index - tile_begin] *
                    weight_value;
            }
        }
        simd::div_scalar(
            output_data + tile_begin,
            output_data + tile_end,
            output_data + tile_begin,
            total_weight);
    }
    return output;
}

template <typename Array, typename Value, typename Real>
typename OuterReductionExecutor<Array, Value, Real>::real_array_type
OuterReductionExecutor<Array, Value, Real>::variance(
    Array const & input,
    ReductionPlan const & plan,
    size_t ddof)
{
    size_t const count = plan.inner().size();
    if (count <= ddof)
    {
        throw std::runtime_error(
            "planned variance ddof exceeds reduction size");
    }

    Array const means = mean(input, plan);
    value_type const * mean_data = means.logical_data();
    real_array_type output(plan.output_shape());
    real_type * output_data = output.logical_data();
    value_type const * input_data = input.logical_data();
    size_t const output_tile_size = tile_size(plan);
    ssize_t const base_offset =
        plan.outer_input().base_offset() +
        plan.inner_input().base_offset();
    ssize_t const inner_stride = plan.inner_input().stride(0);
    for (size_t tile_begin = 0;
         tile_begin < plan.outer().size();
         tile_begin += output_tile_size)
    {
        size_t const tile_end = std::min(
            tile_begin + output_tile_size, plan.outer().size());
        std::fill(
            output_data + tile_begin,
            output_data + tile_end,
            real_type{});
        for (size_t reduced = 0;
             reduced < plan.inner().size();
             ++reduced)
        {
            value_type const * input_begin =
                input_data + base_offset +
                static_cast<ssize_t>(tile_begin) +
                static_cast<ssize_t>(reduced) * inner_stride;
            for (size_t output_index = tile_begin;
                 output_index < tile_end;
                 ++output_index)
            {
                value_type const delta =
                    input_begin[output_index - tile_begin] -
                    mean_data[output_index];
                if constexpr (is_complex_v<value_type>)
                {
                    output_data[output_index] += delta.norm();
                }
                else
                {
                    output_data[output_index] += delta * delta;
                }
            }
        }
        simd::div_scalar(
            output_data + tile_begin,
            output_data + tile_end,
            output_data + tile_begin,
            static_cast<real_type>(count - ddof));
    }
    return output;
}

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
