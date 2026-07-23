#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/reduction_outer.hpp>
#include <solvcon/buffer/execution/reduction_schedule.hpp>
#include <solvcon/math/math.hpp>
#include <solvcon/simd/simd.hpp>

#include <functional>
#include <stdexcept>

namespace solvcon
{

namespace detail
{

namespace execution
{

template <typename T>
struct reduction_real
{
    using type = T;
}; /* end struct reduction_real */

template <typename T>
struct reduction_real<Complex<T>>
{
    using type = T;
}; /* end struct reduction_real */

template <typename Array, typename T>
class ReductionExecutor
{
public:
    using value_type = T;
    using real_type = typename reduction_real<value_type>::type;
    using real_array_type = typename Array::template rebind<real_type>;
    using outer_executor =
        OuterReductionExecutor<Array, value_type, real_type>;

    static Array mean(Array const & input, ReductionPlan const & plan);
    static value_type mean(Array const & input);
    static Array average(Array const & input,
                         Array const & weight,
                         ReductionPlan const & plan);
    static value_type average(Array const & input, Array const & weight);
    static real_array_type variance(Array const & input,
                                    ReductionPlan const & plan,
                                    size_t ddof);
    static real_type variance(Array const & input, size_t ddof);

    template <typename Finalize>
    static Array collect(Array const & input,
                         ReductionPlan const & plan,
                         Finalize finalize);

    template <typename Finalize>
    static value_type collect(Array const & input, Finalize finalize);

private:
    static OperandMapping make_weight_mapping(
        Array const & weight, ReductionPlan const & plan);
    static value_type execute_sum(Array const & input,
                                  ReductionPlan const & plan,
                                  ReductionSchedule const & schedule,
                                  ssize_t outer_offset);
    static value_type execute_mean(Array const & input,
                                   ReductionPlan const & plan,
                                   ReductionSchedule const & schedule,
                                   ssize_t outer_offset);
    static value_type execute_weight_sum(
        Array const & weight,
        ReductionPlan const & plan,
        OperandMapping const & weight_mapping);
    static value_type execute_average(
        Array const & input,
        Array const & weight,
        ReductionPlan const & plan,
        OperandMapping const & weight_mapping,
        InnerLoopPlan const & inner_loop,
        value_type total_weight,
        ssize_t outer_offset);
    static real_type execute_squared_difference(
        Array const & input,
        ReductionPlan const & plan,
        ReductionSchedule const & schedule,
        ssize_t outer_offset,
        value_type mean_value);
    static real_type execute_variance(Array const & input,
                                      ReductionPlan const & plan,
                                      ReductionSchedule const & schedule,
                                      ssize_t outer_offset,
                                      size_t ddof);
}; /* end class ReductionExecutor */

template <typename Array, typename T>
OperandMapping ReductionExecutor<Array, T>::make_weight_mapping(
    Array const & weight, ReductionPlan const & plan)
{
    if (weight.shape() == plan.inner().shape())
    {
        return OperandMapping::exact(weight.stride());
    }
    if (weight.ndim() == 1 && weight.size() == plan.inner().size())
    {
        stride_type strides = LoopDomain::row_major_strides(
            plan.inner().shape());
        for (ssize_t & stride : strides)
        {
            stride *= weight.stride(0);
        }
        return OperandMapping(std::move(strides));
    }
    throw std::invalid_argument(
        "weight shape does not match the reduced domain");
}

template <typename Array, typename T>
typename ReductionExecutor<Array, T>::value_type
ReductionExecutor<Array, T>::execute_sum(
    Array const & input,
    ReductionPlan const & plan,
    ReductionSchedule const & schedule,
    ssize_t outer_offset)
{
    value_type const * data = input.logical_data();
    OperandMapping const & mapping = plan.inner_input();
    if (mapping.is_dense(plan.inner()))
    {
        MappingSpan const span = mapping.span(plan.inner());
        value_type const * start =
            data + outer_offset + span.minimum();
        return simd::sum(start, start + plan.inner().size());
    }
    value_type total{};
    InnerLoopPlan const & inner = schedule.inner_loop();
    for (InnerLoopCursor cursor(inner); cursor;
         cursor.advance())
    {
        value_type const * ptr =
            data + outer_offset + cursor.offset(0);
        for (size_t index = 0; index < inner.size(); ++index)
        {
            total += *ptr;
            ptr += inner.stride(0);
        }
    }
    return total;
}

template <typename Array, typename T>
typename ReductionExecutor<Array, T>::value_type
ReductionExecutor<Array, T>::execute_mean(
    Array const & input,
    ReductionPlan const & plan,
    ReductionSchedule const & schedule,
    ssize_t outer_offset)
{
    if (plan.inner().empty())
    {
        throw std::runtime_error(
            "planned mean of an empty reduction domain");
    }

    return execute_sum(input, plan, schedule, outer_offset) /
           static_cast<value_type>(plan.inner().size());
}

template <typename Array, typename T>
Array ReductionExecutor<Array, T>::mean(
    Array const & input, ReductionPlan const & plan)
{
    ReductionSchedule const schedule = ReductionSchedule::make(plan);
    if (schedule.traversal() == ReductionTraversal::outer_contiguous)
    {
        return outer_executor::mean(input, plan);
    }

    Array output(plan.output_shape());
    value_type * output_data = output.logical_data();
    for (ReductionSliceCursor cursor(plan); cursor; cursor.advance())
    {
        output_data[cursor.output_index()] = execute_mean(
            input, plan, schedule, cursor.input_offset());
    }
    return output;
}

template <typename Array, typename T>
typename ReductionExecutor<Array, T>::value_type
ReductionExecutor<Array, T>::mean(Array const & input)
{
    if (input.size() == 0)
    {
        throw std::runtime_error(
            "planned mean of an empty reduction domain");
    }
    ReductionPlan const plan = ReductionPlan::make_all(input);
    ReductionSchedule const schedule = ReductionSchedule::make(plan);
    return execute_mean(input, plan, schedule, 0);
}

template <typename Array, typename T>
typename ReductionExecutor<Array, T>::value_type
ReductionExecutor<Array, T>::execute_weight_sum(
    Array const & weight,
    ReductionPlan const & plan,
    OperandMapping const & weight_mapping)
{
    value_type const * data = weight.logical_data();
    if (weight_mapping.is_dense(plan.inner()))
    {
        MappingSpan const span = weight_mapping.span(plan.inner());
        value_type const * start = data + span.minimum();
        return simd::sum(start, start + plan.inner().size());
    }
    value_type total{};
    small_vector<OperandMapping> const mappings{weight_mapping};
    InnerLoopPlan const inner(plan.inner(), mappings);
    for (InnerLoopCursor cursor(inner); cursor;
         cursor.advance())
    {
        value_type const * ptr = data + cursor.offset(0);
        for (size_t index = 0; index < inner.size(); ++index)
        {
            total += *ptr;
            ptr += inner.stride(0);
        }
    }
    return total;
}

template <typename Array, typename T>
typename ReductionExecutor<Array, T>::value_type
ReductionExecutor<Array, T>::execute_average(
    Array const & input,
    Array const & weight,
    ReductionPlan const & plan,
    OperandMapping const & weight_mapping,
    InnerLoopPlan const & inner_loop,
    value_type total_weight,
    ssize_t outer_offset)
{
    if (plan.inner().empty())
    {
        throw std::runtime_error(
            "planned average of an empty reduction domain");
    }

    value_type const * input_data = input.logical_data();
    value_type const * weight_data = weight.logical_data();
    OperandMapping const & input_mapping = plan.inner_input();
    value_type weighted_sum{};
    if (input_mapping.strides() == weight_mapping.strides() &&
        input_mapping.is_dense(plan.inner()))
    {
        MappingSpan const input_span = input_mapping.span(plan.inner());
        MappingSpan const weight_span = weight_mapping.span(plan.inner());
        value_type const * input_start =
            input_data + outer_offset + input_span.minimum();
        value_type const * weight_start =
            weight_data + weight_span.minimum();
        weighted_sum = simd::sum_product(
            input_start,
            input_start + plan.inner().size(),
            weight_start);
    }
    else
    {
        for (InnerLoopCursor cursor(inner_loop); cursor;
             cursor.advance())
        {
            value_type const * input_ptr =
                input_data + outer_offset + cursor.offset(0);
            value_type const * weight_ptr =
                weight_data + cursor.offset(1);
            for (size_t index = 0; index < inner_loop.size(); ++index)
            {
                weighted_sum += *input_ptr * *weight_ptr;
                input_ptr += inner_loop.stride(0);
                weight_ptr += inner_loop.stride(1);
            }
        }
    }
    return weighted_sum / total_weight;
}

template <typename Array, typename T>
Array ReductionExecutor<Array, T>::average(
    Array const & input,
    Array const & weight,
    ReductionPlan const & plan)
{
    ReductionSchedule const schedule = ReductionSchedule::make(plan);
    OperandMapping const weight_mapping = make_weight_mapping(weight, plan);
    small_vector<OperandMapping> const inner_mappings{
        plan.inner_input(), weight_mapping};
    InnerLoopPlan const inner_loop(plan.inner(), inner_mappings);
    value_type const total_weight = execute_weight_sum(
        weight, plan, weight_mapping);
    if (total_weight == value_type{})
    {
        throw std::runtime_error(
            "planned average total weight is zero");
    }
    if (schedule.traversal() == ReductionTraversal::outer_contiguous)
    {
        return outer_executor::average(
            input, weight, plan, weight_mapping, total_weight);
    }

    Array output(plan.output_shape());
    value_type * output_data = output.logical_data();
    for (ReductionSliceCursor cursor(plan); cursor; cursor.advance())
    {
        output_data[cursor.output_index()] = execute_average(
            input,
            weight,
            plan,
            weight_mapping,
            inner_loop,
            total_weight,
            cursor.input_offset());
    }
    return output;
}

template <typename Array, typename T>
typename ReductionExecutor<Array, T>::value_type
ReductionExecutor<Array, T>::average(
    Array const & input, Array const & weight)
{
    if (input.shape() != weight.shape())
    {
        throw std::invalid_argument(
            "weight shape does not match input shape");
    }
    ReductionPlan const plan = ReductionPlan::make_all(input);
    OperandMapping const weight_mapping = OperandMapping::exact(
        weight.stride());
    small_vector<OperandMapping> const inner_mappings{
        plan.inner_input(), weight_mapping};
    InnerLoopPlan const inner_loop(plan.inner(), inner_mappings);
    value_type const total_weight = execute_weight_sum(
        weight, plan, weight_mapping);
    if (total_weight == value_type{})
    {
        throw std::runtime_error(
            "planned average total weight is zero");
    }
    return execute_average(
        input,
        weight,
        plan,
        weight_mapping,
        inner_loop,
        total_weight,
        0);
}

template <typename Array, typename T>
typename ReductionExecutor<Array, T>::real_type
ReductionExecutor<Array, T>::execute_squared_difference(
    Array const & input,
    ReductionPlan const & plan,
    ReductionSchedule const & schedule,
    ssize_t outer_offset,
    value_type mean_value)
{
    value_type const * data = input.logical_data();
    OperandMapping const & mapping = plan.inner_input();
    if constexpr (!is_complex_v<value_type>)
    {
        if (mapping.is_dense(plan.inner()))
        {
            MappingSpan const span = mapping.span(plan.inner());
            value_type const * start =
                data + outer_offset + span.minimum();
            return simd::sum_squared_difference(
                start, start + plan.inner().size(), mean_value);
        }
    }

    real_type total{};
    InnerLoopPlan const & inner = schedule.inner_loop();
    for (InnerLoopCursor cursor(inner); cursor;
         cursor.advance())
    {
        value_type const * ptr =
            data + outer_offset + cursor.offset(0);
        for (size_t index = 0; index < inner.size(); ++index)
        {
            value_type const delta = *ptr - mean_value;
            if constexpr (is_complex_v<value_type>)
            {
                total += delta.norm();
            }
            else
            {
                total += delta * delta;
            }
            ptr += inner.stride(0);
        }
    }
    return total;
}

template <typename Array, typename T>
typename ReductionExecutor<Array, T>::real_type
ReductionExecutor<Array, T>::execute_variance(
    Array const & input,
    ReductionPlan const & plan,
    ReductionSchedule const & schedule,
    ssize_t outer_offset,
    size_t ddof)
{
    size_t const count = plan.inner().size();
    if (count <= ddof)
    {
        throw std::runtime_error(
            "planned variance ddof exceeds reduction size");
    }

    value_type const mean_value = execute_mean(
        input, plan, schedule, outer_offset);
    real_type const total = execute_squared_difference(
        input, plan, schedule, outer_offset, mean_value);
    return total / static_cast<real_type>(count - ddof);
}

template <typename Array, typename T>
typename ReductionExecutor<Array, T>::real_array_type
ReductionExecutor<Array, T>::variance(
    Array const & input,
    ReductionPlan const & plan,
    size_t ddof)
{
    ReductionSchedule const schedule = ReductionSchedule::make(plan);
    if (schedule.traversal() == ReductionTraversal::outer_contiguous)
    {
        return outer_executor::variance(input, plan, ddof);
    }

    real_array_type output(plan.output_shape());
    real_type * output_data = output.logical_data();
    for (ReductionSliceCursor cursor(plan); cursor; cursor.advance())
    {
        output_data[cursor.output_index()] = execute_variance(
            input, plan, schedule, cursor.input_offset(), ddof);
    }
    return output;
}

template <typename Array, typename T>
typename ReductionExecutor<Array, T>::real_type
ReductionExecutor<Array, T>::variance(Array const & input, size_t ddof)
{
    ReductionPlan const plan = ReductionPlan::make_all(input);
    ReductionSchedule const schedule = ReductionSchedule::make(plan);
    return execute_variance(input, plan, schedule, 0, ddof);
}

template <typename Array, typename T>
template <typename Finalize>
Array ReductionExecutor<Array, T>::collect(
    Array const & input,
    ReductionPlan const & plan,
    Finalize finalize)
{
    if (plan.inner().empty())
    {
        throw std::runtime_error(
            "planned collection of an empty reduction domain");
    }

    Array output(plan.output_shape());
    value_type * output_data = output.logical_data();
    value_type const * input_data = input.logical_data();
    ReductionSchedule const schedule = ReductionSchedule::make(plan);
    for (ReductionSliceCursor slice(plan); slice; slice.advance())
    {
        small_vector<value_type> values(plan.inner().size());
        size_t value_index = 0;
        for (ReducedOffsetCursor cursor(
                 plan, schedule, slice.input_offset());
             cursor;
             cursor.advance())
        {
            values[value_index++] = input_data[cursor.offset()];
        }
        output_data[slice.output_index()] = std::invoke(finalize, values);
    }
    return output;
}

template <typename Array, typename T>
template <typename Finalize>
typename ReductionExecutor<Array, T>::value_type
ReductionExecutor<Array, T>::collect(Array const & input, Finalize finalize)
{
    ReductionPlan const plan = ReductionPlan::make_all(input);
    if (plan.inner().empty())
    {
        throw std::runtime_error(
            "planned collection of an empty reduction domain");
    }

    small_vector<value_type> values(plan.inner().size());
    value_type const * input_data = input.logical_data();
    ReductionSchedule const schedule = ReductionSchedule::make(plan);
    size_t value_index = 0;
    for (ReducedOffsetCursor cursor(plan, schedule, 0); cursor;
         cursor.advance())
    {
        values[value_index++] = input_data[cursor.offset()];
    }
    return std::invoke(finalize, values);
}

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
