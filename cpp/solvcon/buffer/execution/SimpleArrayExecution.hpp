#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/elementwise.hpp>
#include <solvcon/buffer/execution/matmul.hpp>
#include <solvcon/buffer/execution/reduction.hpp>

#include <cmath>
#include <format>
#include <stdexcept>
#include <type_traits>

namespace solvcon
{

namespace detail
{

template <typename A, typename T>
class SimpleArrayExecution
{
public:
    using value_type = T;
    using shape_type = execution::shape_type;
    using real_type = typename execution::reduction_real<value_type>::type;
    using real_array_type = typename A::template rebind<real_type>;

    static A planned_add(A const & self, A const & other);
    static A planned_add(A const & self, value_type scalar);
    static A planned_sub(A const & self, A const & other);
    static A planned_sub(A const & self, value_type scalar);
    static A planned_mul(A const & self, A const & other);
    static A planned_mul(A const & self, value_type scalar);
    static A planned_div(A const & self, A const & other);
    static A planned_div(A const & self, value_type scalar);

    static void planned_iadd(A & self, A const & other);
    static void planned_iadd(A & self, value_type scalar);
    static void planned_isub(A & self, A const & other);
    static void planned_isub(A & self, value_type scalar);
    static void planned_imul(A & self, A const & other);
    static void planned_imul(A & self, value_type scalar);
    static void planned_idiv(A & self, A const & other);
    static void planned_idiv(A & self, value_type scalar);

    static A planned_mean(A const & self, shape_type const & axes);
    static value_type planned_mean(A const & self);
    static A planned_average(A const & self,
                             shape_type const & axes,
                             A const & weight);
    static value_type planned_average(A const & self, A const & weight);
    static real_array_type planned_var(A const & self,
                                       shape_type const & axes,
                                       size_t ddof);
    static real_type planned_var(A const & self, size_t ddof);
    static real_array_type planned_std(A const & self,
                                       shape_type const & axes,
                                       size_t ddof);
    static real_type planned_std(A const & self, size_t ddof);
    static A planned_median(A const & self, shape_type const & axes);
    static value_type planned_median(A const & self);

    static A planned_matmul(A const & self, A const & other);

private:
    using add_executor_type = execution::ElementwiseExecutor<
        A,
        value_type,
        execution::AddKernel<value_type>>;
    using subtract_executor_type = execution::ElementwiseExecutor<
        A,
        value_type,
        execution::SubtractKernel<value_type>>;
    using multiply_executor_type = execution::ElementwiseExecutor<
        A,
        value_type,
        execution::MultiplyKernel<value_type>>;
    using divide_executor_type = execution::ElementwiseExecutor<
        A,
        value_type,
        execution::DivideKernel<value_type>>;
    using matmul_executor_type =
        execution::MatmulExecutor<A, value_type>;

    class MedianFinalize
    {
    public:
        explicit MedianFinalize(A const & array)
            : m_array(array)
        {
        }

        value_type operator()(small_vector<value_type> & values) const
        {
            return m_array.median_op(values);
        }

    private:
        A const & m_array;
    }; /* end class MedianFinalize */

    static void reject_boolean(char const * operation);
}; /* end class SimpleArrayExecution */

template <typename A, typename T>
void SimpleArrayExecution<A, T>::reject_boolean(char const * operation)
{
    if constexpr (std::is_same_v<bool, std::remove_const_t<value_type>>)
    {
        throw std::runtime_error(std::format(
            "SimpleArray<bool>::planned_{}(): unsupported operation",
            operation));
    }
}

template <typename A, typename T>
A SimpleArrayExecution<A, T>::planned_add(
    A const & self, A const & other)
{
    return add_executor_type::transform(
        self, other, execution::AddKernel<value_type>{});
}

template <typename A, typename T>
A SimpleArrayExecution<A, T>::planned_add(
    A const & self, value_type scalar)
{
    return add_executor_type::transform(
        self, scalar, execution::AddKernel<value_type>{});
}

template <typename A, typename T>
A SimpleArrayExecution<A, T>::planned_sub(
    A const & self, A const & other)
{
    reject_boolean("sub");
    return subtract_executor_type::transform(
        self, other, execution::SubtractKernel<value_type>{});
}

template <typename A, typename T>
A SimpleArrayExecution<A, T>::planned_sub(
    A const & self, value_type scalar)
{
    reject_boolean("sub");
    return subtract_executor_type::transform(
        self, scalar, execution::SubtractKernel<value_type>{});
}

template <typename A, typename T>
A SimpleArrayExecution<A, T>::planned_mul(
    A const & self, A const & other)
{
    return multiply_executor_type::transform(
        self, other, execution::MultiplyKernel<value_type>{});
}

template <typename A, typename T>
A SimpleArrayExecution<A, T>::planned_mul(
    A const & self, value_type scalar)
{
    return multiply_executor_type::transform(
        self, scalar, execution::MultiplyKernel<value_type>{});
}

template <typename A, typename T>
A SimpleArrayExecution<A, T>::planned_div(
    A const & self, A const & other)
{
    reject_boolean("div");
    return divide_executor_type::transform(
        self, other, execution::DivideKernel<value_type>{});
}

template <typename A, typename T>
A SimpleArrayExecution<A, T>::planned_div(
    A const & self, value_type scalar)
{
    reject_boolean("div");
    return divide_executor_type::transform(
        self, scalar, execution::DivideKernel<value_type>{});
}

template <typename A, typename T>
void SimpleArrayExecution<A, T>::planned_iadd(
    A & self, A const & other)
{
    add_executor_type::transform_into(
        self, other, execution::AddKernel<value_type>{});
}

template <typename A, typename T>
void SimpleArrayExecution<A, T>::planned_iadd(
    A & self, value_type scalar)
{
    add_executor_type::transform_into(
        self, scalar, execution::AddKernel<value_type>{});
}

template <typename A, typename T>
void SimpleArrayExecution<A, T>::planned_isub(
    A & self, A const & other)
{
    reject_boolean("isub");
    subtract_executor_type::transform_into(
        self, other, execution::SubtractKernel<value_type>{});
}

template <typename A, typename T>
void SimpleArrayExecution<A, T>::planned_isub(
    A & self, value_type scalar)
{
    reject_boolean("isub");
    subtract_executor_type::transform_into(
        self, scalar, execution::SubtractKernel<value_type>{});
}

template <typename A, typename T>
void SimpleArrayExecution<A, T>::planned_imul(
    A & self, A const & other)
{
    multiply_executor_type::transform_into(
        self, other, execution::MultiplyKernel<value_type>{});
}

template <typename A, typename T>
void SimpleArrayExecution<A, T>::planned_imul(
    A & self, value_type scalar)
{
    multiply_executor_type::transform_into(
        self, scalar, execution::MultiplyKernel<value_type>{});
}

template <typename A, typename T>
void SimpleArrayExecution<A, T>::planned_idiv(
    A & self, A const & other)
{
    reject_boolean("idiv");
    divide_executor_type::transform_into(
        self, other, execution::DivideKernel<value_type>{});
}

template <typename A, typename T>
void SimpleArrayExecution<A, T>::planned_idiv(
    A & self, value_type scalar)
{
    reject_boolean("idiv");
    divide_executor_type::transform_into(
        self, scalar, execution::DivideKernel<value_type>{});
}

template <typename A, typename T>
A SimpleArrayExecution<A, T>::planned_mean(
    A const & self, shape_type const & axes)
{
    auto const plan = execution::ReductionPlan::make(
        self, axes, false);
    return execution::ReductionExecutor<A, value_type>::mean(self, plan);
}

template <typename A, typename T>
typename SimpleArrayExecution<A, T>::value_type
SimpleArrayExecution<A, T>::planned_mean(A const & self)
{
    return execution::ReductionExecutor<A, value_type>::mean(self);
}

template <typename A, typename T>
A SimpleArrayExecution<A, T>::planned_average(
    A const & self,
    shape_type const & axes,
    A const & weight)
{
    auto const plan = execution::ReductionPlan::make(
        self, axes, false);
    return execution::ReductionExecutor<A, value_type>::average(
        self, weight, plan);
}

template <typename A, typename T>
typename SimpleArrayExecution<A, T>::value_type
SimpleArrayExecution<A, T>::planned_average(
    A const & self, A const & weight)
{
    return execution::ReductionExecutor<A, value_type>::average(
        self, weight);
}

template <typename A, typename T>
typename SimpleArrayExecution<A, T>::real_array_type
SimpleArrayExecution<A, T>::planned_var(
    A const & self,
    shape_type const & axes,
    size_t ddof)
{
    auto const plan = execution::ReductionPlan::make(
        self, axes, false);
    return execution::ReductionExecutor<A, value_type>::variance(
        self, plan, ddof);
}

template <typename A, typename T>
typename SimpleArrayExecution<A, T>::real_type
SimpleArrayExecution<A, T>::planned_var(
    A const & self, size_t ddof)
{
    return execution::ReductionExecutor<A, value_type>::variance(self, ddof);
}

template <typename A, typename T>
typename SimpleArrayExecution<A, T>::real_array_type
SimpleArrayExecution<A, T>::planned_std(
    A const & self,
    shape_type const & axes,
    size_t ddof)
{
    auto result = planned_var(self, axes, ddof);
    for (size_t index = 0; index < result.size(); ++index)
    {
        result.logical_data()[index] = std::sqrt(
            result.logical_data()[index]);
    }
    return result;
}

template <typename A, typename T>
typename SimpleArrayExecution<A, T>::real_type
SimpleArrayExecution<A, T>::planned_std(
    A const & self, size_t ddof)
{
    return std::sqrt(planned_var(self, ddof));
}

template <typename A, typename T>
A SimpleArrayExecution<A, T>::planned_median(
    A const & self, shape_type const & axes)
{
    auto const plan = execution::ReductionPlan::make(
        self, axes, false);
    return execution::ReductionExecutor<A, value_type>::collect(
        self, plan, MedianFinalize(self));
}

template <typename A, typename T>
typename SimpleArrayExecution<A, T>::value_type
SimpleArrayExecution<A, T>::planned_median(A const & self)
{
    return execution::ReductionExecutor<A, value_type>::collect(
        self, MedianFinalize(self));
}

template <typename A, typename T>
A SimpleArrayExecution<A, T>::planned_matmul(
    A const & self, A const & other)
{
    return matmul_executor_type::multiply(self, other);
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
