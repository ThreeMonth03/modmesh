#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/loop.hpp>

#include <concepts>
#include <stdexcept>
#include <utility>

namespace solvcon
{

namespace detail
{

namespace execution
{

class CommonBroadcastingRule
{
public:
    template <typename... Inputs>
    static LoopDomain make_domain(Inputs const &... inputs);

    template <typename Input>
    static OperandMapping make_mapping(Input const & input,
                                       LoopDomain const & domain);
}; /* end class CommonBroadcastingRule */

template <typename... Inputs>
LoopDomain CommonBroadcastingRule::make_domain(Inputs const &... inputs)
{
    shape_type shape;
    ((shape = LoopDomain::common_shape(shape, inputs.shape())), ...);
    return LoopDomain(std::move(shape));
}

template <typename Input>
OperandMapping CommonBroadcastingRule::make_mapping(
    Input const & input, LoopDomain const & domain)
{
    return OperandMapping::broadcast(
        input.shape(), input.stride(), domain);
}

class NoBroadcastingRule
{
public:
    template <typename Input>
    static OperandMapping make_mapping(Input const & input)
    {
        return OperandMapping::exact(input.stride());
    }
}; /* end class NoBroadcastingRule */

class BatchBroadcastingRule
{
public:
    static LoopDomain make_domain(shape_type const & lhs_shape,
                                  shape_type const & rhs_shape)
    {
        return LoopDomain(LoopDomain::common_shape(
            lhs_shape, rhs_shape));
    }

    static OperandMapping make_mapping(
        shape_type const & shape,
        stride_type const & strides,
        LoopDomain const & domain)
    {
        return OperandMapping::broadcast(shape, strides, domain);
    }
}; /* end class BatchBroadcastingRule */

class AxisReductionPartition
{
public:
    AxisReductionPartition(shape_type output_shape,
                           LoopDomain outer,
                           LoopDomain inner,
                           OperandMapping outer_input,
                           OperandMapping inner_input)
        : m_output_shape(std::move(output_shape))
        , m_outer(std::move(outer))
        , m_inner(std::move(inner))
        , m_outer_input(std::move(outer_input))
        , m_inner_input(std::move(inner_input))
    {
    }

    shape_type const & output_shape() const noexcept
    {
        return m_output_shape;
    }
    LoopDomain const & outer() const noexcept { return m_outer; }
    LoopDomain const & inner() const noexcept { return m_inner; }
    OperandMapping const & outer_input() const noexcept
    {
        return m_outer_input;
    }
    OperandMapping const & inner_input() const noexcept
    {
        return m_inner_input;
    }

private:
    shape_type m_output_shape;
    LoopDomain m_outer;
    LoopDomain m_inner;
    OperandMapping m_outer_input;
    OperandMapping m_inner_input;
}; /* end class AxisReductionPartition */

class AxisReductionRule
{
public:
    static AxisReductionPartition make(
        shape_type const & input_shape,
        OperandMapping const & input_mapping,
        shape_type const & axes,
        bool allow_all_axes);
}; /* end class AxisReductionRule */

class NoReductionRule
{
}; /* end class NoReductionRule */

class ContractionReductionRule
{
public:
    static bool compatible(ssize_t lhs_size, ssize_t rhs_size) noexcept
    {
        return lhs_size == rhs_size;
    }
}; /* end class ContractionReductionRule */

class ElementwiseIterationRule
{
public:
    template <typename Operation, typename Array, typename... Inputs>
    static auto make(Array const & output, Inputs const &... inputs);

    template <typename Operation, typename Array>
    static auto make_scalar(Array const & output, Array const & input);
}; /* end class ElementwiseIterationRule */

class ReductionIterationRule
{
public:
    template <typename Operation, typename Array>
    static auto make(Array const & input,
                     shape_type const & axes,
                     bool allow_all_axes);

    template <typename Operation, typename Array>
    static auto make_all(Array const & input);
}; /* end class ReductionIterationRule */

class MatmulIterationRule
{
public:
    template <typename Operation, typename Array>
    static auto make(Array const & lhs, Array const & rhs);
}; /* end class MatmulIterationRule */

template <typename T>
class MeanComputeRule
{
public:
    static T finalize(T total, size_t count)
    {
        return total / static_cast<T>(count);
    }
}; /* end class MeanComputeRule */

template <typename T>
class AverageComputeRule
{
public:
    static T finalize(T weighted_sum, T total_weight)
    {
        return weighted_sum / total_weight;
    }
}; /* end class AverageComputeRule */

template <typename T>
class VarianceComputeRule
{
public:
    static T finalize(T squared_difference,
                      size_t count,
                      size_t ddof)
    {
        return squared_difference / static_cast<T>(count - ddof);
    }
}; /* end class VarianceComputeRule */

class CollectionComputeRule
{
}; /* end class CollectionComputeRule */

class MatmulComputeRule
{
}; /* end class MatmulComputeRule */

template <typename BroadcastingRule,
          typename ReductionRule,
          typename IterationRule,
          typename ComputeRule>
class OperationDefinition
{
public:
    using broadcasting_rule = BroadcastingRule;
    using reduction_rule = ReductionRule;
    using iteration_rule = IterationRule;
    using compute_rule = ComputeRule;
}; /* end class OperationDefinition */

template <typename Operation>
concept PlannedOperation = requires {
    typename Operation::broadcasting_rule;
    typename Operation::reduction_rule;
    typename Operation::iteration_rule;
    typename Operation::compute_rule;
};

template <typename Operation>
concept ElementwiseOperation =
    PlannedOperation<Operation> &&
    std::same_as<typename Operation::broadcasting_rule,
                 CommonBroadcastingRule> &&
    std::same_as<typename Operation::reduction_rule,
                 NoReductionRule> &&
    std::same_as<typename Operation::iteration_rule,
                 ElementwiseIterationRule>;

template <typename Operation>
concept ReductionOperation =
    PlannedOperation<Operation> &&
    std::same_as<typename Operation::broadcasting_rule,
                 NoBroadcastingRule> &&
    std::same_as<typename Operation::reduction_rule,
                 AxisReductionRule> &&
    std::same_as<typename Operation::iteration_rule,
                 ReductionIterationRule>;

template <typename Operation>
concept MatmulOperation =
    PlannedOperation<Operation> &&
    std::same_as<typename Operation::broadcasting_rule,
                 BatchBroadcastingRule> &&
    std::same_as<typename Operation::reduction_rule,
                 ContractionReductionRule> &&
    std::same_as<typename Operation::iteration_rule,
                 MatmulIterationRule>;

template <typename ComputeRule>
using elementwise_operation_type = OperationDefinition<
    CommonBroadcastingRule,
    NoReductionRule,
    ElementwiseIterationRule,
    ComputeRule>;

template <typename ComputeRule>
using reduction_operation_type = OperationDefinition<
    NoBroadcastingRule,
    AxisReductionRule,
    ReductionIterationRule,
    ComputeRule>;

template <typename ComputeRule>
using matmul_operation_type = OperationDefinition<
    BatchBroadcastingRule,
    ContractionReductionRule,
    MatmulIterationRule,
    ComputeRule>;

template <typename T>
using mean_operation_type =
    reduction_operation_type<MeanComputeRule<T>>;

template <typename T>
using average_operation_type =
    reduction_operation_type<AverageComputeRule<T>>;

template <typename T>
using variance_operation_type =
    reduction_operation_type<VarianceComputeRule<T>>;

using collection_operation_type =
    reduction_operation_type<CollectionComputeRule>;

using matrix_multiply_operation_type =
    matmul_operation_type<MatmulComputeRule>;

template <PlannedOperation Operation, typename... Arguments>
auto make_operation_plan(Arguments const &... arguments)
{
    return Operation::iteration_rule::template make<Operation>(
        arguments...);
}

template <ElementwiseOperation Operation, typename Array>
auto make_scalar_operation_plan(Array const & output,
                                Array const & input)
{
    return Operation::iteration_rule::template make_scalar<Operation>(
        output, input);
}

template <ReductionOperation Operation, typename Array>
auto make_all_operation_plan(Array const & input)
{
    return Operation::iteration_rule::template make_all<Operation>(input);
}

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
