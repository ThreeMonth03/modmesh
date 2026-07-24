#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/elementwise_kernel.hpp>
#include <solvcon/buffer/execution/loop.hpp>

#include <cstdint>
#include <stdexcept>

namespace solvcon
{

namespace detail
{

namespace execution
{

enum class ElementwiseLayout : uint8_t
{
    contiguous,
    inner_strided,
    strided,
}; /* end enum class ElementwiseLayout */

class ElementwisePlan
{
public:
    LoopDomain const & domain() const noexcept { return m_domain; }
    OperandMapping const & output() const noexcept { return m_output; }
    OperandMapping const & input(size_t index) const noexcept
    {
        return m_inputs[index];
    }
    ElementwiseLayout layout() const noexcept { return m_layout; }

    template <typename... Inputs>
    static shape_type broadcast_shape(Inputs const &... inputs);

    template <typename Array, typename... Inputs>
    static ElementwisePlan make(Array const & output,
                                Inputs const &... inputs);

    template <typename Array>
    static ElementwisePlan make_scalar(Array const & output,
                                       Array const & lhs);

private:
    LoopDomain m_domain;
    OperandMapping m_output;
    small_vector<OperandMapping> m_inputs;
    ElementwiseLayout m_layout = ElementwiseLayout::strided;
}; /* end class ElementwisePlan */

template <typename... Inputs>
shape_type ElementwisePlan::broadcast_shape(Inputs const &... inputs)
{
    shape_type shape;
    ((shape = LoopDomain::common_shape(shape, inputs.shape())), ...);
    return shape;
}

template <typename Array, typename... Inputs>
ElementwisePlan ElementwisePlan::make(
    Array const & output, Inputs const &... inputs)
{
    LoopDomain const domain(broadcast_shape(inputs...));
    if (output.shape() != domain.shape())
    {
        throw std::invalid_argument(
            "elementwise output shape does not match result shape");
    }

    ElementwisePlan plan;
    plan.m_domain = domain;
    plan.m_output = OperandMapping::exact(output.stride());
    plan.m_inputs = small_vector<OperandMapping>{
        OperandMapping::broadcast(
            inputs.shape(), inputs.stride(), domain)...};

    bool row_major = plan.m_output.is_row_major(domain);
    bool common_dense_layout = plan.m_output.is_dense(domain);
    for (OperandMapping const & input : plan.m_inputs)
    {
        row_major = row_major && input.is_row_major(domain);
        common_dense_layout = common_dense_layout &&
                              plan.m_output.strides() == input.strides();
    }
    if (row_major || common_dense_layout)
    {
        plan.m_layout = ElementwiseLayout::contiguous;
    }
    else if (domain.rank() != 0)
    {
        plan.m_layout = ElementwiseLayout::inner_strided;
    }
    return plan;
}

template <typename Array>
ElementwisePlan ElementwisePlan::make_scalar(
    Array const & output, Array const & lhs)
{
    if (output.shape() != lhs.shape())
    {
        throw std::invalid_argument(
            "scalar elementwise output shape mismatch");
    }

    ElementwisePlan plan;
    plan.m_domain = LoopDomain(output.shape());
    plan.m_output = OperandMapping::exact(output.stride());
    plan.m_inputs = small_vector<OperandMapping>{
        OperandMapping::broadcast(
            lhs.shape(), lhs.stride(), plan.m_domain)};
    bool const row_major = plan.m_output.is_row_major(plan.m_domain) &&
                           plan.m_inputs[0].is_row_major(plan.m_domain);
    bool const common_dense_layout =
        plan.m_output.strides() == plan.m_inputs[0].strides() &&
        plan.m_output.is_dense(plan.m_domain);
    if (row_major || common_dense_layout)
    {
        plan.m_layout = ElementwiseLayout::contiguous;
    }
    else if (plan.m_domain.rank() != 0)
    {
        plan.m_layout = ElementwiseLayout::inner_strided;
    }
    return plan;
}

template <typename Array, typename T, typename Kernel>
requires ElementwiseKernel<Kernel, T>
class ElementwiseExecutor
{
public:
    using value_type = T;
    using kernel_type = Kernel;

    static Array transform(Array const & lhs,
                           Array const & rhs,
                           kernel_type kernel);
    static Array transform(Array const & lhs,
                           value_type scalar,
                           kernel_type kernel);
    static void transform_into(Array & destination,
                               Array const & rhs,
                               kernel_type kernel);
    static void transform_into(Array & destination,
                               value_type scalar,
                               kernel_type kernel);

private:
    static void execute(ElementwisePlan const & plan,
                        Array & output,
                        Array const & lhs,
                        Array const & rhs,
                        kernel_type kernel);
    static void execute_scalar(ElementwisePlan const & plan,
                               Array & output,
                               Array const & lhs,
                               value_type scalar,
                               kernel_type kernel);
    static void execute_into(Array & destination,
                             Array const & rhs,
                             kernel_type kernel);
    static bool storage_overlaps(Array const & lhs,
                                 Array const & rhs);
}; /* end class ElementwiseExecutor */

template <typename Array, typename T, typename Kernel>
requires ElementwiseKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::execute(
    ElementwisePlan const & plan,
    Array & output,
    Array const & lhs,
    Array const & rhs,
    kernel_type kernel)
{
    value_type * output_data = output.logical_data();
    value_type const * lhs_data = lhs.logical_data();
    value_type const * rhs_data = rhs.logical_data();
    OperandMapping const & lhs_mapping = plan.input(0);
    OperandMapping const & rhs_mapping = plan.input(1);
    if (plan.layout() == ElementwiseLayout::contiguous)
    {
        ssize_t const output_offset = plan.output().span(
                                                       plan.domain())
                                          .minimum();
        ssize_t const lhs_offset = lhs_mapping.span(
                                                  plan.domain())
                                       .minimum();
        ssize_t const rhs_offset = rhs_mapping.span(
                                                  plan.domain())
                                       .minimum();
        kernel_type::contiguous(output_data + output_offset,
                                plan.domain().size(),
                                lhs_data + lhs_offset,
                                rhs_data + rhs_offset);
        return;
    }

    if (plan.layout() == ElementwiseLayout::inner_strided)
    {
        small_vector<OperandMapping> const mappings{
            plan.output(), lhs_mapping, rhs_mapping};
        InnerLoopPlan const inner(plan.domain(), mappings);
        ssize_t const output_stride = inner.stride(0);
        ssize_t const lhs_stride = inner.stride(1);
        ssize_t const rhs_stride = inner.stride(2);
        for (InnerLoopCursor cursor(inner); cursor;
             cursor.advance())
        {
            if (output_stride == 1 &&
                lhs_stride == 1 &&
                rhs_stride == 1)
            {
                kernel_type::contiguous(
                    output_data + cursor.offset(0),
                    inner.size(),
                    lhs_data + cursor.offset(1),
                    rhs_data + cursor.offset(2));
                continue;
            }
            if (output_stride == 1 &&
                lhs_stride == 1 &&
                rhs_stride == 0)
            {
                kernel_type::contiguous_scalar(
                    output_data + cursor.offset(0),
                    inner.size(),
                    lhs_data + cursor.offset(1),
                    rhs_data[cursor.offset(2)]);
                continue;
            }
            ssize_t output_offset = cursor.offset(0);
            ssize_t lhs_offset = cursor.offset(1);
            ssize_t rhs_offset = cursor.offset(2);
            for (size_t index = 0; index < inner.size(); ++index)
            {
                output_data[output_offset] = kernel(
                    lhs_data[lhs_offset], rhs_data[rhs_offset]);
                output_offset += output_stride;
                lhs_offset += lhs_stride;
                rhs_offset += rhs_stride;
            }
        }
        return;
    }

    small_vector<OperandMapping> const mappings{
        plan.output(), lhs_mapping, rhs_mapping};
    for (MappedOffsetCursor cursor(plan.domain(), mappings); cursor;
         cursor.advance())
    {
        output_data[cursor.offset(0)] = kernel(
            lhs_data[cursor.offset(1)], rhs_data[cursor.offset(2)]);
    }
}

template <typename Array, typename T, typename Kernel>
requires ElementwiseKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::execute_scalar(
    ElementwisePlan const & plan,
    Array & output,
    Array const & lhs,
    value_type scalar,
    kernel_type kernel)
{
    value_type * output_data = output.logical_data();
    value_type const * lhs_data = lhs.logical_data();
    OperandMapping const & lhs_mapping = plan.input(0);
    if (plan.layout() == ElementwiseLayout::contiguous)
    {
        output_data += plan.output().span(plan.domain()).minimum();
        lhs_data += lhs_mapping.span(plan.domain()).minimum();
        size_t const count = plan.domain().size();
        kernel_type::contiguous_scalar(
            output_data, count, lhs_data, scalar);
        return;
    }

    if (plan.layout() == ElementwiseLayout::inner_strided)
    {
        small_vector<OperandMapping> const mappings{
            plan.output(), lhs_mapping};
        InnerLoopPlan const inner(plan.domain(), mappings);
        ssize_t const output_stride = inner.stride(0);
        ssize_t const lhs_stride = inner.stride(1);
        for (InnerLoopCursor cursor(inner); cursor;
             cursor.advance())
        {
            ssize_t output_offset = cursor.offset(0);
            ssize_t lhs_offset = cursor.offset(1);
            for (size_t index = 0; index < inner.size(); ++index)
            {
                output_data[output_offset] =
                    kernel(lhs_data[lhs_offset], scalar);
                output_offset += output_stride;
                lhs_offset += lhs_stride;
            }
        }
        return;
    }

    small_vector<OperandMapping> const mappings{
        plan.output(), lhs_mapping};
    for (MappedOffsetCursor cursor(plan.domain(), mappings); cursor;
         cursor.advance())
    {
        output_data[cursor.offset(0)] = kernel(
            lhs_data[cursor.offset(1)], scalar);
    }
}

template <typename Array, typename T, typename Kernel>
requires ElementwiseKernel<Kernel, T>
Array ElementwiseExecutor<Array, T, Kernel>::transform(
    Array const & lhs, Array const & rhs, kernel_type kernel)
{
    LoopDomain const result_domain(
        ElementwisePlan::broadcast_shape(lhs, rhs));
    shape_type const & result_shape = result_domain.shape();
    OperandMapping const lhs_mapping = OperandMapping::exact(lhs.stride());
    bool const preserve_layout = result_shape == lhs.shape() &&
                                 result_shape == rhs.shape() &&
                                 lhs.stride() == rhs.stride() &&
                                 lhs_mapping.is_dense(result_domain);
    Array output = preserve_layout
                       ? LayoutAllocator<Array>::allocate(
                             result_shape, lhs.stride())
                       : Array(result_shape);
    ElementwisePlan const plan = ElementwisePlan::make(
        output, lhs, rhs);
    execute(plan, output, lhs, rhs, kernel);
    return output;
}

template <typename Array, typename T, typename Kernel>
requires ElementwiseKernel<Kernel, T>
Array ElementwiseExecutor<Array, T, Kernel>::transform(
    Array const & lhs, value_type scalar, kernel_type kernel)
{
    LoopDomain const domain(lhs.shape());
    OperandMapping const mapping = OperandMapping::exact(lhs.stride());
    Array output = mapping.is_dense(domain)
                       ? LayoutAllocator<Array>::allocate(
                             lhs.shape(), lhs.stride())
                       : Array(lhs.shape());
    ElementwisePlan const plan =
        ElementwisePlan::make_scalar(output, lhs);
    execute_scalar(plan, output, lhs, scalar, kernel);
    return output;
}

template <typename Array, typename T, typename Kernel>
requires ElementwiseKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::transform_into(
    Array & destination, Array const & rhs, kernel_type kernel)
{
    bool const exact_alias =
        destination.logical_data() == rhs.logical_data() &&
        destination.shape() == rhs.shape() &&
        destination.stride() == rhs.stride();
    if (storage_overlaps(destination, rhs) && !exact_alias)
    {
        Array const safe_rhs(rhs); // NOLINT(performance-unnecessary-copy-initialization)
        execute_into(destination, safe_rhs, kernel);
        return;
    }
    execute_into(destination, rhs, kernel);
}

template <typename Array, typename T, typename Kernel>
requires ElementwiseKernel<Kernel, T>
bool ElementwiseExecutor<Array, T, Kernel>::storage_overlaps(
    Array const & lhs, Array const & rhs)
{
    if (lhs.size() == 0 || rhs.size() == 0)
    {
        return false;
    }

    MappingSpan const lhs_span =
        OperandMapping::span(lhs.shape(), lhs.stride());
    MappingSpan const rhs_span =
        OperandMapping::span(rhs.shape(), rhs.stride());
    std::uintptr_t const lhs_begin = reinterpret_cast<std::uintptr_t>(
        lhs.logical_data() + lhs_span.minimum());
    std::uintptr_t const lhs_end = reinterpret_cast<std::uintptr_t>(
        lhs.logical_data() + lhs_span.maximum() + 1);
    std::uintptr_t const rhs_begin = reinterpret_cast<std::uintptr_t>(
        rhs.logical_data() + rhs_span.minimum());
    std::uintptr_t const rhs_end = reinterpret_cast<std::uintptr_t>(
        rhs.logical_data() + rhs_span.maximum() + 1);
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

template <typename Array, typename T, typename Kernel>
requires ElementwiseKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::execute_into(
    Array & destination, Array const & rhs, kernel_type kernel)
{
    if (destination.shape() == rhs.shape() &&
        destination.stride() == rhs.stride() &&
        ((destination.is_c_contiguous() && rhs.is_c_contiguous()) ||
         (destination.is_f_contiguous() && rhs.is_f_contiguous())))
    {
        kernel_type::inplace(destination.logical_data(),
                             destination.size(),
                             rhs.logical_data());
        return;
    }

    if (destination.shape() == rhs.shape() &&
        destination.stride() == rhs.stride() &&
        OperandMapping::is_dense(
            destination.shape(), destination.stride()))
    {
        ssize_t const offset = OperandMapping::span(
                                   destination.shape(),
                                   destination.stride())
                                   .minimum();
        kernel_type::inplace(destination.logical_data() + offset,
                             destination.size(),
                             rhs.logical_data() + offset);
        return;
    }

    ElementwisePlan const plan = ElementwisePlan::make(
        destination, destination, rhs);
    execute(plan, destination, destination, rhs, kernel);
}

template <typename Array, typename T, typename Kernel>
requires ElementwiseKernel<Kernel, T>
void ElementwiseExecutor<Array, T, Kernel>::transform_into(
    Array & destination, value_type scalar, kernel_type kernel)
{
    if (destination.is_c_contiguous() || destination.is_f_contiguous())
    {
        kernel_type::scalar(destination.logical_data(),
                            destination.size(),
                            scalar);
        return;
    }

    if (OperandMapping::is_dense(
            destination.shape(), destination.stride()))
    {
        value_type * data = destination.logical_data() +
                            OperandMapping::span(
                                destination.shape(),
                                destination.stride())
                                .minimum();
        kernel_type::scalar(data, destination.size(), scalar);
        return;
    }

    ElementwisePlan const plan =
        ElementwisePlan::make_scalar(destination, destination);
    execute_scalar(plan, destination, destination, scalar, kernel);
}

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
