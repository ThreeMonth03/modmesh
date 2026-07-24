/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/loop.hpp>

#include <algorithm>
#include <format>
#include <stdexcept>
#include <utility>

namespace solvcon
{

namespace detail
{

namespace execution
{

LoopDomain::LoopDomain(shape_type shape)
    : m_shape(std::move(shape))
{
}

size_t LoopDomain::size() const noexcept
{
    size_t count = 1;
    for (ssize_t const extent : m_shape)
    {
        count *= static_cast<size_t>(extent);
    }
    return count;
}

stride_type LoopDomain::row_major_strides(shape_type const & shape)
{
    stride_type strides(shape.size(), 1);
    for (size_t axis = shape.size(); axis > 1; --axis)
    {
        strides[axis - 2] = strides[axis - 1] * shape[axis - 1];
    }
    return strides;
}

shape_type LoopDomain::common_shape(shape_type const & lhs,
                                    shape_type const & rhs)
{
    size_t const rank = std::max(lhs.size(), rhs.size());
    shape_type result(rank, 1);
    for (size_t offset = 0; offset < rank; ++offset)
    {
        ssize_t const lhs_extent = offset < lhs.size()
                                       ? lhs[lhs.size() - 1 - offset]
                                       : 1;
        ssize_t const rhs_extent = offset < rhs.size()
                                       ? rhs[rhs.size() - 1 - offset]
                                       : 1;
        if (lhs_extent != rhs_extent && lhs_extent != 1 && rhs_extent != 1)
        {
            throw std::invalid_argument(std::format(
                "cannot broadcast dimensions {} and {}",
                lhs_extent,
                rhs_extent));
        }
        result[rank - 1 - offset] = lhs_extent == 1
                                        ? rhs_extent
                                        : lhs_extent;
    }
    return result;
}

size_t MappingSpan::size() const noexcept
{
    return static_cast<size_t>(m_maximum - m_minimum + 1);
}

OperandMapping::OperandMapping(stride_type strides, ssize_t base_offset)
    : m_base_offset(base_offset)
    , m_strides(std::move(strides))
{
}

MappingSpan OperandMapping::span(LoopDomain const & domain) const
{
    if (domain.empty())
    {
        return MappingSpan(0, -1);
    }
    return span(domain.shape(), m_strides, m_base_offset);
}

MappingSpan OperandMapping::span(
    shape_type const & shape,
    stride_type const & strides,
    ssize_t base_offset)
{
    ssize_t minimum = base_offset;
    ssize_t maximum = base_offset;
    for (size_t axis = 0; axis < shape.size(); ++axis)
    {
        ssize_t const end = (shape[axis] - 1) * strides[axis];
        if (end < 0)
        {
            minimum += end;
        }
        else
        {
            maximum += end;
        }
    }
    return MappingSpan(minimum, maximum);
}

bool OperandMapping::is_row_major(LoopDomain const & domain) const
{
    if (domain.rank() != m_strides.size())
    {
        return false;
    }
    stride_type const expected = LoopDomain::row_major_strides(
        domain.shape());
    for (size_t axis = 0; axis < domain.rank(); ++axis)
    {
        if (domain.shape()[axis] > 1 && m_strides[axis] != expected[axis])
        {
            return false;
        }
    }
    return true;
}

bool OperandMapping::is_dense(LoopDomain const & domain) const
{
    return is_dense(domain.shape(), m_strides);
}

bool OperandMapping::is_dense(
    shape_type const & shape, stride_type const & strides)
{
    size_t element_count = 1;
    for (ssize_t const extent : shape)
    {
        element_count *= static_cast<size_t>(extent);
    }
    return element_count == 0 ||
           span(shape, strides).size() == element_count;
}

OperandMapping OperandMapping::without_last_axis() const
{
    return OperandMapping(
        stride_type(m_strides.begin(), m_strides.end() - 1),
        m_base_offset);
}

OperandMapping OperandMapping::exact(stride_type const & strides)
{
    return OperandMapping(strides);
}

OperandMapping OperandMapping::broadcast(
    shape_type const & operand_shape,
    stride_type const & operand_strides,
    LoopDomain const & domain)
{
    if (operand_shape.size() > domain.rank())
    {
        throw std::invalid_argument("operand rank exceeds result rank");
    }

    stride_type strides(domain.rank(), 0);
    size_t const rank_delta = domain.rank() - operand_shape.size();
    for (size_t axis = 0; axis < operand_shape.size(); ++axis)
    {
        ssize_t const source_extent = operand_shape[axis];
        ssize_t const result_extent = domain.shape()[rank_delta + axis];
        if (source_extent == result_extent)
        {
            strides[rank_delta + axis] = operand_strides[axis];
        }
        else if (source_extent != 1)
        {
            throw std::invalid_argument(std::format(
                "cannot broadcast dimension {} to {}",
                source_extent,
                result_extent));
        }
    }
    return OperandMapping(std::move(strides));
}

MappedOffsetCursor::MappedOffsetCursor(
    LoopDomain const & domain,
    small_vector<OperandMapping> const & mappings)
    : m_domain(&domain)
    , m_mappings(&mappings)
    , m_index(domain.rank(), 0)
    , m_offsets(mappings.size(), 0)
    , m_valid(!domain.empty())
{
    for (size_t operand = 0; operand < mappings.size(); ++operand)
    {
        m_offsets[operand] = mappings[operand].base_offset();
    }
}

InnerLoopPlan::InnerLoopPlan(
    LoopDomain const & domain,
    small_vector<OperandMapping> const & mappings)
{
    if (domain.rank() == 0)
    {
        throw std::invalid_argument(
            "inner loop requires a positive-rank domain");
    }
    size_t const inner_axis = domain.rank() - 1;
    shape_type outer_shape(
        domain.shape().begin(), domain.shape().end() - 1);
    m_outer = LoopDomain(std::move(outer_shape));
    m_size = static_cast<size_t>(domain.shape()[inner_axis]);
    m_strides = stride_type(mappings.size());
    m_outer_mappings = small_vector<OperandMapping>(mappings.size());
    for (size_t operand = 0; operand < mappings.size(); ++operand)
    {
        if (mappings[operand].strides().size() != domain.rank())
        {
            throw std::invalid_argument(
                "inner loop mapping rank does not match domain");
        }
        m_strides[operand] = mappings[operand].stride(inner_axis);
        m_outer_mappings[operand] =
            mappings[operand].without_last_axis();
    }
}

InnerLoopCursor::InnerLoopCursor(InnerLoopPlan const & plan)
    : m_cursor(plan.outer(), plan.outer_mappings())
{
}

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
