#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/small_vector.hpp>

#include <cstddef>

namespace solvcon
{

namespace detail
{

namespace execution
{

using shape_type = small_vector<ssize_t>;
using stride_type = small_vector<ssize_t>;

class LoopDomain
{
public:
    LoopDomain() = default;
    explicit LoopDomain(shape_type shape);

    shape_type const & shape() const noexcept { return m_shape; }
    size_t rank() const noexcept { return m_shape.size(); }
    size_t size() const noexcept;
    bool empty() const noexcept { return size() == 0; }

    static stride_type row_major_strides(shape_type const & shape);
    static shape_type common_shape(shape_type const & lhs,
                                   shape_type const & rhs);

private:
    shape_type m_shape;
}; /* end class LoopDomain */

class MappingSpan
{
public:
    MappingSpan() = default;
    MappingSpan(ssize_t minimum, ssize_t maximum)
        : m_minimum(minimum)
        , m_maximum(maximum)
    {
    }

    ssize_t minimum() const noexcept { return m_minimum; }
    ssize_t maximum() const noexcept { return m_maximum; }
    size_t size() const noexcept;

private:
    ssize_t m_minimum = 0;
    ssize_t m_maximum = 0;
}; /* end class MappingSpan */

class OperandMapping
{
public:
    OperandMapping() = default;
    explicit OperandMapping(stride_type strides, ssize_t base_offset = 0);

    ssize_t base_offset() const noexcept { return m_base_offset; }
    stride_type const & strides() const noexcept { return m_strides; }
    ssize_t stride(size_t axis) const noexcept { return m_strides[axis]; }

    MappingSpan span(LoopDomain const & domain) const;
    bool is_row_major(LoopDomain const & domain) const;
    bool is_dense(LoopDomain const & domain) const;
    OperandMapping without_last_axis() const;

    static MappingSpan span(shape_type const & shape,
                            stride_type const & strides,
                            ssize_t base_offset = 0);
    static bool is_dense(shape_type const & shape,
                         stride_type const & strides);
    static OperandMapping exact(stride_type const & strides);
    static OperandMapping broadcast(shape_type const & operand_shape,
                                    stride_type const & operand_strides,
                                    LoopDomain const & domain);

private:
    ssize_t m_base_offset = 0;
    stride_type m_strides;
}; /* end class OperandMapping */

class MappedOffsetCursor
{
public:
    MappedOffsetCursor(LoopDomain const & domain,
                       small_vector<OperandMapping> const & mappings);

    explicit operator bool() const noexcept { return m_valid; }
    ssize_t offset(size_t operand) const noexcept { return m_offsets[operand]; }
    stride_type const & offsets() const noexcept { return m_offsets; }
    void advance()
    {
        for (size_t axis_plus_one = m_domain->rank(); axis_plus_one > 0;
             --axis_plus_one)
        {
            size_t const axis = axis_plus_one - 1;
            ++m_index[axis];
            for (size_t operand = 0; operand < m_mappings->size(); ++operand)
            {
                m_offsets[operand] += (*m_mappings)[operand].stride(axis);
            }
            if (m_index[axis] < m_domain->shape()[axis])
            {
                return;
            }
            m_index[axis] = 0;
            for (size_t operand = 0; operand < m_mappings->size(); ++operand)
            {
                m_offsets[operand] -= (*m_mappings)[operand].stride(axis) *
                                      m_domain->shape()[axis];
            }
        }
        m_valid = false;
    }

private:
    LoopDomain const * m_domain;
    small_vector<OperandMapping> const * m_mappings;
    shape_type m_index;
    stride_type m_offsets;
    bool m_valid;
}; /* end class MappedOffsetCursor */

class InnerLoopPlan
{
public:
    InnerLoopPlan() = default;
    InnerLoopPlan(LoopDomain const & domain,
                  small_vector<OperandMapping> const & mappings);

    LoopDomain const & outer() const noexcept { return m_outer; }
    size_t size() const noexcept { return m_size; }
    ssize_t stride(size_t operand) const noexcept
    {
        return m_strides[operand];
    }
    small_vector<OperandMapping> const & outer_mappings() const noexcept
    {
        return m_outer_mappings;
    }

private:
    LoopDomain m_outer;
    size_t m_size = 0;
    stride_type m_strides;
    small_vector<OperandMapping> m_outer_mappings;
}; /* end class InnerLoopPlan */

class InnerLoopCursor
{
public:
    explicit InnerLoopCursor(InnerLoopPlan const & plan);

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(m_cursor);
    }
    ssize_t offset(size_t operand) const noexcept
    {
        return m_cursor.offset(operand);
    }
    void advance() { m_cursor.advance(); }

private:
    MappedOffsetCursor m_cursor;
}; /* end class InnerLoopCursor */

template <typename Array>
class LayoutAllocator
{
public:
    static Array allocate(shape_type const & shape,
                          stride_type const & strides)
    {
        LoopDomain const domain(shape);
        if (domain.empty())
        {
            return Array(shape);
        }
        OperandMapping const mapping(strides);
        MappingSpan const span = mapping.span(domain);
        auto buffer = Array::buffer_type::construct(
            span.size() * Array::ITEMSIZE);
        size_t const data_offset = static_cast<size_t>(-span.minimum()) *
                                   Array::ITEMSIZE;
        return Array(shape, strides, buffer, data_offset);
    }
}; /* end class LayoutAllocator */

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
