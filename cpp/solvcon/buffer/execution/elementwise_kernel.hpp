#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/simd/simd.hpp>

#include <concepts>
#include <cstddef>

namespace solvcon
{

namespace detail
{

namespace execution
{

template <typename Derived, typename T>
class BinaryKernelBase
{
public:
    static void scalar(T * output, size_t count, T rhs);
    static void inplace(T * output, size_t count, T const * rhs);
}; /* end class BinaryKernelBase */

template <typename Derived, typename T>
void BinaryKernelBase<Derived, T>::scalar(T * output, size_t count, T rhs)
{
    Derived const kernel;
    T const * const end = output + count;
    while (output < end)
    {
        *output = kernel(*output, rhs);
        ++output;
    }
}

template <typename Derived, typename T>
void BinaryKernelBase<Derived, T>::inplace(
    T * output, size_t count, T const * rhs)
{
    Derived const kernel;
    T const * const end = output + count;
    while (output < end)
    {
        *output = kernel(*output, *rhs);
        ++output;
        ++rhs;
    }
}

template <typename T>
class AddKernel : public BinaryKernelBase<AddKernel<T>, T>
{
public:
    T operator()(T lhs, T rhs) const { return lhs + rhs; }

    static void contiguous(T * output,
                           size_t count,
                           T const * lhs,
                           T const * rhs)
    {
        simd::add<T>(output, output + count, lhs, rhs);
    }
    static void contiguous_scalar(T * output,
                                  size_t count,
                                  T const * lhs,
                                  T rhs)
    {
        simd::add_scalar<T>(output, output + count, lhs, rhs);
    }
}; /* end class AddKernel */

template <typename T>
class SubtractKernel : public BinaryKernelBase<SubtractKernel<T>, T>
{
public:
    T operator()(T lhs, T rhs) const { return lhs - rhs; }

    static void contiguous(T * output,
                           size_t count,
                           T const * lhs,
                           T const * rhs)
    {
        simd::sub<T>(output, output + count, lhs, rhs);
    }
    static void contiguous_scalar(T * output,
                                  size_t count,
                                  T const * lhs,
                                  T rhs)
    {
        simd::sub_scalar<T>(output, output + count, lhs, rhs);
    }
}; /* end class SubtractKernel */

template <typename T>
class MultiplyKernel : public BinaryKernelBase<MultiplyKernel<T>, T>
{
public:
    T operator()(T lhs, T rhs) const { return lhs * rhs; }

    static void contiguous(T * output,
                           size_t count,
                           T const * lhs,
                           T const * rhs)
    {
        simd::mul<T>(output, output + count, lhs, rhs);
    }
    static void contiguous_scalar(T * output,
                                  size_t count,
                                  T const * lhs,
                                  T rhs)
    {
        simd::mul_scalar<T>(output, output + count, lhs, rhs);
    }
}; /* end class MultiplyKernel */

template <typename T>
class DivideKernel : public BinaryKernelBase<DivideKernel<T>, T>
{
public:
    T operator()(T lhs, T rhs) const { return lhs / rhs; }

    static void contiguous(T * output,
                           size_t count,
                           T const * lhs,
                           T const * rhs)
    {
        simd::div<T>(output, output + count, lhs, rhs);
    }
    static void contiguous_scalar(T * output,
                                  size_t count,
                                  T const * lhs,
                                  T rhs)
    {
        simd::div_scalar<T>(output, output + count, lhs, rhs);
    }
}; /* end class DivideKernel */

template <typename Kernel, typename T>
concept ElementwiseKernel = requires(Kernel kernel,
                                     T lhs,
                                     T rhs,
                                     T * output,
                                     T const * lhs_data,
                                     T const * rhs_data) {
    {
        kernel(lhs, rhs)
    } -> std::convertible_to<T>;
    Kernel::scalar(output, size_t{}, rhs);
    Kernel::inplace(output, size_t{}, rhs_data);
    Kernel::contiguous(output, size_t{}, lhs_data, rhs_data);
    Kernel::contiguous_scalar(output, size_t{}, lhs_data, rhs);
};

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
