#pragma once

/*
 * Copyright (c) 2026, Chun-Shih Chang <austin20463@gmail.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <modmesh/math/Complex.hpp>

#include <cstddef>
#include <type_traits>

namespace modmesh
{

namespace simd
{

namespace detail
{

namespace accelerate
{

#if defined(__APPLE__) && defined(__arm64__)
inline constexpr bool is_available_v = true;
#else
inline constexpr bool is_available_v = false;
#endif

template <typename T>
inline constexpr bool supports_matmul_v = is_available_v &&
                                          (std::is_same_v<T, float> ||
                                           std::is_same_v<T, double> ||
                                           std::is_same_v<T, Complex<float>> ||
                                           std::is_same_v<T, Complex<double>>);

void matmul(size_t m,
            size_t n,
            size_t k,
            float const * lhs,
            float const * rhs,
            float * result);
void matmul(size_t m,
            size_t n,
            size_t k,
            double const * lhs,
            double const * rhs,
            double * result);
void matmul(size_t m,
            size_t n,
            size_t k,
            Complex<float> const * lhs,
            Complex<float> const * rhs,
            Complex<float> * result);
void matmul(size_t m,
            size_t n,
            size_t k,
            Complex<double> const * lhs,
            Complex<double> const * rhs,
            Complex<double> * result);

} /* namespace accelerate */

} /* namespace detail */

} /* namespace simd */

} /* namespace modmesh */
