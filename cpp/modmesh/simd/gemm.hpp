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

#include <modmesh/simd/accelerate/accelerate.hpp>
#include <modmesh/simd/gemm_generic.hpp>

#include <cstddef>

namespace modmesh
{

namespace simd
{

namespace detail
{

enum class GemmBackend
{
    Generic,
    Accelerate,
};

template <typename T>
inline constexpr GemmBackend gemm_backend_v = accelerate::supports_matmul_v<T>
                                                  ? GemmBackend::Accelerate
                                                  : GemmBackend::Generic;

} /* namespace detail */

template <typename T>
void matmul(size_t m, size_t n, size_t k, T const * lhs, T const * rhs, T * result)
{
    if constexpr (detail::gemm_backend_v<T> == detail::GemmBackend::Accelerate)
    {
        detail::accelerate::matmul(m, n, k, lhs, rhs, result);
    }
    else
    {
        generic::matmul(m, n, k, lhs, rhs, result);
    }
}

} /* namespace simd */

} /* namespace modmesh */
