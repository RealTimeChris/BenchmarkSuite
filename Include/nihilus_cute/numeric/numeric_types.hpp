/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <nihilus_cute/config.hpp>          // CUTE_HOST_DEVICE
#include <nihilus_cute/numeric/int.hpp>     // nihilus_cute::int2_t, nihilus_cute::int4_t, etc

#include <nihilus_gemm/numeric_size.h>   // nihilus_gemm::sizeof_bits
#include <nihilus_gemm/numeric_types.h>  // nihilus_gemm::float_e4m3_t, nihilus_gemm::float_e5m2_t, etc

namespace nihilus_cute {

template <class T>
struct sizeof_bits : nihilus_gemm::sizeof_bits<T> {};

template <class T>
struct sizeof_bits<T const> : sizeof_bits<T> {};

template <class T>
struct sizeof_bits<T volatile> : sizeof_bits<T> {};

template <class T>
struct sizeof_bits<T const volatile> : sizeof_bits<T> {};

// DO NOT change auto to int, sizeof_bits<sparse_elem> use integral_ratio instead of int
template <class T>
static constexpr auto sizeof_bits_v = sizeof_bits<T>::value;

using nihilus_gemm::bits_to_bytes;
using nihilus_gemm::bytes_to_bits;

using nihilus_gemm::is_subbyte;

template <class T>
static constexpr auto is_subbyte_v = is_subbyte<T>::value;

//
// Integral
//

using nihilus_gemm::bin1_t;
using nihilus_gemm::uint1b_t;
using nihilus_gemm::int2b_t;
using nihilus_gemm::uint2b_t;
using nihilus_gemm::int4b_t;
using nihilus_gemm::uint4b_t;
using nihilus_gemm::int6b_t;
using nihilus_gemm::uint6b_t;

//
// Floating Point
//

using nihilus_gemm::half_t;
using nihilus_gemm::bfloat16_t;

using nihilus_gemm::tfloat32_t;

// Umbrella floating-point 8-bit data type : type_erased_dynamic_float8_t
// This umbrella datatype can be enabled when a user provides a specific
// datatype in runtime argument list.
using nihilus_gemm::type_erased_dynamic_float8_t;
using nihilus_gemm::float_e4m3_t;
using nihilus_gemm::float_e5m2_t;




using nihilus_gemm::float_ue4m3_t;
using nihilus_gemm::float_ue8m0_t;

using nihilus_gemm::float_e2m1_t;
using nihilus_gemm::float_e2m3_t;
using nihilus_gemm::float_e3m2_t;

using nihilus_gemm::type_erased_dynamic_float6_t;
using nihilus_gemm::type_erased_dynamic_float4_t;

namespace detail {
using nihilus_gemm::detail::float_e2m1_unpacksmem_t;
using nihilus_gemm::detail::float_e2m3_unpacksmem_t;
using nihilus_gemm::detail::float_e3m2_unpacksmem_t;
using nihilus_gemm::detail::float_e2m3_unpack8bits_t;
using nihilus_gemm::detail::float_e3m2_unpack8bits_t;
using nihilus_gemm::detail::type_erased_dynamic_float4_unpacksmem_t;
using nihilus_gemm::detail::type_erased_dynamic_float6_unpacksmem_t;
};

//
// Print utility
//

CUTE_HOST_DEVICE
void
print(half_t a) {
  printf("%f", static_cast<float>(a));
}

CUTE_HOST_DEVICE
void
print(bfloat16_t a) {
  printf("%f", static_cast<float>(a));
}

CUTE_HOST_DEVICE
void
print(tfloat32_t a) {
  printf("%f", static_cast<float>(a));
}

CUTE_HOST_DEVICE
void
print(float_e4m3_t a) {
  printf("%f", static_cast<float>(a));
}

CUTE_HOST_DEVICE
void
print(float_e5m2_t a) {
  printf("%f", static_cast<float>(a));
}

template <nihilus_gemm::detail::FpEncoding Encoding, class Derived>
CUTE_HOST_DEVICE
void
print(nihilus_gemm::float_exmy_base<Encoding, Derived> a) {
  printf("%f", static_cast<float>(a));
}

// Pretty Print utility

CUTE_HOST_DEVICE void
pretty_print(bfloat16_t v) {
  printf("%*.2f", 8, float(v));
}

CUTE_HOST_DEVICE void
pretty_print(half_t v) {
  printf("%*.2f", 8, float(v));
}

CUTE_HOST_DEVICE void
pretty_print(tfloat32_t v) {
  printf("%*.2e", 10, static_cast<float>(v));
}

CUTE_HOST_DEVICE void
pretty_print(float_e4m3_t t) {
  printf("%*.2f", 8, static_cast<float>(t));
}

CUTE_HOST_DEVICE void
pretty_print(float_e5m2_t t) {
  printf("%*.2f", 8, static_cast<float>(t));
}

template <nihilus_gemm::detail::FpEncoding Encoding, class Derived>
CUTE_HOST_DEVICE
void
pretty_print_float_exmy_base(nihilus_gemm::float_exmy_base<Encoding, Derived> t) {
  printf("%*.2f", 8, static_cast<float>(t));
}

} // namespace nihilus_cute
