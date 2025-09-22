/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*! 
    \file
    \brief Top-level include for all CUTLASS numeric types.
*/
#pragma once

#include "cute_rt_tm/util/type_traits.hpp"

#include "nihilus_gemm/numeric_size.h"
#include "nihilus_gemm/integer_subbyte.h"
#include "nihilus_gemm/half.h"
#include "nihilus_gemm/bfloat16.h"
#include "nihilus_gemm/tfloat32.h"
#include "nihilus_gemm/float8.h"
#include "nihilus_gemm/uint128.h"
#include "nihilus_gemm/uint256.h"
#include "nihilus_gemm/exmy_base.h"
#include "nihilus_gemm/float_subbyte.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace nihilus_gemm {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <size_t... Seq>
struct index_sequence;

template <size_t N, size_t... Next>
struct index_sequence_helper : index_sequence_helper<N - 1, N - 1, Next...> {};

template <size_t... Next>
struct index_sequence_helper<0, 0, Next...> {
  using type = index_sequence<0, Next...>;
};

template <size_t N>
using make_index_sequence = typename index_sequence_helper<N>::type;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Default case - no negative zero
template <typename T>
struct has_negative_zero : CUTE_RT_TM_STL_NAMESPACE::false_type{};

// Float types that support negative zero
template <> struct has_negative_zero<mx_float4_t<float_e2m1_t>> : CUTE_RT_TM_STL_NAMESPACE::true_type{};
template <> struct has_negative_zero<mx_float6_t<float_e2m3_t>> : CUTE_RT_TM_STL_NAMESPACE::true_type{};
template <> struct has_negative_zero<mx_float8_t<float_e4m3_t>> : CUTE_RT_TM_STL_NAMESPACE::true_type{};
template <> struct has_negative_zero<mx_float8_t<float_e5m2_t>> : CUTE_RT_TM_STL_NAMESPACE::true_type{};
template <> struct has_negative_zero<float_e2m1_t> : CUTE_RT_TM_STL_NAMESPACE::true_type{};
template <> struct has_negative_zero<float_e2m3_t> : CUTE_RT_TM_STL_NAMESPACE::true_type{};
template <> struct has_negative_zero<float_e4m3_t> : CUTE_RT_TM_STL_NAMESPACE::true_type{};
template <> struct has_negative_zero<float_e5m2_t> : CUTE_RT_TM_STL_NAMESPACE::true_type{};
template <> struct has_negative_zero<half_t> : CUTE_RT_TM_STL_NAMESPACE::true_type{};
template <> struct has_negative_zero<bfloat16_t> : CUTE_RT_TM_STL_NAMESPACE::true_type{};
template <> struct has_negative_zero<float> : CUTE_RT_TM_STL_NAMESPACE::true_type{};
template <> struct has_negative_zero<double> : CUTE_RT_TM_STL_NAMESPACE::true_type{};
template <> struct has_negative_zero<tfloat32_t> : CUTE_RT_TM_STL_NAMESPACE::true_type{};

// Helper variable template 
template <typename T>
inline constexpr bool has_negative_zero_v = has_negative_zero<T>::value;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Get the register type used in kernel
//
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T>
struct get_unpacked_element_type {
  using type = T;
};

} // namespace detail

}  // namespace nihilus_gemm

/////////////////////////////////////////////////////////////////////////////////////////////////



