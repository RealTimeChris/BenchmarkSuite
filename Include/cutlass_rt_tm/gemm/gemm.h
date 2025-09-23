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
/*! \file
    \brief Defines common types used for all GEMM-like operators.
*/
#pragma once

#include "cutlass_rt_tm/cutlass.h"
#include "cutlass_rt_tm/coord.h"
#include "cutlass_rt_tm/gemm_coord.h"
#include "cutlass_rt_tm/layout/matrix.h"
#include "cutlass_rt_tm/gemm/gemm_enumerated_types.h"
#include "cute_rt_tm/layout.hpp"
#include "cutlass_rt_tm/detail/layout.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_rt_tm {
namespace gemm {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Scaling kind
enum class ScalingKind {
  kTensorwise,   // Accumulated GEMM result is scaled per tensor (default alpha scaling)
  kBlockwise     // Accumulated GEMM result is scaled per CTA tile (blockwise)
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using cutlass_rt_tm::detail::TagToStrideA;
using cutlass_rt_tm::detail::TagToStrideB;
using cutlass_rt_tm::detail::TagToStrideC;
using cutlass_rt_tm::detail::TagToStrideA_t;
using cutlass_rt_tm::detail::TagToStrideB_t;
using cutlass_rt_tm::detail::TagToStrideC_t;

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

using cutlass_rt_tm::detail::StrideToLayoutTagA;
using cutlass_rt_tm::detail::StrideToLayoutTagB;
using cutlass_rt_tm::detail::StrideToLayoutTagC;
using cutlass_rt_tm::detail::StrideToLayoutTagA_t;
using cutlass_rt_tm::detail::StrideToLayoutTagB_t;
using cutlass_rt_tm::detail::StrideToLayoutTagC_t;

template<int ModeIndex, class Stride>
constexpr bool
is_major(Stride = {}) {
  return ::cutlass_rt_tm::detail::is_major<ModeIndex>(Stride{});
}

template<class Stride>
constexpr bool
is_mn_major() {
  return is_major<0,Stride>();
}

template<class Stride>
constexpr
bool
is_k_major() {
  return is_major<1,Stride>();
}

template<class LayoutA>
constexpr bool
is_mn_major_A() {
  return is_mn_major<TagToStrideA_t<LayoutA>>();
}

template<class LayoutB>
constexpr bool
is_mn_major_B() {
  return is_mn_major<TagToStrideB_t<LayoutB>>();
}

template<class LayoutA>
constexpr bool
is_k_major_A() {
  return is_k_major<TagToStrideA_t<LayoutA>>();
}

template<class LayoutB>
constexpr bool
is_k_major_B() {
  return is_k_major<TagToStrideB_t<LayoutB>>();
}

///////////////////////////////////////////////////////////////////////////////

// The following two metafunctions are used to detect whether a `kernel::Gemm` or `kernel::GemmUniversal`
// is implementing the CUTLASS 3.x API or not, by checking if the problem shape type is aliased within or not.
template <class GemmKernel, class = void>
struct IsCutlass3GemmKernel : cute_rt_tm::false_type { };

template <typename GemmKernel>
struct IsCutlass3GemmKernel<GemmKernel, cute_rt_tm::void_t<typename GemmKernel::ProblemShape>>
    : cute_rt_tm::true_type { };

///////////////////////////////////////////////////////////////////////////////

} // namespace detail

///////////////////////////////////////////////////////////////////////////////

} // namespace gemm
} // namespace cutlass_rt_tm

////////////////////////////////////////////////////////////////////////////////////////////////////
