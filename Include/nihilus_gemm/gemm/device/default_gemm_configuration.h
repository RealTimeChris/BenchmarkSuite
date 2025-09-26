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
    \brief Definitions for GEMM structures
*/

#pragma once

#include "nihilus_gemm/nihilus_gemm.h"

#include "nihilus_gemm/arch/arch.h"
#include "nihilus_gemm/arch/mma.h"
#include "nihilus_gemm/arch/wmma.h"

#include "nihilus_gemm/gemm/gemm.h"
#include "nihilus_gemm/epilogue/thread/linear_combination.h"
#include "nihilus_gemm/epilogue/thread/linear_combination_clamp.h"

////////////////////////////////////////////////////////////////////////////////

namespace nihilus_gemm {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

template <
  typename OperatorClass,
  typename ArchTag,
  typename ElementA, 
  typename ElementB, 
  typename ElementC,
  typename ElementAccumulator
>
struct DefaultGemmConfiguration;

////////////////////////////////////////////////////////////////////////////////

template <
  typename ArchTag,
  typename ElementA, 
  typename ElementB, 
  typename ElementC, 
  typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassSimt, 
  ArchTag,
  ElementA, 
  ElementB, 
  ElementC, 
  ElementAccumulator> {
  
  static constexpr int kAlignmentA = 1;
  static constexpr int kAlignmentB = 1;
  using ThreadblockShape = GemmShape<128, 128, 8>;
  using WarpShape = GemmShape<32, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static constexpr int kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    1,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ArchTag,
  typename ElementC>
struct DefaultGemmConfiguration<arch::OpClassSimt, ArchTag, int8_t, int8_t, ElementC, int32_t> {
  
  static constexpr int kAlignmentA = 4;
  static constexpr int kAlignmentB = 4;
  using ThreadblockShape = GemmShape<128, 128, 32>;
  using WarpShape = GemmShape<32, 64, 32>;
  using InstructionShape = GemmShape<1, 1, 4>;
  static constexpr int kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
    ElementC,
    1,
    int32_t,
    float
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////


} // namespace device
} // namespace gemm
} // namespace nihilus_gemm

////////////////////////////////////////////////////////////////////////////////
