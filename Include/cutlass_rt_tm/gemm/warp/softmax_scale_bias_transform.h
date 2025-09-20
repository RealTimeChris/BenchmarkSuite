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
    \brief Templates implementing warp-level per-channel softmax before
   matrix multiply-accumulate operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass_rt_tm/cutlass.h"
#include "cutlass_rt_tm/array.h"
#include "cutlass_rt_tm/platform/platform.h"

#include "cutlass_rt_tm/numeric_conversion.h"
#include "cutlass_rt_tm/numeric_types.h"
#include "cutlass_rt_tm/matrix_shape.h"

#include "cutlass_rt_tm/arch/memory_sm75.h"
#include "cutlass_rt_tm/arch/mma_sm75.h"
#include "cutlass_rt_tm/arch/mma_sm80.h"

#include "cutlass_rt_tm/gemm/gemm.h"
#include "cutlass_rt_tm/gemm/warp/mma.h"

#include "cutlass_rt_tm/gemm/warp/mma_tensor_op_policy.h"

#include "cutlass_rt_tm/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass_rt_tm/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_rt_tm {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FragmentActivations, typename FragmentNormSum>
struct SoftmaxScaleBiasTransform {

  using T = typename FragmentActivations::Element;

  static int constexpr  NumActivations = FragmentActivations::kElements;
  static int constexpr  NumNormSum = FragmentNormSum::kElements;
  static int constexpr  MmaElements = 2;
  // One element has one scale and one bias
  static int constexpr  MmaScaleBiasPair = 2;
  // 16816 has 2 columns and 2 rows
  static int constexpr  MmaCols = 2;
  static int constexpr  MmaRows = 2;

  using MmaOperand = Array<T, MmaElements>;
  using NormSumOperand = Array<__half2, MmaScaleBiasPair>;

  CUTLASS_RT_TM_DEVICE
  void transform(MmaOperand &activations,
                 NormSumOperand const &norm_sum) {

    __half2* packed_activations = reinterpret_cast<__half2*>(&activations);

    CUTLASS_RT_TM_PRAGMA_UNROLL
    for (int i = 0; i < MmaElements / 2; ++i) {
      __half2 out = ::h2exp(__hsub2(packed_activations[i], norm_sum[2*i]));
      packed_activations[i] = __hmul2(out, norm_sum[2*i + 1]);
    }
  }

  CUTLASS_RT_TM_DEVICE
  void operator()(FragmentActivations &activations,
                  FragmentNormSum const &norm_sum) {
    MmaOperand *ptr_activations = reinterpret_cast<MmaOperand *>(&activations);
    NormSumOperand const *ptr_norm_sum =
        reinterpret_cast<NormSumOperand const *>(&norm_sum);

    CUTLASS_RT_TM_PRAGMA_UNROLL
    for (int i = 0; i < (NumActivations / MmaElements); ++i) {
      transform(ptr_activations[i],
                ptr_norm_sum[i / (MmaCols * MmaRows) * MmaRows + i % MmaRows]);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass_rt_tm

/////////////////////////////////////////////////////////////////////////////////////////////////
