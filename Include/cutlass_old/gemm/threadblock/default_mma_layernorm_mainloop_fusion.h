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
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass_old/cutlass.h"
#include "cutlass_old/numeric_types.h"
#include "cutlass_old/arch/arch.h"

#include "cutlass_old/layout/matrix.h"
#include "cutlass_old/gemm/threadblock/default_mma_core.h"
#include "cutlass_old/gemm/threadblock/mma_layernorm_mainloop_fusion_multistage.h"
#include "cutlass_old/transform/threadblock/predicated_scale_bias_vector_iterator.h"
#include "cutlass_old/transform/threadblock/predicated_scale_bias_vector_access_iterator.h"
#include "cutlass_old/transform/threadblock/regular_scale_bias_vector_access_iterator.h"
#include "cutlass_old/gemm/warp/scale_bias_tile_iterator.h"
#include "cutlass_old/transform/threadblock/predicated_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass_old {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for Scale/Bias vectors
    typename ElementScaleBias,
    /// Layout type for Scale/Bias vectors
    typename LayoutScaleBias,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Use zfill or predicate for SM80 out-of-bound cp.async 
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone
    >
struct DefaultMmaLayernormMainloopFusion {

  static cutlass_old::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? cutlass_old::arch::CacheOperation::Global
          : cutlass_old::arch::CacheOperation::Always;

  static cutlass_old::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? cutlass_old::arch::CacheOperation::Global
          : cutlass_old::arch::CacheOperation::Always;

  static cutlass_old::arch::CacheOperation::Kind const CacheOpGammaBeta = CacheOpA;

  // Define the MmaCore components
  using MmaCore = typename cutlass_old::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass_old::Array<ElementA, kAlignmentA>;
  using IteratorA =
      cutlass_old::transform::threadblock::PredicatedTileAccessIterator<
          cutlass_old::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass_old::Array<ElementB, kAlignmentB>;
  using IteratorB =
      cutlass_old::transform::threadblock::PredicatedTileAccessIterator<
          cutlass_old::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

  /// Define iterators over tiles from scale/bias vectors
  using IteratorVarMean =
      cutlass_old::transform::threadblock::PredicatedScaleBiasVectorIterator<
          cutlass_old::MatrixShape<1, WarpShape::kN>,
          ElementScaleBias,
          LayoutScaleBias>;

  /// Define iterators over tiles from scale/bias vectors
  using IteratorGammaBeta =
      cutlass_old::transform::threadblock::PredicatedScaleBiasVectorAccessIterator<
          cutlass_old::MatrixShape<1, ThreadblockShape::kK>, ElementScaleBias,
          LayoutScaleBias>;

  using SmemIteratorGammaBeta =
      cutlass_old::transform::threadblock::RegularScaleBiasVectorAccessIterator<
          cutlass_old::MatrixShape<1, ThreadblockShape::kK>, ElementScaleBias,
          LayoutScaleBias>;

  static int const kThreadCount = 32;

  // Warp-level iterators to load scale and bias vectors
  using WarpIteratorGammaBeta = cutlass_old::gemm::warp::ScaleBiasTileIterator<
      MatrixShape<WarpShape::kM, WarpShape::kK>, ElementScaleBias,
      LayoutScaleBias, MatrixShape<InstructionShape::kM, InstructionShape::kK>,
      typename MmaCore::MmaTensorOp::IteratorA::Base::Policy, kThreadCount,
      MmaCore::WarpCount::kK>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = cutlass_old::gemm::threadblock::MmaLayernormMainloopFusionMultistage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, IteratorVarMean, IteratorGammaBeta, SmemIteratorGammaBeta,
      CacheOpGammaBeta,
      ElementAccumulator, layout::RowMajor,
      typename MmaCore::MmaPolicy, WarpIteratorGammaBeta, Stages, SharedMemoryClear>;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass_old 

////////////////////////////////////////////////////////////////////////////////
