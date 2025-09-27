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
    \brief Defines basic structures needed for implementing the warp-scoped phase of the epilogue.
          These quantities assume a 'column-major' arrangement of TensorOp instructions, of which
          a row-oriented slice is visible per iteration.
*/

#pragma once

#include "nihilus_gemm/matrix_shape.h"
#include "nihilus_gemm/layout/matrix.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

/// Policy details related to the epilogue
template <
  typename WarpShape,     ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape, ///< matrix multiply operation shape (concept: gemm:GemmShape)
  typename Layout         ///< target shared memory layout
>
struct TensorOpPolicy; 

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for row-major
template <
  typename WarpShape,           ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape        ///< matrix multiply operation shape (concept: gemm::GemmShape)
>
struct TensorOpPolicy<WarpShape, OperatorShape, layout::RowMajor> {

  /// Number of operations
  using OperatorCount = MatrixShape<
    (WarpShape::kM + OperatorShape::kM - 1) / OperatorShape::kM,
    (WarpShape::kN + OperatorShape::kN - 1) / OperatorShape::kN
  >;

  //
  // Hard-coded constants regarding Tensor Operations
  //

  static constexpr int kElementsPerAccess = 2;
  static constexpr int kRowsPerIteration = 8;
  static constexpr bool kDivisible = 
    !(WarpShape::kM % OperatorShape::kM) && !(WarpShape::kN % OperatorShape::kN);

  //
  // Derived quantities
  //

  // Number of 'externally visible' iterations per actual instruction
  static constexpr int kIterationsPerInstruction = OperatorShape::kM / kRowsPerIteration;

  // Number of externally visible iterations
  static constexpr int kIterations = OperatorCount::kRow * kIterationsPerInstruction;

  using TileIterations = MatrixShape<kIterations, 1>;

  static constexpr int kAccumulatorRowStride = kElementsPerAccess;
  static constexpr int kAccumulatorColumnStride = kElementsPerAccess * OperatorCount::kRow * kIterationsPerInstruction;

};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for row-major
template <
  typename WarpShape,           ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape        ///< matrix multiply operation shape (concept: gemm::GemmShape)
>
struct TensorOpPolicy<WarpShape, OperatorShape, layout::ColumnMajor> {

  /// Number of operations
  using OperatorCount = MatrixShape<
    (WarpShape::kM + OperatorShape::kM - 1) / OperatorShape::kM,
    (WarpShape::kN + OperatorShape::kN - 1) / OperatorShape::kN
  >;

  //
  // Hard-coded constants regarding Tensor Operations
  //

  static constexpr int kElementsPerAccess = 1;
  static constexpr int kColumnsPerIteration = 8;
  static constexpr bool kDivisible = 
    !(WarpShape::kM % OperatorShape::kM) && !(WarpShape::kN % OperatorShape::kN);

  //
  // Derived quantities
  //

  // Number of 'externally visible' iterations per actual instruction
  static constexpr int kIterationsPerInstruction = OperatorShape::kN / kColumnsPerIteration;

  // Number of externally visible iterations
  static constexpr int kIterations = OperatorCount::kColumn * kIterationsPerInstruction;

  using TileIterations = MatrixShape<kIterations, 1>;

  // Hard code for 16x8
  static constexpr int kAccumulatorRowStride = 2;
  static constexpr int kAccumulatorColumnStride = 4 * OperatorCount::kRow;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for column-major-interleaved
template <
    typename WarpShape,  ///< shape of warp-level GEMM (concept: MatrixShape)
    typename OperatorShape,   ///< matrix multiply operation (concept: arch::Mma)
    int InterleavedK     ///< number of interleaved k
    >
struct TensorOpPolicy<WarpShape, OperatorShape,
                      layout::ColumnMajorInterleaved<InterleavedK> > {
  /// Number of operations
  using OperatorCount = MatrixShape<WarpShape::kM / OperatorShape::kM,
                                    WarpShape::kN / OperatorShape::kN>;

  //
  // Hard-coded constants regarding Tensor Operations
  //

  static constexpr int kElementsPerAccess = 2;
  static constexpr int kRowsPerIteration = 8;

  //
  // Derived quantities
  //

  // Number of 'externally visible' iterations per actual instruction
  static constexpr int kIterationsPerInstruction =
      OperatorShape::kM / kRowsPerIteration;

  // Number of externally visible iterations
  static constexpr int kIterations = WarpShape::kN / InterleavedK *
                                 OperatorCount::kRow *
                                 kIterationsPerInstruction;

  static constexpr int kElementsPerIteration = InterleavedK / OperatorShape::kN * kElementsPerAccess;

  static constexpr int kAccessPerIteration = kElementsPerIteration / kElementsPerAccess;

  // Number of externally visible iterations
  //static constexpr int kTileIterations = OperatorCount::kRow * kIterationsPerInstruction;
  using TileIterations = MatrixShape<1, WarpShape::kN / InterleavedK>;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
