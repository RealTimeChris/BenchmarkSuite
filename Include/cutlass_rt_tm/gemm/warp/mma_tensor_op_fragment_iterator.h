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
    \brief This defines a "fragment" iterator for visiting the fragments of a warp tile
      that participate in one warp-level mma operation.

      Typically, this is used to access the accumulator tile/fragment of a warp-level mma operation.
      The accumulator tile is then partitioned into smaller tiles/fragments that can be fed into 
      next warp-level mma operation. 

      This iterator is necessary to accomplish warp-level mma fusion where the accumulator tile is 
      reused as multiplicand tile for the next mma.

*/

#pragma once

#include "cutlass_rt_tm/cutlass.h"

#include "cutlass_rt_tm/array.h"
#include "cutlass_rt_tm/matrix_shape.h"
#include "cutlass_rt_tm/layout/matrix.h"
#include "cutlass_rt_tm/layout/tensor.h"
#include "cutlass_rt_tm/numeric_conversion.h"

namespace cutlass_rt_tm {
namespace gemm {
namespace warp {


////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Size of the accumulation tile shape (concept: MatrixShape)
    typename AccumulatorShape_,
    /// KBlocks columns to compute residual
    int KBlocksColumn_,
    /// Accumulator Element type
    typename ElementAccumulator_,    
    /// Element type
    typename Element_,
    /// Layout of operand in memory
    typename Layout_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Output operation on the fragment
    typename OutputOp_>
class MmaTensorOpFragmentIterator;


// Partial specialization for col-major accumulator tile

template <
    /// Shape of warp tile to load (concept: MatrixShape)
    typename Shape_,
    /// Shape of the warp accumulation tile (concept: MatrixShape)
    typename AccumulatorShape_,
    /// KBlocks columns to compute residual
    int KBlocksColumn_,    
    /// Accumulator Element type
    typename ElementAccumulator_,
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Output operation on fragment
    typename OutputOp_>
class MmaTensorOpFragmentIterator<Shape_, AccumulatorShape_, KBlocksColumn_, ElementAccumulator_, Element_,
                                         cutlass_rt_tm::layout::ColumnMajor,
                                         InstructionShape_, OutputOp_> {
 public:

  /// Shape of warp tile to load (concept: MatrixShape)
  using Shape = Shape_;
    
  /// Shape of the warp accumulation tile (concept: MatrixShape)
  using AccumulatorShape = AccumulatorShape_;

  /// KBlocks columns to compute residual
  static constexpr int kKBlockColumn = KBlocksColumn_;

  /// Accumulator Element type
  using ElementAccumulator = ElementAccumulator_;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass_rt_tm::layout::ColumnMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Output operation on fragment
  using OutputOp = OutputOp_;

  /// Number of participating threads
  static constexpr int kThreads = 32;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN),
        "Shape of warp-level Mma must be divisible by operator shape.");
    static_assert(
        AccumulatorShape::kRow == Shape::kRow, 
        "Rows of Warp Accumulator must be the same as rows of warp");
    static_assert(
        !(AccumulatorShape::kColumn % Shape::kColumn),
        "Shape of Warp Accumulator must be divisible by warp shape.");
    static_assert(
        !(kKBlockColumn % Shape::kColumn),
        "KBlock size must be divisible by warp shape.");

    /// Number of times this iterator can be incremented
    static constexpr int kIterations = AccumulatorShape::kCount / Shape::kCount;
  };

private:

  static constexpr int kElementsPerAccess = InstructionShape::kM * InstructionShape::kN / kThreads;

  /// Number of mma operations performed by a warp
  using MmaIterations = MatrixShape<Shape::kRow / InstructionShape::kM,
                                    Shape::kColumn / InstructionShape::kN>;
  /// Number of mma operations performed by the entire accumulator
  using AccumulatorIterations = MatrixShape<AccumulatorShape::kRow / InstructionShape::kM,
                                              AccumulatorShape::kColumn / InstructionShape::kN>;

  /// Number of K iterations    
  static constexpr int kKBlockIterations = (AccumulatorShape::kColumn + kKBlockColumn - 1) / kKBlockColumn;
  static constexpr int kResidualColumn = AccumulatorShape::kColumn - (kKBlockIterations - 1) * kKBlockColumn;
  static constexpr int kKBlockColumnIterations = kKBlockColumn / Shape::kColumn 
                                     * (AccumulatorShape::kRow / Shape::kRow);
  static constexpr int kResidualIndex = kResidualColumn / Shape::kColumn
                                     * (AccumulatorShape::kRow / Shape::kRow);

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<Element, Shape::kCount / kThreads>;

  /// Accumulator Fragment object
  using AccumulatorFragment = Array<ElementAccumulator, AccumulatorShape::kCount / kThreads>;

  /// Scale Bias Element Type
  using ElementScaleBias = typename OutputOp::ElementCompute;

  /// Scale Bias Fragment object
  using ScaleBiasFragment = Array<ElementScaleBias, InstructionShape::kM * InstructionShape::kK / kThreads>;


private:

  /// Internal access type
  using AccessType = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentAccessType = Array<Element, kElementsPerAccess>;

  using ScaleBiasAccessType = Array<ElementScaleBias, kElementsPerAccess>;

private:
  //
  // Data members
  //

  /// Accumulator tile
  AccessType const *accumulators_;

  /// Internal index
  int index_;

  /// Used to access residual tile first
  bool is_residual_tile_;

public:
  /// Constructs an iterator
  CUTLASS_RT_TMHOST_DEVICE
  MmaTensorOpFragmentIterator(AccumulatorFragment const &accum)
      : accumulators_(reinterpret_cast<AccessType const *>(&accum)),
        index_(0), is_residual_tile_(true) {}

  /// Add offset
  CUTLASS_RT_TMHOST_DEVICE
  void add_offset(int index_offset) {
    index_ += index_offset; 
    if(is_residual_tile_ && index_ >= kKBlockColumnIterations) {
      index_ = index_ - kKBlockColumnIterations + kResidualIndex;
      is_residual_tile_ = false;
    }
  }

  /// Increments
  CUTLASS_RT_TMHOST_DEVICE
  MmaTensorOpFragmentIterator &operator++() {
    add_offset(1);
    return *this;
  }

  /// Decrements
  CUTLASS_RT_TMHOST_DEVICE
  MmaTensorOpFragmentIterator &operator--() {
    add_offset(-1);
    return *this;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_RT_TMHOST_DEVICE
  void load(Fragment &frag, OutputOp output_op) const {

    if (output_op.is_source_needed()) //beta must be zero
      assert(0);

    FragmentAccessType *frag_ptr = reinterpret_cast<FragmentAccessType *>(&frag);

    int index = index_ * MmaIterations::kCount;

    CUTLASS_RT_TMPRAGMA_UNROLL
    for (int n = 0; n < MmaIterations::kColumn; n++) {
      for (int m = 0; m < MmaIterations::kRow; m++) {
        int accumulator_access_offset = 
            n * AccumulatorIterations::kRow + m + index;
            
        frag_ptr[m * MmaIterations::kColumn + n].clear();
        if(!(is_residual_tile_ && index_ >= kResidualIndex))
            frag_ptr[m * MmaIterations::kColumn + n] = output_op(accumulators_[accumulator_access_offset]);
      }
    }
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  /// Then apply per-channel scale and bias
  CUTLASS_RT_TMHOST_DEVICE
  void load(Fragment &frag, ScaleBiasFragment &scale, 
        ScaleBiasFragment &bias, OutputOp output_op) const {

    if (output_op.is_source_needed()) //beta must be zero
      assert(0);

    FragmentAccessType *frag_ptr = reinterpret_cast<FragmentAccessType *>(&frag);
    ScaleBiasAccessType * scale_ptr = reinterpret_cast<ScaleBiasAccessType *>(&scale);
    ScaleBiasAccessType * bias_ptr = reinterpret_cast<ScaleBiasAccessType *>(&bias);

    int index = index_ * MmaIterations::kCount;

    CUTLASS_RT_TMPRAGMA_UNROLL
    for (int n = 0; n < MmaIterations::kColumn; n++) {
      for (int m = 0; m < MmaIterations::kRow; m++) {
        int accumulator_access_offset = 
            n * AccumulatorIterations::kRow + m + index;
            
        frag_ptr[m * MmaIterations::kColumn + n].clear();
        if(!(is_residual_tile_ && index_ >= kResidualIndex))
            frag_ptr[m * MmaIterations::kColumn + n] = 
                output_op(accumulators_[accumulator_access_offset], 
                    scale_ptr[n] /*scale*/, bias_ptr[n] /*bias*/);
      }
    }
  }



};

// Partial specialization for row-major accumulator tile

template <
    /// Shape of warp tile to load (concept: MatrixShape)
    typename Shape_,
    /// Shape of the warp accumulation tile (concept: MatrixShape)
    typename AccumulatorShape_,
    /// KBlocks columns to compute residual
    int KBlocksColumn_,    
    /// Accumulator Element type
    typename ElementAccumulator_,    
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Output operation on fragment
    typename OutputOp_>
class MmaTensorOpFragmentIterator<Shape_, AccumulatorShape_, KBlocksColumn_, ElementAccumulator_, Element_,
                                         cutlass_rt_tm::layout::RowMajor,
                                         InstructionShape_, OutputOp_> {
 public:

  /// Shape of warp tile to load (concept: MatrixShape)
  using Shape = Shape_;
    
  /// Shape of the warp accumulation tile (concept: MatrixShape)
  using AccumulatorShape = AccumulatorShape_;

  /// KBlocks columns to compute residual
  static constexpr int kKBlockColumn = KBlocksColumn_;

  /// Accumulator Element type
  using ElementAccumulator = ElementAccumulator_;

  /// Element type
  using Element = Element_;
  
  /// Layout of source tile
  using Layout = cutlass_rt_tm::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Output operation on fragment
  using OutputOp = OutputOp_;

  /// Number of participating threads
  static constexpr int kThreads = 32;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN),
        "Shape of warp-level Mma must be divisible by operator shape.");
    static_assert(
        AccumulatorShape::kRow == Shape::kRow, 
        "Rows of Warp Accumulator must be the same as rows of warp");
    static_assert(
        !(AccumulatorShape::kColumn % Shape::kColumn),
        "Shape of Warp Accumulator must be divisible by warp shape.");
    static_assert(
        !(kKBlockColumn % Shape::kColumn),
        "KBlock size must be divisible by warp shape.");

    /// Number of times this iterator can be incremented
    static constexpr int kIterations = AccumulatorShape::kCount / Shape::kCount;
  };

private:

  static constexpr int kRowsPerIteration = 8;
  static constexpr int kColumnsPerIteration = 16;
  static constexpr int kElementsPerIteration = kRowsPerIteration * InstructionShape::kN / kThreads;
  static constexpr int kElementsPerAccess = kRowsPerIteration * kColumnsPerIteration / kThreads;
  static constexpr int kIterationsPerAccess = kElementsPerAccess / kElementsPerIteration;
  
  // Number of iterations per actual instruction
  static constexpr int kIterationsPerInstruction = InstructionShape::kM / kRowsPerIteration;

  static constexpr int kAccessStride = kIterationsPerInstruction;

  /// Number of mma operations performed by a warp
  using MmaIterations = MatrixShape<Shape::kRow / InstructionShape::kM,
                                    Shape::kColumn / InstructionShape::kN>;
  /// Number of mma operations performed by the entire accumulator
  using AccumulatorIterations = MatrixShape<AccumulatorShape::kRow / InstructionShape::kM,
                                              AccumulatorShape::kColumn / InstructionShape::kN>;

  /// Number of Accesses in a warp
  using AccessIterations = MatrixShape<MmaIterations::kRow * kIterationsPerInstruction, 
                                        MmaIterations::kColumn / kIterationsPerAccess>;

  /// Number of K iterations    
  static constexpr int kKBlockIterations = (AccumulatorShape::kColumn + kKBlockColumn - 1) / kKBlockColumn;
  static constexpr int kResidualColumn = AccumulatorShape::kColumn - (kKBlockIterations - 1) * kKBlockColumn;
  static constexpr int kKBlockColumnIterations = kKBlockColumn / Shape::kColumn;
  static constexpr int kResidualIndex = kResidualColumn / Shape::kColumn;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<Element, Shape::kCount / kThreads>;

  /// Accumulator Fragment object
  using AccumulatorFragment = Array<ElementAccumulator, AccumulatorShape::kCount / kThreads>;

  /// Scale Bias Element Type
  using ElementScaleBias = typename OutputOp::ElementCompute;

  /// Scale Bias Fragment object
  using ScaleBiasFragment = Array<ElementScaleBias, InstructionShape::kM * InstructionShape::kK / kThreads>;


private:

  /// Internal access type
  using AccessType = Array<ElementAccumulator, kElementsPerIteration>;
  using FragmentAccessType = Array<Element, kElementsPerIteration>;
  using ScaleBiasAccessType = Array<ElementScaleBias, kElementsPerIteration>;

private:
  //
  // Data members
  //

  /// Accumulator tile
  AccessType const *accumulators_;

  /// Internal index
  int index_;

  /// Used to access residual tile first
  bool is_residual_tile_;

public:
  /// Constructs an iterator
  CUTLASS_RT_TMHOST_DEVICE
  MmaTensorOpFragmentIterator(AccumulatorFragment const &accum)
      : accumulators_(reinterpret_cast<AccessType const *>(&accum)),
        index_(0), is_residual_tile_(true) {}

  /// Add offset
  CUTLASS_RT_TMHOST_DEVICE
  void add_offset(int index_offset) {
    index_ += index_offset; 
    if(is_residual_tile_ && index_ >= kKBlockColumnIterations) {
      index_ = index_ - kKBlockColumnIterations + kResidualIndex;
      is_residual_tile_ = false;
    }
  }

  /// Increments
  CUTLASS_RT_TMHOST_DEVICE
  MmaTensorOpFragmentIterator &operator++() {
    add_offset(1);
    return *this;
  }

  /// Decrements
  CUTLASS_RT_TMHOST_DEVICE
  MmaTensorOpFragmentIterator &operator--() {
    add_offset(-1);
    return *this;
  }

  CUTLASS_RT_TMHOST_DEVICE
  void set_index(int idx) {
    index_ = idx;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_RT_TMHOST_DEVICE
  void load(Fragment &frag, OutputOp output_op) const {

    if (output_op.is_source_needed()) //beta must be zero
      assert(0);

    FragmentAccessType *frag_ptr = reinterpret_cast<FragmentAccessType *>(&frag);

    int index = index_ * AccessIterations::kCount;

    CUTLASS_RT_TMPRAGMA_UNROLL
    for (int i = 0; i < AccessIterations::kCount; i++) {

      int accumulator_access_offset = index / AccessIterations::kCount * (MmaIterations::kColumn * kIterationsPerInstruction) +
                                    (index % AccessIterations::kCount) / (AccessIterations::kColumn * kIterationsPerInstruction) *
                                    AccumulatorIterations::kColumn * kIterationsPerInstruction +
                                    (index % (AccessIterations::kColumn * kIterationsPerInstruction)) / kIterationsPerInstruction *
                                    (kIterationsPerInstruction * kIterationsPerAccess) +
                                    (index % kIterationsPerInstruction);
      CUTLASS_RT_TMPRAGMA_UNROLL
      for (int j = 0; j < kIterationsPerAccess; j++) {
  
        frag_ptr[i*kIterationsPerAccess + j].clear();
        if(!(is_residual_tile_ && index_ >= kResidualIndex))
              frag_ptr[i*kIterationsPerAccess + j] = output_op(accumulators_[accumulator_access_offset + j * kAccessStride]);
      }
      index++;
    }
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  /// Then apply per-channel scale and bias
  CUTLASS_RT_TMHOST_DEVICE
  void load(Fragment &frag, ScaleBiasFragment &scale, 
        ScaleBiasFragment & bias, OutputOp output_op) const {

    if (output_op.is_source_needed()) //beta must be zero
      assert(0);

    FragmentAccessType *frag_ptr = reinterpret_cast<FragmentAccessType *>(&frag);
    ScaleBiasAccessType * scale_ptr = reinterpret_cast<ScaleBiasAccessType *>(&scale);
    ScaleBiasAccessType * bias_ptr = reinterpret_cast<ScaleBiasAccessType *>(&bias);

    int index = index_ * AccessIterations::kCount;

    CUTLASS_RT_TMPRAGMA_UNROLL
    for (int i = 0; i < AccessIterations::kCount; i++) {

      int accumulator_access_offset = index / AccessIterations::kCount * (MmaIterations::kColumn * kIterationsPerInstruction) +
                                    (index % AccessIterations::kCount) / (AccessIterations::kColumn * kIterationsPerInstruction) *
                                    AccumulatorIterations::kColumn * kIterationsPerInstruction +
                                    (index % (AccessIterations::kColumn * kIterationsPerInstruction)) / kIterationsPerInstruction *
                                    (kIterationsPerInstruction * kIterationsPerAccess) +
                                    (index % kIterationsPerInstruction);

      int scale_bias_offset = (index 
                    % (kIterationsPerInstruction * AccessIterations::kColumn))
                    * kIterationsPerAccess;

      CUTLASS_RT_TMPRAGMA_UNROLL
      for (int j = 0; j < kIterationsPerAccess; j++) {

  
        frag_ptr[i*kIterationsPerAccess + j].clear();
        if(!(is_residual_tile_ && index_ >= kResidualIndex))
              frag_ptr[i*kIterationsPerAccess + j] = output_op(
                    accumulators_[accumulator_access_offset + j * kAccessStride], 
                    scale_ptr[scale_bias_offset + j], bias_ptr[scale_bias_offset + j]);
      }
      index++;
    }
  }

};

////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass_rt_tm

////////////////////////////////////////////////////////////////////////////////
