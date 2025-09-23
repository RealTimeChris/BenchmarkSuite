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
    \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass_rt_tm/cutlass.h"

#include "cutlass_rt_tm/array.h"
#include "cutlass_rt_tm/numeric_types.h"
#include "cutlass_rt_tm/tensor_ref.h"
#include "cutlass_rt_tm/matrix_shape.h"

#include "cutlass_rt_tm/arch/memory_sm75.h"
#include "cutlass_rt_tm/gemm/gemm.h"

#include "cutlass_rt_tm/layout/matrix.h"
#include "cutlass_rt_tm/layout/tensor.h"
#include "cutlass_rt_tm/layout/pitch_linear.h"
#include "cutlass_rt_tm/layout/tensor_op_multiplicand_sm80.h"
#include "cutlass_rt_tm/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h"

#include "cutlass_rt_tm/platform/platform.h"
#include "cutlass_rt_tm/fast_math.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_rt_tm {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Layout of operand in memory
    typename Layout_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpGaussianComplexAccumulatorTileIterator;

////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
/// 
/// Partial specialization for complex<T>
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of underlying field of reals.
    typename RealElement,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpGaussianComplexAccumulatorTileIterator<
    Shape_, complex<RealElement>, cutlass_rt_tm::layout::RowMajor, InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static constexpr Operand kOperand = Operand::kC;

  /// Element type
  using Element = complex<RealElement>;

  /// Layout of source tile
  using Layout = cutlass_rt_tm::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static constexpr int kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN),
        "Shape of warp-level Mma must be divisible by operator shape.");

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<Shape::kRow / InstructionShape::kM,
                                      Shape::kColumn / InstructionShape::kN>;
  };

private:

  // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire
  // shape, with each quad mapped to one row and each thread mapped to 1/4 of the elements
  // of that row. The accumulators within one row are assumed to be consecutive.
 static constexpr int kElementsPerAccess = InstructionShape::kN / 4;
 static constexpr int kRowsPerTile = 8;
 static constexpr int kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile. It is assumed that the accumulators
  /// are stored in a gaussian complex arrangement with parts 1, 2, and 3 as entirely contiguous
  /// arranged as [part1, part2, part3]
  using Fragment = Array<RealElement, (Shape::kCount / kThreads) * 3>;

  static constexpr int kPart1Index = (Shape::kCount / kThreads) * 0;
  static constexpr int kPart2Index = (Shape::kCount / kThreads) * 1;
  static constexpr int kPart3Index = (Shape::kCount / kThreads) * 2;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_RT_TMHOST_DEVICE
  MmaTensorOpGaussianComplexAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_RT_TMHOST_DEVICE
  MmaTensorOpGaussianComplexAccumulatorTileIterator(
    TensorRef const &ref, 
    int lane_id
  ):
    ref_(ref) {

    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);

    MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);

    ref_.add_coord_offset(lane_offset);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_RT_TMHOST_DEVICE
  MmaTensorOpGaussianComplexAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_RT_TMHOST_DEVICE
  MmaTensorOpGaussianComplexAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_RT_TMHOST_DEVICE
  MmaTensorOpGaussianComplexAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_RT_TMHOST_DEVICE
  MmaTensorOpGaussianComplexAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_RT_TMDEVICE
  MmaTensorOpGaussianComplexAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_RT_TMDEVICE
  MmaTensorOpGaussianComplexAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_RT_TMHOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_RT_TMDEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    CUTLASS_RT_TMPRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      CUTLASS_RT_TMPRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        
        int mma_accum_start = kAccumulatorRows * kElementsPerAccess * 
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        CUTLASS_RT_TMPRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          CUTLASS_RT_TMPRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            Element z = offset_ref.at({accum_m, accum_n});

            frag[mma_accum_start + row * kElementsPerAccess + col + kPart1Index] = z.real() + z.imag();
            frag[mma_accum_start + row * kElementsPerAccess + col + kPart2Index] = -z.real();
            frag[mma_accum_start + row * kElementsPerAccess + col + kPart3Index] = z.imag();
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_RT_TMDEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_RT_TMDEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_RT_TMDEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  CUTLASS_RT_TMHOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_RT_TMDEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    CUTLASS_RT_TMPRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      CUTLASS_RT_TMPRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        
        int mma_accum_start = kAccumulatorRows * kElementsPerAccess * 
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        CUTLASS_RT_TMPRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          CUTLASS_RT_TMPRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;
            int idx = mma_accum_start + row * kElementsPerAccess + col;

            Element z(frag[kPart1Index + idx] - frag[kPart3Index + idx], 
                      frag[kPart1Index + idx] + frag[kPart2Index + idx]);

            offset_ref.at({accum_m, accum_n}) = z;
          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_RT_TMDEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  CUTLASS_RT_TMDEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_RT_TMDEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass_rt_tm

/////////////////////////////////////////////////////////////////////////////////////////////////
