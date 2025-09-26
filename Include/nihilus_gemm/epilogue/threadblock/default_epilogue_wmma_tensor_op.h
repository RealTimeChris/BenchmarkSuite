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
  \brief Epilogue for threadblock scoped GEMMs using WMMA.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include "nihilus_gemm/nihilus_gemm.h"

#include "nihilus_gemm/array.h"

#include "nihilus_gemm/gemm/gemm.h"

#include "nihilus_gemm/epilogue/thread/linear_combination.h"
#include "nihilus_gemm/epilogue/thread/linear_combination_clamp.h"
#include "nihilus_gemm/epilogue/thread/linear_combination_relu.h"
#include "nihilus_gemm/epilogue/thread/linear_combination_gelu.h"
#include "nihilus_gemm/epilogue/thread/linear_combination_sigmoid.h"
#include "nihilus_gemm/epilogue/thread/linear_combination_planar_complex.h"

#include "nihilus_gemm/epilogue/thread/conversion_op.h"
#include "nihilus_gemm/epilogue/thread/reduction_op.h"

#include "nihilus_gemm/transform/threadblock/regular_tile_iterator_pitch_linear.h"

#include "nihilus_gemm/epilogue/warp/fragment_iterator_wmma_tensor_op.h"
#include "nihilus_gemm/epilogue/warp/tile_iterator_wmma_tensor_op.h"
#include "nihilus_gemm/epilogue/threadblock/default_thread_map_wmma_tensor_op.h"
#include "nihilus_gemm/epilogue/threadblock/predicated_tile_iterator.h"
#include "nihilus_gemm/epilogue/threadblock/shared_load_iterator.h"

#include "nihilus_gemm/epilogue/threadblock/epilogue.h"

#include "nihilus_gemm/layout/permute.h"

////////////////////////////////////////////////////////////////////////////////

namespace nihilus_gemm {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for WMMA TensorOps.
template <
  typename Shape_,
  typename WarpMmaTensorOp_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute
>
struct DefaultEpilogueWmmaTensorOp {

  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static constexpr int kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static constexpr int kElementsPerAccess = ElementsPerAccess;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaTensorOp::LayoutC;
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;

  //
  // Thread map
  //

  using OutputTileThreadMap = typename nihilus_gemm::epilogue::threadblock::DefaultThreadMapWmmaTensorOp<
    Shape,
    typename WarpMmaTensorOp::Shape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  using OutputTileIterator = nihilus_gemm::epilogue::threadblock::PredicatedTileIterator<
    OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout
  >;

  using AccumulatorFragmentIterator = nihilus_gemm::epilogue::warp::FragmentIteratorWmmaTensorOp<
    typename WarpMmaTensorOp::Shape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    typename WarpMmaTensorOp::Policy::Operator::ElementC,
    typename WarpMmaTensorOp::Policy::Operator::FragmentC,
    LayoutC
  >;

  using WarpTileIterator = nihilus_gemm::epilogue::warp::TileIteratorWmmaTensorOp<
    typename WarpMmaTensorOp::Shape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    typename WarpMmaTensorOp::Policy::Operator::FragmentC,
    LayoutC
  >;

  using SharedLoadIterator = nihilus_gemm::epilogue::threadblock::SharedLoadIterator<
    typename OutputTileThreadMap::CompactedThreadMap,
    ElementAccumulator
  >;

  /// Hard-coded padding elements added 
  using Padding = typename WarpTileIterator::Padding;

  //
  // Define the epilogue
  //
  using Epilogue = nihilus_gemm::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};


////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace nihilus_gemm

////////////////////////////////////////////////////////////////////////////////
