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
    \brief Implements several possible threadblock-swizzling functions mapping blockIdx to 
      GEMM problems.
*/

#pragma once

#include "cutlass_new/cutlass.h"
#include "cutlass_new/layout/matrix.h"
#include "cutlass_new/platform/platform.h"
#include "cutlass_new/gemm/gemm.h"
#include "cutlass_new/conv/conv2d_problem_size.h"
#include "cutlass_new/conv/conv3d_problem_size.h"
#include "cutlass_new/gemm/threadblock/index_remat.h"
#include "cutlass_new/gemm/threadblock/threadblock_swizzle_streamk.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for GEMMs
template <int32_t N = 1>
struct GemmIdentityThreadblockSwizzle {

  CUTLASS_HOST_DEVICE
  GemmIdentityThreadblockSwizzle() { }

  /// Returns the shape of the problem in units of logical tiles
  /// *Gemm* problem size: gemm(M, N, K)
  CUTLASS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int32_t split_k_slices) {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      split_k_slices);
  }

  /// Returns the shape of the problem in units of logical tiles
  /// *ImplicitGemm* Conv2d problem size: conv_operator(NPQK, NHWC, KRSC)
  CUTLASS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    cutlass::conv::Operator conv_operator,
    cutlass::conv::Conv2dProblemSize const &problem_size,
    GemmCoord tile_size,
    int32_t split_k_slices) {

    gemm::GemmCoord implicit_gemm_problem_size = 
    cutlass::conv::implicit_gemm_problem_size(conv_operator, problem_size);

    return get_tiled_shape(
      implicit_gemm_problem_size, tile_size, split_k_slices);
  }

  /// Returns the shape of the problem in units of logical tiles
  /// *ImplicitGemm* Conv3d problem size: conv_operator(NZPQK, NDHWC, KTRSC)
  CUTLASS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    cutlass::conv::Operator conv_operator,
    cutlass::conv::Conv3dProblemSize const &problem_size,
    GemmCoord tile_size,
    int32_t split_k_slices) {

    gemm::GemmCoord implicit_gemm_problem_size = 
    cutlass::conv::implicit_gemm_problem_size(conv_operator, problem_size);

    return get_tiled_shape(
      implicit_gemm_problem_size, tile_size, split_k_slices);
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  static dim3 get_grid_shape(GemmCoord tiled_shape) {
    int32_t tile = 1 << get_log_tile(tiled_shape);
    return dim3(tiled_shape.m() * tile, (tiled_shape.n() + tile - 1) / tile, tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  static int32_t get_log_tile(GemmCoord tiled_shape) {
    auto n = tiled_shape.n();
    // Thresholds picked so that it doesn't cause too many no-op CTAs
    if (N >= 8 && n >= 6)
      return 3;
    else if (N >= 4 && n >= 3)
      return 2;
    else if (N >= 2 && n >= 2)
      return 1;
    else
      return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static GemmCoord get_tile_offset(int32_t log_tile) {
    int32_t block_idx_x = RematerializeBlockIdxX();
    int32_t block_idx_y = RematerializeBlockIdxY();
    int32_t block_idx_z = RematerializeBlockIdxZ();

    return GemmCoord{(block_idx_x >> log_tile),  //
                     (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static GemmCoord get_tile_offset(GemmCoord tiled_shape) {

    int32_t const kTile = N;
    int32_t block_idx_x = RematerializeBlockIdxX();
    int32_t block_idx_y = RematerializeBlockIdxY();

    if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile))
      return GemmCoord{block_idx_x, block_idx_y, RematerializeBlockIdxZ()};

    return GemmCoord{
      (block_idx_x / kTile),
      (block_idx_y * kTile) + (block_idx_x % kTile),
      RematerializeBlockIdxZ()
    };
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

