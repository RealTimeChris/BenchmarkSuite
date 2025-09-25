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

#include "nihilus_gemm/nihilus_gemm.h"
#include "nihilus_gemm/layout/matrix.h"
#include "nihilus_gemm/platform/platform.h"
#include "nihilus_gemm/gemm/gemm.h"
#include "nihilus_gemm/conv/conv2d_problem_size.h"
#include "nihilus_gemm/conv/conv3d_problem_size.h"
#include "nihilus_gemm/gemm/threadblock/index_remat.h"
#include "nihilus_gemm/gemm/threadblock/threadblock_swizzle_streamk.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace nihilus_gemm {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for GEMMs
template<uint64_t M, uint64_t K, int N = 1>
struct GemmIdentityThreadblockSwizzle {

  NIHILUS_HOST_DEVICE
	constexpr GemmIdentityThreadblockSwizzle() {}

  /// Returns the shape of the problem in units of logical tiles
  /// *Gemm* problem size: gemm(M, N, K)
  NIHILUS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices) {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      split_k_slices);
  }

  template<int32_t split_k_slices, uint64_t M_new, uint64_t K_new, uint64_t M_newer, uint64_t K_newer> NIHILUS_HOST_DEVICE constexpr static decltype(auto) get_tiled_shape(constexpresh_gemm_coord<M_new, K_new> problem_size, constexpresh_gemm_coord<M_newer, K_newer> tile_size) {
	  return constexpresh_gemm_coord<(constexpresh_gemm_coord<M_new, K_new>::M + constexpresh_gemm_coord<M_newer, K_newer>::M - 1) / constexpresh_gemm_coord<M_newer, K_newer>::M,
		  split_k_slices>{};
  }

  /// Returns the shape of the problem in units of logical tiles
  /// *ImplicitGemm* Conv2d problem size: conv_operator(NPQK, NHWC, KRSC)
  NIHILUS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    nihilus_gemm::conv::Operator conv_operator,
    nihilus_gemm::conv::Conv2dProblemSize const &problem_size,
    GemmCoord tile_size,
    int split_k_slices) {

    gemm::GemmCoord implicit_gemm_problem_size = 
    nihilus_gemm::conv::implicit_gemm_problem_size(conv_operator, problem_size);

    return get_tiled_shape(
      implicit_gemm_problem_size, tile_size, split_k_slices);
  }

  /// Returns the shape of the problem in units of logical tiles
  /// *ImplicitGemm* Conv3d problem size: conv_operator(NZPQK, NDHWC, KTRSC)
  NIHILUS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    nihilus_gemm::conv::Operator conv_operator,
    nihilus_gemm::conv::Conv3dProblemSize const &problem_size,
    GemmCoord tile_size,
    int split_k_slices) {

    gemm::GemmCoord implicit_gemm_problem_size = 
    nihilus_gemm::conv::implicit_gemm_problem_size(conv_operator, problem_size);

    return get_tiled_shape(
      implicit_gemm_problem_size, tile_size, split_k_slices);
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  NIHILUS_HOST_DEVICE
  static dim3 get_grid_shape(GemmCoord tiled_shape) {
    int tile = 1 << get_log_tile(tiled_shape);
    return dim3(tiled_shape.m() * tile, (tiled_shape.n() + tile - 1) / tile, tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  NIHILUS_HOST_DEVICE
  constexpr static int get_log_tile(const GemmCoord& tiled_shape) {
	  if constexpr (N < 2) {
		  return 0;
	  } else {
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
  }

  template<uint64_t M_new, uint64_t K_new>NIHILUS_HOST_DEVICE static constexpr int get_log_tile(constexpresh_gemm_coord<M_new, K_new> tiled_shape) {
	  if constexpr (N < 2) {
		  return 0;
	  } else {
		  auto n = constexpresh_gemm_coord<M_new, K_new>::N.n();
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
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  NIHILUS_DEVICE
  static GemmCoord get_tile_offset(int log_tile) {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    int block_idx_z = RematerializeBlockIdxZ();

    return GemmCoord{(block_idx_x >> log_tile),  //
                     (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }

  template<int32_t log_tile> NIHILUS_DEVICE static GemmCoord get_tile_offset() {
	  int block_idx_x = RematerializeBlockIdxX();
	  int block_idx_y = RematerializeBlockIdxY();
	  int block_idx_z = RematerializeBlockIdxZ();

	  return GemmCoord{ (block_idx_x >> log_tile),//
		  (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)), block_idx_z };
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  NIHILUS_DEVICE
  static GemmCoord get_tile_offset(GemmCoord tiled_shape) {

    int const kTile = N;
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();

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

/// Threadblock swizzling function for GEMMs
struct GemmHorizontalThreadblockSwizzle {

  NIHILUS_HOST_DEVICE
  GemmHorizontalThreadblockSwizzle() { }

  /// Returns the shape of the problem in units of logical tiles
  NIHILUS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices) {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      split_k_slices);
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  NIHILUS_HOST_DEVICE
  static dim3 get_grid_shape(GemmCoord tiled_shape) {
    return dim3(tiled_shape.n(), tiled_shape.m(), tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  NIHILUS_HOST_DEVICE
  static int get_log_tile(GemmCoord tiled_shape) {
    return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  NIHILUS_DEVICE
  static GemmCoord get_tile_offset(GemmCoord tiled_shape) {
    return GemmCoord{
      RematerializeBlockIdxY(),
      RematerializeBlockIdxX(),
      RematerializeBlockIdxZ()
    };
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for batched GEMMs
struct GemmBatchedIdentityThreadblockSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  NIHILUS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int batch_count) {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      batch_count % (1 << 16));
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  NIHILUS_HOST_DEVICE
  static dim3 get_grid_shape(GemmCoord tiled_shape) {
    return dim3(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  NIHILUS_HOST_DEVICE
  static int get_log_tile(GemmCoord tiled_shape) {
    return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  NIHILUS_DEVICE
  static GemmCoord get_tile_offset(GemmCoord tiled_shape) {
    return GemmCoord{
      RematerializeBlockIdxX(),
      RematerializeBlockIdxY(),
      RematerializeBlockIdxZ()
    };
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  NIHILUS_DEVICE
  static GemmCoord get_tile_offset(int log_tile) {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    int block_idx_z = RematerializeBlockIdxZ();

    return GemmCoord{(block_idx_x >> log_tile),  //
                     (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }

  /// Gets the batch index
  NIHILUS_DEVICE
  static int get_batch_idx() {
    return RematerializeBlockIdxZ();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for split-K GEMMs
template <int N = 1>
struct GemmSplitKIdentityThreadblockSwizzle {

  int const kTile = N;

  /// Returns the shape of the problem in units of logical tiles
  NIHILUS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int partitions) {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      partitions);
  }

  /// Calculates optimal swizzle width
  NIHILUS_HOST_DEVICE
  static int get_log_tile(GemmCoord tiled_shape) {
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

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  NIHILUS_HOST_DEVICE
  static dim3 get_grid_shape(GemmCoord tiled_shape) {
    int tile = 1 << get_log_tile(tiled_shape);
    return dim3(tiled_shape.m() * tile, (tiled_shape.n() + tile - 1) / tile, tiled_shape.k());
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  NIHILUS_DEVICE
  static GemmCoord get_tile_offset(int log_tile) {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    int block_idx_z = RematerializeBlockIdxZ();

    return GemmCoord{(block_idx_x >> log_tile),  //
                     (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  NIHILUS_DEVICE
  static GemmCoord get_tile_offset(GemmCoord tiled_shape) {

    int const kTile = N;
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();

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

/// Threadblock swizzling function for split-K GEMMs
struct GemmSplitKHorizontalThreadblockSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  NIHILUS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int partitions) {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      partitions);
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  NIHILUS_HOST_DEVICE
  static dim3 get_grid_shape(GemmCoord tiled_shape) {
    return dim3(tiled_shape.n(), tiled_shape.m(), tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  NIHILUS_HOST_DEVICE
  static int get_log_tile(GemmCoord tiled_shape) {
    return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  NIHILUS_DEVICE
  static GemmCoord get_tile_offset(int log_tile) {
    return GemmCoord{
      RematerializeBlockIdxY(),
      RematerializeBlockIdxX(),
      RematerializeBlockIdxZ()
    };
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  NIHILUS_DEVICE
  static GemmCoord get_tile_offset(GemmCoord tiled_shape) {
    return GemmCoord{
      RematerializeBlockIdxY(),
      RematerializeBlockIdxX(),
      RematerializeBlockIdxZ()
    };
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for batched GEMVs
struct GemvBatchedStridedThreadblockDefaultSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  NIHILUS_HOST_DEVICE
  static BatchedGemmCoord get_tiled_shape(
    BatchedGemmCoord problem_size,
    BatchedGemmCoord tile_size) {

    return BatchedGemmCoord(
      1, // M is always 1
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      (problem_size.k() + tile_size.k() - 1) / tile_size.k(),
      (problem_size.batch() + tile_size.batch() - 1) / tile_size.batch());
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  NIHILUS_HOST_DEVICE
  static dim3 get_grid_shape(BatchedGemmCoord tiled_shape) {
    return dim3(tiled_shape.n(), tiled_shape.batch(), tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  NIHILUS_HOST_DEVICE
  static int get_log_tile(GemmCoord tiled_shape) {
    return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  NIHILUS_DEVICE
  static BatchedGemmCoord get_tile_offset(int log_tile) {
    return BatchedGemmCoord{
      0, // M is always 1
      RematerializeBlockIdxX(),
      RematerializeBlockIdxZ(),
      RematerializeBlockIdxY(),
    };
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  NIHILUS_DEVICE
  static BatchedGemmCoord get_tile_offset() {
    return BatchedGemmCoord{
      0, // M is always 1
      RematerializeBlockIdxX(),
      RematerializeBlockIdxZ(),
      RematerializeBlockIdxY(),
    };
  }

  /// Gets the batch tile index
  NIHILUS_DEVICE
  static int get_batch_tile_idx() {
    return RematerializeBlockIdxY();
  }

  /// Gets the absolute batch index
  NIHILUS_DEVICE
  static int get_batch_idx() {
    return RematerializeBlockDimY()*RematerializeBlockIdxY() + RematerializeThreadIdxY();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace nihilus_gemm

