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
  \brief Metaprogram for determining the mapping of output elements to threads for epilogue tiles.

  
*/

#pragma once

#include "cutlass_rt_tm/cutlass.h"
#include "cutlass_rt_tm/numeric_types.h"
#include "cutlass_rt_tm/array.h"
#include "cutlass_rt_tm/layout/matrix.h"
#include "cutlass_rt_tm/matrix_shape.h"
#include "cutlass_rt_tm/tensor_ref.h"
#include "cutlass_rt_tm/fast_math.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass_rt_tm {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tuple defining point in output tile
template <
  int Column,
  int Row,
  int Group,
  int Cluster,
  int Tile
>
struct OutputTileShape {
  static constexpr int kColumn = Column;
  static constexpr int kRow = Row;
  static constexpr int kGroup = Group;
  static constexpr int kCluster = Cluster;
  static constexpr int kTile = Tile;

  static constexpr int kCount = kColumn * kRow * kGroup * kCluster * kTile;
};

////////////////////////////////////////////////////////////////////////////////

template <typename Iterations, typename Delta>
struct OutputTileThreadMapHelpers {

  /// Determines the iteration index of a vector access according to the thread map
  CUTLASS_RT_TM_HOST_DEVICE
  static void iteration_index(
    int &column_idx,
    int &row_idx,
    int &group_idx,
    int &cluster_idx,
    int &tile_idx,
    int iter_idx) {

    column_idx = iter_idx % Iterations::kColumn;
    int residual   = iter_idx / Iterations::kColumn;

    row_idx    = residual % Iterations::kRow;
    residual       = residual / Iterations::kRow;

    group_idx  = residual % Iterations::kGroup;
    residual       = residual / Iterations::kGroup;

    cluster_idx = residual % Iterations::kCluster;
    tile_idx    = residual / Iterations::kCluster;
  }

  /// Computes the offset of a given vector access
  CUTLASS_RT_TM_HOST_DEVICE
  static MatrixCoord iteration_offset(int iter_idx) {

    int column_idx;
    int row_idx;
    int group_idx;
    int cluster_idx;
    int tile_idx;

    iteration_index(column_idx, row_idx, group_idx, cluster_idx, tile_idx, iter_idx);

    return
      MatrixCoord(
        row_idx     * Delta::kRow     +
        group_idx   * Delta::kGroup   +
        cluster_idx * Delta::kCluster +
        tile_idx    * Delta::kTile,

        column_idx  * Delta::kColumn);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////


template <
  typename ThreadMap_,
  typename Shape_,
  typename Iterations_,
  typename Delta_,
  typename Count_
>
struct OutputTileThreadMap : public OutputTileThreadMapHelpers<Iterations_, Delta_> {

  /// Conventional thread map (concept: ThreadMap)
  using ThreadMap = ThreadMap_;

  /// Number of threads participating in the operation
  static constexpr int kThreads = ThreadMap::kThreads;

  /// Number of scalar elements per access
  static constexpr int kElementsPerAccess = ThreadMap::kElementsPerAccess;

  /// Shape of the tile
  using Shape = Shape_;

  /// Iterations performed by each thread
  using Iterations = Iterations_;

  /// Delta between accesses
  using Delta = Delta_;

  /// Number of iterator iterations 
  using Count = Count_;

  /// Initial offset function
  CUTLASS_RT_TM_HOST_DEVICE
  static MatrixCoord initial_offset(int thread_idx) {

    using Index = typename layout::PitchLinearCoord::Index;
    
    layout::PitchLinearCoord coord = ThreadMap::initial_offset(thread_idx);

    Index cluster = coord.strided() / (Shape::kGroup * Shape::kRow);
    Index cluster_residual = coord.strided() % (Shape::kGroup * Shape::kRow);

    Index group = cluster_residual / (Shape::kRow);
    Index row = cluster_residual % (Shape::kRow);

    return MatrixCoord{
      row + group * Shape::kRow * Count::kRow 
        + cluster * Shape::kGroup * Count::kGroup * Shape::kRow * Count::kRow,
      coord.contiguous()
    };
  }
};

////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// RowArrangement determines how one or more warps cover a region of consecutive rows.
template <
  typename Shape,
  int WarpsRemaining,
  int ElementsPerAccess,
  int ElementSize,
  bool Is2dTile
>
struct RowArrangement;

/// RowArrangement in which each warp's access is a 1D tiled arrangement.
template <
  typename Shape,
  int WarpsRemaining,
  int ElementsPerAccess,
  int ElementSize
>
struct RowArrangement<Shape, WarpsRemaining, ElementsPerAccess, ElementSize, false> {
  static constexpr int kWarpSize = 32;
  static constexpr int kElementsPerAccess = ElementsPerAccess;
  static constexpr int kElementSize = ElementSize;

  static constexpr int kIterationsRow = 1;
  static constexpr int kDeltaRow = 1;
  static constexpr int kIterationsColumn = Shape::kColumn / kElementsPerAccess / kWarpSize;
  static constexpr int kDeltaColumn = kWarpSize * kElementsPerAccess;

  static constexpr int kAccessWidth = kWarpSize;
  static constexpr int kAccessRows = 1;
  static constexpr int kWarpPartitionsRow = 1;
  static constexpr int kWarpPartitionsColumn = WarpsRemaining;
};

/// RowArrangement in which each warp's access is a 2D tiled arrangement.
template <
  typename Shape,
  int WarpsRemaining,
  int ElementsPerAccess,
  int ElementSize
>
struct RowArrangement<Shape, WarpsRemaining, ElementsPerAccess, ElementSize, true> {

  static constexpr int kMemoryAccessSize = 256; // Preferred access size
  static constexpr int kWarpSize = 32;

  static constexpr int kElementsPerAccess = ElementsPerAccess;
  static constexpr int kElementSize = ElementSize;

  struct Detail {
    static constexpr int kShapeRow = Shape::kRow / WarpsRemaining;
    static constexpr int kShapeWidth = Shape::kColumn / kElementsPerAccess;

    static constexpr int kTargetMemoryAccessWidth = 
      kMemoryAccessSize / (kElementsPerAccess * kElementSize / 8);

    static constexpr int kTargetAccessRows = kWarpSize / kTargetMemoryAccessWidth;
  };

  static constexpr int kAccessWidth = 
    (Detail::kTargetAccessRows > Detail::kShapeRow ?
      kWarpSize / Detail::kShapeRow
      : const_min(
          Detail::kShapeWidth,
        const_min(kWarpSize, kMemoryAccessSize / (kElementsPerAccess * kElementSize / 8))
        ));

  static constexpr int kAccessRows =
    (Detail::kTargetAccessRows > Detail::kShapeRow ?
      Detail::kShapeRow
      : const_min(Shape::kRow, kWarpSize / kAccessWidth));

  static constexpr int kIterationsRow = Detail::kShapeRow / kAccessRows;
  static constexpr int kDeltaRow = kAccessRows;

  static constexpr int kIterationsColumn = Detail::kShapeWidth / kAccessWidth;
  static constexpr int kDeltaColumn = kAccessWidth * kElementsPerAccess;

  static_assert( kAccessWidth * kElementsPerAccess <= Shape::kColumn, "Accessing too many elements per access");
  static_assert( kIterationsColumn > 0, "Iteration Count Column must be > 0" );
  static_assert( kIterationsRow > 0, "Iteration Count Row must be > 0" );

  static constexpr int kWarpPartitionsRow = 1;
  static constexpr int kWarpPartitionsColumn = 1;
};

}

////////////////////////////////////////////////////////////////////////////////

/// Template metaprogram for partitioning a 4D space across warps to achieve several performance
/// objectives:
///
///   - coalesced memory accesses in units of 128 Byte lines
///   - minimal address arithmetic
///   - minimal predicate calculations
///
template <
  typename Shape_,
  typename Count_,
  int Threads,
  int ElementsPerAccess,
  int ElementSize
>
struct OutputTileOptimalThreadMap {

  using Shape = Shape_;
  using Count = Count_;

  static constexpr int kWarpSize = 32;
  static constexpr int kThreads = Threads;
  static constexpr int kWarpCount = kThreads / kWarpSize;

  static constexpr int kElementsPerAccess = ElementsPerAccess;
  static constexpr int kElementSize = ElementSize;

  //
  // Metaprogram computation
  //

  struct Detail {

    // Clusters
    static constexpr int kIterationsCluster = 
      ((Shape::kCluster > kWarpCount) ?
        Shape::kCluster / kWarpCount
        : 1);

    static constexpr int kDeltaCluster =
      ((Shape::kCluster > kWarpCount) ?
        Shape::kRow * Count::kRow * Shape::kGroup * Count::kGroup * Shape::kCluster / kIterationsCluster
        : 1);

    static constexpr int kCompactedDeltaCluster =
      ((Shape::kCluster > kWarpCount) ?
        Shape::kRow * Shape::kGroup * Shape::kCluster / kIterationsCluster
        : 1);

    static constexpr int kWarpPartitionsCluster =
      ((Shape::kCluster > kWarpCount) ?
        kWarpCount
        : kWarpCount / Shape::kCluster);

    static constexpr int kWarpsRemainingForGroups =
      ((Shape::kCluster > kWarpCount) ? 1 : kWarpCount / Shape::kCluster);

    // Groups
    static constexpr int kIterationsGroup =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        Shape::kGroup / kWarpsRemainingForGroups
        : 1);

    static constexpr int kDeltaGroup =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        Shape::kRow * Count::kRow * Shape::kGroup / kIterationsGroup
        : 1);

    static constexpr int kCompactedDeltaGroup =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        Shape::kRow * Shape::kGroup / kIterationsGroup
        : 1);

    static constexpr int kWarpPartitionsGroup =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        1
        : kWarpsRemainingForGroups / Shape::kGroup);

    static constexpr int kWarpsRemainingForRows =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        1
        : kWarpsRemainingForGroups / Shape::kGroup);
    
    // Rows
    using RowArrangement = detail::RowArrangement<
      Shape,
      kWarpsRemainingForRows,
      kElementsPerAccess,
      kElementSize,
      (Shape::kRow > kWarpsRemainingForRows)
    >;

    // Warp partitions
    using WarpPartitions = OutputTileShape<
      RowArrangement::kWarpPartitionsColumn,
      RowArrangement::kWarpPartitionsRow,
      kWarpPartitionsGroup,
      kWarpPartitionsCluster,
      1>;

    static constexpr int kAccessWidth = RowArrangement::kAccessWidth;
    static constexpr int kAccessRows = RowArrangement::kAccessRows;
  };

  //
  // Output
  //

  using Iterations = OutputTileShape<
    Detail::RowArrangement::kIterationsColumn, 
    Detail::RowArrangement::kIterationsRow, 
    Detail::kIterationsGroup, 
    Detail::kIterationsCluster, 
    1>;

  using Delta = OutputTileShape<
    Detail::RowArrangement::kDeltaColumn,
    Detail::RowArrangement::kDeltaRow,
    Detail::kDeltaGroup,
    Detail::kDeltaCluster,
    1>;

  /// Initial offset function
  CUTLASS_RT_TM_HOST_DEVICE
  static MatrixCoord initial_offset(int thread_idx) {

//    int warp_idx = __shfl_sync(0xffffffff, thread_idx / kWarpSize, 0);
    int warp_idx = thread_idx / kWarpSize;
    int lane_idx = thread_idx % kWarpSize;

    // Compute warp location
    int cluster_idx = warp_idx / Detail::WarpPartitions::kCluster;
    int residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

    int group_idx = residual_cluster / Detail::WarpPartitions::kGroup;
    int residual_group = residual_cluster % Detail::WarpPartitions::kGroup;

    int row_idx = residual_group / Detail::WarpPartitions::kRow;
    int col_idx = residual_group % Detail::WarpPartitions::kRow;

    // Compute per-lane offset
    int lane_row_offset = lane_idx / Detail::kAccessWidth;
    int lane_col_offset = lane_idx % Detail::kAccessWidth;

    // Compute coordinate in output space
    int cluster_offset = cluster_idx * Shape::kRow * Count::kRow * Shape::kGroup * Count::kGroup;
    int group_offset = group_idx * Shape::kRow * Count::kRow;
    int row_offset = row_idx * Iterations::kRow * Detail::kAccessRows;
    int column_offset = col_idx * Iterations::kColumn * Detail::kAccessWidth * kElementsPerAccess;

    return MatrixCoord(
      cluster_offset + group_offset + row_offset + lane_row_offset,
      column_offset + lane_col_offset * kElementsPerAccess
    );
  }

  /// Computes the offset of a given vector access
  CUTLASS_RT_TM_HOST_DEVICE
  static MatrixCoord iteration_offset(int iter_idx) {
    return OutputTileThreadMapHelpers<Iterations, Delta>::iteration_offset(iter_idx);
  }

  /// Compacted thread map in which the 4D region is contiguous
  struct CompactedThreadMap {


    using Shape = Shape_;

    using TileShape = MatrixShape<
      Shape::kTile * Shape::kCluster * Shape::kGroup * Shape::kRow,
      Shape::kColumn
    >;

    using Iterations = OutputTileShape<
      Detail::RowArrangement::kIterationsColumn,
      Detail::RowArrangement::kIterationsRow,
      Detail::kIterationsGroup,
      Detail::kIterationsCluster,
      1>;

    using Delta = OutputTileShape<
      Detail::RowArrangement::kDeltaColumn,
      Detail::RowArrangement::kDeltaRow,
      Detail::kCompactedDeltaGroup,
      Detail::kCompactedDeltaCluster,
      1>;

    /// Number of elements within each vector access
    static constexpr int kElementsPerAccess = ElementsPerAccess;

    /// Number  of threads
    static constexpr int kThreads = Threads;

    /// Function to compute each thread's initial offset
    CUTLASS_RT_TM_HOST_DEVICE
    static MatrixCoord initial_offset(int thread_idx) {

//      int warp_idx = __shfl_sync(0xffffffff, thread_idx / kWarpSize, 0);
      int warp_idx = thread_idx / kWarpSize;
      int lane_idx = thread_idx % kWarpSize;

      // Compute warp location
      int cluster_idx = warp_idx / Detail::WarpPartitions::kCluster;
      int residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

      int group_idx = residual_cluster / Detail::WarpPartitions::kGroup;
      int residual_group = residual_cluster % Detail::WarpPartitions::kGroup;

      int row_idx = residual_group / Detail::WarpPartitions::kRow;
      int col_idx = residual_group % Detail::WarpPartitions::kRow;

      // Compute per-lane offset
      int lane_row_offset = lane_idx / Detail::kAccessWidth;
      int lane_col_offset = lane_idx % Detail::kAccessWidth;

      // Compute coordinate in output space
      int cluster_offset = cluster_idx * Shape::kRow * Shape::kGroup;
      int group_offset = group_idx * Shape::kRow;
      int row_offset = row_idx * Iterations::kRow * Detail::kAccessRows;
      int column_offset = col_idx * Iterations::kColumn * Detail::kAccessWidth * kElementsPerAccess;

      MatrixCoord coord(
        cluster_offset + group_offset + row_offset + lane_row_offset,
        column_offset + lane_col_offset * kElementsPerAccess
      );

      return coord;
    }
  };
};

////////////////////////////////////////////////////////////////////////////////

/// Template metaprogram for partitioning a 3D interleaved layout across warps
/// to achieve several performance objectives:
///
///   - coalesced memory accesses in units of 64 Byte lines
///   - minimal address arithmetic
///   - minimal predicate calculations
///
template <typename WarpCount_, typename Iterations_, int Threads,
          int ElementsPerAccess, int ElementSize>
struct InterleavedOutputTileThreadMap {
  using WarpCount = WarpCount_;

  static constexpr int kWarpSize = 32;
  static constexpr int kThreads = Threads;
  static constexpr int kWarpCount = kThreads / kWarpSize;

  static constexpr int kElementsPerAccess = ElementsPerAccess;
  static constexpr int kElementSize = ElementSize;

  //
  // Metaprogram computation
  //

  struct Detail {};

  //
  // Output
  //

  using Iterations = Iterations_;

  using Delta = layout::PitchLinearShape<kWarpSize * kElementsPerAccess, 1>;

  /// Initial offset function
  CUTLASS_RT_TM_HOST_DEVICE
  static layout::PitchLinearCoord initial_offset(int thread_idx) {
    int warp_idx = thread_idx / kWarpSize;
    int lane_idx = thread_idx % kWarpSize;

    // Compute warp location
    layout::PitchLinearCoord warp_footprint{
        Delta::kContiguous * Iterations::kContiguous,
        Delta::kStrided * Iterations::kStrided};

    layout::PitchLinearCoord warp_offset{warp_idx % WarpCount::kContiguous,
                                         warp_idx / WarpCount::kContiguous};

    // Compute per-lane offset
    layout::PitchLinearCoord thread_offset_in_warp{
        lane_idx * kElementsPerAccess, 0};

    layout::PitchLinearCoord thread_offset_in_threadblock_tile =
        warp_footprint * warp_offset + thread_offset_in_warp;

    return thread_offset_in_threadblock_tile;
  }
};


////////////////////////////////////////////////////////////////////////////////

/// Template metaprogram for partitioning a 4D interleaved layout across warps
/// to achieve several performance objectives:
///
///   - coalesced memory accesses in units of 64 Byte lines
///   - minimal address arithmetic
///   - minimal predicate calculations
///
template <typename WarpCount_, typename Iterations_, int Threads,
          int ElementsPerAccess, int ElementSize>
struct InterleavedConvOutputTileThreadMap {
  using WarpCount = WarpCount_;

  static constexpr int kWarpSize = 32;
  static constexpr int kThreads = Threads;
  static constexpr int kWarpCount = kThreads / kWarpSize;

  static constexpr int kElementsPerAccess = ElementsPerAccess;
  static constexpr int kElementSize = ElementSize;

  //
  // Metaprogram computation
  //

  struct Detail {};

  //
  // Output
  //

  using Iterations = Iterations_;

  using Delta = MatrixShape<kWarpSize / 4, 4 * kElementsPerAccess>;

  /// Initial offset function
  CUTLASS_RT_TM_HOST_DEVICE
  static MatrixCoord initial_offset(int thread_idx) {
    int warp_idx = thread_idx / kWarpSize;
    int lane_idx = thread_idx % kWarpSize;

    // Compute warp location
    MatrixCoord warp_footprint{
        Delta::kRow * Iterations::kRow,
        Delta::kColumn * Iterations::kColumn,
    };

    MatrixCoord warp_offset{warp_idx % WarpCount::kRow,
                            warp_idx / WarpCount::kRow};

    // Compute per-lane offset
    MatrixCoord thread_offset_in_warp{lane_idx / 4,
                                      (lane_idx % 4) * kElementsPerAccess};

    MatrixCoord thread_offset_in_threadblock_tile =
        warp_footprint * warp_offset + thread_offset_in_warp;

    return thread_offset_in_threadblock_tile;
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass_rt_tm
