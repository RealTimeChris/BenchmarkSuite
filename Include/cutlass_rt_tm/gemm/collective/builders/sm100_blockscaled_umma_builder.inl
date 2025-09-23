/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include "cutlass_rt_tm/gemm/collective/builders/sm100_common.inl"
#include "cutlass_rt_tm/gemm/collective/builders/sm100_pipeline_carveout.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_rt_tm::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template <
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class TileShapeMNK,
  class TileShapeSFA,
  class TileShapeSFB,
  int stages
>
constexpr int
sm100_compute_stage_count_or_override_blockscaled(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template <
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class TileShapeMNK,
  class TileShapeSFA,
  class TileShapeSFB,
  int carveout_bytes
>
constexpr int
sm100_compute_stage_count_or_override_blockscaled(StageCountAutoCarveout<carveout_bytes> stage_count) {
  // For MXF8F6F4 MMA, ElementA/B will be passed in as uint8_t
  // Each stage include (CollectiveMma::SharedStorage)
  // 1. smem for A and smem for B (CollectiveMma::SharedStorage::TensorStorage)
  // 2. one MainloopPipeline = PipelineTmaUmmaAsync (CollectiveMma::SharedStorage::SharedStorage)
  // 3. smem for SFB and smem for SFB (CollectiveMma::SharedStorage::TensorStorage, independent of input size b.c. sizeof(sf) is fixed)
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass_rt_tm::PipelineTmaUmmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute_rt_tm::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute_rt_tm::sizeof_bits_v<ElementB>;
  constexpr auto stage_sfa_bytes = size(filter_zeros(TileShapeSFA{}));
  constexpr auto stage_sfb_bytes = size(filter_zeros(TileShapeSFB{}));

  constexpr int stage_bytes =
    cutlass_rt_tm::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass_rt_tm::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    static_cast<int>(mainloop_pipeline_bytes + stage_sfa_bytes + stage_sfb_bytes);

  return (CapacityBytes - carveout_bytes) / stage_bytes;
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ElementPairA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementPairB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,        // (MmaAtomShapeM, MmaAtomShapeN, TileK)
  class ClusterShape_MNK,     // Static cluster shape or dynamic (int, int, _1)
  class StageCountType,
  class BuilderScheduleTag
>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassBlockScaledTensorOp,
    ElementPairA,
    GmemLayoutATag,
    AlignmentA,
    ElementPairB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    BuilderScheduleTag,
    cute_rt_tm::enable_if_t<
      // Blockscaled Gemm
      (not cute_rt_tm::is_same_v<KernelMixedTmaCpAsyncWarpSpecialized1SmBlockScaledSm100, BuilderScheduleTag>) &&
      (cute_rt_tm::is_base_of_v<KernelScheduleBlockScaledGemmSm100, BuilderScheduleTag> ||
       cute_rt_tm::is_same_v<KernelScheduleAuto, BuilderScheduleTag>) 
       &&
      // Alignment check
      detail::sm1xx_blockscaled_gemm_is_aligned<typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::data_type,
                                                AlignmentA,
                                                typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::data_type,
                                                AlignmentB,
                                                BuilderScheduleTag>()>>
{
  using ElementSFA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::sf_type;
  using ElementSFB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::sf_type;
  using ElementA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::data_type;
  using ElementB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::data_type;
  using ElementSF = ElementSFA;

  static constexpr cute_rt_tm::UMMA::Major UmmaMajorA = cutlass_rt_tm::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute_rt_tm::UMMA::Major UmmaMajorB = cutlass_rt_tm::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();

  static_assert(cute_rt_tm::is_static_v<TileShape_MNK>, "TileShape has to be static");
  static_assert(detail::blockscaled::check_input_datatypes<BuilderScheduleTag, ElementPairA, ElementPairB, UmmaMajorA, UmmaMajorB>(), "Incorrect input types");

  static constexpr bool is_2sm = detail::blockscaled::is_2sm<TileShape_MNK, ClusterShape_MNK, BuilderScheduleTag>();
  static constexpr auto Instr = detail::blockscaled::select_instr<ElementPairA, ElementPairB, ElementAccumulator, UmmaMajorA, UmmaMajorB, BuilderScheduleTag>();

  using TiledMma = typename cutlass_rt_tm::gemm::collective::detail::TrivialBlockscaledMma<ElementPairA, ElementPairB, ElementAccumulator,
                                                                  TileShape_MNK, ClusterShape_MNK,
                                                                  UmmaMajorA, UmmaMajorB, Instr, BuilderScheduleTag, is_2sm>::type;

  static constexpr bool UseMxf8f6f4 = Instr == detail::blockscaled::BlockScaledInstr::MXF4F6F8;

  static_assert(UseMxf8f6f4 || (cutlass_rt_tm::gemm::detail::is_k_major_A<GmemLayoutATag>() && cutlass_rt_tm::gemm::detail::is_k_major_B<GmemLayoutBTag>()), "Only MMA.MXF8F6F4 supports non-K major inputs");

  // Data type used by MMA instruction
  using ElementAMma = decltype(cutlass_rt_tm::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA, UseMxf8f6f4>());
  using ElementBMma = decltype(cutlass_rt_tm::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB, UseMxf8f6f4>());

  static_assert(detail::sm1xx_gemm_check_for_f8f6f4_mix8bit_requirement<ElementAMma, ElementBMma,
                                                                        TileShape_MNK, ClusterShape_MNK,
                                                                        GmemLayoutATag, GmemLayoutBTag, false /*is_sparse*/, is_2sm>(),
                "TileSize and MNK Major does not met with MMA Mix 8-bit TMA load requirement" );

  static constexpr uint32_t SFVectorSize = TiledMma::SFVecSize;

  // Basic storage block for new Scaling Factor Layouts
  using AtomThrID = typename TiledMma::AtomThrID;
  using Sm1xxBlkScaledConfig = cutlass_rt_tm::detail::Sm1xxBlockScaledConfig<SFVectorSize>;

  using ElementAMma_SmemAllocType = cute_rt_tm::conditional_t<UseMxf8f6f4, uint8_t, ElementAMma>;
  using ElementBMma_SmemAllocType = cute_rt_tm::conditional_t<UseMxf8f6f4, uint8_t, ElementBMma>;

  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute_rt_tm::size<0>(TileShape_MNK{}),
                                                                         cute_rt_tm::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute_rt_tm::size<1>(TileShape_MNK{}),
                                                                         cute_rt_tm::size<2>(TileShape_MNK{}))));

  using GmemTiledCopyA = decltype(cutlass_rt_tm::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(
      ClusterShape_MNK{}, AtomThrID{}));

  using GmemTiledCopyB = decltype(cutlass_rt_tm::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_B(
      ClusterShape_MNK{}, AtomThrID{}));

  using GmemTiledCopySFA = decltype(cutlass_rt_tm::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(
      ClusterShape_MNK{}, AtomThrID{}));

  using GmemTiledCopySFB = decltype(cutlass_rt_tm::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_SFB(
      ClusterShape_MNK{}, AtomThrID{})); 

  using GmemTiledCopyPairA = decltype(cute_rt_tm::make_tuple(GmemTiledCopyA{}, GmemTiledCopySFA{}));
  using GmemTiledCopyPairB = decltype(cute_rt_tm::make_tuple(GmemTiledCopyB{}, GmemTiledCopySFB{}));

  //
  // Construct SMEM layout (SmemLayoutAtom) for A and SFA
  //
  using BlockTileA_M = decltype(cute_rt_tm::size<0,0>(MmaShapeA_MK{}) * cute_rt_tm::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute_rt_tm::size<0,1>(MmaShapeA_MK{}) * cute_rt_tm::size<2>(MmaShapeA_MK{}));
  using SmemLayoutAtomA = decltype(cutlass_rt_tm::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorA, ElementAMma_SmemAllocType, BlockTileA_M, BlockTileA_K>());

  // A single indivisible block will hold 4 scale factors of 128 rows/columns (A/B matrix).
  // 4 is chosen to make consecutive 32bits of data to have scale factors for only a single row (col). 32bits corresponds to the TMEM word size 
  using Blk_MN    = typename Sm1xxBlkScaledConfig::Blk_MN;
  using Blk_SF    = typename Sm1xxBlkScaledConfig::Blk_SF; 
  using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
  using SmemLayoutAtomSFA = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape_MNK{}));
  using SmemLayoutAtomsA = decltype(cute_rt_tm::make_tuple(SmemLayoutAtomA{}, SmemLayoutAtomSFA{}));

  //
  // Construct SMEM layout (SmemLayoutAtom) for B and SFB
  //
  using BlockTileB_N = decltype(cute_rt_tm::size<0,0>(MmaShapeB_NK{}) * cute_rt_tm::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute_rt_tm::size<0,1>(MmaShapeB_NK{}) * cute_rt_tm::size<2>(MmaShapeB_NK{}));
  using SmemLayoutAtomB = decltype(cutlass_rt_tm::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, ElementBMma_SmemAllocType, BlockTileB_N, BlockTileB_K>());
  using SmemLayoutAtomSFB = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape_MNK{}));
  using SmemLayoutAtomsB = decltype(cute_rt_tm::make_tuple(SmemLayoutAtomB{}, SmemLayoutAtomSFB{}));

  //
  // Construct Strides for A, SFA, B, and SFB
  //
  using StrideA = cutlass_rt_tm::gemm::TagToStrideA_t<GmemLayoutATag>;
  using StrideB = cutlass_rt_tm::gemm::TagToStrideB_t<GmemLayoutBTag>;
  using InternalStrideA  = cute_rt_tm::remove_pointer_t<StrideA>;
  using InternalStrideB  = cute_rt_tm::remove_pointer_t<StrideB>;
  using InternalLayoutSFA = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFA());
  using InternalLayoutSFB = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFB());
  using LayoutSFA = cute_rt_tm::conditional_t<cute_rt_tm::is_same_v<InternalStrideA, StrideA>, InternalLayoutSFA, InternalLayoutSFA *>;
  using LayoutSFB = cute_rt_tm::conditional_t<cute_rt_tm::is_same_v<InternalStrideB, StrideB>, InternalLayoutSFB, InternalLayoutSFB *>;
  using StridePairA = decltype(cute_rt_tm::make_tuple(StrideA{}, LayoutSFA{}));
  using StridePairB = decltype(cute_rt_tm::make_tuple(StrideB{}, LayoutSFB{}));

  static constexpr int MMA_N = cute_rt_tm::size<1>(TileShape_MNK{});
  static constexpr uint32_t AccumulatorPipelineStageCount = (MMA_N == 256) ? 1 : 2;
  static constexpr bool IsArrayOfPointersGemm = cute_rt_tm::is_base_of_v<KernelSchedulePtrArrayBlockScaledGemmSm100, BuilderScheduleTag>;
  // Grouped GEMM(where Stride type is Stride*) uses specific static tile scheduler.  
  static constexpr bool IsGroupGemm = !cute_rt_tm::is_same_v<StrideA, InternalStrideA>;
  static constexpr uint32_t SchedulerPipelineStageCount = cute_rt_tm::conditional_return<IsGroupGemm>(8, 2);

  static constexpr uint32_t KernelSmemCarveout = detail::Sm100DenseGemmTmaUmmaCarveout<
      ClusterShape_MNK,
      AccumulatorPipelineStageCount,
      SchedulerPipelineStageCount,
      detail::CLCResponseSize,
      IsArrayOfPointersGemm,
      4 // 4 Tensor maps for A, SFA, B and SFB
    >::KernelSmemCarveout;
  // Reduce SMEM capacity available for buffers considering barrier allocations.
  static constexpr int Sm100ReducedSmemCapacityBytes = cutlass_rt_tm::gemm::collective::detail::sm100_smem_capacity_bytes - KernelSmemCarveout;

  using SmemTileShape = cute_rt_tm::Shape<BlockTileA_M, BlockTileB_N, BlockTileA_K>;

  static constexpr int PipelineStages = cutlass_rt_tm::gemm::collective::detail::sm100_compute_stage_count_or_override_blockscaled<
      Sm100ReducedSmemCapacityBytes, ElementAMma_SmemAllocType, ElementBMma_SmemAllocType, SmemTileShape, SmemLayoutAtomSFA, SmemLayoutAtomSFB>(StageCountType{});
  static_assert(PipelineStages > 0, "Smem usage is too high. Can't create any SMEM buffers for A, B, SFA, and SFB.");

  using DispatchPolicy = 
    cute_rt_tm::conditional_t<IsArrayOfPointersGemm,
      cutlass_rt_tm::gemm::MainloopSm100ArrayTmaUmmaWarpSpecializedBlockScaled<
          PipelineStages,
          SchedulerPipelineStageCount,
          AccumulatorPipelineStageCount,
          ClusterShape_MNK
      >,
      cutlass_rt_tm::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled<
          PipelineStages,
          SchedulerPipelineStageCount,
          AccumulatorPipelineStageCount,
          ClusterShape_MNK
      >
    >;

  using CollectiveOp = cutlass_rt_tm::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      cute_rt_tm::tuple<ElementA, ElementSF>,
      StridePairA,
      cute_rt_tm::tuple<ElementB, ElementSF>,
      StridePairB,
      TiledMma,
      GmemTiledCopyPairA,
      SmemLayoutAtomsA,
      void,
      cute_rt_tm::identity,
      GmemTiledCopyPairB,
      SmemLayoutAtomsB,
      void,
      cute_rt_tm::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass_rt_tm::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
