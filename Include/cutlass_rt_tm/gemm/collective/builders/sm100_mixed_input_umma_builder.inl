/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_rt_tm::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<
  int CapacityBytes,
  class ElementA,
  class ElementAMma,
  class ElementScale,
  class ElementZero,
  class ElementB,
  class CtaTileShape_MNK,
  class TiledMma,
  class KernelScheduleType,
  UMMA::Major UmmaMajorA,
  int ScaleGranularityK,
  int stages
>
constexpr cute_rt_tm::tuple<int, int, int>
sm100_compute_stage_count_or_override_mixed_input(StageCount<stages> stage_count) {
  constexpr int Load2TransformStageCount = stages;
  constexpr int Transform2MmaStageCount = stages;
  constexpr int AccumulatorStageCount = stages;
  return cute_rt_tm::make_tuple(Load2TransformStageCount, Transform2MmaStageCount, AccumulatorStageCount);
}

template<
  int CapacityBytes,
  class ElementA,
  class ElementAMma,
  class ElementScale,
  class ElementZero,
  class ElementB,
  class CtaTileShape_MNK,
  class TiledMma,
  class KernelScheduleType,
  UMMA::Major UmmaMajorA,
  int ScaleGranularityK,
  int carveout_bytes
>
constexpr cute_rt_tm::tuple<int, int, int>
sm100_compute_stage_count_or_override_mixed_input(StageCountAutoCarveout<carveout_bytes> stage_count) {

  constexpr int CtaM = get<0>(CtaTileShape_MNK{});
  constexpr int CtaN = get<1>(CtaTileShape_MNK{});
  static_assert(CtaN <= 128, "Can't support CtaN>128 tiles");
  constexpr int CtaK = get<2>(CtaTileShape_MNK{});
  using AtomThrID = typename TiledMma::AtomThrID;

  constexpr int TmemColumns = 512;

  constexpr bool IsAComputeinTmem = UmmaMajorA == cute_rt_tm::UMMA::Major::K && !cute_rt_tm::is_base_of_v<KernelTmaWarpSpecializedMixedInputSmemSm100, KernelScheduleType>;
  constexpr bool IsAComputeinSmem = !IsAComputeinTmem;

  // Detect 2x2 TMEM layout
  constexpr int TmemAccWordsPerDP = (CtaM == 64 && size(AtomThrID{}) == 2) ? CtaN/2 : CtaN;
  constexpr int TmemAWordsPerDP = CtaK / 2;

  constexpr int AccumulatorStageCount = (IsAComputeinTmem) ? ((TmemAccWordsPerDP == 128) ? 2 : 3) : (TmemColumns / TmemAccWordsPerDP);
  
  constexpr int SmemCapacityAfterMma2AccumCarveout = CapacityBytes - (carveout_bytes + AccumulatorStageCount * 32);

  constexpr int TmemInAStageCount_Potential = (IsAComputeinTmem) ? (TmemColumns - AccumulatorStageCount * TmemAccWordsPerDP) / TmemAWordsPerDP : 10000;
  
  // Mainload2Transform Pipeline
  constexpr auto load2transform_pipeline_bytes = sizeof(typename cutlass_rt_tm::PipelineTmaTransformAsync<1>::SharedStorage);
  constexpr auto a_bits = cute_rt_tm::sizeof_bits_v<ElementA>; // ElementA introduce here
  constexpr auto s_bits = cute_rt_tm::is_void_v<ElementScale> ? 0 : cute_rt_tm::sizeof_bits_v<ElementScale>;
  constexpr auto z_bits = cute_rt_tm::is_void_v<ElementZero> ? 0 : cute_rt_tm::sizeof_bits_v<ElementZero>;

  constexpr auto load2mma_pipeline_bytes = sizeof(typename cutlass_rt_tm::PipelineTmaUmmaAsync<1>::SharedStorage);
  constexpr auto b_bits = cute_rt_tm::sizeof_bits_v<ElementB>; // ElementB introduce here

  constexpr int ab_stage_bytes =
    cutlass_rt_tm::bits_to_bytes(a_bits * size<0>(CtaTileShape_MNK{}) * size<2>(CtaTileShape_MNK{})) +
    cutlass_rt_tm::bits_to_bytes(s_bits * size<0>(CtaTileShape_MNK{}) * size<2>(CtaTileShape_MNK{}) / ScaleGranularityK) +
    cutlass_rt_tm::bits_to_bytes(z_bits * size<0>(CtaTileShape_MNK{}) * size<2>(CtaTileShape_MNK{}) / ScaleGranularityK) +
    cutlass_rt_tm::bits_to_bytes(b_bits * size<1>(CtaTileShape_MNK{}) / size(AtomThrID{}) * size<2>(CtaTileShape_MNK{})) +
    static_cast<int>(load2transform_pipeline_bytes) + static_cast<int>(load2mma_pipeline_bytes);

  // Transform2Mma Pipeline
  constexpr auto transform2mma_pipeline_bytes = sizeof(typename cutlass_rt_tm::PipelineUmmaConsumerAsync<1>::SharedStorage);
  constexpr auto a_compute_bits = cute_rt_tm::sizeof_bits_v<ElementAMma>;
  constexpr int ab_compute_stage_bytes =
    cutlass_rt_tm::bits_to_bytes(a_compute_bits * int(IsAComputeinSmem) * size<0>(CtaTileShape_MNK{})  * size<2>(CtaTileShape_MNK{})) + // If ACompute is in TMEM, Acompute buffer has 0 bytes.
    static_cast<int>(transform2mma_pipeline_bytes);

  constexpr int ABComputeStageCount_Potential = SmemCapacityAfterMma2AccumCarveout / (ab_stage_bytes + ab_compute_stage_bytes);

  // The number of SMEM buffers for A, B. ACompute (if in SMEM), BCompute should be at least Transform2MmaStageCount
  constexpr int Transform2MmaStageCount = std::min(TmemInAStageCount_Potential, ABComputeStageCount_Potential);

  constexpr int SmemCapacityAfterABComputeCarveout = SmemCapacityAfterMma2AccumCarveout - (Transform2MmaStageCount * ab_compute_stage_bytes);

  // Can we boost the number of buffers for A and B?
  constexpr int Load2TransformStageCount = SmemCapacityAfterABComputeCarveout / ab_stage_bytes;

  static_assert(Load2TransformStageCount >= 2 && Transform2MmaStageCount >= 2 && AccumulatorStageCount >= 2, "Not enough SMEM or TMEM capacity for selected tile size");
  return cute_rt_tm::make_tuple(Load2TransformStageCount, Transform2MmaStageCount, AccumulatorStageCount);
}

} // namespace detail

template <typename LayoutScale>
constexpr int get_ScaleGranularityK() {
  if constexpr (cute_rt_tm::is_void_v<LayoutScale>) {
    return 1;
  } else {
    return size<1,0>(LayoutScale{});
  }
}


// Mixed Input MMA kernels builder
template <
  class ElementAOptionalTuple,
  class GmemLayoutATagTuple,
  int AlignmentA,
  class ElementBOptionalTuple,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,  // The Cluster-level TileShape
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassTensorOp,
    ElementAOptionalTuple, // ElementA
    GmemLayoutATagTuple,  // LayoutA 
    AlignmentA,
    ElementBOptionalTuple, // ElementB
    GmemLayoutBTag,  // LayoutB
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,    // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    ClusterShape_MNK, // Static cluster shape or dynamic (int, int, int)
    StageCountType,
    KernelScheduleType,
    cute_rt_tm::enable_if_t<
      (cute_rt_tm::is_base_of_v<KernelScheduleSm100MixedInputGemm, KernelScheduleType>) &&
      ((sizeof(float) * AlignmentA) % detail::tma_alignment_bytes == 0) &&
      ((sizeof(float) * AlignmentB) % detail::tma_alignment_bytes == 0)>>
{
  using GmemLayoutATag = detail::deduce_mixed_width_dtype_t<0, GmemLayoutATagTuple>;
  using GmemLayoutScaleTag = detail::deduce_mixed_width_dtype_t<1, GmemLayoutATagTuple>;

  static constexpr cute_rt_tm::UMMA::Major UmmaMajorA = cutlass_rt_tm::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute_rt_tm::UMMA::Major UmmaMajorB = cutlass_rt_tm::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();

  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  using ElementScale = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;
  using ElementZero = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;

  static constexpr bool NeitherIsTuple = !cute_rt_tm::is_tuple<ElementAOptionalTuple>::value && !cute_rt_tm::is_tuple<ElementBOptionalTuple>::value;
  static constexpr bool IsANarrow = cute_rt_tm::sizeof_bits_v<ElementA> < cute_rt_tm::sizeof_bits_v<ElementB>;
  static constexpr bool IsMixedInput = cute_rt_tm::sizeof_bits_v<ElementA> != cute_rt_tm::sizeof_bits_v<ElementB>;
  static_assert(IsMixedInput, "Mixed Input GEMM Kernel doesn't support regular gemm.");

  static_assert((cute_rt_tm::is_tuple<ElementAOptionalTuple>::value ^ cute_rt_tm::is_tuple<ElementBOptionalTuple>::value ||
               (NeitherIsTuple && (cute_rt_tm::sizeof_bits<ElementA>::value != cute_rt_tm::sizeof_bits<ElementB>::value))),
    "Either A OR B must be a tuple or the widths of A and B must be different.");
  using ElementPairA = cute_rt_tm::conditional_t<IsMixedInput && IsANarrow && NeitherIsTuple, cute_rt_tm::tuple<ElementA>, ElementAOptionalTuple>;
  using ElementPairB = cute_rt_tm::conditional_t<IsMixedInput && !IsANarrow && NeitherIsTuple, cute_rt_tm::tuple<ElementB>, ElementBOptionalTuple>;
  static constexpr bool IsATransformed = cute_rt_tm::is_tuple<ElementPairA>::value;
  static_assert(IsATransformed, "A matrix should be transformed.");

  // For fp32 types, map to tf32 MMA value type.
  using ElementMma = cute_rt_tm::conditional_t<cute_rt_tm::is_same_v<ElementB, float>, tfloat32_t, ElementB>;


  using ElementAMma = ElementMma;
  using ElementBMma = ElementMma;
  
  static constexpr int IsSubbyteA = cute_rt_tm::sizeof_bits_v<ElementA> < 8;
  using TmaElementA = cute_rt_tm::conditional_t<IsSubbyteA, uint8_t, ElementA>;

  static constexpr int ScalingFactor = 1;

  using TiledMma = decltype(detail::sm100_make_trivial_mixed_input_tiled_mma<ElementAMma, ElementB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, KernelScheduleType>());
  using AtomThrID = typename TiledMma::AtomThrID;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;
  using CtaTileShape_MNK = decltype(shape_div(TileShape_MNK{}, AtomThrShapeMNK{}));

  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute_rt_tm::size<0>(TileShape_MNK{}),
                                                                         cute_rt_tm::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute_rt_tm::size<1>(TileShape_MNK{}),
                                                                         cute_rt_tm::size<2>(TileShape_MNK{}))));

  using BlockTileA_M = decltype(cute_rt_tm::size<0,0>(MmaShapeA_MK{}) * cute_rt_tm::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute_rt_tm::size<0,1>(MmaShapeA_MK{}) * cute_rt_tm::size<2>(MmaShapeA_MK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(cute_rt_tm::size<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm100_cluster_shape_to_tma_atom_B(ClusterShape_MNK{}, AtomThrID{}));

  // Input transform kernel can not use TMA 2SM instructions.
  using SmemLayoutAtomA = decltype(cutlass_rt_tm::gemm::collective::detail::sm100_smem_selector<UmmaMajorA, ElementA,
    BlockTileA_M, BlockTileA_K>());
  using SmemLayoutAtomACompute =  decltype(cutlass_rt_tm::gemm::collective::detail::sm100_smem_selector<UmmaMajorA, ElementAMma, BlockTileA_M, BlockTileA_K>());
  using SmemLayoutAtomPairA = cutlass_rt_tm::gemm::collective::detail::CollectiveMmaEmulatedLayoutAtomType<
    SmemLayoutAtomA, SmemLayoutAtomACompute>;
  static constexpr int MMA_M = cute_rt_tm::size<0,0>(MmaShapeA_MK{});
  using CopyAtomPairA = cutlass_rt_tm::gemm::collective::detail::CollectiveMmaEmulatedCopyType<
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementA>,
    cute_rt_tm::conditional_t<(UmmaMajorA == cute_rt_tm::UMMA::Major::K && !cute_rt_tm::is_base_of_v<KernelTmaWarpSpecializedMixedInputSmemSm100, KernelScheduleType>),
                        cute_rt_tm::conditional_t<(MMA_M == 64 && size(AtomThrID{}) == 1), SM100_TMEM_STORE_16dp256b1x, SM100_TMEM_STORE_32dp32b8x>, // TS Implementation
                        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementA>>                                         // SS Implementation
  >;

  using BlockTileB_N = decltype(cute_rt_tm::size<0,0>(MmaShapeB_NK{}) * cute_rt_tm::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute_rt_tm::size<0,1>(MmaShapeB_NK{}) * cute_rt_tm::size<2>(MmaShapeB_NK{}));
  
  // Input transform kernel can not use TMA 2SM instructions.
  using SmemLayoutAtomB = decltype(cutlass_rt_tm::gemm::collective::detail::sm100_smem_selector<UmmaMajorB, ElementB,
    BlockTileB_N, BlockTileB_K>());
  using SmemLayoutAtomBCompute = decltype(cutlass_rt_tm::gemm::collective::detail::sm100_smem_selector<UmmaMajorB, ElementBMma, BlockTileB_N, BlockTileB_K>());
  using SmemLayoutAtomPairB = cutlass_rt_tm::gemm::collective::detail::CollectiveMmaEmulatedLayoutAtomType<
    SmemLayoutAtomB, SmemLayoutAtomBCompute>;
  using CopyAtomPairB = cutlass_rt_tm::gemm::collective::detail::CollectiveMmaEmulatedCopyType<
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementB>, 
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementMma>
  >;

  //Creating the stride of Transformed Input
  using StrideA = cutlass_rt_tm::gemm::TagToStrideA_t<GmemLayoutATag>;
  using LayoutScale = cutlass_rt_tm::gemm::TagToStrideA_t<GmemLayoutScaleTag>;
  
  using VoidShapeScale = Shape<Shape<Int<128>, _1>, Shape<Int<64>, _1>, _1>; //Dummy Value to create a dummy ScaleConfig
  using VoidStrideScale = Stride<Stride<_0,_1>,Stride<_0, _1>, _1>;
  using VoidLayoutScale = Layout<VoidShapeScale, VoidStrideScale>;
 
  using NonVoidLayoutScale = cute_rt_tm::conditional_t<
    cute_rt_tm::is_void_v<LayoutScale>, VoidLayoutScale, LayoutScale>;
  
  using StridePairA = decltype(cute_rt_tm::make_tuple(StrideA{}, NonVoidLayoutScale{}));

  // SmemCarveout
  static constexpr int SchedulerPipelineStageCount = 3;
  static constexpr bool IsArrayOfPointersGemm = (cute_rt_tm::is_base_of_v<KernelScheduleSm100PtrArrayFastFP32Gemm, KernelScheduleType>);

  // CLCPipeline = PipelineCLCFetchAsync
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass_rt_tm::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage);
  // CLC (scheduler) response
  static constexpr auto CLCResponseStorage = SchedulerPipelineStageCount * detail::CLCResponseSize;
  // CLC Throttle pipeline storage
  static constexpr auto CLCThrottlePipelineStorage = sizeof(typename cutlass_rt_tm::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage);
  // Tmem dealloc
  static constexpr auto TmemDeallocStorage = sizeof(cutlass_rt_tm::arch::ClusterBarrier);
  // Tmem ptr storage
  static constexpr auto TmemBasePtrsStorage = sizeof(uint32_t);
  // Tensormap Storage
  static constexpr size_t TensorMapStorage = IsArrayOfPointersGemm ? sizeof(cute_rt_tm::TmaDescriptor) * 2 /* for A and B */ : 0;

  // Smem usage that's not part of CollectiveEpilogue::SharedStorage & CollectiveMainloop::SharedStorage
  static constexpr auto KernelSmemCarveout = static_cast<int>( CLCPipelineStorage +
                                                               CLCResponseStorage +
                                                               CLCThrottlePipelineStorage +
                                                               TmemDeallocStorage +
                                                               TmemBasePtrsStorage +
                                                               TensorMapStorage);

  // Reduce SMEM capacity available for buffers considering extra B smem and barrier smem allocations
  static constexpr int Sm100ReducedSmemCapacityBytes = detail::sm100_smem_capacity_bytes - KernelSmemCarveout;

  static constexpr int ScaleGranularityK = get_ScaleGranularityK<LayoutScale>();

  static constexpr auto stage_info = cutlass_rt_tm::gemm::collective::detail::sm100_compute_stage_count_or_override_mixed_input<
      Sm100ReducedSmemCapacityBytes, TmaElementA, ElementAMma, ElementScale, ElementZero, ElementB, CtaTileShape_MNK, TiledMma, KernelScheduleType, UmmaMajorA, ScaleGranularityK>(StageCountType{});
  
  static constexpr int Load2TransformPipelineStageCount = get<0>(stage_info);
  static constexpr int Transform2MmaPipelineStageCount = get<1>(stage_info);
  static constexpr int AccumulatorPipelineStageCount = get<2>(stage_info);

  static_assert(!IsArrayOfPointersGemm, "mixed input does not support grouped gemm on Blackwell");

  using DispatchPolicy = cutlass_rt_tm::gemm::MainloopSm100TmaUmmaWarpSpecializedMixedInput<
    Load2TransformPipelineStageCount,
    Transform2MmaPipelineStageCount,
    SchedulerPipelineStageCount,
    AccumulatorPipelineStageCount,
    ClusterShape_MNK
  >;
  using CollectiveOp = cutlass_rt_tm::gemm::collective::CollectiveMma<
    DispatchPolicy,
    TileShape_MNK,
    ElementPairA,
    StridePairA,
    ElementPairB,
    cutlass_rt_tm::gemm::TagToStrideB_t<GmemLayoutBTag>,
    TiledMma,
    GmemTiledCopyA,
    SmemLayoutAtomPairA,
    CopyAtomPairA,
    cute_rt_tm::identity,
    GmemTiledCopyB,
    SmemLayoutAtomPairB,
    CopyAtomPairB,
    cute_rt_tm::identity
  >;
};

} // namespace cutlass_rt_tm::gemm::collective
