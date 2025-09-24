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
//

//

#include "cutlass_old/gemm/collective/builders/sm100_common.inl"
#include "cutlass_old/gemm/collective/builders/sm100_pipeline_carveout.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_old::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class TileShapeMNK,
  class MainloopPipelineStorage,
  int stages
>
constexpr int
sm100_compute_stage_count_or_override(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class TileShapeMNK,
  class MainloopPipelineStorage,
  int stages
>
constexpr int
sm100_compute_stage_count_or_override(cute_old::Int<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class TileShapeMNK,
  class MainloopPipelineStorage,
  int carveout_bytes>
constexpr int
sm100_compute_stage_count_or_override(StageCountAutoCarveout<carveout_bytes> stage_count) {
  // For F8/F6/F4 sub-bytes, ElementA/B will be passed in as uint8_t
  // For Planar Complex, ElementA/B will be passed in as cutlass_old::complex<ElementARaw>
  // Each stage include (CollectiveMma::SharedStorage)
  // 1. smem for A and smem for B (CollectiveMma::SharedStorage::TensorStorage)
  // 2. one MainloopPipeline = (CollectiveMma::SharedStorage::PipelineStorage = PipelineTmaUmmaAsync)
  constexpr auto mainloop_pipeline_bytes = sizeof(MainloopPipelineStorage);
  constexpr auto a_bits = cute_old::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute_old::sizeof_bits_v<ElementB>;
  constexpr int stage_bytes =
    cutlass_old::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass_old::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    static_cast<int>(mainloop_pipeline_bytes);

  return (CapacityBytes - carveout_bytes) / stage_bytes;
}

template <class ElementA, class ElementB>
CUTLASS_HOST_DEVICE
static constexpr bool
check_input_datatypes() {
  auto is_non_f4f6f8_input = [&]() {
    return (cute_old::is_same_v<ElementA, cutlass_old::tfloat32_t> ||
            cute_old::is_same_v<ElementA,               float> ||
            cute_old::is_same_v<ElementA,     cutlass_old::half_t> ||
            cute_old::is_same_v<ElementA, cutlass_old::bfloat16_t> ||
            cute_old::is_same_v<ElementA,  int8_t> ||
            cute_old::is_same_v<ElementA, uint8_t>) &&
           (cute_old::is_same_v<ElementA, ElementB>); // For all MMA instrs except F4F6F8, A and B types should be the same.
  };
  auto is_f4f6f8_input = [&]() {
    // Allowed input element datatype for narrow precision GEMM
    return (
            (
              cute_old::is_same_v<ElementA, cutlass_old::type_erased_dynamic_float8_t> ||
              cute_old::is_same_v<ElementA, cutlass_old::type_erased_dynamic_float6_t> ||
              cute_old::is_same_v<ElementA, cutlass_old::type_erased_dynamic_float4_t>
            ) &&
            (
              cute_old::is_same_v<ElementB, cutlass_old::type_erased_dynamic_float8_t> ||
              cute_old::is_same_v<ElementB, cutlass_old::type_erased_dynamic_float6_t> ||
              cute_old::is_same_v<ElementB, cutlass_old::type_erased_dynamic_float4_t>
            )
           ) || 
           (
            (
              cute_old::is_same_v<ElementA, cutlass_old::float_e2m1_t> ||
              cute_old::is_same_v<ElementA, cutlass_old::float_e2m3_t> ||
              cute_old::is_same_v<ElementA, cutlass_old::float_e3m2_t> ||
              cute_old::is_same_v<ElementA, cutlass_old::float_e4m3_t> ||
              cute_old::is_same_v<ElementA, cutlass_old::float_e5m2_t>
            ) &&
            ( 
              cute_old::is_same_v<ElementB, cutlass_old::float_e2m1_t> ||
              cute_old::is_same_v<ElementB, cutlass_old::float_e2m3_t> ||
              cute_old::is_same_v<ElementB, cutlass_old::float_e3m2_t> ||
              cute_old::is_same_v<ElementB, cutlass_old::float_e4m3_t> ||
              cute_old::is_same_v<ElementB, cutlass_old::float_e5m2_t>
            )
           );
  };

  static_assert(is_f4f6f8_input() || is_non_f4f6f8_input(), "Unsupported data type for ElementA");

  return true;
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class BuilderScheduleTag
>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,    // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    ClusterShape_MNK, // Static cluster shape or dynamic (int, int, _1)
    StageCountType,
    BuilderScheduleTag,
    cute_old::enable_if_t<
      not cute_old::is_tuple_v<ElementA>   && not cute_old::is_tuple_v<ElementB> &&
      not cute_old::is_complex_v<ElementA> && not cute_old::is_complex_v<ElementB> &&
      // Dense Gemm / PtrArrayDenseGemm
      (
       (not cute_old::is_same_v<KernelMixedTmaCpAsyncWarpSpecialized1SmSm100, BuilderScheduleTag>) && 
       (not cute_old::is_same_v<KernelWarpSpecialized1SmSm100, BuilderScheduleTag>) && 
       (cute_old::is_base_of_v<KernelScheduleSm100DenseGemm, BuilderScheduleTag> ||
        cute_old::is_same_v<KernelScheduleAuto, BuilderScheduleTag>)) &&
      // Alignment check
      detail::sm1xx_gemm_is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, BuilderScheduleTag>()>>
{
  static_assert(cute_old::is_static_v<TileShape_MNK>, "TileShape has to be static");
  static_assert(detail::check_input_datatypes<ElementA, ElementB>(), "Incorrect input types");

  static constexpr cute_old::UMMA::Major UmmaMajorA = cutlass_old::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute_old::UMMA::Major UmmaMajorB = cutlass_old::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();

  // Data type used by MMA instruction
  using ElementAMma = decltype(cutlass_old::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA>());
  using ElementBMma = decltype(cutlass_old::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB>());

  static constexpr bool is_2sm = cute_old::is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag> ||
                        (not cute_old::is_base_of_v<KernelSchedule1Sm, BuilderScheduleTag> &&
                          not cute_old::is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag> &&
                          cute_old::is_static_v<ClusterShape_MNK> &&
                          cute_old::get<0>(ClusterShape_MNK{}) % 2 == 0 );

  static_assert(detail::sm1xx_gemm_check_for_f8f6f4_mix8bit_requirement<ElementAMma, ElementBMma,
                                                                        TileShape_MNK, ClusterShape_MNK,
                                                                        GmemLayoutATag, GmemLayoutBTag, false /*is_sparse*/, is_2sm>(),
                "TileSize and MNK Major does not met with MMA Mix 8-bit TMA load requirement" );

  using TiledMma = decltype(detail::sm100_make_trivial_tiled_mma<
      ElementAMma, ElementBMma, ElementAccumulator,
      decltype(cute_old::product_each(TileShape_MNK{})), ClusterShape_MNK,
      UmmaMajorA, UmmaMajorB, BuilderScheduleTag>());

  using ElementAMma_SmemAllocType = cute_old::conditional_t<cute_old::sizeof_bits_v<ElementAMma> < 8, uint8_t, ElementAMma>;
  using ElementBMma_SmemAllocType = cute_old::conditional_t<cute_old::sizeof_bits_v<ElementBMma> < 8, uint8_t, ElementBMma>;

  using AtomThrID = typename TiledMma::AtomThrID;

  using AtomThrShapeMNK = cute_old::Shape<decltype(cute_old::shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;
  using CtaTileShape_MNK = decltype(cute_old::shape_div(TileShape_MNK{}, AtomThrShapeMNK{}));

  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute_old::size<0>(TileShape_MNK{}),
                                                                         cute_old::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute_old::size<1>(TileShape_MNK{}),
                                                                         cute_old::size<2>(TileShape_MNK{}))));

  using BlockTileA_M = decltype(cute_old::size<0,0>(MmaShapeA_MK{}) * cute_old::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute_old::size<0,1>(MmaShapeA_MK{}) * cute_old::size<2>(MmaShapeA_MK{}));
  using BlockTileB_N = decltype(cute_old::size<0,0>(MmaShapeB_NK{}) * cute_old::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute_old::size<0,1>(MmaShapeB_NK{}) * cute_old::size<2>(MmaShapeB_NK{}));

  // Kludged right divide to divide TileShape_M/N by 1SM/2SM
  // Future work: fix partition_shape to account for hierarchies and
  // contiguity so we can pass BlockTileA/B to sm100_smem_selector instead
  using SmemShape_M = decltype(shape_div(shape<0>(TileShape_MNK{}), shape_div(shape<0>(TileShape_MNK{}), size<0>(TileShape_MNK{}) / size(AtomThrID{}))));
  using SmemShape_N = decltype(shape_div(shape<1>(TileShape_MNK{}), shape_div(shape<1>(TileShape_MNK{}), size<1>(TileShape_MNK{}) / size(AtomThrID{}))));
  using SmemShape_K = decltype(cute_old::get<2>(TileShape_MNK{}));

  using GmemTiledCopyA = decltype(cutlass_old::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(
    ClusterShape_MNK{}, AtomThrID{}));
  using GmemTiledCopyB = decltype(cutlass_old::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_B(
      ClusterShape_MNK{}, AtomThrID{}));

  using SmemLayoutAtomA = decltype(cutlass_old::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorA, ElementAMma_SmemAllocType, SmemShape_M, SmemShape_K>());
  using SmemLayoutAtomB = decltype(cutlass_old::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, ElementBMma_SmemAllocType, SmemShape_N, SmemShape_K>());
  static constexpr uint32_t TotalTmemRows = 128;
  static constexpr uint32_t Sm100TmemCapacityColumns = 512;
  static constexpr uint32_t TotalTmem = TotalTmemRows * Sm100TmemCapacityColumns;
  static constexpr uint32_t AccumulatorPipelineStageCount_ = (is_2sm || (!is_2sm && size(shape<0,0>(MmaShapeA_MK{}) > 64))) ? 
                                                              TotalTmem / (cute_old::size<0>(CtaTileShape_MNK{}) * cute_old::size<1>(CtaTileShape_MNK{}))
                                                            : (Sm100TmemCapacityColumns / cute_old::size<1>(CtaTileShape_MNK{})) * 2;                       // 1SM MMA_M = 64 case
  // 4 accumulator stages works well to buffer the accumulators, while also preventing overhead in the epilogue tail on small tile sizes.
  static constexpr uint32_t AccumulatorPipelineStageCount = cute_old::min(4u, AccumulatorPipelineStageCount_);                                              // Cap at 4 accumulator stages
  static_assert(AccumulatorPipelineStageCount > 0, "Accumulator pipeline stage count must be positive.  This error probably means that TileShape_MNK and/or TiledMma::ThrLayoutVMNK are wrong.");

  // Calculate scheduler pipeline stages. Having one more stage than the accumulator allows more latency hiding.
  using StrideA = cutlass_old::gemm::TagToStrideA_t<GmemLayoutATag>;
  using InternalStrideA  = cute_old::remove_pointer_t<StrideA>;
  static constexpr bool IsArrayOfPointersGemm = (cute_old::is_base_of_v<KernelScheduleSm100PtrArrayDenseGemm, BuilderScheduleTag>);
  // Grouped GEMM(where Stride type is Stride*) uses specific static tile scheduler.
  static constexpr bool IsGroupGemm = !cute_old::is_same_v<StrideA, InternalStrideA>;
  static constexpr uint32_t SchedulerPipelineStageCount = cute_old::conditional_return<IsGroupGemm>(8, 2);
  
  static constexpr uint32_t KernelSmemCarveout = detail::Sm100DenseGemmTmaUmmaCarveout<
      ClusterShape_MNK,
      AccumulatorPipelineStageCount,
      SchedulerPipelineStageCount,
      detail::CLCResponseSize,
      IsArrayOfPointersGemm
    >::KernelSmemCarveout;
  // Reduce SMEM capacity available for buffers considering barrier allocations.
  static constexpr int Sm100ReducedSmemCapacityBytes = cutlass_old::gemm::collective::detail::sm100_smem_capacity_bytes - KernelSmemCarveout;

  using SmemTileShape = cute_old::Shape<BlockTileA_M, BlockTileB_N, BlockTileA_K>;
  using MainloopPipelineStorage = typename cutlass_old::PipelineTmaUmmaAsync<1>::SharedStorage;

  static constexpr int PipelineStages = cutlass_old::gemm::collective::detail::sm100_compute_stage_count_or_override<
      Sm100ReducedSmemCapacityBytes, ElementAMma_SmemAllocType, ElementBMma_SmemAllocType, SmemTileShape, MainloopPipelineStorage>(StageCountType{});
  static_assert(PipelineStages > 0, "Smem usage is too high. Can't create any SMEM buffers for A, and B.");

  using DispatchPolicy = 
      cute_old::conditional_t<IsArrayOfPointersGemm, 
      cutlass_old::gemm::MainloopSm100ArrayTmaUmmaWarpSpecialized<
          PipelineStages,
          SchedulerPipelineStageCount,
          AccumulatorPipelineStageCount,
          ClusterShape_MNK
      >,
      cutlass_old::gemm::MainloopSm100TmaUmmaWarpSpecialized<
          PipelineStages,
          SchedulerPipelineStageCount,
          AccumulatorPipelineStageCount,
          ClusterShape_MNK
      >
    >;

  using CollectiveOp = cutlass_old::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      cutlass_old::gemm::TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      cutlass_old::gemm::TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      void,
      cute_old::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      void,
      cute_old::identity
    >;
};

} // namespace cutlass_old::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
