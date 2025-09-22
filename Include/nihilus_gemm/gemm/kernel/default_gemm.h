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
    \brief 
      Default kernel-level GEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.
  
      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.
*/

#pragma once

#include "nihilus_gemm/cutlass.h"

#include "nihilus_gemm/layout/matrix.h"
#include "nihilus_gemm/numeric_types.h"
#include "nihilus_gemm/arch/wmma.h"

#include "nihilus_gemm/epilogue/threadblock/epilogue.h"
#include "nihilus_gemm/epilogue/thread/linear_combination.h"

#include "nihilus_gemm/gemm/gemm.h"
#include "nihilus_gemm/gemm/kernel/gemm.h"
#include "nihilus_gemm/gemm/kernel/gemm_pipelined.h"
#include "nihilus_gemm/gemm/threadblock/default_mma_core_sm80.h"
#include "nihilus_gemm/gemm/threadblock/default_mma.h"
#include "nihilus_gemm/gemm/threadblock/default_mma_core_simt.h"
#include "nihilus_gemm/gemm/threadblock/threadblock_swizzle.h"

#include "nihilus_gemm/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "nihilus_gemm/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "nihilus_gemm/epilogue/threadblock/default_epilogue_simt.h"
#include "nihilus_gemm/transform/threadblock/predicated_tile_iterator.h"

#include "nihilus_gemm/layout/permute.h"

#if defined(CUTLASS_RT_TM_ARCH_WMMA_ENABLED)
	#include "nihilus_gemm/epilogue/threadblock/default_epilogue_wmma_tensor_op.h"
#endif//CUTLASS_RT_TM_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////

namespace nihilus_gemm {
	namespace gemm {
		namespace kernel {

			////////////////////////////////////////////////////////////////////////////////

			template<uint64_t M, uint64_t K,
				/// Element type for A matrix operand
				typename ElementA_,
				/// Layout type for A matrix operand
				typename LayoutA_,
				/// Access granularity of A matrix in units of elements
				int kAlignmentA,
				/// Element type for B matrix operand
				typename ElementB_,
				/// Layout type for B matrix operand
				typename LayoutB_,
				/// Access granularity of B matrix in units of elements
				int kAlignmentB,
				/// Element type for C and D matrix operands
				typename ElementC_,
				/// Layout type for C and D matrix operands
				typename LayoutC_,
				/// Element type for internal accumulation
				typename ElementAccumulator,
				/// Operator class tag
				typename OperatorClass,
				/// Tag indicating architecture to tune for
				typename ArchTag,
				/// Threadblock-level tile size (concept: GemmShape)
				typename ThreadblockShape,
				/// Warp-level tile size (concept: GemmShape)
				typename WarpShape,
				/// Warp-level tile size (concept: GemmShape)
				typename InstructionShape,
				/// Epilogue output operator
				typename EpilogueOutputOp,
				/// Threadblock-level swizzling operator
				typename ThreadblockSwizzle,
				/// Number of stages used in the pipelined mainloop
				int Stages,
				/// If true, kernel is configured to support serial reduction in the
				/// epilogue
				bool SplitKSerial,
				/// Operation performed by GEMM
				typename Operator,
				/// Use zfill or predicate for out-of-bound cp.async
				SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
				/// Gather operand A by using an index array
				bool gather_a = false,
				/// Gather operand B by using an index array
				bool gather_b = false,
				/// Scatter result D by using an index array
				bool ScatterD = false,
				/// Permute result D
				typename PermuteDLayout = layout::NoPermute,
				/// Permute operand A
				typename PermuteALayout = layout::NoPermute,
				/// Permute operand B
				typename PermuteBLayout = layout::NoPermute,
				///
				typename Enable = void>
			struct DefaultGemm;

			////////////////////////////////////////////////////////////////////////////////

			/// Partial specialization for Ada Architecture
			template<uint64_t M, uint64_t K,
				/// Element type for A matrix operand
				typename ElementA,
				/// Layout type for A matrix operand
				typename LayoutA,
				/// Access granularity of A matrix in units of elements
				int kAlignmentA,
				/// Element type for B matrix operand
				typename ElementB,
				/// Layout type for B matrix operand
				typename LayoutB,
				/// Access granularity of A matrix in units of elements
				int kAlignmentB,
				/// Element type for C and D matrix operands
				typename ElementC,
				/// Element type for internal accumulation
				typename ElementAccumulator,
				/// Threadblock-level tile size (concept: GemmShape)
				typename ThreadblockShape,
				/// Warp-level tile size (concept: GemmShape)
				typename WarpShape,
				/// Warp-level tile size (concept: GemmShape)
				typename InstructionShape,
				/// Epilogue output operator
				typename EpilogueOutputOp,
				/// Threadblock-level swizzling operator
				typename ThreadblockSwizzle,
				/// Number of stages used in the pipelined mainloop
				int Stages,
				/// If true, kernel is configured to support serial reduction in the
				/// epilogue
				bool SplitKSerial,
				/// Operation performed by GEMM
				typename Operator,
				/// Use zfill or predicate for out-of-bound cp.async
				SharedMemoryClearOption SharedMemoryClear,
				/// Gather operand A by using an index array
				bool gather_a,
				/// Gather operand B by using an index array
				bool gather_b,
				/// Scatter result D by using an index array
				bool ScatterD,
				/// Permute result D
				typename PermuteDLayout,
				/// Permute operand A
				typename PermuteALayout,
				/// Permute operand B
				typename PermuteBLayout>
			struct DefaultGemm<M, K, ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC, layout::RowMajor, ElementAccumulator, arch::OpClassTensorOp,
				arch::Sm89, ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages, SplitKSerial, Operator, SharedMemoryClear, gather_a,
				gather_b, ScatterD, PermuteDLayout, PermuteALayout, PermuteBLayout> {
				/// Define the threadblock-scoped matrix multiply-accumulate
				using Mma = typename nihilus_gemm::gemm::threadblock::DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator,
					layout::RowMajor, arch::OpClassTensorOp, arch::Sm89, ThreadblockShape, WarpShape, InstructionShape, Stages, Operator, false, SharedMemoryClear, gather_a,
					gather_b, PermuteALayout, PermuteBLayout>::ThreadblockMma;

				static constexpr int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

				/// Define the epilogue
				using Epilogue = typename nihilus_gemm::epilogue::threadblock::DefaultEpilogueTensorOp<ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
					EpilogueOutputOp::kCount, ScatterD, PermuteDLayout>::Epilogue;

				/// Define the kernel-level GEMM operator.
				using GemmKernel = kernel::Gemm<M, K, Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
			};
			////////////////////////////////////////////////////////////////////////////////

			/// Partial specialization for SIMT
			template<uint64_t M, uint64_t K,
				/// Element type for A matrix operand
				typename ElementA,
				/// Layout type for A matrix operand
				typename LayoutA,
				/// Access granularity of A matrix in units of elements
				int kAlignmentA,
				/// Element type for B matrix operand
				typename ElementB,
				/// Layout type for B matrix operand
				typename LayoutB,
				/// Access granularity of A matrix in units of elements
				int kAlignmentB,
				/// Element type for C and D matrix operands
				typename ElementC,
				/// Layout type for C and D matrix operand
				typename LayoutC,
				/// Element type for internal accumulation
				typename ElementAccumulator,
				/// Tag indicating architecture to tune for
				typename ArchTag,
				/// Threadblock-level tile size (concept: GemmShape)
				typename ThreadblockShape,
				/// Warp-level tile size (concept: GemmShape)
				typename WarpShape,
				/// Epilogue output operator
				typename EpilogueOutputOp,
				/// Threadblock-level swizzling operator
				typename ThreadblockSwizzle,
				/// If true, kernel is configured to support serial reduction in the epilogue
				bool SplitKSerial,
				/// Operation performed by GEMM
				typename Operator,
				/// Use zfill or predicate for out-of-bound cp.async
				SharedMemoryClearOption SharedMemoryClear,
				/// Gather operand A by using an index array
				bool gather_a,
				/// Gather operand B by using an index array
				bool gather_b,
				/// Scatter result D by using an index array
				bool ScatterD,
				/// Permute result D
				typename PermuteDLayout,
				/// Permute operand A
				typename PermuteALayout,
				/// Permute operand B
				typename PermuteBLayout>
			struct DefaultGemm<M, K, ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC, LayoutC, ElementAccumulator, arch::OpClassSimt, ArchTag,
				ThreadblockShape, WarpShape, GemmShape<1, 1, 1>, EpilogueOutputOp, ThreadblockSwizzle, 2, SplitKSerial, Operator, SharedMemoryClear, gather_a, gather_b, ScatterD,
				PermuteDLayout, PermuteALayout, PermuteBLayout, typename platform::enable_if<!platform::is_same<ArchTag, arch::Sm80>::value>::type> {
				static_assert((platform::is_same<LayoutC, layout::RowMajor>::value || platform::is_same<LayoutC, layout::AffineRankN<2>>::value),
					"Epilogue in the kernel level must be row major");

				/// Define the threadblock-scoped matrix multiply-accumulate
				using Mma = typename nihilus_gemm::gemm::threadblock::DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator, LayoutC,
					arch::OpClassSimt, arch::Sm50, ThreadblockShape, WarpShape, GemmShape<1, 1, 1>, 2, Operator, false, SharedMemoryClear, gather_a, gather_b, PermuteALayout,
					PermuteBLayout>::ThreadblockMma;

				static constexpr int kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
				static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

				/// Define the epilogue
				using RegularEpilogue = typename nihilus_gemm::epilogue::threadblock::DefaultEpilogueSimt<ThreadblockShape, typename Mma::Operator, EpilogueOutputOp,
					kEpilogueElementsPerAccess, ScatterD, PermuteDLayout>::Epilogue;

				using Affine2Epilogue = typename nihilus_gemm::epilogue::threadblock::DefaultEpilogueSimtAffineRankN<2, ThreadblockShape, typename Mma::Operator, EpilogueOutputOp,
					kEpilogueElementsPerAccess>::Epilogue;

				using Epilogue = typename platform::conditional<platform::is_same<LayoutC, layout::RowMajor>::value, RegularEpilogue, Affine2Epilogue>::type;

				/// Define the kernel-level GEMM operator.
				using GemmKernel = kernel::Gemm<M, K, Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
			};


			template<uint64_t M, uint64_t K,
				/// Layout type for A matrix operand
				typename LayoutA,
				/// Access granularity of A matrix in units of elements
				int kAlignmentA,
				/// Element type for B matrix operand
				typename ElementB,
				/// Layout type for B matrix operand
				typename LayoutB,
				/// Access granularity of A matrix in units of elements
				int kAlignmentB,
				/// Element type for C and D matrix operands
				typename ElementC,
				/// Layout type for C and D matrix operand
				typename LayoutC,
				/// Element type for internal accumulation
				typename ElementAccumulator,
				/// Tag indicating architecture to tune for
				typename ArchTag,
				/// Threadblock-level tile size (concept: GemmShape)
				typename ThreadblockShape,
				/// Warp-level tile size (concept: GemmShape)
				typename WarpShape,
				/// Epilogue output operator
				typename EpilogueOutputOp,
				/// Threadblock-level swizzling operator
				typename ThreadblockSwizzle,
				/// If true, kernel is configured to support serial reduction in the epilogue
				bool SplitKSerial,
				/// Operation performed by GEMM
				typename Operator,
				/// Use zfill or predicate for out-of-bound cp.async
				SharedMemoryClearOption SharedMemoryClear,
				/// Gather operand A by using an index array
				bool gather_a,
				/// Gather operand B by using an index array
				bool gather_b,
				/// Scatter result D by using an index array
				bool ScatterD,
				/// Permute result D
				typename PermuteDLayout,
				/// Permute operand A
				typename PermuteALayout,
				/// Permute operand B
				typename PermuteBLayout>
			struct DefaultGemm<M, K, block_q8_0, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC, LayoutC, ElementAccumulator, arch::OpClassSimt, ArchTag,
				ThreadblockShape, WarpShape, GemmShape<1, 1, 1>, EpilogueOutputOp, ThreadblockSwizzle, 2, SplitKSerial, Operator, SharedMemoryClear, gather_a, gather_b, ScatterD,
				PermuteDLayout, PermuteALayout, PermuteBLayout, typename platform::enable_if<!platform::is_same<ArchTag, arch::Sm80>::value>::type> {
				static_assert((platform::is_same<LayoutC, layout::RowMajor>::value || platform::is_same<LayoutC, layout::AffineRankN<2>>::value),
					"Epilogue in the kernel level must be row major");

				/// Define the threadblock-scoped matrix multiply-accumulate
				using Mma = typename nihilus_gemm::gemm::threadblock::DefaultMma<block_q8_0, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator, LayoutC,
					arch::OpClassSimt, arch::Sm50, ThreadblockShape, WarpShape, GemmShape<1, 1, 1>, 2, Operator, false, SharedMemoryClear, gather_a, gather_b, PermuteALayout,
					PermuteBLayout>::ThreadblockMma;

				static constexpr int kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
				static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

				/// Define the epilogue
				using RegularEpilogue = typename nihilus_gemm::epilogue::threadblock::DefaultEpilogueSimt<ThreadblockShape, typename Mma::Operator, EpilogueOutputOp,
					kEpilogueElementsPerAccess, ScatterD, PermuteDLayout>::Epilogue;

				using Affine2Epilogue = typename nihilus_gemm::epilogue::threadblock::DefaultEpilogueSimtAffineRankN<2, ThreadblockShape, typename Mma::Operator, EpilogueOutputOp,
					kEpilogueElementsPerAccess>::Epilogue;

				using Epilogue = typename platform::conditional<platform::is_same<LayoutC, layout::RowMajor>::value, RegularEpilogue, Affine2Epilogue>::type;

				/// Define the kernel-level GEMM operator.
				using GemmKernel = kernel::Gemm<M, K, Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
			};


			////////////////////////////////////////////////////////////////////////////////

		}// namespace kernel
	}// namespace gemm
}// namespace nihilus_gemm
