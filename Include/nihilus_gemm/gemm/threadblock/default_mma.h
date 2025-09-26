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
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "nihilus_gemm/nihilus_gemm.h"

#include "nihilus_gemm/arch/arch.h"
#include "nihilus_gemm/arch/wmma.h"

#include "nihilus_gemm/layout/matrix.h"
#include "nihilus_gemm/layout/permute.h"
#include "nihilus_gemm/transform/threadblock/predicated_tile_iterator.h"
#include "nihilus_gemm/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"

#include "nihilus_gemm/gemm/gemm.h"
#include "nihilus_gemm/gemm/threadblock/default_mma_core_simt.h"
#include "nihilus_gemm/gemm/threadblock/default_mma_core_sm70.h"
#include "nihilus_gemm/gemm/threadblock/default_mma_core_sm75.h"
#include "nihilus_gemm/gemm/threadblock/default_mma_core_sm80.h"

#if defined(NIHILUS_ARCH_WMMA_ENABLED)
	#include "nihilus_gemm/gemm/threadblock/default_mma_core_wmma.h"
#endif//NIHILUS_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////

namespace nihilus_gemm {
	namespace gemm {
		namespace threadblock {

			////////////////////////////////////////////////////////////////////////////////

			template<
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
				/// Element type for internal accumulation
				typename ElementAccumulator_,
				/// Layout type for C and D matrix operands
				typename LayoutC_,
				/// Operator class tag
				typename OperatorClass_,
				/// Tag indicating architecture to tune for
				typename ArchTag_,
				/// Threadblock-level tile size (concept: GemmShape)
				typename ThreadblockShape_,
				/// Warp-level tile size (concept: GemmShape)
				typename WarpShape_,
				/// Instruction-level tile size (concept: GemmShape)
				typename InstructionShape_,
				/// Number of stages used in the pipelined mainloop
				int Stages,
				/// Operation performed by GEMM
				typename Operator,
				/// Store the accumulators in row major or column major.  Row major is used
				/// when output layout is interleaved.
				bool AccumulatorsInRowMajor = false,
				/// Use zfill or predicate for out-of-bound cp.async
				SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
				/// Gather operand A by using an index array
				bool GatherA = false,
				/// Gather operand B by using an index array
				bool GatherB = false,
				/// Permute operand A
				typename PermuteALayout = layout::NoPermute,
				/// Permute operand B
				typename PermuteBLayout = layout::NoPermute>
			struct DefaultMma;

			////////////////////////////////////////////////////////////////////////////////

			/// Specialization for row-major output (OperatorClass Simt)
			template<
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
				/// Access granularity of B matrix in units of elements
				int kAlignmentB,
				/// Element type for internal accumulation
				typename ElementAccumulator,
				/// Layout type for C and D matrix operand
				typename LayoutC,
				/// Tag indicating architecture to tune for
				typename ArchTag,
				/// Threadblock-level tile size (concept: GemmShape)
				typename ThreadblockShape,
				/// Warp-level tile size (concept: GemmShape)
				typename WarpShape,
				/// Instruction-level tile size (concept: GemmShape)
				typename InstructionShape,
				/// Operation performed by GEMM
				typename Operator,
				/// Gather operand A by using an index array
				bool GatherA,
				/// Gather operand B by using an index array
				bool GatherB,
				/// Permute operand A
				typename PermuteALayout,
				/// Permute operand B
				typename PermuteBLayout>
			struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator, LayoutC, arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
				InstructionShape, 2, Operator, false, SharedMemoryClearOption::kNone, GatherA, GatherB, PermuteALayout, PermuteBLayout> {
				static_assert(platform::is_same<LayoutC, layout::RowMajor>::value || platform::is_same<LayoutC, layout::AffineRankN<2>>::value, "simt epilogue must be row major");

				// Define the MmaCore components
				using MmaCore = typename nihilus_gemm::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB,
					ElementAccumulator, LayoutC, arch::OpClassSimt, 2, Operator>;

				// Define iterators over tiles from the A operand
				using IteratorA = nihilus_gemm::transform::threadblock::PredicatedTileIterator<nihilus_gemm::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, ElementA, LayoutA,
					1, typename MmaCore::IteratorThreadMapA, kAlignmentA, GatherA, PermuteALayout>;

				// Define iterators over tiles from the B operand
				using IteratorB = nihilus_gemm::transform::threadblock::PredicatedTileIterator<nihilus_gemm::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, ElementB, LayoutB,
					0, typename MmaCore::IteratorThreadMapB, kAlignmentB, GatherB, PermuteBLayout>;

				// Define the threadblock-scoped pipelined matrix multiply
				using ThreadblockMma = nihilus_gemm::gemm::threadblock::MmaPipelined<typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA, IteratorB,
					typename MmaCore::SmemIteratorB, ElementAccumulator, LayoutC, typename MmaCore::MmaPolicy>;
			};

		}// namespace threadblock
	}// namespace gemm
}// namespace nihilus_gemm

////////////////////////////////////////////////////////////////////////////////
