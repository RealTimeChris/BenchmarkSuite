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

#include "nihilus_gemm/cutlass.h"
#include "nihilus_gemm/numeric_types.h"
#include "nihilus_gemm/arch/arch.h"
#include "nihilus_gemm/arch/wmma.h"

#include "nihilus_gemm/layout/matrix.h"
#include "nihilus_gemm/layout/permute.h"
#include "nihilus_gemm/transform/threadblock/predicated_tile_iterator.h"
#include "nihilus_gemm/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"

#include "nihilus_gemm/gemm/gemm.h"
#include "nihilus_gemm/gemm/threadblock/default_mma_core_simt.h"
#include "nihilus_gemm/gemm/threadblock/default_mma_core_sm80.h"

#if defined(CUTLASS_RT_TM_ARCH_WMMA_ENABLED)
#include "nihilus_gemm/gemm/threadblock/default_mma_core_wmma.h"
#endif //CUTLASS_RT_TM_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////

namespace nihilus_gemm {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <
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
    bool gather_a = false,
    /// Gather operand B by using an index array
    bool gather_b = false,
    /// Permute operand A
    typename PermuteALayout = layout::NoPermute,
    /// Permute operand B
    typename PermuteBLayout = layout::NoPermute
    >
struct DefaultMma;

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass Simt)
template <
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
    bool gather_a,
    /// Gather operand B by using an index array
    bool gather_b,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
    >
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false, SharedMemoryClearOption::kNone,
                  gather_a, gather_b, PermuteALayout, PermuteBLayout> {

  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
             "simt epilogue must be row major");

  // Define the MmaCore components
  using MmaCore = typename nihilus_gemm::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      arch::OpClassSimt, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      nihilus_gemm::transform::threadblock::PredicatedTileIterator<
          nihilus_gemm::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA,
          gather_a, PermuteALayout>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      nihilus_gemm::transform::threadblock::PredicatedTileIterator<
          nihilus_gemm::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB,
          gather_b, PermuteBLayout>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = nihilus_gemm::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      LayoutC, typename MmaCore::MmaPolicy>;
};

// STEP 1: First, create a specialized iterator that handles block_q8_0 loading and dequantization

template<
	/// Threadblock-level tile size
	typename Shape_,
	/// Data type of A elements - float after dequantization
	typename Element_,
	/// Layout of A matrix
	typename Layout_,
	/// Advance rank
	int AdvanceRank,
	/// ThreadMap of iterator
	typename ThreadMap_,
	/// Access granularity of A matrix in units of elements
	int AccessSize,
	/// Gather indices
	bool Gather,
	/// Permute layout
	typename PermuteLayout>
class PredicatedTileIteratorQ8 {
  public:
	using Shape		= Shape_;
	using Element	= Element_;// This will be float (after dequantization)
	using Layout	= Layout_;
	using ThreadMap = ThreadMap_;

	// The actual storage type we read from memory
	using QuantizedElement = block_q8_0;

	// Fragment after dequantization
	using Fragment = nihilus_gemm::Array<Element, ThreadMap::kElementsPerAccess>;

  private:
	// Pointer to quantized data in global memory
	QuantizedElement const* quantized_pointer_;

	// Current position and bounds
	nihilus_gemm::MatrixCoord extent_;
	nihilus_gemm::MatrixCoord thread_offset_;

	// Thread-local dequantization cache to avoid redundant work
	mutable Element cache_[QuantizedElement::kBlockSize];
	mutable int cached_block_idx_;
	mutable bool cache_valid_;

  public:
	CUTLASS_RT_TM_HOST_DEVICE
	PredicatedTileIteratorQ8(QuantizedElement const* ptr, nihilus_gemm::MatrixCoord extent, int thread_idx,
		nihilus_gemm::MatrixCoord threadblock_offset = nihilus_gemm::MatrixCoord(0, 0))
		: quantized_pointer_(ptr), extent_(extent), cached_block_idx_(-1), cache_valid_(false) {
		// Calculate thread offset based on ThreadMap
		// This would need to be implemented based on your specific ThreadMap
		thread_offset_ = ThreadMap::initial_offset(thread_idx) + threadblock_offset;
	}

	/// Load a fragment with dequantization
	CUTLASS_RT_TM_HOST_DEVICE
	void load_with_pointer_offset(Fragment& frag, int pointer_offset) const {
		// Calculate global position
		int global_offset = thread_offset_.row() * extent_.column() + thread_offset_.column() + pointer_offset;

		// Convert to block coordinates
		int block_idx		   = global_offset / QuantizedElement::kBlockSize;
		int intra_block_offset = global_offset % QuantizedElement::kBlockSize;

		// Check if we need to dequantize a new block
		if (!cache_valid_ || cached_block_idx_ != block_idx) {
			// Load and dequantize the block
			const QuantizedElement& quant_block = quantized_pointer_[block_idx];

			// Dequantize: scale * int8_value
			float scale_f = __half2float(quant_block.scale);

#pragma unroll
			for (int i = 0; i < QuantizedElement::kBlockSize; ++i) {
				cache_[i] = scale_f * static_cast<float>(quant_block.quants[i]);
			}

			cached_block_idx_ = block_idx;
			cache_valid_	  = true;
		}

// Copy dequantized values to fragment
#pragma unroll
		for (int i = 0; i < Fragment::kElements; ++i) {
			int cache_idx = intra_block_offset + i;
			if (cache_idx < QuantizedElement::kBlockSize) {
				frag[i] = cache_[cache_idx];
			} else {
				// Handle cross-block access if needed
				// Load next block and dequantize
				int next_block_idx				   = block_idx + 1;
				const QuantizedElement& next_block = quantized_pointer_[next_block_idx];
				float next_scale				   = __half2float(next_block.scale);
				int next_offset					   = cache_idx - QuantizedElement::kBlockSize;
				frag[i]							   = next_scale * static_cast<float>(next_block.quants[next_offset]);
			}
		}
	}

	CUTLASS_RT_TM_HOST_DEVICE
	void load(Fragment& frag) const {
		load_with_pointer_offset(frag, 0);
	}

	/// Advance iterator
	CUTLASS_RT_TM_HOST_DEVICE
	PredicatedTileIteratorQ8& operator++() {
		thread_offset_ += ThreadMap::delta();
		return *this;
	}

	/// Add pointer offset
	CUTLASS_RT_TM_HOST_DEVICE
	PredicatedTileIteratorQ8& add_pointer_offset(int offset) {
		// Implementation depends on your memory layout
		return *this;
	}
};

// STEP 2: Create the DefaultMma specialization for block_q8_0
template<
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
	bool gather_a,
	/// Gather operand B by using an index array
	bool gather_b,
	/// Permute operand A
	typename PermuteALayout,
	/// Permute operand B
	typename PermuteBLayout>
struct DefaultMma<block_q8_0,// ElementA - quantized input
	LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator, LayoutC,
	arch::OpClassSimt,// OperatorClass
	ArchTag, ThreadblockShape, WarpShape, InstructionShape,
	2,// Stages
	Operator,
	false,// AccumulatorsInRowMajor
	SharedMemoryClearOption::kNone,// SharedMemoryClear
	gather_a, gather_b, PermuteALayout, PermuteBLayout> {
	static_assert(platform::is_same<LayoutC, layout::RowMajor>::value || platform::is_same<LayoutC, layout::AffineRankN<2>>::value, "simt epilogue must be row major");

	// KEY INSIGHT: Define MmaCore with FLOAT, not block_q8_0
	// The dequantization happens in the iterator, so downstream everything sees float
	using MmaCore = typename nihilus_gemm::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape,
		float,// Use float here, not block_q8_0!
		LayoutA, ElementB, LayoutB, ElementAccumulator, LayoutC, arch::OpClassSimt, 2, Operator>;

	// Use our custom iterator for A operand (handles dequantization)
	using IteratorA = PredicatedTileIteratorQ8<nihilus_gemm::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
		float,// Output type after dequantization
		LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA, gather_a, PermuteALayout>;

	// Standard iterator for B operand (no changes needed)
	using IteratorB = nihilus_gemm::transform::threadblock::PredicatedTileIterator<nihilus_gemm::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, ElementB, LayoutB, 0,
		typename MmaCore::IteratorThreadMapB, kAlignmentB, gather_b, PermuteBLayout>;

	// Standard ThreadblockMma (works with float data from our custom IteratorA)
	using ThreadblockMma = nihilus_gemm::gemm::threadblock::MmaPipelined<typename MmaCore::Shape,
		IteratorA,// Our custom quantized iterator
		typename MmaCore::SmemIteratorA, IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator, LayoutC, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp)
template <
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
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool gather_a,
    /// Gather operand B by using an index array
    bool gather_b,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
    >
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false, SharedMemoryClear,
                  gather_a, gather_b, PermuteALayout, PermuteBLayout> {
  // Define the MmaCore components
  using MmaCore = typename nihilus_gemm::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      arch::OpClassTensorOp, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      nihilus_gemm::transform::threadblock::PredicatedTileIterator<
          nihilus_gemm::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA,
          gather_a, PermuteALayout>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      nihilus_gemm::transform::threadblock::PredicatedTileIterator<
          nihilus_gemm::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB,
          gather_b, PermuteBLayout>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = nihilus_gemm::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major output (OperatorClass TensorOp)
template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
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
    bool gather_a,
    /// Gather operand B by using an index array
    bool gather_b,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
    >
struct DefaultMma<float, LayoutA, kAlignmentA, float, LayoutB,
                  kAlignmentB, float, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false, SharedMemoryClearOption::kNone,
                  gather_a, gather_b, PermuteALayout, PermuteBLayout> {
  // Define the MmaCore components
  using MmaCore = typename nihilus_gemm::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, float, LayoutA, float,
      LayoutB, float, layout::RowMajor, arch::OpClassTensorOp, 2,
      arch::OpMultiplyAddFastF16>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      nihilus_gemm::transform::threadblock::PredicatedTileIterator<
          nihilus_gemm::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          float, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA,
          gather_a, PermuteALayout>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      nihilus_gemm::transform::threadblock::PredicatedTileIterator<
          nihilus_gemm::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          float, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB,
          gather_b, PermuteBLayout>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = nihilus_gemm::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, float,
      layout::RowMajor, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for column-major-interleaved output
template <
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
    /// Tag indicating architecture to tune for
    typename OperatorClass,
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
    /// Number of Interleaved K
    int InterleavedK>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator,
                  layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass,
                  ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2,
                  Operator, true, SharedMemoryClearOption::kNone, false, false,
                  layout::NoPermute, layout::NoPermute> {
  // Define the MmaCore components
  using MmaCore = typename nihilus_gemm::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator,
      layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass, 2, Operator,
      true>;

  static_assert(kAlignmentA == 128 / sizeof_bits<ElementA>::value, 
    "Alignment must match thread data map's vector length");

  static_assert(kAlignmentB ==128 / sizeof_bits<ElementB>::value,
    "Alignment must match thread data map's vector length");

  // Define iterators over tiles from the A operand
  using IteratorA = nihilus_gemm::transform::threadblock::PredicatedTileIterator<
      nihilus_gemm::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, ElementA,
      LayoutA, 1, typename MmaCore::IteratorThreadMapA>;

  // Define iterators over tiles from the B operand
  using IteratorB = nihilus_gemm::transform::threadblock::PredicatedTileIterator<
      nihilus_gemm::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, ElementB,
      LayoutB, 0, typename MmaCore::IteratorThreadMapB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = nihilus_gemm::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::ColumnMajorInterleaved<InterleavedK>,
      typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output
template <
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
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Gather operand A by using an index array
    bool gather_a,
    /// Gather operand B by using an index array
    bool gather_b,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
    >
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, Stages, Operator, false, SharedMemoryClearOption::kNone,
                  gather_a, gather_b, PermuteALayout, PermuteBLayout> {

  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
             "simt epilogue must be row major");

  // Define the MmaCore components
  using MmaCore = typename nihilus_gemm::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC, arch::OpClassSimt,
      Stages, Operator>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = nihilus_gemm::Array<ElementA, kAlignmentA>;
  using IteratorA =
      nihilus_gemm::transform::threadblock::PredicatedTileAccessIterator<
          nihilus_gemm::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA, gather_a, PermuteALayout>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = nihilus_gemm::Array<ElementB, kAlignmentB>;
  using IteratorB =
      nihilus_gemm::transform::threadblock::PredicatedTileAccessIterator<
          nihilus_gemm::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB, gather_b, PermuteBLayout>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = nihilus_gemm::gemm::threadblock::MmaMultistage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, LayoutC,
      typename MmaCore::MmaPolicy, Stages>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp)
template <
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
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool gather_a,
    /// Gather operand B by using an index array
    bool gather_b,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
    >
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, Stages, Operator, false, SharedMemoryClear,
                  gather_a, gather_b, PermuteALayout, PermuteBLayout> {

  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
             "simt epilogue must be row major");

  static nihilus_gemm::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? nihilus_gemm::arch::CacheOperation::Global
          : nihilus_gemm::arch::CacheOperation::Always;

  static nihilus_gemm::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? nihilus_gemm::arch::CacheOperation::Global
          : nihilus_gemm::arch::CacheOperation::Always;

  // Define the MmaCore components
  using MmaCore = typename nihilus_gemm::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      Stages, Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = nihilus_gemm::Array<ElementA, kAlignmentA>;
  using IteratorA =
      nihilus_gemm::transform::threadblock::PredicatedTileAccessIterator<
          nihilus_gemm::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA, gather_a, PermuteALayout>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = nihilus_gemm::Array<ElementB, kAlignmentB>;
  using IteratorB =
      nihilus_gemm::transform::threadblock::PredicatedTileAccessIterator<
          nihilus_gemm::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB, gather_b, PermuteBLayout>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = nihilus_gemm::gemm::threadblock::MmaMultistage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, LayoutC,
      typename MmaCore::MmaPolicy, Stages, SharedMemoryClear>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for column-major-interleaved output
template <
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
    /// Tag indicating architecture to tune for
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Number of Interleaved K
    int InterleavedK>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator,
                  layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass,
                  ArchTag, ThreadblockShape, WarpShape, InstructionShape,
                  Stages, Operator, true, SharedMemoryClearOption::kNone, 
                  false, false, layout::NoPermute, layout::NoPermute> {
  // Define the MmaCore components
  using MmaCore = typename nihilus_gemm::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator,
      layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass, Stages,
      Operator, true>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = nihilus_gemm::Array<ElementA, kAlignmentA>;
  using IteratorA =
      nihilus_gemm::transform::threadblock::PredicatedTileAccessIterator<
          nihilus_gemm::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = nihilus_gemm::Array<ElementB, kAlignmentB>;
  using IteratorB =
      nihilus_gemm::transform::threadblock::PredicatedTileAccessIterator<
          nihilus_gemm::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = nihilus_gemm::gemm::threadblock::MmaMultistage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for SIMT IDP4A Kernels
template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape>
struct DefaultMma<int8_t, LayoutA, kAlignmentA, int8_t, LayoutB, kAlignmentB,
                  ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
                  ArchTag, ThreadblockShape, WarpShape, GemmShape<1, 1, 4>, 2,
                  Operator, false, SharedMemoryClearOption::kNone,
                  false, false, layout::NoPermute, layout::NoPermute> {
  using InstructionShape = GemmShape<1, 1, 4>;
  using ElementA = int8_t;
  using ElementB = int8_t;
  using OperatorClass =  arch::OpClassSimt;

  static constexpr  bool transposeA = platform::is_same< LayoutA, layout::ColumnMajor >::value;
  static constexpr  bool transposeB = platform::is_same< LayoutB, layout::RowMajor >::value;

  // Define the MmaCore components
  using MmaCore = typename nihilus_gemm::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      OperatorClass, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      nihilus_gemm::transform::threadblock::PredicatedTileIterator2dThreadTile<
          nihilus_gemm::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, transposeA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      nihilus_gemm::transform::threadblock::PredicatedTileIterator2dThreadTile<
          nihilus_gemm::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, transposeB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = nihilus_gemm::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_RT_TM_ARCH_WMMA_ENABLED)
/// Specialization for Wmma TensorOp operator with 2 staged pipeline
template <
    ///< Element type for A matrix operand
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
    /// Layout type for C and D matrix operands
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
    typename Operator>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassWmmaTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false, SharedMemoryClearOption::kNone,
                  false, false, layout::NoPermute, layout::NoPermute> {
  // Define the MmaCore components
  using MmaCore = typename nihilus_gemm::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      arch::OpClassWmmaTensorOp, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      nihilus_gemm::transform::threadblock::PredicatedTileIterator<
          nihilus_gemm::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      nihilus_gemm::transform::threadblock::PredicatedTileIterator<
          nihilus_gemm::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = nihilus_gemm::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      LayoutC, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for Wmma TensorOp operator with 1 staged pipeline
template <
    ///< Element type for A matrix operand
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
    /// Layout type for C and D matrix operands
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
    typename Operator>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassWmmaTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 1, Operator, false, SharedMemoryClearOption::kNone,
                  false, false, layout::NoPermute, layout::NoPermute> {
  // Define the MmaCore components
  using MmaCore = typename nihilus_gemm::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      arch::OpClassWmmaTensorOp, 1, Operator>; 

  // Define iterators over tiles from the A operand
  using IteratorA =
      nihilus_gemm::transform::threadblock::PredicatedTileIterator<
          nihilus_gemm::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      nihilus_gemm::transform::threadblock::PredicatedTileIterator<
          nihilus_gemm::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  // Define the threadblock-scoped singlestage matrix multiply
  using ThreadblockMma = nihilus_gemm::gemm::threadblock::MmaSingleStage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      LayoutC, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////
#endif //CUTLASS_RT_TM_ARCH_WMMA_ENABLED

} // namespace threadblock
} // namespace gemm
} // namespace nihilus_gemm 

////////////////////////////////////////////////////////////////////////////////
