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

#include "cutlass_rt_tm/cutlass.h"

#include "cutlass_rt_tm/gemm/gemm.h"
#include "cutlass_rt_tm/matrix_coord.h"
#include "cutlass_rt_tm/semaphore.h"
#include "cutlass_rt_tm/arch/arch.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_rt_tm {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

// STEP 1: Modify the core kernel::Gemm to use NTTP for M and K

template<uint64_t M_, uint64_t K_, typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_, bool SplitKSerial> struct Gemm {
	using Mma				 = Mma_;
	using Epilogue			 = Epilogue_;
	using OutputOp			 = typename Epilogue::OutputOp;
	using ThreadblockSwizzle = ThreadblockSwizzle_;

	static constexpr bool kSplitKSerial = SplitKSerial;

	// Compile-time dimensions
	static constexpr uint64_t kM = M_;
	static constexpr uint64_t kK = K_;

	// CONSTEXPR OPTIMIZATION 1: Pre-compute grid calculations
	static constexpr int kThreadblockM = Mma::Shape::kM;
	static constexpr int kThreadblockN = Mma::Shape::kN;
	static constexpr int kThreadblockK = Mma::Shape::kK;

	// Grid dimensions for M and K are now compile-time!
	static constexpr int kGridM = (kM + kThreadblockM - 1) / kThreadblockM;
	static constexpr int kGridK = (kK + kThreadblockK - 1) / kThreadblockK;

	// CONSTEXPR OPTIMIZATION 2: K iteration counts
	static constexpr int kTotalGemmKIterations = (kK + kThreadblockK - 1) / kThreadblockK;

	// CONSTEXPR OPTIMIZATION 3: Alignment checks
	static constexpr int kAlignmentA = (platform::is_same<typename Mma::IteratorA::Layout, layout::ColumnMajorInterleaved<32>>::value) ? 32
		: (platform::is_same<typename Mma::IteratorA::Layout, layout::ColumnMajorInterleaved<64>>::value)							   ? 64
																																	   : Mma::IteratorA::AccessType::kElements;

	static constexpr int kAlignmentB = (platform::is_same<typename Mma::IteratorB::Layout, layout::RowMajorInterleaved<32>>::value) ? 32
		: (platform::is_same<typename Mma::IteratorB::Layout, layout::RowMajorInterleaved<64>>::value)								? 64
																																	: Mma::IteratorB::AccessType::kElements;

	static constexpr int kAlignmentC = (platform::is_same<typename Epilogue::OutputTileIterator::Layout, layout::ColumnMajorInterleaved<32>>::value) ? 32
		: (platform::is_same<typename Epilogue::OutputTileIterator::Layout, layout::ColumnMajorInterleaved<64>>::value)								 ? 64
																														: Epilogue::OutputTileIterator::kElementsPerAccess;

	using WarpCount					  = typename Mma::WarpCount;
	static constexpr int kThreadCount = 32 * WarpCount::kCount;

	struct Params {
		uint64_t N;// Only N dimension remains runtime
		cutlass_rt_tm::gemm::GemmCoord grid_tiled_shape;
		int swizzle_log_tile;
		typename Mma::IteratorA::Params params_A;
		typename Mma::IteratorA::TensorRef ref_A;
		typename Mma::IteratorB::Params params_B;
		typename Mma::IteratorB::TensorRef ref_B;
		typename Epilogue::OutputTileIterator::Params params_C;
		typename Epilogue::OutputTileIterator::TensorRef ref_C;
		typename Epilogue::OutputTileIterator::Params params_D;
		typename Epilogue::OutputTileIterator::TensorRef ref_D;
		typename OutputOp::Params output_op;
		int* semaphore;
		int gemm_k_size;

		int const* gather_A_indices;
		int const* gather_B_indices;
		int const* scatter_D_indices;

		CUTLASS_RT_TM_HOST_DEVICE
		__forceinline__ Params() : N(0), swizzle_log_tile(0), semaphore(0), gemm_k_size(0) {
		}

		CUTLASS_RT_TM_HOST_DEVICE
		__forceinline__ Params(uint64_t N_, cutlass_rt_tm::gemm::GemmCoord const& grid_tiled_shape, typename Mma::IteratorA::TensorRef ref_A,
			typename Mma::IteratorB::TensorRef ref_B,
			typename Epilogue::OutputTileIterator::TensorRef ref_C, typename Epilogue::OutputTileIterator::TensorRef ref_D,
			typename OutputOp::Params output_op = typename OutputOp::Params(), int* workspace = nullptr, int const* gather_A_indices = nullptr,
			int const* gather_B_indices = nullptr, int const* scatter_D_indices = nullptr)
			: N(N_), grid_tiled_shape(grid_tiled_shape), swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)), params_A(ref_A.layout()), ref_A(ref_A),
			  params_B(ref_B.layout()), ref_B(ref_B), params_C(ref_C.layout()), ref_C(ref_C), params_D(ref_D.layout()), ref_D(ref_D), output_op(output_op),
			  gather_A_indices(gather_A_indices), gather_B_indices(gather_B_indices), scatter_D_indices(scatter_D_indices) {
			// CONSTEXPR OPTIMIZATION 4: Use pre-computed constants
			int gemm_k_iterations = (kTotalGemmKIterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
			gemm_k_size			  = gemm_k_iterations * kThreadblockK;
			semaphore			  = workspace;
		}
	};

	union SharedStorage {
		typename Mma::SharedStorage main_loop;
		typename Epilogue::SharedStorage epilogue;
	};

	CUTLASS_RT_TM_HOST_DEVICE
	__forceinline__ Gemm() {
	}

	/// CONSTEXPR OPTIMIZATION 5: Simplified alignment check
	CUTLASS_RT_TM_HOST_DEVICE
	__forceinline__ static Status can_implement(uint64_t N, typename Mma::IteratorA::TensorRef ref_A, typename Mma::IteratorB::TensorRef ref_B,
		typename Epilogue::OutputTileIterator::TensorRef ref_C, typename Epilogue::OutputTileIterator::TensorRef ref_D) {
		// Use constexpr alignment values
		if (!TensorRef_aligned(ref_A, kAlignmentA)) {
			return Status::kErrorMisalignedOperand;
		}
		if (!TensorRef_aligned(ref_B, kAlignmentB)) {
			return Status::kErrorMisalignedOperand;
		}
		if (!TensorRef_aligned(ref_C, kAlignmentC)) {
			return Status::kErrorMisalignedOperand;
		}
		if (!TensorRef_aligned(ref_D, kAlignmentC)) {
			return Status::kErrorMisalignedOperand;
		}
		return Status::kSuccess;
	}

	CUTLASS_RT_TM_DEVICE
	__forceinline__ void operator()(Params const& params, SharedStorage& shared_storage) {
		ThreadblockSwizzle threadblock_swizzle;
		cutlass_rt_tm::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

		// Early exit - grid bounds check can use constexpr kGridM
		if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() || params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
			return;
		}

		// CONSTEXPR OPTIMIZATION 6: Threadblock offset calculations
		cutlass_rt_tm::MatrixCoord tb_offset_A{
			threadblock_tile_offset.m() * kThreadblockM,// constexpr multiply
			threadblock_tile_offset.k() * params.gemm_k_size,
		};

		cutlass_rt_tm::MatrixCoord tb_offset_B{
			threadblock_tile_offset.k() * params.gemm_k_size,
			threadblock_tile_offset.n() * kThreadblockN// constexpr multiply
		};

		// CONSTEXPR OPTIMIZATION 7: Problem size K is compile-time constant
		static constexpr int kProblemSizeK = static_cast<int>(kK);
		int problem_size_k			= min(kProblemSizeK, (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

		// CONSTEXPR OPTIMIZATION 8: K iteration calculation
		int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + kThreadblockK - 1) / kThreadblockK;

		int thread_idx = threadIdx.x;

		// CONSTEXPR OPTIMIZATION 9: Iterator extents use compile-time M
		constexpr cutlass_rt_tm::MatrixCoord kExtentA{ kM, kProblemSizeK };
		typename Mma::IteratorA iterator_A(params.params_A, params.ref_A.data(), { static_cast<int>(kM), problem_size_k },// M is constexpr
			thread_idx, tb_offset_A, params.gather_A_indices);

		typename Mma::IteratorB iterator_B(params.params_B, params.ref_B.data(), { problem_size_k, static_cast<int>(params.N) },// K constexpr, N runtime
			thread_idx, tb_offset_B, params.gather_B_indices);

		int warp_idx = canonical_warp_idx_sync();
		int lane_idx = threadIdx.x % 32;

		// Main loop (unchanged)
		Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
		typename Mma::FragmentC accumulators;
		accumulators.clear();

		if (!kSplitKSerial || gemm_k_iterations > 0) {
			mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
		}

		// Epilogue
		OutputOp output_op(params.output_op);
		threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

		// CONSTEXPR OPTIMIZATION 10: Threadblock offset calculations
		MatrixCoord threadblock_offset(threadblock_tile_offset.m() * kThreadblockM,// constexpr
			threadblock_tile_offset.n() * kThreadblockN// constexpr
		);

		int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

		Semaphore semaphore(params.semaphore + block_idx, thread_idx);

		if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
			semaphore.fetch();
			output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
		}

		// CONSTEXPR OPTIMIZATION 11: Output iterator extents
		static constexpr cutlass_rt_tm::MatrixCoord kExtentCD{ static_cast<int>(kM), 0 };// N filled at runtime

		typename Epilogue::OutputTileIterator iterator_C(params.params_C, params.ref_C.data(), { static_cast<int>(kM), static_cast<int>(params.N) },// M constexpr
			thread_idx, threadblock_offset, params.scatter_D_indices);

		typename Epilogue::OutputTileIterator iterator_D(params.params_D, params.ref_D.data(), { static_cast<int>(kM), static_cast<int>(params.N) },// M constexpr
			thread_idx, threadblock_offset, params.scatter_D_indices);

		Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

		if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
			if (threadblock_tile_offset.k()) {
				iterator_C = iterator_D;
			}
			semaphore.wait(threadblock_tile_offset.k());
		}

		epilogue(output_op, iterator_D, accumulators, iterator_C);

		if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
			int lock = (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) ? 0 : threadblock_tile_offset.k() + 1;
			semaphore.release(lock);
		}
	}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass_rt_tm

