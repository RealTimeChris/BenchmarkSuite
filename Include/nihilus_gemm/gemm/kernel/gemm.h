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

#include "nihilus_gemm/gemm/gemm.h"
#include "nihilus_gemm/matrix_coord.h"
#include "nihilus_gemm/semaphore.h"
#include "nihilus_gemm/arch/arch.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace nihilus_gemm {
	namespace gemm {
		namespace kernel {

			/////////////////////////////////////////////////////////////////////////////////////////////////

			template<uint64_t M, uint64_t K, typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_, bool SplitKSerial> struct Gemm {
				using Mma							= Mma_;
				using Epilogue						= Epilogue_;
				using OutputOp						= typename Epilogue::OutputOp;
				using ThreadblockSwizzle			= ThreadblockSwizzle_;
				static constexpr bool kSplitKSerial = SplitKSerial;

				/// Warp count (concept: GemmShape)
				using WarpCount					  = typename Mma::WarpCount;
				static constexpr int kThreadCount = 32 * WarpCount::kCount;

				/// Parameters structure
				struct Params {
					nihilus_gemm::gemm::constexpresh_gemm_coord<M, K> problem_size;
					nihilus_gemm::gemm::GemmCoord grid_tiled_shape;
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

					//
					// Methods
					//

					NIHILUS_HOST_DEVICE
					Params() : swizzle_log_tile(0), semaphore(0), gemm_k_size(0) {
					}

					NIHILUS_HOST_DEVICE
					Params(nihilus_gemm::gemm::constexpresh_gemm_coord<M, K> const& problem_size, nihilus_gemm::gemm::GemmCoord const& grid_tiled_shape,
						typename Mma::IteratorA::TensorRef ref_A, typename Mma::IteratorB::TensorRef ref_B, typename Epilogue::OutputTileIterator::TensorRef ref_C,
						typename Epilogue::OutputTileIterator::TensorRef ref_D, typename OutputOp::Params output_op = typename OutputOp::Params(), int* workspace = nullptr,
						int const* gather_A_indices = nullptr, int const* gather_B_indices = nullptr, int const* scatter_D_indices = nullptr)
						: problem_size(problem_size), grid_tiled_shape(grid_tiled_shape), swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
						  params_A(ref_A.layout()), ref_A(ref_A), params_B(ref_B.layout()), ref_B(ref_B), params_C(ref_C.layout()), ref_C(ref_C), params_D(ref_D.layout()),
						  ref_D(ref_D), output_op(output_op) {
						constexpr int total_gemm_k_iterations = (K + Mma::Shape::kK - 1) / Mma::Shape::kK;
						int gemm_k_iterations				  = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();

						gemm_k_size = gemm_k_iterations * Mma::Shape::kK;

						semaphore = workspace;
					}
				};

				/// Shared memory storage structure
				union SharedStorage {
					typename Mma::SharedStorage main_loop;
					typename Epilogue::SharedStorage epilogue;
				};

				//
				// Methods
				//

				NIHILUS_HOST_DEVICE
				Gemm() {
				}

				/// Executes one GEMM
				NIHILUS_DEVICE
				void operator()(Params const& params, SharedStorage& shared_storage) {
					// Compute threadblock location
					ThreadblockSwizzle threadblock_swizzle;

					nihilus_gemm::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

					// Early exit if CTA is out of range
					if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() || params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
						return;
					}

					// Compute initial location in logical coordinates
					nihilus_gemm::MatrixCoord tb_offset_A{
						threadblock_tile_offset.m() * Mma::Shape::kM,
						threadblock_tile_offset.k() * params.gemm_k_size,
					};

					nihilus_gemm::MatrixCoord tb_offset_B{ threadblock_tile_offset.k() * params.gemm_k_size, threadblock_tile_offset.n() * Mma::Shape::kN };

					// Problem size is a function of threadblock index in the K dimension
					int problem_size_k = min(static_cast<int32_t>(K), (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

					// Compute threadblock-scoped matrix multiply-add
					int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

					// Compute position within threadblock
					int thread_idx = threadIdx.x;

					// Construct iterators to A and B operands
					typename Mma::IteratorA iterator_A(params.params_A, params.ref_A.data(), { M, problem_size_k }, thread_idx, tb_offset_A);

					typename Mma::IteratorB iterator_B(params.params_B, params.ref_B.data(), { problem_size_k, params.problem_size.n() }, thread_idx, tb_offset_B);

					// Broadcast the warp_id computed by lane 0 to ensure dependent code
					// is compiled as warp-uniform.
					int warp_idx = canonical_warp_idx_sync();
					int lane_idx = threadIdx.x % 32;

					//
					// Main loop
					//

					// Construct thread-scoped matrix multiply
					Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

					typename Mma::FragmentC accumulators;

					accumulators.clear();

					if (!kSplitKSerial || gemm_k_iterations > 0) {
						// Compute threadblock-scoped matrix multiply-add
						mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
					}

					//
					// Epilogue
					//

					OutputOp output_op(params.output_op);

					//
					// Masked tile iterators constructed from members
					//

					threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

					//assume identity swizzle
					MatrixCoord threadblock_offset(threadblock_tile_offset.m() * Mma::Shape::kM, threadblock_tile_offset.n() * Mma::Shape::kN);

					int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

					// Construct the semaphore.
					Semaphore semaphore(params.semaphore + block_idx, thread_idx);

					// Tile iterator loading from source tensor.
					typename Epilogue::OutputTileIterator iterator_C(params.params_C, params.ref_C.data(), static_cast<Coord<2>>(params.problem_size.mn()), thread_idx,
						threadblock_offset);

					// Tile iterator writing to destination tensor.
					typename Epilogue::OutputTileIterator iterator_D(params.params_D, params.ref_D.data(), static_cast<Coord<2>>(params.problem_size.mn()), thread_idx,
						threadblock_offset);

					Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

					// Wait on the semaphore - this latency may have been covered by iterator construction
					if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
						// For subsequent threadblocks, the source matrix is held in the 'D' tensor.
						if (threadblock_tile_offset.k()) {
							iterator_C = iterator_D;
						}

						semaphore.wait(threadblock_tile_offset.k());
					}

					// Execute the epilogue operator to update the destination tensor.
					epilogue(output_op, iterator_D, accumulators, iterator_C);

					//
					// Release the semaphore
					//

					if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
						int lock = 0;
						if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {
							// The final threadblock resets the semaphore for subsequent grids.
							lock = 0;
						} else {
							// Otherwise, the semaphore is incremented
							lock = threadblock_tile_offset.k() + 1;
						}

						semaphore.release(lock);
					}
				}
			};

			/////////////////////////////////////////////////////////////////////////////////////////////////

		}// namespace kernel
	}// namespace gemm
}// namespace nihilus_gemm
