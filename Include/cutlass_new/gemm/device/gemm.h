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

#include "cutlass_new/cutlass.h"
#include "cutlass_new/numeric_types.h"
#include "cutlass_new/arch/arch.h"

#include "cutlass_new/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass_new/gemm/kernel/gemm.h"

#include "cutlass_new/gemm/kernel/default_gemm.h"

namespace cutlass {
	namespace gemm {
		namespace device {

			template<typename OperatorClass, typename ArchTag, typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
			struct DefaultGemmConfiguration;

			////////////////////////////////////////////////////////////////////////////////

			template<typename ArchTag, typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
			struct DefaultGemmConfiguration<arch::OpClassSimt, ArchTag, ElementA, ElementB, ElementC, ElementAccumulator> {
				static constexpr int32_t kAlignmentA = 1;
				static constexpr int32_t kAlignmentB = 1;
				using ThreadblockShape				 = GemmShape<128, 128, 8>;
				using WarpShape						 = GemmShape<32, 64, 8>;
				using InstructionShape				 = GemmShape<1, 1, 1>;
				static constexpr int32_t kStages	 = 2;

				using EpilogueOutputOp = epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>;

				using Operator = arch::OpMultiplyAdd;
			};

			template<typename Operator> CUTLASS_GLOBAL void Kernel(typename Operator::Params params) {
				// Dynamic shared memory base pointer
				extern __shared__ int32_t SharedStorageBase[];
				// Declare pointer to dynamic shared memory.
				typename Operator::SharedStorage* shared_storage = reinterpret_cast<typename Operator::SharedStorage*>(SharedStorageBase);

				Operator op;

				op(params, *shared_storage);
				cutlass::arch::synclog_print();
			}

			template<int32_t M_, int32_t K_, typename ElementA_, typename LayoutA_, typename ElementB_, typename LayoutB_, typename ElementC_, typename LayoutC_,
				typename ElementAccumulator_ = ElementC_, typename OperatorClass_ = arch::OpClassSimt, typename ArchTag_ = arch::Sm120,
				typename ThreadblockShape_	 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::ThreadblockShape,
				typename WarpShape_			 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::WarpShape,
				typename InstructionShape_	 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::InstructionShape,
				typename EpilogueOutputOp_	 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::EpilogueOutputOp,
				typename ThreadblockSwizzle_ = typename threadblock::GemmIdentityThreadblockSwizzle<>,
				int32_t Stages					 = DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kStages,
				int32_t AlignmentA				 = DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kAlignmentA,
				int32_t AlignmentB = DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kAlignmentB, bool SplitKSerial = false,
				typename Operator_ = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::Operator,
				bool GatherA = false, bool GatherB = false, bool ScatterD = false, typename PermuteDLayout = layout::NoPermute>
			class Gemm {
			  public:
				static constexpr int32_t kM = M_;
				static constexpr int32_t kK = K_;

				using ElementA								  = ElementA_;
				using LayoutA								  = LayoutA_;
				using TensorRefA							  = TensorRef<ElementA const, LayoutA>;
				using ElementB								  = ElementB_;
				using LayoutB								  = LayoutB_;
				using TensorRefB							  = TensorRef<ElementB const, LayoutB>;
				using ElementC								  = ElementC_;
				using LayoutC								  = LayoutC_;
				using TensorRefC							  = TensorRef<ElementC const, LayoutC>;
				using TensorRefD							  = TensorRef<ElementC, LayoutC>;
				using ElementAccumulator					  = ElementAccumulator_;
				using OperatorClass							  = OperatorClass_;
				using ArchTag								  = ArchTag_;
				using ThreadblockShape						  = ThreadblockShape_;
				using WarpShape								  = WarpShape_;
				using InstructionShape						  = InstructionShape_;
				using EpilogueOutputOp						  = EpilogueOutputOp_;
				using ThreadblockSwizzle					  = ThreadblockSwizzle_;
				using Operator								  = Operator_;
				static constexpr int32_t kStages				  = Stages;
				static constexpr int32_t kAlignmentA			  = AlignmentA;
				static constexpr int32_t kAlignmentB			  = AlignmentB;
				static constexpr int32_t kAlignmentC			  = EpilogueOutputOp::kCount;
				static constexpr bool kSplitKSerial			  = SplitKSerial;
				static constexpr ComplexTransform kTransformA = ComplexTransform::kNone;
				static constexpr ComplexTransform kTransformB = ComplexTransform::kNone;

				static constexpr int32_t kTiledM = (kM + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
				static constexpr int32_t kTiledK = (kK + ThreadblockShape::kK - 1) / ThreadblockShape::kK;

				using GemmKernel = typename kernel::DefaultGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC, LayoutC, ElementAccumulator,
					OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, kStages, kSplitKSerial, Operator,
					SharedMemoryClearOption::kNone, GatherA, GatherB, ScatterD, PermuteDLayout>::GemmKernel;

				struct Arguments {
					int32_t N;
					TensorRef<ElementA const, LayoutA> ref_A;
					TensorRef<ElementB const, LayoutB> ref_B;
					TensorRef<ElementC const, LayoutC> ref_C;
					TensorRef<ElementC, LayoutC> ref_D;
					typename EpilogueOutputOp::Params epilogue;
					int32_t split_k_slices;
					int32_t const* gather_A_indices;
					int32_t const* gather_B_indices;
					int32_t const* scatter_D_indices;


					CUTLASS_HOST_DEVICE
					Arguments() : N(0), split_k_slices(1) {
					}

					CUTLASS_HOST_DEVICE
					Arguments(int32_t N_, TensorRef<ElementA const, LayoutA> ref_A_, TensorRef<ElementB const, LayoutB> ref_B_, TensorRef<ElementC const, LayoutC> ref_C_,
						TensorRef<ElementC, LayoutC> ref_D_, typename EpilogueOutputOp::Params epilogue_ = typename EpilogueOutputOp::Params(), int32_t split_k_slices = 1,
						int32_t const* gather_A_indices_ = nullptr, int32_t const* gather_B_indices_ = nullptr, int32_t const* scatter_D_indices_ = nullptr)
						: N(N_), ref_A(ref_A_), ref_B(ref_B_), ref_C(ref_C_), ref_D(ref_D_), epilogue(epilogue_), split_k_slices(split_k_slices),
						  gather_A_indices(gather_A_indices_), gather_B_indices(gather_B_indices_), scatter_D_indices(scatter_D_indices_) {
					}

					CUTLASS_HOST_DEVICE
					GemmCoord problem_size() const {
						return GemmCoord(kM, N, kK);
					}
				};

			  private:
				typename GemmKernel::Params params_;

			  public:
				Gemm() {
				}

				static Status can_implement(Arguments const& args) {
					if (!kSplitKSerial && args.split_k_slices > 1) {
						return Status::kErrorInvalidProblem;
					}

					GemmCoord problem_size(kM, args.N, kK);

					Status status = GemmKernel::can_implement(problem_size, args.ref_A.non_const_ref(), args.ref_B.non_const_ref(), args.ref_C.non_const_ref(), args.ref_D);

					if (status != Status::kSuccess) {
						return status;
					}

					return Status::kSuccess;
				}

				static size_t get_workspace_size(Arguments const& args) {
					size_t bytes = 0;

					ThreadblockSwizzle threadblock_swizzle;

					int32_t tiled_n = (args.N + ThreadblockShape::kN - 1) / ThreadblockShape::kN;

					if (kSplitKSerial && args.split_k_slices > 1) {
						bytes += sizeof(int32_t) * size_t(kTiledM) * size_t(tiled_n);
					}

					return bytes;
				}

				Status initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
					GemmCoord problem_size(kM, args.N, kK);

					ThreadblockSwizzle threadblock_swizzle;

					cutlass::gemm::GemmCoord grid_shape =
						threadblock_swizzle.get_tiled_shape(problem_size, { ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK }, args.split_k_slices);

					if (kSplitKSerial) {
						if (args.split_k_slices > 1) {
							if (!workspace) {
								return Status::kErrorWorkspaceNull;
							}

							size_t bytes = get_workspace_size(args);

							cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);

							if (result != cudaSuccess) {
								return Status::kErrorInternal;
							}
						}
					} else {
						if (args.split_k_slices > 1) {
							return Status::kErrorInvalidProblem;
						}
					}

					params_ = typename GemmKernel::Params{ problem_size, grid_shape, args.ref_A.non_const_ref(), args.ref_B.non_const_ref(), args.ref_C.non_const_ref(), args.ref_D,
						args.epilogue, static_cast<int32_t*>(workspace), args.gather_A_indices, args.gather_B_indices, args.scatter_D_indices };

					return Status::kSuccess;
				}

				Status update(Arguments const& args, void* workspace = nullptr) {
					if (kSplitKSerial && args.split_k_slices > 1) {
						if (!workspace) {
							return Status::kErrorWorkspaceNull;
						}
					}

					params_.ref_A.reset(args.ref_A.non_const_ref().data());
					params_.ref_B.reset(args.ref_B.non_const_ref().data());
					params_.ref_C.reset(args.ref_C.non_const_ref().data());
					params_.ref_D.reset(args.ref_D.data());
					params_.output_op = args.epilogue;
					params_.semaphore = static_cast<int32_t*>(workspace);

					return Status::kSuccess;
				}

				Status run(cudaStream_t stream = nullptr) {
					ThreadblockSwizzle threadblock_swizzle;

					dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
					dim3 block(GemmKernel::kThreadCount, 1, 1);

					cudaError_t result;

					int32_t smem_size = int32_t(sizeof(typename GemmKernel::SharedStorage));

					if (smem_size >= (48 << 10)) {
						result = cudaFuncSetAttribute(Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

						if (result != cudaSuccess) {
							return Status::kErrorInternal;
						}
					}

					cutlass::arch::synclog_setup();
					Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);

					result = cudaGetLastError();

					return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
				}

				Status operator()(cudaStream_t stream = nullptr) {
					return run(stream);
				}

				Status operator()(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
					Status status = initialize(args, workspace, stream);

					if (status == Status::kSuccess) {
						status = run(stream);
					}

					return status;
				}
			};




		}
	}
}
