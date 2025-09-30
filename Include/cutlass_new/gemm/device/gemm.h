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
#include "cutlass_new/layout/matrix.h"
#include "cutlass_new/platform/platform.h"
#include "cutlass_new/gemm/gemm.h"
#include "cutlass_new/conv/conv2d_problem_size.h"
#include "cutlass_new/conv/conv3d_problem_size.h"
#include "cutlass_new/gemm/threadblock/index_remat.h"
#include "cutlass_new/gemm/threadblock/threadblock_swizzle_streamk.h"

#include "cutlass_new/numeric_types.h"
#include "cutlass_new/arch/arch.h"

#include "cutlass_new/gemm/kernel/gemm.h"

#include "cutlass_new/arch/wmma.h"

#include "cutlass_new/epilogue/threadblock/epilogue.h"
#include "cutlass_new/epilogue/thread/linear_combination.h"

#include "cutlass_new/gemm/kernel/gemm_pipelined.h"
#include "cutlass_new/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass_new/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass_new/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass_new/gemm/threadblock/default_mma.h"
#include "cutlass_new/gemm/threadblock/default_mma_core_simt.h"

#include "cutlass_new/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass_new/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass_new/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass_new/transform/threadblock/predicated_tile_iterator.h"

#include "cutlass_new/layout/permute.h"

namespace cutlass {

	namespace gemm {

		namespace device {

			template<int N = 1> struct GemmIdentityThreadblockSwizzle {
				CUTLASS_HOST_DEVICE
				GemmIdentityThreadblockSwizzle() {
				}

				/// Returns the shape of the problem in units of logical tiles
				/// *Gemm* problem size: gemm(M, N, K)
				CUTLASS_HOST_DEVICE
				static GemmCoord get_tiled_shape(GemmCoord problem_size, GemmCoord tile_size, int split_k_slices) {
					return GemmCoord((problem_size.m() + tile_size.m() - 1) / tile_size.m(), (problem_size.n() + tile_size.n() - 1) / tile_size.n(), split_k_slices);
				}

				/// Returns the shape of the problem in units of logical tiles
				/// *ImplicitGemm* Conv2d problem size: conv_operator(NPQK, NHWC, KRSC)
				CUTLASS_HOST_DEVICE
				static GemmCoord get_tiled_shape(cutlass::conv::Operator conv_operator, cutlass::conv::Conv2dProblemSize const& problem_size, GemmCoord tile_size,
					int split_k_slices) {
					gemm::GemmCoord implicit_gemm_problem_size = cutlass::conv::implicit_gemm_problem_size(conv_operator, problem_size);

					return get_tiled_shape(implicit_gemm_problem_size, tile_size, split_k_slices);
				}

				/// Returns the shape of the problem in units of logical tiles
				/// *ImplicitGemm* Conv3d problem size: conv_operator(NZPQK, NDHWC, KTRSC)
				CUTLASS_HOST_DEVICE
				static GemmCoord get_tiled_shape(cutlass::conv::Operator conv_operator, cutlass::conv::Conv3dProblemSize const& problem_size, GemmCoord tile_size,
					int split_k_slices) {
					gemm::GemmCoord implicit_gemm_problem_size = cutlass::conv::implicit_gemm_problem_size(conv_operator, problem_size);

					return get_tiled_shape(implicit_gemm_problem_size, tile_size, split_k_slices);
				}

				/// Computes CUDA grid dimensions given a size in units of logical tiles
				CUTLASS_HOST_DEVICE
				static dim3 get_grid_shape(GemmCoord tiled_shape) {
					int tile = 1 << get_log_tile(tiled_shape);
					return dim3(tiled_shape.m() * tile, (tiled_shape.n() + tile - 1) / tile, tiled_shape.k());
				}

				/// Calculates optimal swizzle width
				CUTLASS_HOST_DEVICE
				static int get_log_tile(GemmCoord tiled_shape) {
					auto n = tiled_shape.n();
					// Thresholds picked so that it doesn't cause too many no-op CTAs
					if (N >= 8 && n >= 6)
						return 3;
					else if (N >= 4 && n >= 3)
						return 2;
					else if (N >= 2 && n >= 2)
						return 1;
					else
						return 0;
				}

				/// Obtains the threadblock offset (in units of threadblock-scoped tiles)
				CUTLASS_DEVICE
				static GemmCoord get_tile_offset(int log_tile) {
					int block_idx_x = threadblock::RematerializeBlockIdxX();
					int block_idx_y = threadblock::RematerializeBlockIdxY();
					int block_idx_z = threadblock::RematerializeBlockIdxZ();

					return GemmCoord{ (block_idx_x >> log_tile),//
						(block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)), block_idx_z };
				}

				/// Obtains the threadblock offset (in units of threadblock-scoped tiles)
				CUTLASS_DEVICE
				static GemmCoord get_tile_offset(GemmCoord tiled_shape) {
					int const kTile = N;
					int block_idx_x = threadblock::RematerializeBlockIdxX();
					int block_idx_y = threadblock::RematerializeBlockIdxY();

					if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile))
						return GemmCoord{ block_idx_x, block_idx_y, threadblock::RematerializeBlockIdxZ() };

					return GemmCoord{ (block_idx_x / kTile), (block_idx_y * kTile) + (block_idx_x % kTile), threadblock::RematerializeBlockIdxZ() };
				}
			};

			template<typename Operator> CUTLASS_GLOBAL void Kernel(typename Operator::Params params) {
				// Dynamic shared memory base pointer
				extern __shared__ int SharedStorageBase[];
				// Declare pointer to dynamic shared memory.
				typename Operator::SharedStorage* shared_storage = reinterpret_cast<typename Operator::SharedStorage*>(SharedStorageBase);

				Operator::impl(params, *shared_storage);
				cutlass::arch::synclog_print();
			}

			template<typename OperatorClass, typename ArchTag, typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
			struct DefaultGemmConfiguration;

			////////////////////////////////////////////////////////////////////////////////

			template<typename ArchTag, typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
			struct DefaultGemmConfiguration<arch::OpClassSimt, ArchTag, ElementA, ElementB, ElementC, ElementAccumulator> {
				static constexpr int kAlignmentA = 1;
				static constexpr int kAlignmentB = 1;
				using ThreadblockShape			 = GemmShape<128, 128, 8>;
				using WarpShape					 = GemmShape<32, 64, 8>;
				using InstructionShape			 = GemmShape<1, 1, 1>;
				static constexpr int kStages	 = 2;

				using EpilogueOutputOp = epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>;

				using Operator = arch::OpMultiplyAdd;
			};

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
				bool GatherA = false,
				/// Gather operand B by using an index array
				bool GatherB = false,
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
			////////////////////////////////////////////////////////////////////////////////

			/// Partial specialization for SIMT
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
				bool GatherA,
				/// Gather operand B by using an index array
				bool GatherB,
				/// Scatter result D by using an index array
				bool ScatterD,
				/// Permute result D
				typename PermuteDLayout,
				/// Permute operand A
				typename PermuteALayout,
				/// Permute operand B
				typename PermuteBLayout>
			struct DefaultGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC, LayoutC, ElementAccumulator, arch::OpClassSimt, ArchTag, ThreadblockShape,
				WarpShape, GemmShape<1, 1, 1>, EpilogueOutputOp, ThreadblockSwizzle, 2, SplitKSerial, Operator, SharedMemoryClear, GatherA, GatherB, ScatterD, PermuteDLayout,
				PermuteALayout, PermuteBLayout, typename std::enable_if<!std::is_same<ArchTag, arch::Sm80>::value>::type> {
				static_assert((std::is_same<LayoutC, layout::RowMajor>::value || std::is_same<LayoutC, layout::AffineRankN<2>>::value),
					"Epilogue in the kernel level must be row major");

				/// Define the threadblock-scoped matrix multiply-accumulate
				using Mma = typename cutlass::gemm::threadblock::DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator, LayoutC,
					arch::OpClassSimt, arch::Sm50, ThreadblockShape, WarpShape, GemmShape<1, 1, 1>, 2, Operator, false, SharedMemoryClear, GatherA, GatherB, PermuteALayout,
					PermuteBLayout>::ThreadblockMma;

				static constexpr int kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
				static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

				using RegularEpilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<ThreadblockShape, typename Mma::Operator, EpilogueOutputOp,
					kEpilogueElementsPerAccess, ScatterD, PermuteDLayout>::Epilogue;

				using Epilogue = RegularEpilogue;

				/// Define the kernel-level GEMM operator.
				using GemmKernel = kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
			};

			template<int M_, int K_, typename ElementA_, typename LayoutA_, typename ElementB_, typename LayoutB_, typename ElementC_, typename LayoutC_,
				typename ElementAccumulator_ = ElementC_, typename OperatorClass_ = arch::OpClassSimt, typename ArchTag_ = arch::Sm120,
				typename ThreadblockShape_	 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::ThreadblockShape,
				typename WarpShape_			 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::WarpShape,
				typename InstructionShape_	 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::InstructionShape,
				typename EpilogueOutputOp_	 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::EpilogueOutputOp,
				typename ThreadblockSwizzle_ = typename GemmIdentityThreadblockSwizzle<>,
				int Stages					 = DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kStages,
				int AlignmentA				 = DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kAlignmentA,
				int AlignmentB = DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kAlignmentB, bool SplitKSerial = false,
				typename Operator_ = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::Operator,
				bool GatherA = false, bool GatherB = false, bool ScatterD = false, typename PermuteDLayout = layout::NoPermute>
			class Gemm {
			  public:
				static constexpr int kM = M_;
				static constexpr int kK = K_;

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
				static constexpr int kStages				  = Stages;
				static constexpr int kAlignmentA			  = AlignmentA;
				static constexpr int kAlignmentB			  = AlignmentB;
				static constexpr int kAlignmentC			  = EpilogueOutputOp::kCount;
				static constexpr bool kSplitKSerial			  = SplitKSerial;
				static constexpr ComplexTransform kTransformA = ComplexTransform::kNone;
				static constexpr ComplexTransform kTransformB = ComplexTransform::kNone;

				static constexpr int kTiledM = (kM + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
				static constexpr int kTiledK = (kK + ThreadblockShape::kK - 1) / ThreadblockShape::kK;

				using GemmKernel = typename DefaultGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC, LayoutC, ElementAccumulator, OperatorClass,
					ArchTag, ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, kStages, kSplitKSerial, Operator, SharedMemoryClearOption::kNone,
					GatherA, GatherB, ScatterD, PermuteDLayout>::GemmKernel;

				struct Arguments {
					int N;
					TensorRef<ElementA const, LayoutA> ref_A;
					TensorRef<ElementB const, LayoutB> ref_B;
					TensorRef<ElementC const, LayoutC> ref_C;
					TensorRef<ElementC, LayoutC> ref_D;
					typename EpilogueOutputOp::Params epilogue;
					int split_k_slices;
					int const* gather_A_indices;
					int const* gather_B_indices;
					int const* scatter_D_indices;
					Status status;


					CUTLASS_HOST_DEVICE
					Arguments(Status status_new) : status(status_new), split_k_slices(1) {
					}

					CUTLASS_HOST_DEVICE
					Arguments() : N(0), split_k_slices(1) {
					}

					CUTLASS_HOST_DEVICE
					Arguments(int N_, TensorRef<ElementA const, LayoutA> ref_A_, TensorRef<ElementB const, LayoutB> ref_B_, TensorRef<ElementC const, LayoutC> ref_C_,
						TensorRef<ElementC, LayoutC> ref_D_, typename EpilogueOutputOp::Params epilogue_ = typename EpilogueOutputOp::Params(), int split_k_slices = 1,
						int const* gather_A_indices_ = nullptr, int const* gather_B_indices_ = nullptr, int const* scatter_D_indices_ = nullptr)
						: N(N_), ref_A(ref_A_), ref_B(ref_B_), ref_C(ref_C_), ref_D(ref_D_), epilogue(epilogue_), split_k_slices(split_k_slices),
						  gather_A_indices(gather_A_indices_), gather_B_indices(gather_B_indices_), scatter_D_indices(scatter_D_indices_) {
					}

					CUTLASS_HOST_DEVICE
					GemmCoord problem_size() const {
						return GemmCoord(kM, N, kK);
					}
				};

			  private:
			  public:
				Gemm() {
				}

				static typename GemmKernel::Params initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
					GemmCoord problem_size(kM, args.N, kK);

					ThreadblockSwizzle threadblock_swizzle;

					cutlass::gemm::GemmCoord grid_shape =
						threadblock_swizzle.get_tiled_shape(problem_size, { ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK }, args.split_k_slices);

					if (kSplitKSerial) {
						if (args.split_k_slices > 1) {
							if (!workspace) {
								return Status::kErrorWorkspaceNull;
							}

							size_t bytes{};

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
					typename GemmKernel::Params params_;

					params_ = typename GemmKernel::Params{ problem_size, grid_shape, args.ref_A.non_const_ref(), args.ref_B.non_const_ref(), args.ref_C.non_const_ref(), args.ref_D,
						args.epilogue, static_cast<int*>(workspace), args.gather_A_indices, args.gather_B_indices, args.scatter_D_indices };

					return params_;
				}

				static Status run(typename GemmKernel::Params params_) {
					ThreadblockSwizzle threadblock_swizzle;

					dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
					dim3 block(GemmKernel::kThreadCount, 1, 1);

					cudaError_t result;

					int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

					if (smem_size >= (48 << 10)) {
						result = cudaFuncSetAttribute(Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

						if (result != cudaSuccess) {
							return Status::kErrorInternal;
						}
					}

					cutlass::arch::synclog_setup();
					Kernel<GemmKernel><<<grid, block, smem_size>>>(params_);

					result = cudaGetLastError();

					return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
				}

				static Status impl(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
					return run(initialize(args, workspace, stream));
				}
			};
		}
	}
}
