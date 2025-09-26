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
#include "nihilus_gemm/device_kernel.h"

#include "nihilus_gemm/gemm/threadblock/threadblock_swizzle.h"
#include "nihilus_gemm/gemm/kernel/gemm.h"

#include "nihilus_gemm/gemm/kernel/default_gemm.h"
#include "nihilus_gemm/gemm/device/default_gemm_configuration.h"

#include "nihilus_gemm/layout/permute.h"

////////////////////////////////////////////////////////////////////////////////

namespace nihilus_gemm {
	namespace gemm {
		namespace device {

			/////////////////////////////////////////////////////////////////////////////////////////////////

			/*! Gemm device-level operator. This is an interface to efficient NIHILUS GEMM kernels that may
  be invoked from host code.

  The contributions of this class are:
    
    1. At compile time, it maps data types and high-level structural parameters onto 
       specific NIHILUS components.

    2. At runtime, it maps logical arguments to GEMM problems to kernel parameters.

    3. At runtime, it launches kernels on the device.

  The intent is to provide a convenient mechanism for interacting with most plausible GEMM
  configurations for each supported architecture. Consequently, not all parameters are exposed
  to the top-level interface. Rather, sensible defaults at each level of the NIHILUS hierarchy
  are selected to tradeoff simplicity of the interface with flexibility. We expect 
  most configurations to be specified at this level. Applications with more exotic requirements 
  may construct their kernels of interest using NIHILUS components at the threadblock, warp, 
  and thread levels of abstraction.

  NIHILUS exposes computations using the functor design pattern in which objects compose some
  internal state with an overloaded function call operator. This enables decoupling of
  initialization from execution, possibly reducing overhead during steady state phases of
  application execution.

  NIHILUS device-level operators expose an Arguments structure encompassing each logical
  input to the computation. This is distinct from the kernel-level Params structure pattern
  which contains application-specific precomputed state needed by the device code.

  Example of a NIHILUS GEMM operator implementing the functionality of cuBLAS's SGEMM NN
  is as follows:

    //
    // Instantiate the NIHILUS GEMM operator.
    //

    nihilus_gemm::gemm::device::Gemm<
      float,
      nihilus_gemm::layout::ColumnMajor,
      float,
      nihilus_gemm::layout::ColumnMajor,
      float,
      nihilus_gemm::layout::ColumnMajor
    > gemm_op;

    //
    // Launch the GEMM operation on the device
    //

    nihilus_gemm::Status status = gemm_op({
      {m, n, k},                          // GemmCoord problem_size,
      {A, lda},                           // TensorRef<float, layout::ColumnMajor> ref_A,
      {B, ldb},                           // TensorRef<float, layout::ColumnMajor> ref_B,
      {C, ldc},                           // TensorRef<float, layout::ColumnMajor> ref_C,
      {D, ldd},                           // TensorRef<float, layout::ColumnMajor> ref_D,
      {alpha, beta}                       // EpilogueOutputOp::Params epilogue_op_params
    });


  A simplified view of the template is listed below.

    template <
      /// Element type for A matrix operand
      typename ElementA,
      
      /// Layout type for A matrix operand
      typename LayoutA,
      
      /// Element type for B matrix operand
      typename ElementB,
      
      /// Layout type for B matrix operand
      typename LayoutB,
      
      /// Element type for C and D matrix operands
      typename ElementC,
      
      /// Layout type for C and D matrix operands
      typename LayoutC,
      
      /// Element type for internal accumulation
      typename ElementAccumulator,

      /// Operator class tag
      typename OperatorClass,
      
      /// Tag indicating architecture to tune for.  This is the minimum SM that
      /// supports the intended feature. The device kernel can be built
      /// targeting any SM larger than this number.
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
      int Stages
    >
    class Gemm;
*/
			/// Element type for B matrix operand
			template<uint64_t M, uint64_t K, typename ElementA_, typename LayoutA_, typename ElementB_, typename LayoutB_, typename ElementC_, typename LayoutC_> class Gemm
				: public DefaultGemmConfiguration<arch::OpClassSimt, arch::Sm120, ElementA_, ElementB_, ElementC_, ElementC_> {
				static constexpr bool SplitKSerial = false;
				static constexpr bool GatherA	   = false;
				static constexpr bool GatherB	   = false;
				static constexpr bool ScatterD	   = false;

			  public:
				using base				 = DefaultGemmConfiguration<arch::OpClassSimt, arch::Sm120, ElementA_, ElementB_, ElementC_, ElementC_>;
				using ElementA			 = ElementA_;
				using LayoutA			 = LayoutA_;
				using TensorRefA		 = TensorRef<ElementA const, LayoutA>;
				using ElementB			 = ElementB_;
				using LayoutB			 = LayoutB_;
				using TensorRefB		 = TensorRef<ElementB const, LayoutB>;
				using ElementC			 = ElementC_;
				using LayoutC			 = LayoutC_;
				using TensorRefC		 = TensorRef<ElementC const, LayoutC>;
				using TensorRefD		 = TensorRef<ElementC, LayoutC>;
				using ElementAccumulator = ElementC_;
				using OperatorClass		 = arch::OpClassSimt;
				using ArchTag			 = arch::Sm120;
				using ThreadblockShape	 = base::ThreadblockShape;
				using WarpShape			 = base::WarpShape;
				using InstructionShape	 = base::InstructionShape;
				using EpilogueOutputOp	 = base::EpilogueOutputOp;
				using ThreadblockSwizzle = typename threadblock::GemmIdentityThreadblockSwizzle<M, K>;
				using Operator			 = base::Operator;

				static constexpr int kStages				  = base::kStages;
				static constexpr int kAlignmentA			  = base::kAlignmentA;
				static constexpr int kAlignmentB			  = base::kAlignmentB;
				static constexpr int kAlignmentC			  = EpilogueOutputOp::kCount;
				static constexpr bool kSplitKSerial			  = SplitKSerial;
				static constexpr ComplexTransform kTransformA = ComplexTransform::kNone;
				static constexpr ComplexTransform kTransformB = ComplexTransform::kNone;

				using GemmKernel = typename kernel::DefaultGemm<M, K, ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC, LayoutC, ElementAccumulator,
					OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, kStages, kSplitKSerial, Operator,
					SharedMemoryClearOption::kNone, GatherA, GatherB, ScatterD, layout::NoPermute>::GemmKernel;

				struct Arguments {
					nihilus_gemm::gemm::constexpresh_gemm_coord<M, K> problem_size;
					TensorRef<ElementA const, LayoutA> ref_A;
					TensorRef<ElementB const, LayoutB> ref_B;
					TensorRef<ElementC const, LayoutC> ref_C;
					TensorRef<ElementC, LayoutC> ref_D;
					typename EpilogueOutputOp::Params epilogue;
					static constexpr int split_k_slices{ 1 };
					int const* gather_A_indices;
					int const* gather_B_indices;
					int const* scatter_D_indices;

					NIHILUS_HOST_DEVICE Arguments() : problem_size(0, 0, 0) {
					}

					NIHILUS_HOST_DEVICE
					Arguments(nihilus_gemm::gemm::constexpresh_gemm_coord<M, K> problem_size_, TensorRef<ElementA const, LayoutA> ref_A_, TensorRef<ElementB const, LayoutB> ref_B_,
						TensorRef<ElementC const, LayoutC> ref_C_, TensorRef<ElementC, LayoutC> ref_D_,
						typename EpilogueOutputOp::Params epilogue_ = typename EpilogueOutputOp::Params(), int const* gather_A_indices_ = nullptr,
						int const* gather_B_indices_ = nullptr, int const* scatter_D_indices_ = nullptr)
						: problem_size(problem_size_), ref_A(ref_A_), ref_B(ref_B_), ref_C(ref_C_), ref_D(ref_D_), epilogue(epilogue_), gather_A_indices(gather_A_indices_),
						  gather_B_indices(gather_B_indices_), scatter_D_indices(scatter_D_indices_) {
					}
				};

			  private:
				// Compute grid dimensions - keeping the original constexpr lambda approach
				static constexpr auto new_dimensions = []() {
					ThreadblockSwizzle threadblock_swizzle;
					constexpr auto grid_shape = threadblock_swizzle.template get_tiled_shape<Arguments::split_k_slices>(constexpresh_gemm_coord<M, K>{},
						constexpresh_gemm_coord<ThreadblockShape::kM, ThreadblockShape::kK>{});
					return grid_shape;
				}();

				// Cache kernel configuration constants
				static constexpr int kThreadCount	   = GemmKernel::kThreadCount;
				static constexpr int kSmemSize		   = int(sizeof(typename GemmKernel::SharedStorage));
				static constexpr bool kNeedsSmemConfig = (kSmemSize >= (48 << 10));

				using params_type = typename GemmKernel::Params<decltype(new_dimensions)::M, decltype(new_dimensions)::K>;

			  public:
				Gemm() {
				}

				// Optimized static initialization
				__forceinline__ static params_type initialize(Arguments const& args) noexcept {
					ThreadblockSwizzle threadblock_swizzle;
					auto grid_shape = new_dimensions;// Copy compile-time computed shape

					// Only compute the dynamic part at runtime
					grid_shape.N = (args.problem_size.n() + ThreadblockShape::kN - 1) / ThreadblockShape::kN;

					return params_type{ args.problem_size, grid_shape, args.ref_A.non_const_ref(), args.ref_B.non_const_ref(), args.ref_C.non_const_ref(), args.ref_D,
						args.epilogue, nullptr, args.gather_A_indices, args.gather_B_indices, args.scatter_D_indices };
				}

				// Separate error handling to avoid inlining bloat
				__noinline__ static Status handle_cuda_error(cudaError_t result) noexcept {
					return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
				}

				// Optimized static kernel launch
				__forceinline__ static Status run(params_type const& params) noexcept {
					ThreadblockSwizzle threadblock_swizzle;

					const dim3 grid = threadblock_swizzle.get_grid_shape(params.grid_tiled_shape);
					constexpr dim3 block(kThreadCount, 1, 1);// Make block size constexpr

					cudaError_t result = cudaSuccess;

					// Conditional shared memory configuration
					if constexpr (kNeedsSmemConfig) {
						result = cudaFuncSetAttribute(reinterpret_cast<const void*>(&Kernel<decltype(new_dimensions)::M, decltype(new_dimensions)::K, GemmKernel>),
							cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
						if (result != cudaSuccess) {
							return Status::kErrorInternal;
						}
					}

					nihilus_gemm::arch::synclog_setup();

					nihilus_gemm::Kernel<decltype(new_dimensions)::M, decltype(new_dimensions)::K, GemmKernel><<<grid, block, kSmemSize>>>(params);

					result = cudaGetLastError();
					return handle_cuda_error(result);
				}

				// Combined optimized static implementation
				__forceinline__ static Status impl(Arguments const& args) noexcept {
					const auto params = initialize(args);
					return run(params);
				}
			};


			////////////////////////////////////////////////////////////////////////////////

		}// namespace device
	}// namespace gemm
}// namespace nihilus_gemm

////////////////////////////////////////////////////////////////////////////////
