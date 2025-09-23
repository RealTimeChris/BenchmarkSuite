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
  \brief Kernel performing a reduction over densely packed tensors in global memory
*/

#pragma once

#include "cutlass_rt_tm/device_kernel.h"
#include "cutlass_rt_tm/reduction/kernel/reduce_split_k.h"
#include "cutlass_rt_tm/cuda_host_adapter.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_rt_tm {
namespace reduction {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ReductionKernel_
>
class ReduceSplitK {
public:
  using ReductionKernel = ReductionKernel_;

  using Shape = typename ReductionKernel::Shape;
  using ReductionOp = typename ReductionKernel::ReductionOp;
  using OutputOp = typename ReductionKernel::OutputOp;

  using ElementWorkspace = typename ReductionKernel::ElementWorkspace;
  using ElementAccumulator = typename ReductionKernel::ElementAccumulator;
  using ElementOutput = typename ReductionKernel::ElementOutput;

  using WorkspaceTensorRef = typename ReductionKernel::WorkspaceTensorRef;
  using OutputTensorRef = typename ReductionKernel::OutputTensorRef;

  using StrideIndex = typename ReductionKernel::StrideIndex;

  static constexpr bool kEnableCudaHostAdapter = CUTLASS_RT_TMENABLE_CUDA_HOST_ADAPTER;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    MatrixCoord problem_size{0,0};
    int partitions{1};
    size_t partition_stride{0};
    WorkspaceTensorRef workspace{};
    OutputTensorRef destination{};
    OutputTensorRef source{};
    typename OutputOp::Params output{};
    typename ReductionOp::Params reduction{};

    //
    // Methods
    //

    /// Default ctor
    Arguments() = default;
   
    CUTLASS_RT_TMHOST_DEVICE 
    Arguments(
      MatrixCoord const & problem_size
    ):
      problem_size(problem_size) { }

    CUTLASS_RT_TMHOST_DEVICE
    Arguments(
      MatrixCoord problem_size_,
      int partitions_,
      size_t partition_stride_,
      WorkspaceTensorRef workspace_,
      OutputTensorRef destination_,
      OutputTensorRef source_,
      typename OutputOp::Params output_ = typename OutputOp::Params(),
      typename ReductionOp::Params reduction_ = typename ReductionOp::Params()
    ):
      problem_size(problem_size_),
      partitions(partitions_),
      partition_stride(partition_stride_),
      workspace(workspace_),
      destination(destination_),
      source(source_),
      output(output_),
      reduction(reduction_)
    {

    }

  };

private:
  /// Kernel parameters object
  typename ReductionKernel::Params params_;

public:
  /// Constructs Reduction SplitK
  ReduceSplitK() { }

  /// Determines whether the ReduceSplitK can execute the given problem.
  static Status can_implement(Arguments const &args) {

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    // needs no additional workspace
    return 0;
  }

  /// Initializes Reduction state from arguments.
  Status initialize(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    // initialize the params structure from the arguments
    params_ = typename ReductionKernel::Params(
      args.problem_size,
      args.partitions,
      args.partition_stride,
      args.workspace,
      args.destination,
      args.source,
      args.output,
      args.reduction
    );

    return Status::kSuccess;

   }

  /// Initializes Reduction kernel state from arguments.
  Status update(Arguments const &args, void *workspace = nullptr) {

    // update the params structure from the arguments
    params_.workspace.reset(args.workspace.non_const_ref().data());
    params_.destination.reset(args.destination.non_const_ref().data());
    params_.source.reset(args.source.non_const_ref().data());
    params_.output = args.output;
    params_.reduction = args.reduction;

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr, int32_t kernel_index = 0) {

    //
    // Launch reduction kernel
    //
    dim3 block = ReductionKernel::block_shape();
    dim3 grid = ReductionKernel::grid_shape(params_.problem_size);

    if constexpr (kEnableCudaHostAdapter) {
        CUTLASS_RT_TMASSERT(cuda_adapter);
        if (cuda_adapter) {
          void* kernel_params[] = {&params_};
          cuda_adapter->launch(
              grid, dim3(1,1,1), block, 0, stream, kernel_params, kernel_index);
        }
    }
    else {
      cutlass_rt_tm::arch::synclog_setup();
      Kernel<ReductionKernel><<< grid, block, 0, stream >>>(params_);
    }

    cudaError_t result = cudaGetLastError();
    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }


  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr, int32_t kernel_index = 0) {
    return run(stream, cuda_adapter, kernel_index);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr, int32_t kernel_index = 0) {
    
    Status status = initialize(args, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(stream,cuda_adapter, kernel_index);
    }

    return status;
  }
  
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace reduction
} // namespace cutlass_rt_tm
