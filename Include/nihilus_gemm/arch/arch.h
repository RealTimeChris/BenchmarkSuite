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
    \brief Defines tags for architecture-specific configurations.
*/

#pragma once

#include "nihilus_gemm/nihilus_gemm.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace nihilus_gemm {
namespace arch {

constexpr int sm100_smem_capacity_bytes = 232448;  
constexpr int sm120_smem_capacity_bytes = 101376;

#if defined(__NVCC__) || defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))

/// Computes laneId within a warp
NIHILUS_DEVICE
int LaneId() {
  int ret;
  asm ("mov.u32 %0, %%laneid;" : "=r"(ret) : );
  return ret;
}

/// Computes SM number the thread is running on
NIHILUS_DEVICE
int SmId() {
  int ret;
  asm ("mov.u32 %0, %%smid;" : "=r"(ret) : );
  return ret;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Sm120 {
  static constexpr int kMinComputeCapability = 120;
};

/// Triggers a breakpoint on the device
NIHILUS_DEVICE
void device_breakpoint() {
#if defined(__CUDA_ARCH__)
  asm volatile ("  brkpt;\n");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace nihilus_gemm

////////////////////////////////////////////////////////////////////////////////////////////////////
