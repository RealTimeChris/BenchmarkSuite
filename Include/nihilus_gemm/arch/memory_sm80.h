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
    \brief Architecture-specific operators on memory added for SM80
*/

#pragma once

#include "nihilus_gemm/nihilus_gemm.h"
#include "nihilus_gemm/complex.h"
#include "nihilus_gemm/arch/memory.h"
#include "nihilus_gemm/arch/memory_sm75.h"
#include "nihilus_gemm/arch/cache_operation.h"
#include "nihilus_gemm/arch/synclog.hpp"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  #define CUDA_CP_ASYNC_ACTIVATED 1
#else
  #define CUDA_CP_ASYNC_ACTIVATED 0
#endif

namespace nihilus_gemm {
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Initiates an asynchronous copy from global memory to shared memory.
///
/// cp.async
///
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async;

/// Initiates an asynchronous copy from global memory to shared memory. Rather than predicate
/// the entire transfer, zeros are written to SMEM if the guard predicate is false.
///
/// cp.async
///
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async_zfill;

/// Initiates an asynchronous copy from global memory to shared memory. Rather than predicate
/// the entire transfer, nans (0x7eff) are written to SMEM if the guard predicate is false.
///
/// cp.async
///
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async_nan;

/// Either 0 or 1 are written to SMEM based on input element type
/// Used for diagonal elements of triangular matrix of BLAS3 functions
///
/// st.shared
///
template <
   /// Type of Element
   typename Element,
   /// If the data is for a Hermitian matrix diagonal
   bool IsHermitianData = false>
struct cp_async_diag;

static constexpr uint32_t OOB_NAN_F16 = 0x7eff;
static constexpr uint32_t OOB_NAN_F16x2 = ((OOB_NAN_F16 << 16) | OOB_NAN_F16);

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Establishes an ordering w.r.t previously issued cp.async instructions. Does not block.
NIHILUS_DEVICE
void cp_async_fence() {
  #if CUDA_CP_ASYNC_ACTIVATED
  asm volatile("cp.async.commit_group;\n" ::);
  nihilus_gemm::arch::synclog_emit_cp_async_fence(__LINE__);
  #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Blocks until all but <N> previous cp.async.commit_group operations have committed.
template <int N>
NIHILUS_DEVICE void cp_async_wait() {
  #if CUDA_CP_ASYNC_ACTIVATED
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  nihilus_gemm::arch::synclog_emit_cp_async_wait(__LINE__, N);
  #endif
}

/// Blocks until all previous cp.async.commit_group operations have committed.
template <>
NIHILUS_DEVICE void cp_async_wait<0>() {
  #if CUDA_CP_ASYNC_ACTIVATED
  asm volatile("cp.async.wait_all;\n" ::);
  nihilus_gemm::arch::synclog_emit_cp_async_wait_all(__LINE__);
  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace arch
}  // namespace nihilus_gemm

/////////////////////////////////////////////////////////////////////////////////////////////////
