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
    \brief Architecture-specific operators on memory
*/

#pragma once

#include "nihilus_gemm/cutlass.h"
#include "nihilus_gemm/arch/cache_operation.h"
#include "nihilus_gemm/platform/platform.h"

namespace nihilus_gemm {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store loaded data
    typename AccessType,
    /// The bytes of loading
    int LoadBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always
    >
struct global_load;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11)) &&                                  \
    defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  #define CUTLASS_ENABLE_L2_PREFETCH 1
#else
  #define CUTLASS_ENABLE_L2_PREFETCH 0
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_load<AccessType,
                   4,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  unsigned &data = reinterpret_cast<unsigned &>(D);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  mov.b32 %0, %3;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
        "  @p ld.global.L2::128B.u32 %0, [%1];\n"
#else
        "  @p ld.global.u32 %0, [%1];\n"
#endif
        "}\n"
        : "=r"(data)
        : "l"(ptr), "r"((int)pred_guard), "r"(data));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store data
    typename AccessType,
    /// The bytes of storing
    int StoreBytes
    >
struct global_store;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_store<AccessType, 4> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
  uint32_t const &data = reinterpret_cast<uint32_t const &>(D);
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %2, 0;\n"
      "  @p st.global.u32 [%0], %1;\n"
      "}\n"
      :
      : "l"(ptr), "r"(data), "r"((int)pred_guard));
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace nihilus_gemm

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "nihilus_gemm/arch/memory_sm75.h"
#include "nihilus_gemm/arch/memory_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
