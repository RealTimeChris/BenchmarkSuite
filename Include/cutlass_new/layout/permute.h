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
    \brief Defines layout functions used by GEMM+permute path for common tensor or matrix formats.

    Like Layout functions, permute layout functions map logical coordinates to linear memory. They often require additional
    data to describe strides between elements.

    Permute layout functions must implement all members in the interface of NoPermute<> defined in this file. Address offset
    computation lies in operator() with private member variables  {col_permute_, row_permute_ and stride_} as new addresses after permute op.
*/
#pragma once
#include "cutlass_new/cutlass.h"
#include CUDA_STD_HEADER(cassert)
#include "cutlass_new/fast_math.h"
#include "cutlass_new/layout/pitch_linear.h"
#include "cutlass_new/layout/matrix.h"
#include "cutlass_new/coord.h"
#include "cutlass_new/tensor_coord.h"

namespace cutlass {
namespace layout {

// template<PermuteTag, typename Layout, bool Inverse>
// struct PermuteSelect {
//   // Try to give a reasonable error message to the user
//   static_assert(!platform::is_same<Permute, Permute>::value, // aka always_false<T>
//                 "You've tried to use a layout permutation for which the implementation is not availble. "
//                 "In order to provide an implementation for a particular combination of matrix layout "
//                 "and direction (direct/inverse), please specialize PermuteSelect trait.");
// };

// Base template for defining specializations of permutation inverses
template<typename Permute>
struct InversePermute
{
  // Try to give a reasonable error message to the user
  static_assert(!platform::is_same<Permute, Permute>::value, // aka always_false<T>
                "To apply permutation to a GEMM input operand (A or B), an inverse permutation for the desired "
                "permute class must be defined and enabled by specializing cutlass::layout::InversePermute trait.");
};

class PermuteBase {
public:
  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;
};

class NoPermute : public PermuteBase {
public:
  //
  // Methods
  //

  /// Constructor from matrix extent
  CUTLASS_HOST_DEVICE
  NoPermute(MatrixCoord extent, Index stride) { };

  /// Constructor from pitch-linear extent
  CUTLASS_HOST_DEVICE
  NoPermute(PitchLinearCoord extent, Index stride) { };

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord coord) const { return 0; } // not correct but should never be called

  /// Computes the offset after Permute Op in logical elements
  CUTLASS_HOST_DEVICE
  LongIndex operator()(PitchLinearCoord coord) const { return 0; } // not correct but should never be called
};

/// Helper trait to detect if permute operation is a noop
template<typename Permute>
inline bool constexpr is_trivial_permute = platform::is_same<Permute, cutlass::layout::NoPermute>::value;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines permute layouts of various tensor formats.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//  Tensor4DPermute0213
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Permute layout function for 4-D permuted tensors with matrix (dimensions [M, N]) reshaped
/// as [M/D1, D1, D2, N/D2]. Then perform permute([0, 2, 1, 3]) on the corresponding tensor.


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace layout
} // namespace cutlass
