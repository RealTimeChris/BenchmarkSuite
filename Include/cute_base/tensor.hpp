/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <cute_base/tensor_impl.hpp>

//
// Extended Engines
//

#include <cute_base/pointer_swizzle.hpp>
#include <cute_base/pointer_sparse.hpp>
#include <cute_base/pointer_flagged.hpp>
#include <cute_base/tensor_zip.hpp>

//
// Tensor Algorithms
//

#include <cute_base/algorithm/tensor_algorithms.hpp>
#include <cute_base/algorithm/fill.hpp>
#include <cute_base/algorithm/clear.hpp>
#include <cute_base/algorithm/copy.hpp>
#include <cute_base/algorithm/prefetch.hpp>
#include <cute_base/algorithm/axpby.hpp>
#include <cute_base/algorithm/gemm.hpp>

#include <cute_base/algorithm/cooperative_copy.hpp>
#include <cute_base/algorithm/cooperative_gemm.hpp>

//
// Utilities
//

#include <cute_base/util/print_tensor.hpp>
#include <cute_base/util/print_latex.hpp>
