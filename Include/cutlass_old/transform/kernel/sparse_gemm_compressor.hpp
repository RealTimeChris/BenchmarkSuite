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

/*! \file
  \brief Compress utils for structured sparse kernels
*/

#pragma once

#include <algorithm>                           // std::fill
#include <array>                               // std::array
#include <random>                              // std::mt19937

#include "cute_old/numeric/numeric_types.hpp"      // cute_old::sizeof_bits_v
#include "cute_old/tensor.hpp"                     // cute_old::Tensor, cute_old::make_tensor
#include "cutlass_old/arch/arch.h"                 // cutlass_old::arch::SmXY
#include "cutlass_old/detail/dependent_false.hpp"  // cutlass_old::detail::dependent_false
#include "cutlass_old/gemm/gemm.h"                 // cutlass_old::TagToStrideA_t
#include "cutlass_old/fast_math.h"                 // cutlass_old::ceil_div, cutlass_old::round_up
#include "cutlass_old/numeric_size.h"              // cutlass_old::bits_to_bytes

#include "cutlass_old/transform/kernel/sm90_sparse_gemm_compressor.hpp"

namespace cutlass_old::transform::kernel {

template<
  class ProblemShape_,
  class ElementA_,
  class LayoutATag_,
  class SparseConfig_
>
class StructuredSparseCompressorUtility {
public:
  using SparseConfig = SparseConfig_;
  using ProblemShape = ProblemShape_;

  //* EltA
  using ElementA = ElementA_;
  using LayoutATag = LayoutATag_;
  using StrideA = cutlass_old::gemm::TagToStrideA_t<LayoutATag>;
  using ElementAMmaRaw = typename SparseConfig::ElementAMmaRaw;
  using ElementASparsity = typename SparseConfig::ElementASparsity;
  using ElementAMmaSparsity = typename SparseConfig::ElementAMmaSparsity;

  //* EltE
  using ElementEMmaRaw = typename SparseConfig::ElementEMmaRaw;
  using ElementEMmaSparsity = typename SparseConfig::ElementEMmaSparsity;

  //* AtomE
  using TensorEAtom = typename SparseConfig::TensorEAtom;
  using TensorEAtomK = typename SparseConfig::TensorEAtomK;
  using TensorEAtomM = typename SparseConfig::TensorEAtomM;

  static constexpr int ElemsARawPerElementAMmaRaw = typename SparseConfig::ElemsARawPerElementAMmaRaw{};
  static constexpr int LogicalElemsAPerChunk = typename SparseConfig::LogicalElemsAPerChunk{};
  static constexpr int PhysicalElemsAPerChunk = typename SparseConfig::PhysicalElemsAPerChunk{};
  static constexpr int LogicalElemsAMmaRawPerChunk = cutlass_old::ceil_div(LogicalElemsAPerChunk, ElemsARawPerElementAMmaRaw);
  static constexpr int PhysicalElemsAMmaRawPerChunk = cutlass_old::ceil_div(PhysicalElemsAPerChunk, ElemsARawPerElementAMmaRaw);

  //* Alignment
  static constexpr int TensorEAlignmentM = typename SparseConfig::TensorEAlignmentM{};
  static constexpr int TensorEAlignmentK = typename SparseConfig::TensorEAlignmentK{};
  static constexpr int TensorAAlignmentK = typename SparseConfig::TensorAAlignmentK{};
  static constexpr int TensorAAlignmentM = typename SparseConfig::TensorAAlignmentM{};

  StructuredSparseCompressorUtility() = default;

  StructuredSparseCompressorUtility(ProblemShape problem, StrideA dA) {
    set_problem_size(problem, dA);
  }

  void set_problem_size(ProblemShape problem, StrideA dA_) {
    M = cute_old::size<0>(problem);
    K = cute_old::size<2>(problem);
    L = cute_old::size<3>(problem);

    // The following three vars are logical elem count!
    K_alignedA  = round_up(K, TensorAAlignmentK);
    M_alignedA  = round_up(M, TensorAAlignmentM);
    K_alignedE = round_up(K, TensorEAlignmentK);
    M_alignedE = round_up(M, TensorEAlignmentM);

    dA = dA_;
  }

  /**
   * @brief Get the TensorE number of ElementE along K after alignment requirement
   * 
   * @return int : number of ElementE (uint8_t) along K-dim
   */
  int get_metadata_m_physical() const {
    return M_alignedE;
  }

  /**
   * @brief Get the TensorE number of ElementE along M after alignment requirement
   * 
   * @return int : number of ElementE (uint8_t) along M-dim
   */
  int get_metadata_k_physical() const {
    return K_alignedE / ElementEMmaSparsity{};
  }

  /**
   * @brief Get the TensorACompressed number of ElementA along K after alignment requirement
   * 
   * @return int : number of ElementA along K-dim
   */
  int get_tensorA_k_physical() const {
    return K_alignedA / ElementASparsity{};
  }

  /**
   * @brief Get the TensorACompressed number of ElementA along M after alignment requirement
   * 
   * @return int : number of ElementA along M-dim
   */
  int get_tensorA_m_physical() const {
    return M_alignedA;
  }

  /**
   * @brief Get the TensorACompressed Bytes
   * 
   * @return uint64_t bytes
   */
  uint64_t get_compressed_tensor_A_bytes() const {
    const auto tensor_a_comp_num_elt_a = get_tensorA_m_physical() * get_tensorA_k_physical() * L;
    const auto tensor_a_comp_bytes = cutlass_old::bits_to_bytes<uint64_t>(tensor_a_comp_num_elt_a * cute_old::sizeof_bits_v<ElementA>);
    return tensor_a_comp_bytes;
  }

  /**
   * @brief Get the TensorA Bytes
   * 
   * @return uint64_t bytes
   */
  uint64_t get_raw_tensor_A_bytes() const {
    const auto tensor_a_num_elt_a = uint64_t(M) * uint64_t(K) * uint64_t(L);
    const auto tensor_a_bytes = cutlass_old::bits_to_bytes<uint64_t>(tensor_a_num_elt_a * cute_old::sizeof_bits_v<ElementA>);
    return tensor_a_bytes;
  }

  /**
   * @brief Get the TensorE Bytes
   * 
   * @return uint64_t bytes
   */
  uint64_t get_tensor_E_bytes() const {
    const auto tensor_e_num_elt_a = uint64_t(get_metadata_m_physical()) * uint64_t(get_metadata_k_physical()) * uint64_t(L);
    const auto tensor_e_bytes = cutlass_old::bits_to_bytes<uint64_t>(tensor_e_num_elt_a * cute_old::sizeof_bits_v<ElementEMmaRaw>);
    return tensor_e_bytes;
  }

  constexpr auto fill_layoutA_from_compressor() const {
    return SparseConfig::fill_layoutA(cute_old::make_tuple(M,_1{},K,L));
  }

  constexpr auto fill_layoutE_from_compressor() const {
    return SparseConfig::fill_layoutE(cute_old::make_tuple(M,_1{},K,L));
  }

  void structure_sparse_zero_mask_fill(void* host_a_ptr, uint64_t seed) {
    
    constexpr int ChunkSize = LogicalElemsAMmaRawPerChunk;
    using ChunkElement = cute_old::uint_bit_t<cute_old::sizeof_bits_v<ElementAMmaRaw>>;

    cute_old::Tensor gA_eltA = cute_old::make_tensor(
        cute_old::recast_ptr<ElementA>(host_a_ptr),
        cute_old::make_layout(make_shape(M, K, L), dA));

    // Input TensorA is handled in unit of ElementAMmaRaw instead of ElementA
    cute_old::Tensor gA = cute_old::recast<ChunkElement>(gA_eltA);

    // Extract out the Chunk from K-mode
    Tensor gA_chunk = cute_old::zipped_divide(gA, cute_old::Shape<_1,cute_old::Int<ChunkSize>>{}); // (Chunk, Rest)

    // Half of the data is zero to indicate sparsityA = 2
    std::array<int, ChunkSize> nnzb_indicator{};
    for (size_t i = 1; i < nnzb_indicator.size(); i += 2) {
      nnzb_indicator.at(i) = 1;
    }

    std::mt19937 rng(seed);
    auto rest_shape = cute_old::shape<1>(gA_chunk);
    for (auto iter = cute_old::make_coord_iterator(rest_shape); iter != cute_old::ForwardCoordIteratorSentinel{}; ++iter) {
      std::shuffle(nnzb_indicator.begin(), nnzb_indicator.end(), rng);
      for (int c = 0; c < size<0>(gA_chunk); ++c) {                        // for each elem within chunk
        if (nnzb_indicator[c] == 0) {
          gA_chunk(c, *iter) = ChunkElement{0};
        }
      }  // end of within chunk
    }    // end of chunk_idx
  }

  int M{-1};
  int K{-1};
  int L{-1};
  StrideA dA{};

private:
  int K_alignedA{-1};
  int M_alignedA{-1};
  int K_alignedE{-1};
  int M_alignedE{-1};
};

////////////////////////////////////////////////////////////////////////////////

template<
  class ProblemShape,
  class ElementA,
  class LayoutATag,
  class SparseConfig,
  class ArchTag
>
struct StructuredSparseCompressorSelector {
  static_assert(cutlass_old::detail::dependent_false<ArchTag>,
      "Could not select a structured sparse compressor for given parameters.");
};

template<
  class ProblemShape,
  class ElementA,
  class LayoutATag,
  class SparseConfig
>
struct StructuredSparseCompressorSelector<
    ProblemShape,
    ElementA,
    LayoutATag,
    SparseConfig,
    arch::Sm90> {
  using Compressor = SM90StructuredSparseCompressor<
    ProblemShape,
    ElementA,
    LayoutATag,
    SparseConfig
  >;
};

template<
  class ProblemShape,
  class ElementA,
  class LayoutATag,
  class SparseConfig
>
struct StructuredSparseCompressorSelector<
    ProblemShape,
    ElementA,
    LayoutATag,
    SparseConfig,
    arch::Sm100> {
  using Compressor = SM90StructuredSparseCompressor<
    ProblemShape,
    ElementA,
    LayoutATag,
    SparseConfig
  >;
};

template<
  class ProblemShape,
  class ElementA,
  class LayoutATag,
  class SparseConfig
>
struct StructuredSparseCompressorSelector<
    ProblemShape,
    ElementA,
    LayoutATag,
    SparseConfig,
    arch::Sm120> {
  using Compressor = SM90StructuredSparseCompressor<
    ProblemShape,
    ElementA,
    LayoutATag,
    SparseConfig
  >;
};

template<
  class ProblemShape,
  class ElementA,
  class LayoutATag,
  class SparseConfig,
  class ArchTag
>
using StructuredSparseCompressor = typename StructuredSparseCompressorSelector<
    ProblemShape,
    ElementA,
    LayoutATag,
    SparseConfig,
    ArchTag
>::Compressor;

} // End namespace cutlass_old::transform::kernel
