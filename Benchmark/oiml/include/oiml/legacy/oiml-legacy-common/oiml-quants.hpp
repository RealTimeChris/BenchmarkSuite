#pragma once

#define OIML_COMMON_DECL_CPP
#include <oiml/legacy/oiml-legacy-common/oiml-common.hpp>
#include <oiml/common/common.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-final.hpp>

// OIML internal header

// These are reference versions the quantization functions
// Backends may have more optimized implementations

// Call level quantization functions (take in tensor bindings)
void quantize_row_q8_0_bindings_ref(const float* __restrict x, oiml_tensor_binding* __restrict vy, int64_t offset, int64_t k);
void dequantize_row_q8_0_bindings(const oiml_tensor_binding* __restrict vx, float* __restrict y, int64_t offset, int64_t k);
size_t quantize_q8_0_bindings(const float* __restrict src, oiml_tensor_binding* __restrict dst, int64_t offset, int64_t nrows, int64_t n_per_row, const float* imatrix);

// Low level quantization functions (take in direct memory pointers)
void quantize_row_q8_0_ref(const float* __restrict x, oiml::block_q8_0<oiml_half>* __restrict y, int64_t k);
void dequantize_row_q8_0(const oiml::block_q8_0<oiml_half>* __restrict x, float* __restrict y, int64_t k);
size_t quantize_q8_0(const float* __restrict src, void* __restrict dst, int64_t nrows, int64_t n_per_row, const float* imatrix);
