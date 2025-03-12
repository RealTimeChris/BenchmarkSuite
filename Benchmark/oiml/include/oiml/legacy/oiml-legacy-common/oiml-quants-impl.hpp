#pragma once
#define OIML_COMMON_DECL_CPP
#include <oiml/legacy/oiml-legacy-common/oiml-common.hpp>
#include <oiml/common/common.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-quants.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-impl.hpp>
#include <oiml/legacy/oiml-legacy-cpu/oiml-cpu-impl.hpp>
#include <oiml/legacy/oiml-legacy-cpu/oiml-cpu-quants.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-cpu.hpp>

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>// for qsort
#include <stdio.h>// for OIML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#if defined(_MSC_VER)
	// disable "possible loss of data" to avoid warnings for hundreds of casts
	// we should just be careful :)
	#pragma warning(disable : 4244 4267)
#endif

#define UNUSED OIML_UNUSED

// reference implementation for deterministic creation of model files
OIML_FORCE_INLINE void quantize_row_q8_0_ref(const float* __restrict x, oiml::block_q8_0<oiml_half>* __restrict y, int64_t k) {
	assert(k % oiml::Q_SIZE == 0);
	const uint64_t nb = static_cast<uint64_t>(k) / oiml::Q_SIZE;

	for (uint64_t i = 0; i < nb; i++) {
		float amax = 0.0f;// absolute max

		for (uint64_t j = 0; j < oiml::Q_SIZE; j++) {
			const float v = x[i * oiml::Q_SIZE + j];
			amax		  = MAX(amax, fabsf(v));
		}

		const float d  = amax / ((1 << 7) - 1);
		const float id = d ? 1.0f / d : 0.0f;

		y[i].d = oiml::oiml_fp32_to_fp16(d);

		for (uint64_t j = 0; j < oiml::Q_SIZE; ++j) {
			const float x0 = x[i * oiml::Q_SIZE + j] * id;

			y[i].qs[j] = roundf(x0);
		}
	}
}

OIML_FORCE_INLINE void quantize_row_q8_0_bindings_ref(const float* __restrict x, oiml_tensor_binding* __restrict vy, int64_t offset, int64_t k) {
	OIML_ASSERT(k % oiml::Q_SIZE == 0);
	OIML_ASSERT(vy->type == oiml::oiml_representation_types::q8_0);
	OIML_ASSERT(vy->num_channels == 1);
	OIML_ASSERT(vy->data_channels[0].type == oiml_data_channel_type::block);
	OIML_ASSERT(vy->data_channels[0].data_type == oiml::oiml_representation_types::q8_0);

	oiml::block_q8_0<oiml_half>* y = reinterpret_cast<oiml::block_q8_0<oiml_half>*>(reinterpret_cast<uint8_t*>(vy->data_channels[0].data) + offset);

	quantize_row_q8_0_ref(x, y, k);
}

OIML_FORCE_INLINE size_t quantize_q8_0(const float* __restrict src, void* __restrict dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
	OIML_UNUSED(quant_weights);
	const size_t get_row_size = oiml_row_size(oiml::oiml_representation_types::q8_0, n_per_row);
	quantize_row_q8_0_ref(src, ( oiml::block_q8_0<oiml_half>* )dst, ( int64_t )nrow * n_per_row);
	return nrow * get_row_size;
}

OIML_FORCE_INLINE void dequantize_row_q8_0(const oiml::block_q8_0<oiml_half>* __restrict x, float* __restrict y, int64_t k) {
	static constexpr int qk = oiml::Q_SIZE;

	const uint64_t nb = static_cast<uint64_t>(k) / oiml::Q_SIZE;

	for (uint64_t i = 0; i < nb; i++) {
		const float d = oiml::oiml_lookup_fp16_to_fp32(x[i].d);

		for (uint64_t j = 0; j < oiml::Q_SIZE; ++j) {
			y[i * oiml::Q_SIZE + j] = x[i].qs[j] * d;
		}
	}
}

OIML_FORCE_INLINE void dequantize_row_q8_0_bindings(const oiml_tensor_binding* __restrict vx, float* __restrict y, int64_t offset, int64_t k) {
	OIML_ASSERT(vx->type == oiml::oiml_representation_types::q8_0);
	OIML_ASSERT(vx->num_channels == 1);
	OIML_ASSERT(vx->data_channels[0].type == oiml_data_channel_type::block);
	OIML_ASSERT(vx->data_channels[0].data_type == oiml::oiml_representation_types::q8_0);

	const oiml::block_q8_0<oiml_half>* x = reinterpret_cast<const oiml::block_q8_0<oiml_half>*>(reinterpret_cast<uint8_t*>(vx->data_channels[0].data) + offset);
	dequantize_row_q8_0(x, y, k);
}

// =============================== data validation

static bool validate_float(float f, size_t i) {
	if (isinf(f)) {
		fprintf(stderr, "oiml_validate_row_data: found inf value at oiml::block %zu\n", i);
		return false;
	}

	if (isnan(f)) {
		fprintf(stderr, "oiml_validate_row_data: found nan value at oiml::block %zu\n", i);
		return false;
	}

	return true;
}

static bool isinf_fp16(oiml_fp16_t f) {
	return (f & 0x7c00) == 0x7c00 && (f & 0x03ff) == 0;
}

static bool isnan_fp16(oiml_fp16_t f) {
	return (f & 0x7c00) == 0x7c00 && (f & 0x03ff) != 0;
}

static bool validate_fp16(oiml_fp16_t f, size_t i) {
	if (isinf_fp16(f)) {
		fprintf(stderr, "oiml_validate_row_data: found inf value at oiml::block %zu\n", i);
		return false;
	}

	if (isnan_fp16(f)) {
		fprintf(stderr, "oiml_validate_row_data: found nan value at oiml::block %zu\n", i);
		return false;
	}

	return true;
}

#define VALIDATE_ROW_DATA_D_F16_IMPL(type, data, nb) \
	const type* q = ( const type* )(data); \
	for (size_t i = 0; i < (nb); ++i) { \
		if (!validate_fp16(q[i].d, i)) { \
			return false; \
		} \
	}

#define VALIDATE_ROW_DATA_DM_F16_IMPL(type, data, nb, d, m) \
	const type* q = ( const type* )(data); \
	for (size_t i = 0; i < (nb); ++i) { \
		if (!validate_fp16(q[i].d, i) || !validate_fp16(q[i].m, i)) { \
			return false; \
		} \
	}

#define VALIDATE_ROW_DATA_DVEC_F16_IMPL(type, data, nb, nr) \
	const type* q = ( const type* )(data); \
	for (size_t i = 0; i < (nb); ++i) { \
		for (size_t j = 0; j < (nr); ++j) { \
			if (!validate_fp16(q[i].d[j], i)) { \
				return false; \
			} \
		} \
	}

OIML_FORCE_INLINE bool oiml_validate_row_data(oiml::oiml_representation_types type, const void* data, size_t nbytes) {
	if (static_cast<int32_t>(type) < 0 || type >= oiml::oiml_representation_types::count) {
		fprintf(stderr, "%s: invalid type %d\n", __func__, type);
		return false;
	}

	if (nbytes % oiml_type_size(type) != 0) {
		fprintf(stderr, "%s: invalid size %zu for type %s (type size = %zu)\n", __func__, nbytes, oiml_type_name(type), oiml_type_size(type));
		return false;
	}

	const size_t nb = nbytes / oiml_type_size(type);

	switch (type) {
		case oiml::oiml_representation_types::brain_float_16: {
			int nans				= 0;
			int infs				= 0;
			const unsigned short* f = ( const unsigned short* )data;
			for (size_t i = 0; i < nb; ++i) {
				nans += (f[i] & 0x7fff) > 0x7f80;
				infs += (f[i] & 0x7fff) == 0x7f80;
			}
			if (nans) {
				fprintf(stderr, "%s: found %d NaNs in row of %zu BF16 values\n", __func__, nans, nb);
				return false;
			}
			if (infs) {
				fprintf(stderr, "%s: found %d infinities in row of %zu BF16 values\n", __func__, infs, nb);
				return false;
			}
		} break;
		case oiml::oiml_representation_types::float_16: {
			const oiml_fp16_t* f = ( const oiml_fp16_t* )data;
			size_t i			 = 0;
#if defined(__AVX2__)
			for (; i + 15 < nb; i += 16) {
				__m256i v	 = _mm256_loadu_si256(( const __m256i* )(f + i));
				__m256i vexp = _mm256_and_si256(v, _mm256_set1_epi16(0x7c00));
				__m256i cmp	 = _mm256_cmpeq_epi16(vexp, _mm256_set1_epi16(0x7c00));
				int mask	 = _mm256_movemask_epi8(cmp);
				if (mask) {
					for (size_t j = 0; j < 16; ++j) {
						if (!validate_fp16(f[i + j], i + j)) {
							return false;
						}
					}
					OIML_UNREACHABLE();
				}
			}
#elif defined(__ARM_NEON)
			for (; i + 7 < nb; i += 8) {
				uint16x8_t v	= vld1q_u16(f + i);
				uint16x8_t vexp = vandq_u16(v, vdupq_n_u16(0x7c00));
				uint16x8_t cmp	= vceqq_u16(vexp, vdupq_n_u16(0x7c00));
				uint64_t mask	= vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(cmp, 4)), 0);
				if (mask) {
					for (size_t j = 0; j < 8; ++j) {
						if (!validate_fp16(f[i + j], i + j)) {
							return false;
						}
					}
					OIML_UNREACHABLE();
				}
			}
#endif
			for (; i < nb; ++i) {
				if (!validate_fp16(f[i], i)) {
					return false;
				}
			}
		} break;
		case oiml::oiml_representation_types::float_32: {
			const float* f = ( const float* )data;
			size_t i	   = 0;
#if defined(__AVX2__)
			for (; i + 7 < nb; i += 8) {
				__m256i v	 = _mm256_loadu_si256(( const __m256i* )(f + i));
				__m256i vexp = _mm256_and_si256(v, _mm256_set1_epi32(0x7f800000));
				__m256i cmp	 = _mm256_cmpeq_epi32(vexp, _mm256_set1_epi32(0x7f800000));
				int mask	 = _mm256_movemask_epi8(cmp);
				if (mask) {
					for (size_t j = 0; j < 8; ++j) {
						if (!validate_float(f[i + j], i + j)) {
							return false;
						}
					}
					OIML_UNREACHABLE();
				}
			}
#elif defined(__ARM_NEON)
			for (; i + 3 < nb; i += 4) {
				uint32x4_t v	= vld1q_u32(( const uint32_t* )f + i);
				uint32x4_t vexp = vandq_u32(v, vdupq_n_u32(0x7f800000));
				uint32x4_t cmp	= vceqq_u32(vexp, vdupq_n_u32(0x7f800000));
				uint64_t mask	= vget_lane_u64(vreinterpret_u64_u16(vshrn_n_u32(cmp, 8)), 0);
				if (mask) {
					for (size_t j = 0; j < 4; ++j) {
						if (!validate_float(f[i + j], i + j)) {
							return false;
						}
					}
					OIML_UNREACHABLE();
				}
			}
#endif
			for (; i < nb; ++i) {
				if (!validate_float(f[i], i)) {
					return false;
				}
			}
		} break;
		case oiml::oiml_representation_types::q8_0: {
			VALIDATE_ROW_DATA_D_F16_IMPL(oiml::block_q8_0<oiml_half>, data, nb);
		} break;
		case oiml::oiml_representation_types::int_32:
			// nothing to validate
			break;
		default: {
			fprintf(stderr, "%s: invalid type %d\n", __func__, type);
			return false;
		}
	}

	return true;
}
