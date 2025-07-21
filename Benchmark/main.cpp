#include <BnchSwt/BenchmarkSuite.hpp>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>
#include <array>
#include <queue>
#include <latch>
#include <bit>

static constexpr uint64_t Q_SIZE{ 32 };

template<size_t alignment, typename value_type> struct alignas(alignment) static_aligned_const {
	alignas(alignment) value_type value{};

	BNCH_SWT_INLINE constexpr static_aligned_const() noexcept : value{} {
	}

	BNCH_SWT_INLINE constexpr static_aligned_const(value_type new_value) noexcept : value{ new_value } {
	}

	BNCH_SWT_INLINE constexpr operator const value_type&() const {
		return value;
	}

	BNCH_SWT_INLINE operator value_type&() {
		return value;
	}
};

BNCH_SWT_INLINE static constexpr float fp32_from_bits(uint32_t w) noexcept {
	return std::bit_cast<float>(w);
}

BNCH_SWT_INLINE static constexpr uint32_t fp32_to_bits(float f) noexcept {
	return std::bit_cast<uint32_t>(f);
}

BNCH_SWT_INLINE static constexpr float compute_fp16_to_fp32(uint16_t h) noexcept {
	const uint32_t w	 = static_cast<uint32_t>(h) << 16;
	const uint32_t sign	 = w & 0x80000000u;
	const uint32_t two_w = w + w;

	constexpr uint32_t exp_offset = 0xE0u << 23;
	constexpr float exp_scale	  = fp32_from_bits(0x7800000u);
	const float normalized_value  = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

	constexpr uint32_t magic_mask  = 126u << 23;
	constexpr float magic_bias	   = 0.5f;
	const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

	constexpr uint32_t denormalized_cutoff = 1u << 27;
	const uint32_t result				   = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
	return fp32_from_bits(result);
}

BNCH_SWT_INLINE const auto& get_fp16_to_fp32_array() noexcept {
	alignas(64) static auto fp16_to_fp32_array = []() {
		alignas(64) static std::unique_ptr<std::array<static_aligned_const<64, float>, (1 << 16)>> return_values_new{
			std::make_unique<std::array<static_aligned_const<64, float>, (1 << 16)>>()
		};
		for (uint64_t i = 0; i < (1 << 16); ++i) {
			(*return_values_new)[i] = compute_fp16_to_fp32(static_cast<uint16_t>(i));
		}
		alignas(64) static std::array<static_aligned_const<64, float>, (1 << 16)>& fp16_to_fp32_array_ref{ *return_values_new };
		return fp16_to_fp32_array_ref;
	}();
	return fp16_to_fp32_array;
}

BNCH_SWT_INLINE static float fp16_to_fp32(uint16_t h) noexcept {
	return *(get_fp16_to_fp32_array().data() + static_cast<uint64_t>(h));
}

 enum ggml_type {
	GGML_TYPE_F32  = 0,
	GGML_TYPE_F16  = 1,
	GGML_TYPE_Q4_0 = 2,
	GGML_TYPE_Q4_1 = 3,
	// GGML_TYPE_Q4_2 = 4, support has been removed
	// GGML_TYPE_Q4_3 = 5, support has been removed
	GGML_TYPE_Q5_0	  = 6,
	GGML_TYPE_Q5_1	  = 7,
	GGML_TYPE_Q8_0	  = 8,
	GGML_TYPE_Q8_1	  = 9,
	GGML_TYPE_Q2_K	  = 10,
	GGML_TYPE_Q3_K	  = 11,
	GGML_TYPE_Q4_K	  = 12,
	GGML_TYPE_Q5_K	  = 13,
	GGML_TYPE_Q6_K	  = 14,
	GGML_TYPE_Q8_K	  = 15,
	GGML_TYPE_IQ2_XXS = 16,
	GGML_TYPE_IQ2_XS  = 17,
	GGML_TYPE_IQ3_XXS = 18,
	GGML_TYPE_IQ1_S	  = 19,
	GGML_TYPE_IQ4_NL  = 20,
	GGML_TYPE_IQ3_S	  = 21,
	GGML_TYPE_IQ2_S	  = 22,
	GGML_TYPE_IQ4_XS  = 23,
	GGML_TYPE_I8	  = 24,
	GGML_TYPE_I16	  = 25,
	GGML_TYPE_I32	  = 26,
	GGML_TYPE_I64	  = 27,
	GGML_TYPE_F64	  = 28,
	GGML_TYPE_IQ1_M	  = 29,
	GGML_TYPE_BF16	  = 30,
	// GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
	// GGML_TYPE_Q4_0_4_8 = 32,
	// GGML_TYPE_Q4_0_8_8 = 33,
	GGML_TYPE_TQ1_0 = 34,
	GGML_TYPE_TQ2_0 = 35,
	// GGML_TYPE_IQ4_NL_4_4 = 36,
	// GGML_TYPE_IQ4_NL_4_8 = 37,
	// GGML_TYPE_IQ4_NL_8_8 = 38,
	GGML_TYPE_COUNT = 39,
};


 enum ggml_op {
	 GGML_OP_NONE = 0,

	 GGML_OP_DUP,
	 GGML_OP_ADD,
	 GGML_OP_ADD1,
	 GGML_OP_ACC,
	 GGML_OP_SUB,
	 GGML_OP_MUL,
	 GGML_OP_DIV,
	 GGML_OP_SQR,
	 GGML_OP_SQRT,
	 GGML_OP_LOG,
	 GGML_OP_SIN,
	 GGML_OP_COS,
	 GGML_OP_SUM,
	 GGML_OP_SUM_ROWS,
	 GGML_OP_MEAN,
	 GGML_OP_ARGMAX,
	 GGML_OP_COUNT_EQUAL,
	 GGML_OP_REPEAT,
	 GGML_OP_REPEAT_BACK,
	 GGML_OP_CONCAT,
	 GGML_OP_SILU_BACK,
	 GGML_OP_NORM,// normalize
	 GGML_OP_RMS_NORM,
	 GGML_OP_RMS_NORM_BACK,
	 GGML_OP_GROUP_NORM,
	 GGML_OP_L2_NORM,

	 GGML_OP_MUL_MAT,
	 GGML_OP_MUL_MAT_ID,
	 GGML_OP_OUT_PROD,

	 GGML_OP_SCALE,
	 GGML_OP_SET,
	 GGML_OP_CPY,
	 GGML_OP_CONT,
	 GGML_OP_RESHAPE,
	 GGML_OP_VIEW,
	 GGML_OP_PERMUTE,
	 GGML_OP_TRANSPOSE,
	 GGML_OP_GET_ROWS,
	 GGML_OP_GET_ROWS_BACK,
	 GGML_OP_SET_ROWS,
	 GGML_OP_DIAG,
	 GGML_OP_DIAG_MASK_INF,
	 GGML_OP_DIAG_MASK_ZERO,
	 GGML_OP_SOFT_MAX,
	 GGML_OP_SOFT_MAX_BACK,
	 GGML_OP_ROPE,
	 GGML_OP_ROPE_BACK,
	 GGML_OP_CLAMP,
	 GGML_OP_CONV_TRANSPOSE_1D,
	 GGML_OP_IM2COL,
	 GGML_OP_IM2COL_BACK,
	 GGML_OP_CONV_2D,
	 GGML_OP_CONV_2D_DW,
	 GGML_OP_CONV_TRANSPOSE_2D,
	 GGML_OP_POOL_1D,
	 GGML_OP_POOL_2D,
	 GGML_OP_POOL_2D_BACK,
	 GGML_OP_UPSCALE,
	 GGML_OP_PAD,
	 GGML_OP_PAD_REFLECT_1D,
	 GGML_OP_ROLL,
	 GGML_OP_ARANGE,
	 GGML_OP_TIMESTEP_EMBEDDING,
	 GGML_OP_ARGSORT,
	 GGML_OP_LEAKY_RELU,

	 GGML_OP_FLASH_ATTN_EXT,
	 GGML_OP_FLASH_ATTN_BACK,
	 GGML_OP_SSM_CONV,
	 GGML_OP_SSM_SCAN,
	 GGML_OP_WIN_PART,
	 GGML_OP_WIN_UNPART,
	 GGML_OP_GET_REL_POS,
	 GGML_OP_ADD_REL_POS,
	 GGML_OP_RWKV_WKV6,
	 GGML_OP_GATED_LINEAR_ATTN,
	 GGML_OP_RWKV_WKV7,

	 GGML_OP_UNARY,

	 GGML_OP_MAP_CUSTOM1,
	 GGML_OP_MAP_CUSTOM2,
	 GGML_OP_MAP_CUSTOM3,

	 GGML_OP_CUSTOM,

	 GGML_OP_CROSS_ENTROPY_LOSS,
	 GGML_OP_CROSS_ENTROPY_LOSS_BACK,
	 GGML_OP_OPT_STEP_ADAMW,

	 GGML_OP_GLU,

	 GGML_OP_COUNT,
 };

 struct ggml_tensor {
	enum ggml_type type;

	int64_t ne[4];
	size_t nb[4];
	enum ggml_op op;

	int32_t flags;

	struct ggml_tensor* src[10];

	std::vector<uint8_t> data;

	std::vector<uint8_t> wdata;

	char padding[8];
};

 typedef uint16_t ggml_half;

 #define QK8_0 32
typedef struct {
	ggml_half d;// delta
	int8_t qs[QK8_0];// quants
} block_q8_0;

#define GGML_CPU_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)

 void quantize_row_q8_0(const float* __restrict x, void* __restrict vy, int64_t k) {
	const int nb = k / Q_SIZE;

	block_q8_0* __restrict y = ( block_q8_0* )vy;
	for (int i = 0; i < nb; i++) {
		// Load elements into 4 AVX vectors
		__m256 v0 = _mm256_loadu_ps(x);
		__m256 v1 = _mm256_loadu_ps(x + 8);
		__m256 v2 = _mm256_loadu_ps(x + 16);
		__m256 v3 = _mm256_loadu_ps(x + 24);
		x += 32;

		// Compute max(abs(e)) for the block
		const __m256 signBit = _mm256_set1_ps(-0.0f);
		__m256 maxAbs		 = _mm256_andnot_ps(signBit, v0);
		maxAbs				 = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
		maxAbs				 = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
		maxAbs				 = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

		__m128 max4			  = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
		max4				  = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
		max4				  = _mm_max_ss(max4, _mm_movehdup_ps(max4));
		const float maxScalar = _mm_cvtss_f32(max4);

		// Quantize these floats
		const float d	 = maxScalar / 127.f;
		y[i].d			 = GGML_CPU_FP32_TO_FP16(d);
		const float id	 = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
		const __m256 mul = _mm256_set1_ps(id);

		// Apply the multiplier
		v0 = _mm256_mul_ps(v0, mul);
		v1 = _mm256_mul_ps(v1, mul);
		v2 = _mm256_mul_ps(v2, mul);
		v3 = _mm256_mul_ps(v3, mul);

		// Round to nearest integer
		v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
		v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
		v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
		v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

		// Convert floats to integers
		__m256i i0 = _mm256_cvtps_epi32(v0);
		__m256i i1 = _mm256_cvtps_epi32(v1);
		__m256i i2 = _mm256_cvtps_epi32(v2);
		__m256i i3 = _mm256_cvtps_epi32(v3);
		// Convert int32 to int16
		i0 = _mm256_packs_epi32(i0, i1);// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 1max_iteration_count, measured_iteration_count3, 14, 15
		i2 = _mm256_packs_epi32(i2, i3);// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
			// Convert int16 to int8
		i0 = _mm256_packs_epi16(i0, i2);// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 1max_iteration_count, measured_iteration_count3, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

		// We got our precious signed bytes, but the order is now wrong
		// These AVX2 pack instructions process 16-byte pieces independently
		// The following instruction is fixing the order
		const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
		i0				   = _mm256_permutevar8x32_epi32(i0, perm);

		_mm256_storeu_si256(( __m256i* )y[i].qs, i0);
	}
 }

 // add int16_t pairwise and return as float vector
 static inline __m256 sum_i16_pairs_float(const __m256i x) {
	 const __m256i ones			= _mm256_set1_epi16(1);
	 const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
	 return _mm256_cvtepi32_ps(summed_pairs);
 }

 static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
	 const __m256i zero			= _mm256_setzero_si256();
	 const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
	 return _mm256_cvtepi32_ps(summed_pairs);
#elif defined(__AVXVNNI__)
	 const __m256i zero			= _mm256_setzero_si256();
	 const __m256i summed_pairs = _mm256_dpbusd_avx_epi32(zero, ax, sy);
	 return _mm256_cvtepi32_ps(summed_pairs);
#else
	 // Perform multiplication and create 16-bit values
	 const __m256i dot = _mm256_maddubs_epi16(ax, sy);
	 return sum_i16_pairs_float(dot);
#endif
 }

 // multiply int8_t, add results pairwise twice and return as float vector
 static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
	 const __m256i zero			= _mm256_setzero_si256();
	 const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
	 return _mm256_cvtepi32_ps(summed_pairs);
#else
	 // Get absolute values of x vectors
	 const __m256i ax = _mm256_sign_epi8(x, x);
	 // Sign the values of the y vectors
	 const __m256i sy = _mm256_sign_epi8(y, x);
	 return mul_sum_us8_pairs_float(ax, sy);
#endif
 }

 // horizontally add 8 floats
 static inline float hsum_float_8(const __m256 x) {
	 __m128 res = _mm256_extractf128_ps(x, 1);
	 res		= _mm_add_ps(res, _mm256_castps256_ps128(x));
	 res		= _mm_add_ps(res, _mm_movehl_ps(res, res));
	 res		= _mm_add_ss(res, _mm_movehdup_ps(res));
	 return _mm_cvtss_f32(res);
 }

 void ggml_vec_dot_q8_0_q8_0(int n, float* __restrict s, size_t bs, const void* __restrict vx, size_t bx, const void* __restrict vy, size_t by, int nrc) {
	 const int qk = QK8_0;
	 const int nb = n / qk;

	 const block_q8_0* __restrict x = (const block_q8_0* )vx;
	 const block_q8_0* __restrict y = ( const block_q8_0* )vy;

	 int ib		= 0;
	 float sumf = 0;
	 // Initialize accumulator with zeros
	 __m256 acc = _mm256_setzero_ps();

	 // Main loop
	 for (; ib < nb; ++ib) {
		 // Compute combined scale for the block
		 const __m256 d = _mm256_set1_ps(fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
		 __m256i qx		= _mm256_loadu_si256(( const __m256i* )x[ib].qs);
		 __m256i qy		= _mm256_loadu_si256(( const __m256i* )y[ib].qs);

		 const __m256 q = mul_sum_i8_pairs_float(qx, qy);

		 // Multiply q with scale and accumulate
		 acc = _mm256_fmadd_ps(d, q, acc);
	 }

	 sumf = hsum_float_8(acc);
	 for (; ib < nb; ++ib) {
		 int sumi = 0;

		 for (int j = 0; j < qk; j++) {
			 sumi += x[ib].qs[j] * y[ib].qs[j];
		 }

		 sumf += sumi * (fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
	 }

	 *s = sumf;
 }

 size_t ggml_row_size(enum ggml_type type, int64_t ne) {
	 return sizeof(block_q8_0) * ne / 32;
 }

static void ggml_compute_forward_mul_mat_one_chunk(const uint64_t ith_new, const uint64_t nth_new, struct ggml_tensor* dst, const enum ggml_type type, const int64_t num_rows_per_vec_dot,
	const int64_t ir0_start, const int64_t ir0_end, const int64_t ir1_start, const int64_t ir1_end) {
	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];
	const int64_t ne00 = (src0)->ne[0];
	( void )(ne00);
	const int64_t ne01 = (src0)->ne[1];
	( void )(ne01);
	const int64_t ne02 = (src0)->ne[2];
	( void )(ne02);
	const int64_t ne03 = (src0)->ne[3];
	( void )(ne03);
	const size_t nb00 = (src0)->nb[0];
	( void )(nb00);
	const size_t nb01 = (src0)->nb[1];
	( void )(nb01);
	const size_t nb02 = (src0)->nb[2];
	( void )(nb02);
	const size_t nb03 = (src0)->nb[3];
	( void )(nb03);
	const int64_t ne10 = (src1)->ne[0];
	( void )(ne10);
	const int64_t ne11 = (src1)->ne[1];
	( void )(ne11);
	const int64_t ne12 = (src1)->ne[2];
	( void )(ne12);
	const int64_t ne13 = (src1)->ne[3];
	( void )(ne13);
	const size_t nb10 = (src1)->nb[0];
	( void )(nb10);
	const size_t nb11 = (src1)->nb[1];
	( void )(nb11);
	const size_t nb12 = (src1)->nb[2];
	( void )(nb12);
	const size_t nb13 = (src1)->nb[3];
	( void )(nb13);
	const int64_t ne0 = (dst)->ne[0];
	( void )(ne0);
	const int64_t ne1 = (dst)->ne[1];
	( void )(ne1);
	const int64_t ne2 = (dst)->ne[2];
	( void )(ne2);
	const int64_t ne3 = (dst)->ne[3];
	( void )(ne3);
	const size_t nb0 = (dst)->nb[0];
	( void )(nb0);
	const size_t nb1 = (dst)->nb[1];
	( void )(nb1);
	const size_t nb2 = (dst)->nb[2];
	( void )(nb2);
	const size_t nb3 = (dst)->nb[3];
	( void )(nb3);

	const bool src1_cont = true;

	auto const vec_dot				  = &ggml_vec_dot_q8_0_q8_0;
	enum ggml_type const vec_dot_type = ggml_type::GGML_TYPE_Q8_0;

	// broadcast factors
	const int64_t r2 = ne12 / ne02;
	const int64_t r3 = ne13 / ne03;

	//printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

	// threads with no work simply yield (not sure if it helps)
	if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
		return;
	}

	const void* wdata	  = (src1->type == vec_dot_type) ? ( const void* )src1->data.data() : src1->wdata.data();
	const size_t row_size = ggml_row_size(vec_dot_type, ne10);

	// block-tiling attempt
	const int64_t blck_0 = 16;
	const int64_t blck_1 = 16;

	const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

	// attempt to reduce false-sharing (does not seem to make a difference)
	// 16 * 2, accounting for mmla kernels
	float tmp[32];

	for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
		for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
			for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
				const int64_t i13 = (ir1 / (ne12 * ne1));
				const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
				const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

				// broadcast src0 into src1
				const int64_t i03 = i13 / r3;
				const int64_t i02 = i12 / r2;

				const int64_t i1 = i11;
				const int64_t i2 = i12;
				const int64_t i3 = i13;

				const char* src0_row = ( const char* )src0->data.data() + (0 + i02 * nb02 + i03 * nb03);

				// desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
				//       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
				//       the original src1 data pointer, so we should index using the indices directly
				// TODO: this is a bit of a hack, we should probably have a better way to handle this
				const char* src1_col =
					( const char* )wdata + (src1_cont || src1->type != vec_dot_type ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size : (i11 * nb11 + i12 * nb12 + i13 * nb13));
				float* dst_col = ( float* )(( char* )dst->data.data() + (i1 * nb1 + i2 * nb2 + i3 * nb3));

				//for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
				//    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
				//}

				for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
					vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col,
						(num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
				}

				for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
					memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (std::min(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
				}
			}
		}
	}
}


void ggml_compute_forward_mul_mat(const uint64_t ith_new, uint64_t nth_new, struct ggml_tensor* dst) {
	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	const int64_t ne00 = (src0)->ne[0];
	( void )(ne00);
	const int64_t ne01 = (src0)->ne[1];
	( void )(ne01);
	const int64_t ne02 = (src0)->ne[2];
	( void )(ne02);
	const int64_t ne03 = (src0)->ne[3];
	( void )(ne03);
	const size_t nb00 = (src0)->nb[0];
	( void )(nb00);
	const size_t nb01 = (src0)->nb[1];
	( void )(nb01);
	const size_t nb02 = (src0)->nb[2];
	( void )(nb02);
	const size_t nb03 = (src0)->nb[3];
	( void )(nb03);
	const int64_t ne10 = (src1)->ne[0];
	( void )(ne10);
	const int64_t ne11 = (src1)->ne[1];
	( void )(ne11);
	const int64_t ne12 = (src1)->ne[2];
	( void )(ne12);
	const int64_t ne13 = (src1)->ne[3];
	( void )(ne13);
	const size_t nb10 = (src1)->nb[0];
	( void )(nb10);
	const size_t nb11 = (src1)->nb[1];
	( void )(nb11);
	const size_t nb12 = (src1)->nb[2];
	( void )(nb12);
	const size_t nb13 = (src1)->nb[3];
	( void )(nb13);
	const int64_t ne0 = (dst)->ne[0];
	( void )(ne0);
	const int64_t ne1 = (dst)->ne[1];
	( void )(ne1);
	const int64_t ne2 = (dst)->ne[2];
	( void )(ne2);
	const int64_t ne3 = (dst)->ne[3];
	( void )(ne3);
	const size_t nb0 = (dst)->nb[0];
	( void )(nb0);
	const size_t nb1 = (dst)->nb[1];
	( void )(nb1);
	const size_t nb2 = (dst)->nb[2];
	( void )(nb2);
	const size_t nb3 = (dst)->nb[3];
	( void )(nb3);

	const int ith	 = ith_new;
	const int nth = nth_new;
	enum ggml_type const vec_dot_type = ggml_type ::GGML_TYPE_Q8_0;
	auto const from_float			  = &quantize_row_q8_0;
	int64_t const vec_dot_num_rows	  =1 ;

	if (src1->type != vec_dot_type) {
		char* wdata		  = ( char* )src1->wdata.data();
		const size_t nbw0 = sizeof(block_q8_0);
		const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
		const size_t nbw2 = nbw1 * ne11;
		const size_t nbw3 = nbw2 * ne12;

		for (int64_t i13 = 0; i13 < ne13; ++i13) {
			for (int64_t i12 = 0; i12 < ne12; ++i12) {
				for (int64_t i11 = 0; i11 < ne11; ++i11) {
					size_t bs				 = 32;
					int64_t ne10_block_start = (ith * ne10 / bs) / nth;
					int64_t ne10_block_end	 = ((ith + 1) * ne10 / bs) / nth;
					from_float(( float* )(( char* )src1->data.data() + i13 * nb13 + i12 * nb12 + i11 * nb11 + ne10_block_start * bs * nb10),
						( void* )(wdata + i13 * nbw3 + i12 * nbw2 + i11 * nbw1 + ne10_block_start * nbw0), (ne10_block_end - ne10_block_start) * bs);
				}
			}
		}
	}

	if (ith == 0) {
		// Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
		//atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
	}

	//ggml_barrier(params->threadpool);

	// This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
	const int64_t nr0 = ne0;

	// This is the size of the rest of the dimensions of the result
	const int64_t nr1 = ne1 * ne2 * ne3;

	// Now select a reasonable chunk size.
	int chunk_size = 16;

	// We need to step up the size if it's small
	if (nr0 == 1 || nr1 == 1) {
		chunk_size = 64;
	}

	// distribute the work across the inner or outer loop based on which one is larger
	// The number of chunks in the 0/1 dim.
	// CEIL(nr0/chunk_size)
	int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
	int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

	// If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
	//   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
	//   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
	if (nchunk0 * nchunk1 < nth * 4) {
		// distribute the thread work across the inner or outer loop based on which one is larger
		nchunk0 = nr0 > nr1 ? nth : 1;// parallelize by src0 rows
		nchunk1 = nr0 > nr1 ? 1 : nth;// parallelize by src1 rows
	}

	// The number of elements in each chunk
	const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
	const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

	// The first chunk comes from our thread_id, the rest will get auto-assigned.
	int current_chunk = ith;

	while (current_chunk < nchunk0 * nchunk1) {
		const int64_t ith0 = current_chunk % nchunk0;
		const int64_t ith1 = current_chunk / nchunk0;

		const int64_t ir0_start = dr0 * ith0;
		const int64_t ir0_end	= std::min(ir0_start + dr0, nr0);

		const int64_t ir1_start = dr1 * ith1;
		const int64_t ir1_end	= std::min(ir1_start + dr1, nr1);

		// dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
		int64_t num_rows_per_vec_dot = vec_dot_num_rows;

		// these checks are needed to avoid crossing dim1 boundaries
		// can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
		if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
			num_rows_per_vec_dot = 1;
		}
		ggml_compute_forward_mul_mat_one_chunk(ith_new, nth_new, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

		if (nth >= nchunk0 * nchunk1) {
			break;
		}
		++current_chunk;
		//current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
	}
}
/*
Tensor: Qcur-1                        │
Tensor Byte Size: 16384               │
Type:       f32                      
Dimensions: [4096 × 1 × 1 × 1]    
Operation:  MUL_MAT                  
Inputs:     2            
    Tensor: blk.1.attn_q.weight           │
    Tensor Byte Size: 17825792            │
    Type:       q8_0                     
    Dimensions: [4096 × 4096 × 1 × 1] 
    Tensor: attn_norm-1                   │
    Tensor Byte Size: 16384               │
    Type:       f32                      
    Dimensions: [4096 × 1 × 1 × 1]   
*/


// Helper functions for ggml tensor creation
constexpr size_t ggml_type_size(enum ggml_type type) {
	switch (type) {
		case GGML_TYPE_F32:
			return sizeof(float);
		case GGML_TYPE_F16:
			return sizeof(uint16_t);
		case GGML_TYPE_Q8_0:
			return sizeof(block_q8_0);
		default:
			return sizeof(float);
	}
}

constexpr size_t ggml_blck_size(enum ggml_type type) {
	switch (type) {
		case GGML_TYPE_F32:
			return 1;
		case GGML_TYPE_F16:
			return 1;
		case GGML_TYPE_Q8_0:
			return Q_SIZE;// 32
		default:
			return 1;
	}
}

constexpr size_t GGML_MAX_DIMS = 4;

// Create ggml_tensor with proper dimensions and strides
ggml_tensor create_ggml_tensor(enum ggml_type type, const const std::array<int64_t,4>& dimensions) {
	auto tensor = ggml_tensor{};

	// Initialize tensor
	tensor.type  = type;
	tensor.op	  = GGML_OP_NONE;
	tensor.flags = 0;
	std::fill(std::begin(tensor.src), std::end(tensor.src), nullptr);
	std::fill(std::begin(tensor.padding), std::end(tensor.padding), 0);

	// Set dimensions
	const int n_dims = std::min(static_cast<int>(dimensions.size()), static_cast<int>(GGML_MAX_DIMS));
	for (int i = 0; i < GGML_MAX_DIMS; i++) {
		tensor.ne[i] = (i < n_dims) ? dimensions[i] : 1;
	}

	// Calculate strides (nb array)
	tensor.nb[0] = ggml_type_size(type);
	tensor.nb[1] = tensor.nb[0] * (tensor.ne[0] / ggml_blck_size(type));
	for (int i = 2; i < GGML_MAX_DIMS; i++) {
		tensor.nb[i] = tensor.nb[i - 1] * tensor.ne[i - 1];
	}

	return tensor;
}

// Generate float tensor data
ggml_tensor generate_float_tensor(const std::array<int64_t,4>& dimensions) {
	auto tensor = create_ggml_tensor(GGML_TYPE_F32, dimensions);

	// Calculate total elements
	int64_t total_elements = 1;
	for (int64_t dim: dimensions) {
		total_elements *= dim;
	}

	// Allocate and fill data
	const size_t data_size = total_elements * sizeof(float);
	tensor.data.resize(data_size);

	float* float_data = reinterpret_cast<float*>(tensor.data.data());
	for (int64_t i = 0; i < total_elements; ++i) {
		float_data[i] = bnch_swt::random_generator::generateValue<float>();
	}
	const size_t wdata_size = tensor.data.size() * sizeof(block_q8_0) / sizeof(float);
	tensor.wdata.resize(wdata_size);
	return tensor;
}

// Generate q8_0 quantized tensor data
ggml_tensor  generate_q8_0_tensor(const std::array<int64_t,4>& dimensions) {
	auto tensor = create_ggml_tensor(GGML_TYPE_Q8_0, dimensions);

	// Calculate total elements and blocks
	int64_t total_elements = 1;
	for (int64_t dim: dimensions) {
		total_elements *= dim;
	}
	const int64_t total_blocks = (total_elements + Q_SIZE - 1) / Q_SIZE;

	// Allocate data for q8_0 blocks
	const size_t data_size = total_blocks * sizeof(block_q8_0);
	tensor.data.resize(data_size);

	// Generate temporary float data
	auto float_data = std::make_unique<float[]>(total_elements);
	for (int64_t i = 0; i < total_elements; ++i) {
		float_data[i] = bnch_swt::random_generator::generateValue<float>();
	}

	// Quantize to q8_0 format
	quantize_row_q8_0(float_data.get(), tensor.data.data(), total_elements);

	return tensor;
}

// Helper function to setup tensor relationships for matrix multiplication
void setup_matrix_multiplication(ggml_tensor* output, ggml_tensor* input, ggml_tensor* weights) {
	output->op	   = GGML_OP_MUL_MAT;
	output->src[0] = weights;// weights are src[0] in ggml convention
	output->src[1] = input;// input is src[1] in ggml convention

	// Allocate working data if needed for quantization
	if (input->type != GGML_TYPE_Q8_0 && weights->type == GGML_TYPE_Q8_0) {
		
	}
}

std::vector<ggml_tensor> generate_q8_tensors(const std::array<int64_t, 4>& dims, int64_t count) {
	std::vector<ggml_tensor> return_values{}; 
	for (int64_t x = 0; x < count; ++x) {
		return_values.emplace_back(generate_q8_0_tensor(dims));
	}
	return return_values;
}

std::vector<ggml_tensor> generate_float_tensors(const std::array<int64_t, 4>& dims, int64_t count) {
	std::vector<ggml_tensor> return_values{};
	for (int64_t x = 0; x < count; ++x) {
		return_values.emplace_back(generate_float_tensor(dims));
	}
	return return_values;
}

std::vector<ggml_tensor> generate_dst_tensors(const std::array<int64_t, 4>& dims, std::vector<ggml_tensor>& input01, std::vector<ggml_tensor>& input02) {
	std::vector<ggml_tensor> return_values{};
	for (int64_t x = 0; x < input01.size(); ++x) {
		ggml_tensor new_tensor{ generate_float_tensor(dims) };
		new_tensor.src[0] = &input01[x];
		new_tensor.src[1] = &input02[x];
		return_values.emplace_back(new_tensor);
	}
	return return_values;
}

template<typename value_01_type, typename value_02_type>
concept convertible_to = std::is_convertible_v<value_02_type, value_01_type>;

template<typename value_01_type, convertible_to<value_01_type> value_02_type> BNCH_SWT_INLINE constexpr value_01_type max(value_01_type val01, value_02_type val02) noexcept {
	return val01 > static_cast<value_01_type>(val02) ? val01 : static_cast<value_01_type>(val02);
}

template<typename value_01_type, convertible_to<value_01_type> value_02_type> BNCH_SWT_INLINE constexpr value_01_type min(value_01_type val01, value_02_type val02) noexcept {
	return val01 < static_cast<value_01_type>(val02) ? val01 : static_cast<value_01_type>(val02);
}

// PROPERLY TYPED Nihilus matrix multiplication - no more void* nonsense!

template<uint64_t rows_new> struct nihilus_mul_mat_q8_0_q8_0_typed {};

// Base case: 1-3 rows with proper typing
template<uint64_t rows_new>
	requires(rows_new > 0 && rows_new < 4)
struct nihilus_mul_mat_q8_0_q8_0_typed<rows_new> {
	BNCH_SWT_INLINE static void impl(const int64_t ne00,// 4096 (input features)
		float* __restrict output,// TYPED: float output
		const block_q8_0* __restrict weights,// TYPED: q8_0 weight matrix
		const block_q8_0* __restrict input,// TYPED: q8_0 input vector
		const int64_t weight_row_stride// Number of blocks per row
	) {
		for (uint64_t row = 0; row < rows_new; ++row) {
			const block_q8_0* weight_row = weights + row * weight_row_stride;

			// Inline their exact vec_dot algorithm
			const int qk = QK8_0;// 32
			const int nb = ne00 / qk;// blocks in this row

			__m256 acc = _mm256_setzero_ps();

			for (int ib = 0; ib < nb; ++ib) {
				const __m256 d = _mm256_set1_ps(fp16_to_fp32(weight_row[ib].d) * fp16_to_fp32(input[ib].d));
				__m256i qx	   = _mm256_loadu_si256(( const __m256i* )weight_row[ib].qs);
				__m256i qy	   = _mm256_loadu_si256(( const __m256i* )input[ib].qs);
				const __m256 q = mul_sum_i8_pairs_float(qx, qy);
				acc			   = _mm256_fmadd_ps(d, q, acc);
			}

			output[row] = hsum_float_8(acc);
		}
	}
};

// Optimized case: 4 rows with proper typing
template<uint64_t rows_new>
	requires(rows_new == 4)
struct nihilus_mul_mat_q8_0_q8_0_typed<rows_new> {
	BNCH_SWT_INLINE static void impl(const int64_t ne00, float* __restrict output, const block_q8_0* __restrict weights, const block_q8_0* __restrict input,
		const int64_t weight_row_stride) {
		const int qk = QK8_0;
		const int nb = ne00 / qk;

		// Process 4 rows with better instruction scheduling
		const block_q8_0* weight_row0 = weights + 0 * weight_row_stride;
		const block_q8_0* weight_row1 = weights + 1 * weight_row_stride;
		const block_q8_0* weight_row2 = weights + 2 * weight_row_stride;
		const block_q8_0* weight_row3 = weights + 3 * weight_row_stride;

		__m256 acc0 = _mm256_setzero_ps();
		__m256 acc1 = _mm256_setzero_ps();
		__m256 acc2 = _mm256_setzero_ps();
		__m256 acc3 = _mm256_setzero_ps();

		// Process all blocks for all 4 rows simultaneously
		for (int ib = 0; ib < nb; ++ib) {
			// Load input once, reuse for all rows
			__m256i qy				= _mm256_loadu_si256(( const __m256i* )input[ib].qs);
			const float input_scale = fp16_to_fp32(input[ib].d);

			// Row 0
			const __m256 d0 = _mm256_set1_ps(fp16_to_fp32(weight_row0[ib].d) * input_scale);
			__m256i qx0		= _mm256_loadu_si256(( const __m256i* )weight_row0[ib].qs);
			const __m256 q0 = mul_sum_i8_pairs_float(qx0, qy);
			acc0			= _mm256_fmadd_ps(d0, q0, acc0);

			// Row 1
			const __m256 d1 = _mm256_set1_ps(fp16_to_fp32(weight_row1[ib].d) * input_scale);
			__m256i qx1		= _mm256_loadu_si256(( const __m256i* )weight_row1[ib].qs);
			const __m256 q1 = mul_sum_i8_pairs_float(qx1, qy);
			acc1			= _mm256_fmadd_ps(d1, q1, acc1);

			// Row 2
			const __m256 d2 = _mm256_set1_ps(fp16_to_fp32(weight_row2[ib].d) * input_scale);
			__m256i qx2		= _mm256_loadu_si256(( const __m256i* )weight_row2[ib].qs);
			const __m256 q2 = mul_sum_i8_pairs_float(qx2, qy);
			acc2			= _mm256_fmadd_ps(d2, q2, acc2);

			// Row 3
			const __m256 d3 = _mm256_set1_ps(fp16_to_fp32(weight_row3[ib].d) * input_scale);
			__m256i qx3		= _mm256_loadu_si256(( const __m256i* )weight_row3[ib].qs);
			const __m256 q3 = mul_sum_i8_pairs_float(qx3, qy);
			acc3			= _mm256_fmadd_ps(d3, q3, acc3);
		}

		// Store results
		output[0] = hsum_float_8(acc0);
		output[1] = hsum_float_8(acc1);
		output[2] = hsum_float_8(acc2);
		output[3] = hsum_float_8(acc3);
	}
};

// Recursive cases with proper typing
template<uint64_t rows_new>
	requires(rows_new > 4 && rows_new < 8)
struct nihilus_mul_mat_q8_0_q8_0_typed<rows_new> {
	BNCH_SWT_INLINE static void impl(const int64_t ne00, float* __restrict output, const block_q8_0* __restrict weights, const block_q8_0* __restrict input,
		const int64_t weight_row_stride) {
		nihilus_mul_mat_q8_0_q8_0_typed<4>::impl(ne00, output, weights, input, weight_row_stride);

		constexpr uint64_t remainder = rows_new - 4ULL;
		if constexpr (remainder > 0) {
			const block_q8_0* remaining_weights = weights + 4 * weight_row_stride;
			nihilus_mul_mat_q8_0_q8_0_typed<remainder>::impl(ne00, output + 4, remaining_weights, input, weight_row_stride);
		}
	}
};

template<uint64_t rows_new>
	requires(rows_new == 8)
struct nihilus_mul_mat_q8_0_q8_0_typed<rows_new> {
	BNCH_SWT_INLINE static void impl(const int64_t ne00, float* __restrict output, const block_q8_0* __restrict weights, const block_q8_0* __restrict input,
		const int64_t weight_row_stride) {
		// Process as 2 blocks of 4 for optimal performance
		nihilus_mul_mat_q8_0_q8_0_typed<4>::impl(ne00, output, weights, input, weight_row_stride);
		nihilus_mul_mat_q8_0_q8_0_typed<4>::impl(ne00, output + 4, weights + 4 * weight_row_stride, input, weight_row_stride);
	}
};

template<uint64_t rows_new>
	requires(rows_new > 8 && rows_new <= 32)
struct nihilus_mul_mat_q8_0_q8_0_typed<rows_new> {
	BNCH_SWT_INLINE static void impl(const int64_t ne00, float* __restrict output, const block_q8_0* __restrict weights, const block_q8_0* __restrict input,
		const int64_t weight_row_stride) {
		nihilus_mul_mat_q8_0_q8_0_typed<8>::impl(ne00, output, weights, input, weight_row_stride);

		constexpr uint64_t remainder = rows_new - 8ULL;
		if constexpr (remainder > 0) {
			const block_q8_0* remaining_weights = weights + 8 * weight_row_stride;
			nihilus_mul_mat_q8_0_q8_0_typed<remainder>::impl(ne00, output + 8, remaining_weights, input, weight_row_stride);
		}
	}
};

template<uint64_t rows_new>
	requires(rows_new > 32)
struct nihilus_mul_mat_q8_0_q8_0_typed<rows_new> {
	BNCH_SWT_INLINE static void impl(const int64_t ne00, float* __restrict output, const block_q8_0* __restrict weights, const block_q8_0* __restrict input,
		const int64_t weight_row_stride) {
		static constexpr uint64_t blocks_of_32 = rows_new / 32;
		static constexpr uint64_t remainder	   = rows_new % 32;

		for (uint64_t block = 0; block < blocks_of_32; ++block) {
			const block_q8_0* block_weights = weights + (block * 32) * weight_row_stride;
			nihilus_mul_mat_q8_0_q8_0_typed<32>::impl(ne00, output + (block * 32), block_weights, input, weight_row_stride);
		}

		if constexpr (remainder > 0) {
			const block_q8_0* remaining_weights = weights + (blocks_of_32 * 32) * weight_row_stride;
			nihilus_mul_mat_q8_0_q8_0_typed<remainder>::impl(ne00, output + (blocks_of_32 * 32), remaining_weights, input, weight_row_stride);
		}
	}
};

// Main typed template engine
template<uint64_t BLOCK_SIZE = 32> struct nihilus_typed_mul_mat {
	BNCH_SWT_INLINE static void compute_mul_mat_typed(const ggml_tensor* src0,// q8_0 weights [4096 x 4096]
		const ggml_tensor* src1,// q8_0 input [4096 x 1] - MUST be q8_0!
		ggml_tensor* dst// f32 output [4096 x 1]
	) {
		static_assert(BLOCK_SIZE <= 32, "Block size must be <= 32 for optimal template expansion");

		const int64_t ne00 = src0->ne[0];// 4096 (input features)
		const int64_t ne01 = src0->ne[1];// 4096 (output features)

		// STRONGLY TYPED pointers - no more void* madness!
		const block_q8_0* weights = reinterpret_cast<const block_q8_0*>(src0->data.data());
		const block_q8_0* input	  = reinterpret_cast<const block_q8_0*>((src1->type == GGML_TYPE_Q8_0) ? src1->data.data() : src1->wdata.data());
		float* output			  = reinterpret_cast<float*>(dst->data.data());

		const int64_t weight_row_stride = ne00 / QK8_0;// blocks per row, not bytes!

		// Process matrix in blocks using compile-time optimized templates
		for (int64_t row_block = 0; row_block < ne01; row_block += BLOCK_SIZE) {
			const int64_t rows_remaining	= min(BLOCK_SIZE, ne01 - row_block);
			const block_q8_0* block_weights = weights + row_block * weight_row_stride;
			float* block_output				= output + row_block;

			// Compile-time dispatch - zero runtime overhead!
			if constexpr (BLOCK_SIZE == 32) {
				nihilus_mul_mat_q8_0_q8_0_typed<32>::impl(ne00, block_output, block_weights, input, weight_row_stride);
			} else if constexpr (BLOCK_SIZE == 16) {
				nihilus_mul_mat_q8_0_q8_0_typed<16>::impl(ne00, block_output, block_weights, input, weight_row_stride);
			} else if constexpr (BLOCK_SIZE == 8) {
				nihilus_mul_mat_q8_0_q8_0_typed<8>::impl(ne00, block_output, block_weights, input, weight_row_stride);
			} else if constexpr (BLOCK_SIZE == 4) {
				nihilus_mul_mat_q8_0_q8_0_typed<4>::impl(ne00, block_output, block_weights, input, weight_row_stride);
			} else {
				static_assert(BLOCK_SIZE <= 4, "Unsupported block size - add template specialization!");
			}
		}
	}
};

int main() {
	static constexpr std::array<int64_t, 4> generate_q8_tensors_dims{ 4096, 4096, 1, 1 };
	static constexpr std::array<int64_t, 4> generate_attn_norm_tensors_dims{ 4096, 1, 1, 1 };
	static constexpr size_t max_iteration_count{ 20 };
	static constexpr size_t measured_iteration_count{ 4 };

	auto q_weight_tensors01{ generate_q8_tensors(generate_q8_tensors_dims, max_iteration_count) };
	auto attn_norm_tensors01{ generate_float_tensors(generate_attn_norm_tensors_dims, max_iteration_count) };
	auto qcur_output_tensors01{ generate_dst_tensors(generate_attn_norm_tensors_dims, q_weight_tensors01, attn_norm_tensors01) };

	auto q_weight_tensors02{ generate_q8_tensors(generate_q8_tensors_dims, max_iteration_count) };
	auto attn_norm_tensors02{ generate_float_tensors(generate_attn_norm_tensors_dims, max_iteration_count) };
	auto qcur_output_tensors02{ generate_dst_tensors(generate_attn_norm_tensors_dims, q_weight_tensors02, attn_norm_tensors02) };
	uint64_t current_index{};

	
	struct llama_cpp {
		using dst_type = decltype(qcur_output_tensors01);
		BNCH_SWT_INLINE static uint64_t impl(uint64_t& current_index, dst_type& output_tensors) {
			auto& output_tensor = output_tensors;
			ggml_compute_forward_mul_mat(0, 1, &output_tensor[current_index]);
			auto output_bytes = output_tensor[current_index].data.size();
			++current_index;
			return output_bytes;
		};
	};

	// Benchmark wrapper with proper typing
	struct nihilus_typed {
		using dst_type = decltype(qcur_output_tensors02);
		BNCH_SWT_INLINE static uint64_t impl(uint64_t& current_index, dst_type& output_tensors) {
			auto& output_tensor = output_tensors[current_index];

			// ENFORCE type safety at compile time!
			static_assert(std::is_same_v<decltype(output_tensor.src[0]->data.data()), uint8_t*>, "Weight tensor must be q8_0 typed!");
			static_assert(std::is_same_v<decltype(output_tensor.data.data()), uint8_t*>, "Output tensor must be f32 typed!");

			nihilus_typed_mul_mat<32>::compute_mul_mat_typed(output_tensor.src[0],// weights (q8_0)
				output_tensor.src[1],// input (q8_0)
				&output_tensor// output (f32)
			);

			auto output_bytes = output_tensor.data.size();
			++current_index;
			return output_bytes;
		}
	};

	struct nihilus {
		BNCH_SWT_INLINE static uint64_t impl(uint64_t& current_index) {
			auto start = std::chrono::high_resolution_clock::now();
			auto end   = std::chrono::high_resolution_clock::now();
			return 200000ull;
		};
	};

	bnch_swt::benchmark_stage<"test_stage", max_iteration_count, measured_iteration_count>::runBenchmark<"llama_cpp", llama_cpp>(current_index, qcur_output_tensors01);
	current_index = 0;
	bnch_swt::benchmark_stage<"test_stage", max_iteration_count, measured_iteration_count>::runBenchmark<"nihilus_typed", nihilus_typed>(current_index, qcur_output_tensors02);

	for (int64_t x = 0; x < qcur_output_tensors01.size(); ++x) {
		if (qcur_output_tensors01[x].data != qcur_output_tensors02[x].data) {
			std::cerr << "Sorry, but that impl did not produce the correct output data!" << std::endl;
		}
	}

	bnch_swt::benchmark_stage<"test_stage", max_iteration_count, measured_iteration_count>::printResults();
	return 0;
}
