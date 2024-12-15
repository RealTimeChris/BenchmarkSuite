#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include <glaze/glaze.hpp>
#include "RandomGenerators.hpp"

#include <cstdint>
#include <cstdlib>
#if defined(_MSC_VER)
	#include <intrin.h>
#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
	#include <cpuid.h>
#endif


enum instruction_set {
	DEFAULT		= 0x0,
	NEON		= 0x1,
	AVX2		= 0x4,
	SSE42		= 0x8,
	PCLMULQDQ	= 0x10,
	BMI1		= 0x20,
	BMI2		= 0x40,
	ALTIVEC		= 0x80,
	AVX512F		= 0x100,
	AVX512DQ	= 0x200,
	AVX512IFMA	= 0x400,
	AVX512PF	= 0x800,
	AVX512ER	= 0x1000,
	AVX512CD	= 0x2000,
	AVX512BW	= 0x4000,
	AVX512VL	= 0x8000,
	AVX512VBMI2 = 0x10000,
	LSX			= 0x20000,
	LASX		= 0x40000,
};

#if defined(__PPC64__)

static inline uint32_t detect_supported_architectures() {
	return instruction_set::ALTIVEC;
}

#elif defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)

static inline uint32_t detect_supported_architectures() {
	return instruction_set::NEON;
}

#elif defined(__x86_64__) || defined(_M_AMD64)

namespace {
	// Can be found on Intel ISA Reference for CPUID
	constexpr uint32_t cpuid_avx2_bit		 = 1 << 5;///< @private Bit 5 of EBX for EAX=0x7
	constexpr uint32_t cpuid_bmi1_bit		 = 1 << 3;///< @private bit 3 of EBX for EAX=0x7
	constexpr uint32_t cpuid_bmi2_bit		 = 1 << 8;///< @private bit 8 of EBX for EAX=0x7
	constexpr uint32_t cpuid_avx512f_bit	 = 1 << 16;///< @private bit 16 of EBX for EAX=0x7
	constexpr uint32_t cpuid_avx512dq_bit	 = 1 << 17;///< @private bit 17 of EBX for EAX=0x7
	constexpr uint32_t cpuid_avx512ifma_bit	 = 1 << 21;///< @private bit 21 of EBX for EAX=0x7
	constexpr uint32_t cpuid_avx512pf_bit	 = 1 << 26;///< @private bit 26 of EBX for EAX=0x7
	constexpr uint32_t cpuid_avx512er_bit	 = 1 << 27;///< @private bit 27 of EBX for EAX=0x7
	constexpr uint32_t cpuid_avx512cd_bit	 = 1 << 28;///< @private bit 28 of EBX for EAX=0x7
	constexpr uint32_t cpuid_avx512bw_bit	 = 1 << 30;///< @private bit 30 of EBX for EAX=0x7
	constexpr uint32_t cpuid_avx512vl_bit	 = 1U << 31;///< @private bit 31 of EBX for EAX=0x7
	constexpr uint32_t cpuid_avx512vbmi2_bit = 1 << 6;///< @private bit 6 of ECX for EAX=0x7
	constexpr uint64_t cpuid_avx256_saved	 = uint64_t(1) << 2;///< @private bit 2 = AVX
	constexpr uint64_t cpuid_avx512_saved	 = uint64_t(7) << 5;///< @private bits 5,6,7 = opmask, ZMM_hi256, hi16_ZMM
	constexpr uint32_t cpuid_sse42_bit		 = 1 << 20;///< @private bit 20 of ECX for EAX=0x1
	constexpr uint32_t cpuid_osxsave		 = (uint32_t(1) << 26) | (uint32_t(1) << 27);///< @private bits 26+27 of ECX for EAX=0x1
	constexpr uint32_t cpuid_pclmulqdq_bit	 = 1 << 1;///< @private bit  1 of ECX for EAX=0x1
}

static inline void cpuid(uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
	#if defined(_MSC_VER)
	int cpu_info[4];
	__cpuidex(cpu_info, *eax, *ecx);
	*eax = cpu_info[0];
	*ebx = cpu_info[1];
	*ecx = cpu_info[2];
	*edx = cpu_info[3];
	#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
	uint32_t level = *eax;
	__get_cpuid(level, eax, ebx, ecx, edx);
	#else
	uint32_t a = *eax, b, c = *ecx, d;
	asm volatile("cpuid\n\t" : "+a"(a), "=b"(b), "+c"(c), "=d"(d));
	*eax = a;
	*ebx = b;
	*ecx = c;
	*edx = d;
	#endif
}


static inline uint64_t xgetbv() {
	#if defined(_MSC_VER)
	return _xgetbv(0);
	#else
	uint32_t xcr0_lo, xcr0_hi;
	asm volatile("xgetbv\n\t" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
	return xcr0_lo | (uint64_t(xcr0_hi) << 32);
	#endif
}

static inline uint32_t detect_supported_architectures() {
	uint32_t eax, ebx, ecx, edx;
	uint32_t host_isa = 0x0;

	eax = 0x1;
	ecx = 0x0;
	cpuid(&eax, &ebx, &ecx, &edx);

	if (ecx & cpuid_sse42_bit) {
		host_isa |= instruction_set::SSE42;
	} else {
		return host_isa;
	}

	if (ecx & cpuid_pclmulqdq_bit) {
		host_isa |= instruction_set::PCLMULQDQ;
	}


	if ((ecx & cpuid_osxsave) != cpuid_osxsave) {
		return host_isa;
	}

	uint64_t xcr0 = xgetbv();

	if ((xcr0 & cpuid_avx256_saved) == 0) {
		return host_isa;
	}

	eax = 0x7;
	ecx = 0x0;
	cpuid(&eax, &ebx, &ecx, &edx);
	if (ebx & cpuid_avx2_bit) {
		host_isa |= instruction_set::AVX2;
	}
	if (ebx & cpuid_bmi1_bit) {
		host_isa |= instruction_set::BMI1;
	}

	if (ebx & cpuid_bmi2_bit) {
		host_isa |= instruction_set::BMI2;
	}

	if (!((xcr0 & cpuid_avx512_saved) == cpuid_avx512_saved)) {
		return host_isa;
	}

	if (ebx & cpuid_avx512f_bit) {
		host_isa |= instruction_set::AVX512F;
	}

	if (ebx & cpuid_avx512dq_bit) {
		host_isa |= instruction_set::AVX512DQ;
	}

	if (ebx & cpuid_avx512ifma_bit) {
		host_isa |= instruction_set::AVX512IFMA;
	}

	if (ebx & cpuid_avx512pf_bit) {
		host_isa |= instruction_set::AVX512PF;
	}

	if (ebx & cpuid_avx512er_bit) {
		host_isa |= instruction_set::AVX512ER;
	}

	if (ebx & cpuid_avx512cd_bit) {
		host_isa |= instruction_set::AVX512CD;
	}

	if (ebx & cpuid_avx512bw_bit) {
		host_isa |= instruction_set::AVX512BW;
	}

	if (ebx & cpuid_avx512vl_bit) {
		host_isa |= instruction_set::AVX512VL;
	}

	if (ecx & cpuid_avx512vbmi2_bit) {
		host_isa |= instruction_set::AVX512VBMI2;
	}

	return host_isa;
}

#elif defined(__loongarch_sx) && !defined(__loongarch_asx)

static inline uint32_t detect_supported_architectures() {
	return instruction_set::LSX;
}

#elif defined(__loongarch_asx)

static inline uint32_t detect_supported_architectures() {
	return instruction_set::LASX;
}

#else

static inline uint32_t detect_supported_architectures() {
	return instruction_set::DEFAULT;
}


#endif
static inline float ggml_compute_fp16_to_fp32(float h) {
	return ( float )(*( float* )&h);
}
#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)

static constexpr uint64_t qkCount{ 32 };
static constexpr uint64_t nCount{ 3072 };
static constexpr uint64_t nCount4{ nCount / qkCount };

struct block_q8_0 {
	float d;
	int8_t qs[qkCount];
};

class implementation_base {
  public:
	BNCH_SWT_INLINE virtual void ggml_vec_dot_q8_0_f32(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc) const = 0;
};

#if defined(__ARM_NEON)

class arm_implementation : public implementation_base {
  public:
	BNCH_SWT_INLINE void ggml_vec_dot_q8_0_f32(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc) const override {
		static constexpr int qk = qkCount;
		const int nb			= n / qk;
		( void )(nrc);
		( void )(bx);
		( void )(by);
		( void )(bs);

		float sumf = 0;

		const block_q8_0* x = static_cast<const block_q8_0*>(vx);
		const float* y		= static_cast<const float*>(vy);

		float32x4_t sumv0 = vdupq_n_f32(0.0f);
		float32x4_t sumv1 = vdupq_n_f32(0.0f);
		float32x4_t sumv2 = vdupq_n_f32(0.0f);
		float32x4_t sumv3 = vdupq_n_f32(0.0f);
		float32x4_t sumv4 = vdupq_n_f32(0.0f);
		float32x4_t sumv5 = vdupq_n_f32(0.0f);
		float32x4_t sumv6 = vdupq_n_f32(0.0f);
		float32x4_t sumv7 = vdupq_n_f32(0.0f);

		int ib = 0;
		for (; ib < nb; ib += 1) {
			const block_q8_0* b = x++;
			const float d				 = GGML_FP16_TO_FP32(b->d);

			//        const uint8x16x2_t x0_0 = vld2q_s8(x0->qs);
			int8x16_t vec1 = vld1q_s8(b->qs);// load 16x int8_t
			int8x16_t vec2 = vld1q_s8(b->qs + 16);// load 16x int8_t
			int16x8_t x0_0 = vmovl_s8(vget_low_s8(vec1));// cast the first 8x int8_t to int16_t
			int16x8_t x0_1 = vmovl_s8(vget_high_s8(vec1));// cast the last 8x int8_t to int16_t
			int16x8_t x1_0 = vmovl_s8(vget_low_s8(vec2));// cast the first 8x int8_t to int16_t
			int16x8_t x1_1 = vmovl_s8(vget_high_s8(vec2));// cast the last 8x int8_t to int16_t
			//printf("0:%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", ((int8_t *)&vec)[0], ((int8_t *)&vec)[1], ((int8_t *)&vec)[2], ((int8_t *)&vec)[3], ((int8_t *)&vec)[4], ((int8_t *)&vec)[5], ((int8_t *)&vec)[6], ((int8_t *)&vec)[7], ((int8_t *)&vec)[8], ((int8_t *)&vec)[9], ((int8_t *)&vec)[10], ((int8_t *)&vec)[11], ((int8_t *)&vec)[12], ((int8_t *)&vec)[13], ((int8_t *)&vec)[14], ((int8_t *)&vec)[15]);
			//printf("2:%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", ((int32_t *)&x0_00)[0], ((int32_t *)&x0_00)[1], ((int32_t *)&x0_00)[2], ((int32_t *)&x0_00)[3], ((int32_t *)&x0_01)[0], ((int32_t *)&x0_01)[1], ((int32_t *)&x0_01)[2], ((int32_t *)&x0_01)[3], ((int32_t *)&x0_02)[0], ((int32_t *)&x0_02)[1], ((int32_t *)&x0_02)[2], ((int32_t *)&x0_02)[3], ((int32_t *)&x0_03)[0], ((int32_t *)&x0_03)[1], ((int32_t *)&x0_03)[2], ((int32_t *)&x0_03)[3]);
			//printf("4:%f %f %f %f %d %d %d %d %d %d %d %d %d %d %d %d\n", ((float *)&xx0)[0], ((float *)&xx0)[1], ((float *)&xx0)[2], ((float *)&xx0)[3], ((int32_t *)&x0_01)[0], ((int32_t *)&x0_01)[1], ((int32_t *)&x0_01)[2], ((int32_t *)&x0_01)[3], ((int32_t *)&x0_02)[0], ((int32_t *)&x0_02)[1], ((int32_t *)&x0_02)[2], ((int32_t *)&x0_02)[3], ((int32_t *)&x0_03)[0], ((int32_t *)&x0_03)[1], ((int32_t *)&x0_03)[2], ((int32_t *)&x0_03)[3]);

			const float32x4_t x0 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s8(x0_0))), d);
			const float32x4_t x1 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s8(x0_0))), d);
			const float32x4_t x2 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s8(x0_1))), d);
			const float32x4_t x3 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s8(x0_1))), d);
			const float32x4_t x4 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s8(x1_0))), d);
			const float32x4_t x5 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s8(x1_0))), d);
			const float32x4_t x6 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s8(x1_1))), d);
			const float32x4_t x7 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s8(x1_1))), d);

			sumv0 = vfmaq_f32(sumv0, x0, vld1q_f32(y));
			sumv1 = vfmaq_f32(sumv1, x1, vld1q_f32(y + 4));
			sumv2 = vfmaq_f32(sumv2, x2, vld1q_f32(y + 8));
			sumv3 = vfmaq_f32(sumv3, x3, vld1q_f32(y + 12));
			sumv4 = vfmaq_f32(sumv4, x4, vld1q_f32(y + 16));
			sumv5 = vfmaq_f32(sumv5, x5, vld1q_f32(y + 20));
			sumv6 = vfmaq_f32(sumv6, x6, vld1q_f32(y + 24));
			sumv7 = vfmaq_f32(sumv7, x7, vld1q_f32(y + 28));
			y += 32;
		}

		sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3) + vaddvq_f32(sumv4) + vaddvq_f32(sumv5) + vaddvq_f32(sumv6) + vaddvq_f32(sumv7);
	}
};

static constexpr arm_implementation arm_impl{};

static constexpr const implementation_base* impls[1]{ static_cast<const implementation_base*>(&arm_impl) };

const implementation_base* getImpl() {
	return impls[0];
}

#else

class avx_implementation : public implementation_base {
  public:
	BNCH_SWT_INLINE void ggml_vec_dot_q8_0_f32(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc) const override {
		static constexpr int qk = qkCount;
		const int nb			= n / qk;

		float sumf = 0;

		const block_q8_0* x = static_cast<const block_q8_0*>(vx);
		const float* y		= static_cast<const float*>(vy);

		int ib = 0;
		for (; ib < nb; ++ib) {
			float d			 = GGML_FP16_TO_FP32(x[ib].d);
			const int8_t* qs = x[ib].qs;
			for (int j = 0; j < qk; j++) {
				sumf += (( float )*qs++) * d * *y++;
			}
		}

		*s = sumf;
	}
};

class avx2_implementation : public implementation_base {
  public:
	BNCH_SWT_INLINE void ggml_vec_dot_q8_0_f32(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc) const override {
		static constexpr int qk = qkCount;
		const int nb			= n / qk;
		( void )(nrc);
		( void )(bx);
		( void )(by);
		( void )(bs);

		float sumf = 0;

		const block_q8_0* x = static_cast<const block_q8_0*>(vx);
		const float* y		= static_cast<const float*>(vy);

		__m256 total_sum = _mm256_setzero_ps();

		for (int32_t ib = 0; ib < nb - 4; ib += 4) {
			__m256 local_sum = _mm256_setzero_ps();
			__m256 d_broad00 = _mm256_set1_ps(GGML_FP16_TO_FP32(x->d));

			__m256i x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
			__m256i x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));
			++x;

			__m256 temp0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
			__m256 y0	 = _mm256_loadu_ps(y);
			local_sum	 = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
			y0		  = _mm256_loadu_ps(y + 8);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
			y0		  = _mm256_loadu_ps(y + 16);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
			y0		  = _mm256_loadu_ps(y + 24);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
			local_sum = _mm256_setzero_ps();

			x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
			x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));

			d_broad00 = _mm256_set1_ps(GGML_FP16_TO_FP32(x->d));
			y += 32;
			++x;

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
			y0		  = _mm256_loadu_ps(y);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
			y0		  = _mm256_loadu_ps(y + 8);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
			y0		  = _mm256_loadu_ps(y + 16);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
			y0		  = _mm256_loadu_ps(y + 24);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
			local_sum = _mm256_setzero_ps();

			x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
			x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));

			d_broad00 = _mm256_set1_ps(GGML_FP16_TO_FP32(x->d));
			y += 32;
			++x;

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
			y0		  = _mm256_loadu_ps(y);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
			y0		  = _mm256_loadu_ps(y + 8);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
			y0		  = _mm256_loadu_ps(y + 16);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
			y0		  = _mm256_loadu_ps(y + 24);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
			local_sum = _mm256_setzero_ps();

			x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
			x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));

			d_broad00 = _mm256_set1_ps(GGML_FP16_TO_FP32(x->d));
			y += 32;
			++x;

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
			y0		  = _mm256_loadu_ps(y);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
			y0		  = _mm256_loadu_ps(y + 8);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
			y0		  = _mm256_loadu_ps(y + 16);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
			y0		  = _mm256_loadu_ps(y + 24);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);
			total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
			y += 32;
		}

		__m128 sum = _mm_add_ps(_mm256_castps256_ps128(total_sum), _mm256_extractf128_ps(total_sum, 1));

		sum = _mm_hadd_ps(sum, sum);
		sum = _mm_hadd_ps(sum, sum);

		sumf = _mm_cvtss_f32(sum);

		*s = sumf;
	}
};

class avx512_implementation : public implementation_base {
  public:
	BNCH_SWT_INLINE void ggml_vec_dot_q8_0_f32(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc) const override {
		static constexpr int qk = qkCount;
		const int nb			= n / qk;
		( void )(nrc);
		( void )(bx);
		( void )(by);
		( void )(bs);

		float sumf = 0;

		*s = sumf;
	}
};

static constexpr avx_implementation avx_impl{};
static constexpr avx2_implementation avx2_impl{};
static constexpr avx512_implementation avx512_impl{};

static constexpr const implementation_base* impls[3]{ static_cast<const implementation_base*>(&avx_impl), static_cast<const implementation_base*>(&avx2_impl),
	static_cast<const implementation_base*>(&avx512_impl) };

BNCH_SWT_INLINE const implementation_base* getImpl() {
	auto impl_type = detect_supported_architectures();
	if (impl_type & cpuid_avx512f_bit) {
		return impls[2];
	} else if (impl_type & cpuid_avx2_bit) {
		return impls[1];
	} else {
		return impls[0];
	}
}

#endif

#if defined(__ARM_NEON)

BNCH_SWT_INLINE void arm_work(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc) {
	static constexpr int qk = qkCount;
	const int nb			= n / qk;
	( void )(nrc);
	( void )(bx);
	( void )(by);
	( void )(bs);

	float sumf = 0;

	const block_q8_0* x = static_cast<const block_q8_0*>(vx);
	const float* y		= static_cast<const float*>(vy);
	float32x4_t sumv0 = vdupq_n_f32(0.0f);
	float32x4_t sumv1 = vdupq_n_f32(0.0f);
	float32x4_t sumv2 = vdupq_n_f32(0.0f);
	float32x4_t sumv3 = vdupq_n_f32(0.0f);
	float32x4_t sumv4 = vdupq_n_f32(0.0f);
	float32x4_t sumv5 = vdupq_n_f32(0.0f);
	float32x4_t sumv6 = vdupq_n_f32(0.0f);
	float32x4_t sumv7 = vdupq_n_f32(0.0f);

	int ib = 0;
	for (; ib < nb; ib += 1) {
		const block_q8_0* b = x++;
		const float d				 = GGML_FP16_TO_FP32(b->d);

		//        const uint8x16x2_t x0_0 = vld2q_s8(x0->qs);
		int8x16_t vec1 = vld1q_s8(b->qs);// load 16x int8_t
		int8x16_t vec2 = vld1q_s8(b->qs + 16);// load 16x int8_t
		int16x8_t x0_0 = vmovl_s8(vget_low_s8(vec1));// cast the first 8x int8_t to int16_t
		int16x8_t x0_1 = vmovl_s8(vget_high_s8(vec1));// cast the last 8x int8_t to int16_t
		int16x8_t x1_0 = vmovl_s8(vget_low_s8(vec2));// cast the first 8x int8_t to int16_t
		int16x8_t x1_1 = vmovl_s8(vget_high_s8(vec2));// cast the last 8x int8_t to int16_t
		//printf("0:%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", ((int8_t *)&vec)[0], ((int8_t *)&vec)[1], ((int8_t *)&vec)[2], ((int8_t *)&vec)[3], ((int8_t *)&vec)[4], ((int8_t *)&vec)[5], ((int8_t *)&vec)[6], ((int8_t *)&vec)[7], ((int8_t *)&vec)[8], ((int8_t *)&vec)[9], ((int8_t *)&vec)[10], ((int8_t *)&vec)[11], ((int8_t *)&vec)[12], ((int8_t *)&vec)[13], ((int8_t *)&vec)[14], ((int8_t *)&vec)[15]);
		//printf("2:%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", ((int32_t *)&x0_00)[0], ((int32_t *)&x0_00)[1], ((int32_t *)&x0_00)[2], ((int32_t *)&x0_00)[3], ((int32_t *)&x0_01)[0], ((int32_t *)&x0_01)[1], ((int32_t *)&x0_01)[2], ((int32_t *)&x0_01)[3], ((int32_t *)&x0_02)[0], ((int32_t *)&x0_02)[1], ((int32_t *)&x0_02)[2], ((int32_t *)&x0_02)[3], ((int32_t *)&x0_03)[0], ((int32_t *)&x0_03)[1], ((int32_t *)&x0_03)[2], ((int32_t *)&x0_03)[3]);
		//printf("4:%f %f %f %f %d %d %d %d %d %d %d %d %d %d %d %d\n", ((float *)&xx0)[0], ((float *)&xx0)[1], ((float *)&xx0)[2], ((float *)&xx0)[3], ((int32_t *)&x0_01)[0], ((int32_t *)&x0_01)[1], ((int32_t *)&x0_01)[2], ((int32_t *)&x0_01)[3], ((int32_t *)&x0_02)[0], ((int32_t *)&x0_02)[1], ((int32_t *)&x0_02)[2], ((int32_t *)&x0_02)[3], ((int32_t *)&x0_03)[0], ((int32_t *)&x0_03)[1], ((int32_t *)&x0_03)[2], ((int32_t *)&x0_03)[3]);

		const float32x4_t x0 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s8(x0_0))), d);
		const float32x4_t x1 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s8(x0_0))), d);
		const float32x4_t x2 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s8(x0_1))), d);
		const float32x4_t x3 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s8(x0_1))), d);
		const float32x4_t x4 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s8(x1_0))), d);
		const float32x4_t x5 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s8(x1_0))), d);
		const float32x4_t x6 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s8(x1_1))), d);
		const float32x4_t x7 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s8(x1_1))), d);

		sumv0 = vfmaq_f32(sumv0, x0, vld1q_f32(y));
		sumv1 = vfmaq_f32(sumv1, x1, vld1q_f32(y + 4));
		sumv2 = vfmaq_f32(sumv2, x2, vld1q_f32(y + 8));
		sumv3 = vfmaq_f32(sumv3, x3, vld1q_f32(y + 12));
		sumv4 = vfmaq_f32(sumv4, x4, vld1q_f32(y + 16));
		sumv5 = vfmaq_f32(sumv5, x5, vld1q_f32(y + 20));
		sumv6 = vfmaq_f32(sumv6, x6, vld1q_f32(y + 24));
		sumv7 = vfmaq_f32(sumv7, x7, vld1q_f32(y + 28));
		y += 32;
	}

	sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3) + vaddvq_f32(sumv4) + vaddvq_f32(sumv5) + vaddvq_f32(sumv6) + vaddvq_f32(sumv7);
}

using work_func							= void (*)(int, float*, size_t, const void*, size_t, const void*, size_t, int);
static constexpr work_func work_funcs[] = { arm_work };

const work_func get_work_func() {
	return work_funcs[0];
}

#else

BNCH_SWT_INLINE void avx_work(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc) {
	static constexpr int qk = qkCount;
	const int nb			= n / qk;

	float sumf = 0;

	const block_q8_0* x = static_cast<const block_q8_0*>(vx);
	const float* y		= static_cast<const float*>(vy);

	int ib = 0;
	for (; ib < nb; ++ib) {
		float d			 = GGML_FP16_TO_FP32(x[ib].d);
		const int8_t* qs = x[ib].qs;
		for (int j = 0; j < qk; j++) {
			sumf += (( float )*qs++) * d * *y++;
		}
	}

	*s = sumf;
}

BNCH_SWT_INLINE void avx2_work(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc) {
	static constexpr int qk = qkCount;
	const int nb			= n / qk;
	( void )(nrc);
	( void )(bx);
	( void )(by);
	( void )(bs);

	float sumf = 0;

	const block_q8_0* x = static_cast<const block_q8_0*>(vx);
	const float* y		= static_cast<const float*>(vy);

	__m256 total_sum = _mm256_setzero_ps();

	for (int32_t ib = 0; ib < nb - 4; ib += 4) {
		__m256 local_sum = _mm256_setzero_ps();
		__m256 d_broad00 = _mm256_set1_ps(GGML_FP16_TO_FP32(x->d));

		__m256i x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
		__m256i x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));
		++x;

		__m256 temp0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		__m256 y0	 = _mm256_loadu_ps(y);
		local_sum	 = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_loadu_ps(y + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps(GGML_FP16_TO_FP32(x->d));
		y += 32;
		++x;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_loadu_ps(y);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_loadu_ps(y + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps(GGML_FP16_TO_FP32(x->d));
		y += 32;
		++x;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_loadu_ps(y);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_loadu_ps(y + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps(GGML_FP16_TO_FP32(x->d));
		y += 32;
		++x;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_loadu_ps(y);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_loadu_ps(y + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);
		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		y += 32;
	}

	__m128 sum = _mm_add_ps(_mm256_castps256_ps128(total_sum), _mm256_extractf128_ps(total_sum, 1));

	sum = _mm_hadd_ps(sum, sum);
	sum = _mm_hadd_ps(sum, sum);

	sumf = _mm_cvtss_f32(sum);

	*s = sumf;
}

BNCH_SWT_INLINE void avx512_work(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc) {
	static constexpr int qk = qkCount;
	const int nb			= n / qk;
	( void )(nrc);
	( void )(bx);
	( void )(by);
	( void )(bs);

	float sumf = 0;

	*s = sumf;
}

using work_func							= void (*)(int, float*, size_t, const void*, size_t, const void*, size_t, int);
static constexpr work_func work_funcs[] = { avx_work, avx2_work, avx512_work };
BNCH_SWT_INLINE const work_func get_work_func() {
	auto arch = detect_supported_architectures();
	if (arch & instruction_set::AVX512F) {
		return work_funcs[2];
	} else if (arch & instruction_set::AVX2) {
		return work_funcs[1];
	} else {
		return work_funcs[0];
	}
}

#endif

enum class cpu_id {
	avx	   = 0,
	avx2   = 1,
	avx512 = 2,
	arm	   = 0,
};

BNCH_SWT_INLINE const cpu_id get_cpu_id() {
	auto arch = detect_supported_architectures();
	if (arch & instruction_set::AVX512F) {
		return cpu_id::avx512;
	} else if (arch & instruction_set::AVX2) {
		return cpu_id::avx2;
	} else if (arch & instruction_set::SSE42) {
		return cpu_id::avx;
	} else {
		return cpu_id::arm;
	}
}

static const cpu_id cpuId{ get_cpu_id() };

inline static const implementation_base* impl{ getImpl() };

inline static const work_func impl_func{ get_work_func() };

thread_local static const cpu_id cpuIdThreadLocal{ get_cpu_id() };

thread_local inline static const implementation_base* implThreadLocal{ getImpl() };

thread_local inline static const work_func impl_funcThreadLocal{ get_work_func() };

struct test_data {
	test_data() {
		vy.resize(nCount);
		vx.resize(nCount4);
	}
	~test_data() {
	}
	int n{};
	float s{};
	size_t bs{};
	std::vector<block_q8_0> vx{};
	size_t bx{};
	std::vector<float> vy{};
	size_t by{};
	int nrc{};
};

test_data generate_test_data() {
	test_data data;

	data.n	 = nCount;
	data.s	 = 0.0f;
	data.bs	 = bnch_swt::random_generator::generateValue<size_t>();
	data.bx	 = bnch_swt::random_generator::generateValue<size_t>();
	data.by	 = bnch_swt::random_generator::generateValue<size_t>();
	data.nrc = bnch_swt::random_generator::generateValue<int>();
	for (size_t x = 0; x < nCount4; ++x) {
		for (int i = 0; i < qkCount; ++i) {
			data.vx[x].qs[i] = bnch_swt::random_generator::generateValue<int8_t>();
		}
		data.vx[x].d = bnch_swt::random_generator::generateValue<float>();
	}
	for (int i = 0; i < nCount; ++i) {
		data.vy[i] = bnch_swt::random_generator::generateValue<float>();
	}

	return data;
}

static constexpr uint64_t maxIterations{ 40 };

template<bnch_swt::string_literal testNameNew> BNCH_SWT_INLINE void testFunction() {
	static constexpr bnch_swt::string_literal testName{ testNameNew };
	std::array<test_data, maxIterations> testData{};
	for (size_t x = 0; x < maxIterations; ++x) {
		testData[x] = std::move(generate_test_data());
	}
	size_t currentIndex{};

	currentIndex = 0ull;
	bnch_swt::benchmark_stage<testName, maxIterations, 4>::template runBenchmark<"per-execution-selected-function-ptrs", "CYAN">([&] {
		uint64_t bytesProcessed{};
		work_funcs[static_cast<uint8_t>(cpuId)](testData[currentIndex].n, &testData[currentIndex].s, testData[currentIndex].bs, testData[currentIndex].vx.data(),
			testData[currentIndex].bx, testData[currentIndex].vy.data(), testData[currentIndex].by, testData[currentIndex].nrc);
		++currentIndex;
		bytesProcessed += sizeof(testData[currentIndex].vy) * 128;
		return bytesProcessed;
	});

	currentIndex = 0ull;
	bnch_swt::benchmark_stage<testName, maxIterations, 4>::template runBenchmark<"function-ptrs", "CYAN">([&] {
		uint64_t bytesProcessed{};
		impl_func(testData[currentIndex].n, &testData[currentIndex].s, testData[currentIndex].bs, testData[currentIndex].vx.data(), testData[currentIndex].bx,
			testData[currentIndex].vy.data(), testData[currentIndex].by, testData[currentIndex].nrc);
		++currentIndex;
		bytesProcessed += sizeof(testData[currentIndex].vy) * 128;
		return bytesProcessed;
	});

	currentIndex = 0ull;
	bnch_swt::benchmark_stage<testName, maxIterations, 4>::template runBenchmark<"ptr-to-impl", "CYAN">([&] {
		uint64_t bytesProcessed{};
		impl->ggml_vec_dot_q8_0_f32(testData[currentIndex].n, &testData[currentIndex].s, testData[currentIndex].bs, testData[currentIndex].vx.data(), testData[currentIndex].bx,
			testData[currentIndex].vy.data(), testData[currentIndex].by, testData[currentIndex].nrc);
		++currentIndex;
		bytesProcessed += sizeof(testData[currentIndex].vy) * 128;
		return bytesProcessed;
	});

	currentIndex = 0ull;
	bnch_swt::benchmark_stage<testName, maxIterations, 4>::template runBenchmark<"function-ptrs-thread_local", "CYAN">([&] {
		uint64_t bytesProcessed{};
		impl_funcThreadLocal(testData[currentIndex].n, &testData[currentIndex].s, testData[currentIndex].bs, testData[currentIndex].vx.data(), testData[currentIndex].bx,
			testData[currentIndex].vy.data(), testData[currentIndex].by, testData[currentIndex].nrc);
		++currentIndex;
		bytesProcessed += sizeof(testData[currentIndex].vy) * 128;
		return bytesProcessed;
	});

	currentIndex = 0ull;
	bnch_swt::benchmark_stage<testName, maxIterations, 4>::template runBenchmark<"ptr-to-impl-thread_local", "CYAN">([&] {
		uint64_t bytesProcessed{};
		implThreadLocal->ggml_vec_dot_q8_0_f32(testData[currentIndex].n, &testData[currentIndex].s, testData[currentIndex].bs, testData[currentIndex].vx.data(),
			testData[currentIndex].bx,
			testData[currentIndex].vy.data(), testData[currentIndex].by, testData[currentIndex].nrc);
		++currentIndex;
		bytesProcessed += sizeof(testData[currentIndex].vy) * 128;
		return bytesProcessed;
	});

	currentIndex = 0ull;

	bnch_swt::benchmark_stage<testName, maxIterations, 4>::printResults(true, true);
}

int main() {
	testFunction<"int-to-string-comparisons-1">();
	return 0;
}