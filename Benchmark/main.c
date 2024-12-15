#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>
#include <string.h>

#if defined(_WIN32)
	#include <windows.h>
#else
	#include <sys/time.h>
#endif

#if defined(BNCH_SWT_MSVC)
	#pragma optimize("", off)
void doNotOptimize(const void* value) {
	( void )value;
};
	#pragma optimize("", on)
#else
void doNotOptimize(const void* value) {
	#if defined(BNCH_SWT_CLANG)
	asm volatile("" : "+r,m"(value) : : "memory");
	#elif defined(BNCH_SWT_GNUCXX)
	asm volatile("" : "+m,r"(value) : : "memory");
	#endif
}
#endif

#define QK8_0 32
#define QK8_0_MEGA_D 8
#define QK8_0_MEGA_QS (QK8_0_MEGA_D * QK8_0)
#define dim (2048 * 2048)
#define num_blocks (dim / QK8_0)
#define num_blocks_aligned (dim / QK8_0)
#define num_blocks_mega (dim / QK8_0_MEGA_QS)

typedef struct {
	int8_t qs[QK8_0];
	uint16_t d;
} block_q8_0;

typedef struct {
	_Alignas(32) int8_t qs[QK8_0];
} block_q8_0_aligned_quants;

typedef struct {
	float d;
} block_q8_0_aligned_float;

typedef struct {
	_Alignas(32) int8_t qs[QK8_0_MEGA_QS];
} block_q8_0_mega_quants;

typedef struct {
	_Alignas(32) float d[QK8_0_MEGA_D];
} block_q8_0_mega_float;

// Function prototypes
void oi_vec_dot_q8_0_q8_0(const int ne, float* dst, const block_q8_0* __restrict x, const block_q8_0* __restrict y);
void oi_vec_dot_q8_0_q8_0_aligned(const int ne, float* dst, const block_q8_0_aligned_quants* __restrict x, const block_q8_0_aligned_quants* __restrict y,
	const block_q8_0_aligned_float* x_x, const block_q8_0_aligned_float* y_x);
void oi_vec_dot_q8_0_q8_0_mega_blocks(const int ne, float* dst, const block_q8_0_mega_quants* __restrict x, const block_q8_0_mega_quants* __restrict y,
	const block_q8_0_mega_float* __restrict x_x, const block_q8_0_mega_float* __restrict y_x);
void oi_vec_dot_q8_0_q8_0_mega(const int ne, float* dst, const uint8_t* __restrict x, const uint8_t* __restrict y, const float* __restrict x_x, const float* __restrict y_x);

// Helper functions
float fp32_from_bits(uint32_t w);
uint32_t fp32_to_bits(float f);
float ggml_compute_fp16_to_fp32(uint16_t h);
__m256 sum_i16_pairs_float(const __m256i x);
__m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy);
__m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y);
float hsum_float_8(const __m256 x);

// Random number generator (simple LCG)
uint8_t generate_random_uint8() {
	uint32_t seed = 42;
	seed				 = (1664525 * seed + 1013904223) & 0xFFFFFFFF;
	return ( uint8_t )(seed >> 24);
}

float generate_random_float() {
	return ( float )generate_random_uint8() / 255.0f;
}

// High-resolution timer
double get_time() {
#if defined(_WIN32)
	LARGE_INTEGER frequency, start;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);
	return ( double )start.QuadPart / frequency.QuadPart;
#else
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ( double )ts.tv_sec + ( double )ts.tv_nsec / 1e9;
#endif
}

void benchmark(const char* name, void (*func)(int, float*, const void*, const void*, const void*, const void*), int ne, const void* x, const void* y, const void* x_x,
	const void* y_x, double* time_taken) {
	float dst			 = 0.0f;
	const int iterations = 1000;// Increased iterations for better accuracy
	double start		 = get_time();

	for (int i = 0; i < iterations; ++i) {
		func(ne, &dst, x, y, x_x, y_x);
		doNotOptimize(&dst);// Prevent optimization
	}
	printf("%f\n", dst);

	double end	= get_time();
	*time_taken = (end - start) / iterations;
	printf("%s: %f seconds per iteration\n", name, *time_taken);
}

void print_comparison(double time_mega, double time_mega_blocks) {
	// Calculate the percentage difference
	double mega_vs_mega_blocks = ((time_mega_blocks - time_mega) / time_mega) * 100.0f;

	// Check if mega is faster or mega-blocks is faster
	if (mega_vs_mega_blocks >= 0) {
		printf("Library mega, is faster than library: mega-blocks, by roughly: %.2f%%.\n", mega_vs_mega_blocks);
	} else {
		printf("Library mega-blocks, is faster than library: mega, by roughly: %.2f%%.\n", -mega_vs_mega_blocks);
	}
}


int main() {
	block_q8_0_mega_quants* x_mega_blocks  = ( block_q8_0_mega_quants* )malloc(num_blocks_mega * sizeof(block_q8_0_mega_quants));
	block_q8_0_mega_quants* y_mega_blocks  = ( block_q8_0_mega_quants* )malloc(num_blocks_mega * sizeof(block_q8_0_mega_quants));
	block_q8_0_mega_float* x_x_mega_blocks = ( block_q8_0_mega_float* )malloc(num_blocks_mega * sizeof(block_q8_0_mega_float));
	block_q8_0_mega_float* y_x_mega_blocks = ( block_q8_0_mega_float* )malloc(num_blocks_mega * sizeof(block_q8_0_mega_float));
	_Alignas(32) uint8_t* x_mega		   = ( uint8_t* )malloc(num_blocks_mega * QK8_0_MEGA_QS * sizeof(uint8_t));
	_Alignas(32) uint8_t* y_mega		   = ( uint8_t* )malloc(num_blocks_mega * QK8_0_MEGA_QS * sizeof(uint8_t));
	_Alignas(32) float* x_x_mega		   = ( float* )malloc(num_blocks_mega * QK8_0_MEGA_D * sizeof(float));
	_Alignas(32) float* y_x_mega		   = ( float* )malloc(num_blocks_mega * QK8_0_MEGA_D * sizeof(float));

	for (size_t i = 0; i < num_blocks_mega; ++i) {
		for (size_t j = 0; j < QK8_0_MEGA_QS; ++j) {
			x_mega_blocks[i].qs[j] = generate_random_uint8();
			y_mega_blocks[i].qs[j] = generate_random_uint8();
		}
		for (size_t j = 0; j < QK8_0_MEGA_D; ++j) {
			x_x_mega_blocks[i].d[j] = generate_random_float();
			y_x_mega_blocks[i].d[j] = generate_random_float();
		}
	}

	for (size_t i = 0; i < num_blocks_mega * QK8_0_MEGA_QS; ++i) {
		x_mega[i] = generate_random_uint8();
		y_mega[i] = generate_random_uint8();
	}

	for (size_t i = 0; i < num_blocks_mega * QK8_0_MEGA_D; ++i) {
		x_x_mega[i] = generate_random_float();
		y_x_mega[i] = generate_random_float();
	}

	// Initialize timing variables
	double time_oi_vec_dot_q8_0_q8_0, time_oi_vec_dot_q8_0_q8_0_mega, time_oi_vec_dot_q8_0_q8_0_mega_blocks;
	benchmark("oi_vec_dot_q8_0_q8_0_mega", ( void (*)(int, float*, const void*, const void*, const void*, const void*) )oi_vec_dot_q8_0_q8_0_mega, dim, x_mega, y_mega,
		x_x_mega, y_x_mega, &time_oi_vec_dot_q8_0_q8_0_mega);
	benchmark("oi_vec_dot_q8_0_q8_0_mega_blocks", ( void (*)(int, float*, const void*, const void*, const void*, const void*) )oi_vec_dot_q8_0_q8_0_mega_blocks, dim, x_mega_blocks,
		y_mega_blocks, x_x_mega_blocks, y_x_mega_blocks, &time_oi_vec_dot_q8_0_q8_0_mega_blocks);

	// Print the comparison result
	print_comparison(time_oi_vec_dot_q8_0_q8_0_mega, time_oi_vec_dot_q8_0_q8_0_mega_blocks);
	free(x_mega_blocks);
	free(y_mega_blocks);
	free(x_x_mega_blocks);
	free(y_x_mega_blocks);
	free(x_mega);
	free(y_mega);
	free(x_x_mega);
	free(y_x_mega);

	return 0;
}

// Rest of the functions remain unchanged...

void oi_vec_dot_q8_0_q8_0(const int ne, float* dst, const block_q8_0* __restrict x, const block_q8_0* __restrict y) {
	const int nb = ne / QK8_0;

	// Initialize accumulator with zeros
	__m256 acc = _mm256_setzero_ps();

	// Main loop
	for (int ib = 0; ib < nb; ++ib) {
		// Compute combined scale for the block
		const float xd = ggml_compute_fp16_to_fp32(x[ib].d);
		const float yd = ggml_compute_fp16_to_fp32(y[ib].d);
		const __m256 d = _mm256_set1_ps(xd * yd);

		__m256i qx = _mm256_loadu_si256(( const __m256i* )x[ib].qs);
		__m256i qy = _mm256_loadu_si256(( const __m256i* )y[ib].qs);

		const __m256 q = mul_sum_i8_pairs_float(qx, qy);

		// Multiply q with scale and accumulate
		acc = _mm256_fmadd_ps(d, q, acc);
	}

	*dst = hsum_float_8(acc);
}

void oi_vec_dot_q8_0_q8_0_aligned(const int ne, float* dst, const block_q8_0_aligned_quants* __restrict x, const block_q8_0_aligned_quants* __restrict y,
	const block_q8_0_aligned_float* x_x, const block_q8_0_aligned_float* y_x) {
	const int nb = ne / QK8_0;
	__m256 acc	 = _mm256_setzero_ps();

	// Main loop
	for (int ib = 0; ib < nb; ++ib) {
		const float xd = x_x[ib].d;
		const float yd = y_x[ib].d;
		const __m256 d = _mm256_set1_ps(xd * yd);

		__m256i qx = _mm256_loadu_si256(( const __m256i* )x[ib].qs);
		__m256i qy = _mm256_loadu_si256(( const __m256i* )y[ib].qs);

		const __m256 q = mul_sum_i8_pairs_float(qx, qy);

		// Multiply q with scale and accumulate
		acc = _mm256_fmadd_ps(d, q, acc);
	}

	*dst = hsum_float_8(acc);
}

void oi_vec_dot_q8_0_q8_0_mega_blocks(const int ne, float* dst, const block_q8_0_mega_quants* __restrict x, const block_q8_0_mega_quants* __restrict y,
	const block_q8_0_mega_float* __restrict x_x, const block_q8_0_mega_float* __restrict y_x) {
	const int nb = ne / QK8_0_MEGA_QS;
	// Initialize accumulator with zeros
	__m256 acc = _mm256_setzero_ps();

	// Main loop
	for (int ib = 0; ib < nb; ++ib) {
		float dx   = x_x[ib].d[0];
		float dy   = y_x[ib].d[0];
		__m256 d   = _mm256_set1_ps(dx * dy);
		__m256i qx = _mm256_loadu_si256(( const __m256i* )x[ib].qs);
		__m256i qy = _mm256_loadu_si256(( const __m256i* )y[ib].qs);
		__m256 q   = mul_sum_i8_pairs_float(qx, qy);
		acc		   = _mm256_fmadd_ps(d, q, acc);

		// 1
		dx	= x_x[ib].d[1];
		dy	= y_x[ib].d[1];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[ib].qs[1 * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[ib].qs[1 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 2
		dx	= x_x[ib].d[2];
		dy	= y_x[ib].d[2];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[ib].qs[2 * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[ib].qs[2 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 3
		dx	= x_x[ib].d[3];
		dy	= y_x[ib].d[3];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[ib].qs[3 * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[ib].qs[3 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 4
		dx	= x_x[ib].d[0];
		dy	= y_x[ib].d[0];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[ib].qs[4 * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[ib].qs[4 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 5
		dx	= x_x[ib].d[1];
		dy	= y_x[ib].d[1];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[ib].qs[5 * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[ib].qs[5 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 6
		dx	= x_x[ib].d[2];
		dy	= y_x[ib].d[2];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[ib].qs[6 * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[ib].qs[6 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 7
		dx	= x_x[ib].d[3];
		dy	= y_x[ib].d[3];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[ib].qs[7 * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[ib].qs[7 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
	}

	*dst = hsum_float_8(acc);
}

inline float fp32_from_bits(uint32_t w) {
	union {
		uint32_t as_bits;
		float as_value;
	} fp32;
	fp32.as_bits = w;
	return fp32.as_value;
}

inline uint32_t fp32_to_bits(float f) {
	union {
		float as_value;
		uint32_t as_bits;
	} fp32;
	fp32.as_value = f;
	return fp32.as_bits;
}

float ggml_compute_fp16_to_fp32(uint16_t h) {
	const uint32_t w	 = ( uint32_t )h << 16;
	const uint32_t sign	 = w & UINT32_C(0x80000000);
	const uint32_t two_w = w + w;

	uint32_t exp_offset			 = UINT32_C(0xE0) << 23;
	float exp_scale				 = fp32_from_bits(UINT32_C(0x7800000));
	const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

	const uint32_t magic_mask	   = UINT32_C(126) << 23;
	const float magic_bias		   = 0.5f;
	const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

	uint32_t denormalized_cutoff = UINT32_C(1) << 27;
	const uint32_t result		 = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
	return fp32_from_bits(result);
}

__m256 sum_i16_pairs_float(const __m256i x) {
	const __m256i ones		   = _mm256_set1_epi16(1);
	const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
	return _mm256_cvtepi32_ps(summed_pairs);
}

__m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
	// Perform multiplication and create 16-bit values
	const __m256i dot = _mm256_maddubs_epi16(ax, sy);
	return sum_i16_pairs_float(dot);
}

// multiply int8_t, add results pairwise twice and return as float vector
__m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
	// Get absolute values of x vectors
	const __m256i ax = _mm256_sign_epi8(x, x);
	// Sign the values of the y vectors
	const __m256i sy = _mm256_sign_epi8(y, x);
	return mul_sum_us8_pairs_float(ax, sy);
}

float hsum_float_8(const __m256 x) {
	__m128 res = _mm256_extractf128_ps(x, 1);
	res		   = _mm_add_ps(res, _mm256_castps256_ps128(x));
	res		   = _mm_add_ps(res, _mm_movehl_ps(res, res));
	res		   = _mm_add_ss(res, _mm_movehdup_ps(res));
	return _mm_cvtss_f32(res);
}

void oi_vec_dot_q8_0_q8_0_mega(const int ne, float* dst, const uint8_t* __restrict x, const uint8_t* __restrict y, const float* __restrict x_x, const float* __restrict y_x) {
	const int nb = ne / QK8_0_MEGA_QS;
	// Initialize accumulator with zeros
	__m256 acc = _mm256_setzero_ps();

	// Main loop
	for (int ib = 0; ib < nb; ++ib) {
		float dx   = x_x[ib * 8];
		float dy   = y_x[ib * 8];
		__m256 d   = _mm256_set1_ps(dx * dy);
		__m256i qx = _mm256_loadu_si256(( const __m256i* )&x[ib * 32]);
		__m256i qy = _mm256_loadu_si256(( const __m256i* )&y[ib * 32]);
		__m256 q   = mul_sum_i8_pairs_float(qx, qy);
		acc		   = _mm256_fmadd_ps(d, q, acc);

		// 1
		dx	= x_x[ib * 8 + 1];
		dy	= y_x[ib * 8 + 1];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[(ib + 1) * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[(ib + 1) * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 2
		dx	= x_x[ib * 8 + 2];
		dy	= y_x[ib * 8 + 2];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[(ib + 2) * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[(ib + 2) * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 3
		dx	= x_x[ib * 8 + 3];
		dy	= y_x[ib * 8 + 3];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[(ib + 3) * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[(ib + 3) * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 4
		dx	= x_x[ib * 8 + 0];
		dy	= y_x[ib * 8 + 0];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[(ib + 4) * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[(ib + 4) * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 5
		dx	= x_x[ib * 8 + 1];
		dy	= y_x[ib * 8 + 1];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[(ib + 5) * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[(ib + 5) * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 6
		dx	= x_x[ib * 8 + 2];
		dy	= y_x[ib * 8 + 2];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[(ib + 6) * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[(ib + 6) * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 7
		dx	= x_x[ib * 8 + 3];
		dy	= y_x[ib * 8 + 3];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_loadu_si256(( const __m256i* )&x[(ib + 7) * 32]);
		qy	= _mm256_loadu_si256(( const __m256i* )&y[(ib + 7) * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
	}

	*dst = hsum_float_8(acc);
}