#pragma once

#include <oiml/common/op_traits.hpp>
#include <oiml/common/thread_pool.hpp>
#include <oiml-cpu/cpu_arch.hpp>
#include <oiml/common/tensor.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-common.hpp>
#include <oiml/legacy/oiml-legacy-cpu/sgemm.hpp>

namespace oiml {

	OIML_FORCE_INLINE static constexpr size_t block_size(size_t M, size_t m) {
		const size_t NB_BLOC_M = (m + M - 1) / M;
		return (m % NB_BLOC_M == 0) ? m / NB_BLOC_M : (m / NB_BLOC_M) + 1;
	}

	OIML_FORCE_INLINE static constexpr size_t block_pos(size_t ib, size_t ibN, size_t bloc_size) {
		return ib < ibN ? ib * bloc_size : ibN * bloc_size + (ib - ibN) * (bloc_size - 1);
	}

#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
	template<typename TA, typename TB, typename TC> class tiny_blas_q0_avx {
	  public:
		tiny_blas_q0_avx(int64_t k, const TA* A, int64_t lda, const TB* B, int64_t ldb, TC* C, int64_t ldc, size_t ith, size_t nth)
			: A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
		}

		void matmul(int64_t m, int64_t n) {
			mnpack(0, m, 0, n);
		}

	  private:
		void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
			switch ((MIN(m - m0, 4) << 4) | MIN(n - n0, 4)) {
	#if VECTOR_REGISTERS == 32
				case 0x44: {
					static constexpr uint64_t mc = 4;
					static constexpr uint64_t nc = 4;
		#if defined(__AVX2__) && defined(__F16C__)
					gemm4xN<4>(m0, m, n0, n);
		#else
					gemm<4, 4>(m0, m, n0, n);
		#endif
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x43: {
					static constexpr uint64_t mc = 4;
					static constexpr uint64_t nc = 3;
		#if defined(__AVX2__) && defined(__F16C__)
					gemm4xN<3>(m0, m, n0, n);
		#else
					gemm<4, 3>(m0, m, n0, n);
		#endif
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x34: {
					static constexpr uint64_t mc = 3;
					static constexpr uint64_t nc = 4;
		#if defined(__AVX2__) && defined(__F16C__)
					gemmMx4<3>(m0, m, n0, n);
		#else
					gemm<3, 4>(m0, m, n0, n);
		#endif
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x33: {
					static constexpr uint64_t mc = 3;
					static constexpr uint64_t nc = 3;
					gemm<3, 3>(m0, m, n0, n);
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x42: {
					static constexpr uint64_t mc = 4;
					static constexpr uint64_t nc = 2;
		#if defined(__AVX2__) && defined(__F16C__)
					gemm4xN<2>(m0, m, n0, n);
		#else
					gemm<4, 2>(m0, m, n0, n);
		#endif
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x24: {
					static constexpr uint64_t mc = 2;
					static constexpr uint64_t nc = 4;
		#if defined(__AVX2__) && defined(__F16C__)
					gemmMx4<2>(m0, m, n0, n);
		#else
					gemm<2, 4>(m0, m, n0, n);
		#endif
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
	#else
				case 0x44:
				case 0x43:
				case 0x42: {
					static constexpr uint64_t mc = 4;
					static constexpr uint64_t nc = 2;
		#if defined(__AVX2__) && defined(__F16C__)
					gemm4xN<2>(m0, m, n0, n);
		#else
					gemm<4, 2>(m0, m, n0, n);
		#endif
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x34:
				case 0x24: {
					static constexpr uint64_t mc = 2;
					static constexpr uint64_t nc = 4;
		#if defined(__AVX2__) && defined(__F16C__)
					gemmMx4<2>(m0, m, n0, n);
		#else
					gemm<2, 4>(m0, m, n0, n);
		#endif
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x33:
	#endif
				case 0x32: {
					static constexpr uint64_t mc = 3;
					static constexpr uint64_t nc = 2;
					gemm<3, 2>(m0, m, n0, n);
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x23: {
					static constexpr uint64_t mc = 2;
					static constexpr uint64_t nc = 3;
					gemm<2, 3>(m0, m, n0, n);
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x41: {
					static constexpr uint64_t mc = 4;
					static constexpr uint64_t nc = 1;
	#if defined(__AVX2__) && defined(__F16C__)
					gemm4xN<1>(m0, m, n0, n);
	#else
					gemm<4, 1>(m0, m, n0, n);
	#endif
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x22: {
					static constexpr uint64_t mc = 2;
					static constexpr uint64_t nc = 2;
					gemm<2, 2>(m0, m, n0, n);
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x14: {
					static constexpr uint64_t mc = 1;
					static constexpr uint64_t nc = 4;
	#if defined(__AVX2__) && defined(__F16C__)
					gemmMx4<1>(m0, m, n0, n);
	#else
					gemm<1, 4>(m0, m, n0, n);
	#endif
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x31: {
					static constexpr uint64_t mc = 3;
					static constexpr uint64_t nc = 1;
					gemm<3, 1>(m0, m, n0, n);
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x13: {
					static constexpr uint64_t mc = 1;
					static constexpr uint64_t nc = 3;
					gemm<1, 3>(m0, m, n0, n);
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x21: {
					static constexpr uint64_t mc = 2;
					static constexpr uint64_t nc = 1;
					gemm<2, 1>(m0, m, n0, n);
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x12: {
					static constexpr uint64_t mc = 1;
					static constexpr uint64_t nc = 2;
					gemm<1, 2>(m0, m, n0, n);
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				case 0x11: {
					static constexpr uint64_t mc = 1;
					static constexpr uint64_t nc = 1;
					gemm<1, 1>(m0, m, n0, n);
					uint64_t mp = m0 + (m - m0) / mc * mc;
					uint64_t np = n0 + (n - n0) / nc * nc;
					mnpack(mp, m, n0, np);
					mnpack(m0, m, np, n);
					break;
				}
				default:
					return;
			}
		}

	#if defined(__AVX2__) && defined(__F16C__)
		// Templated functions for gemm of dimensions 4xN
		template<size_t RN> OIML_FORCE_INLINE void gemm4xN(int64_t m0, int64_t m, int64_t n0, int64_t n) {
			int64_t ytiles = (m - m0) / 4;
			int64_t xtiles = (n - n0) / RN;
			int64_t tiles  = xtiles * ytiles;
			int64_t duty   = (tiles + nth - 1) / nth;
			int64_t start  = duty * ith;
			int64_t end	   = start + duty;
			if (end > tiles)
				end = tiles;
			for (int64_t job = start; job < end; ++job) {
				int64_t ii		 = m0 + job / xtiles * 4;
				int64_t jj		 = n0 + job % xtiles * RN;
				__m256 Cv[RN][4] = {};
				for (int64_t l = 0; l < k; ++l) {
					uint64_t a_delta = (( uint64_t )A[lda * (ii + 3) + l].d << 48) | (( uint64_t )A[lda * (ii + 2) + l].d << 32) | (( uint64_t )A[lda * (ii + 1) + l].d << 16) |
						(A[lda * (ii + 0) + l].d);
					// Convert delta values for four blocks to float values
					__m128 da	  = _mm_cvtph_ps(_mm_set_epi64x(0, a_delta));
					__m256i avec0 = load(A + lda * (ii + 0) + l);
					__m256i avec1 = load(A + lda * (ii + 1) + l);
					__m256i avec2 = load(A + lda * (ii + 2) + l);
					__m256i avec3 = load(A + lda * (ii + 3) + l);
					for (int64_t j = 0; j < RN; ++j) {
						__m128 db = _mm_set1_ps(unhalf(B[ldb * (jj + j) + l].d));
						// Computation of product of delta values for four blocks and replicate it across 256 bit lane
						__m256 dvec = _mm256_castps128_ps256(_mm_mul_ps(da, db));
						dvec		= _mm256_permute2f128_ps(dvec, dvec, 0);
						// Computation of dot product and multiplication with appropriate delta value products
						Cv[j][0] = madd(_mm256_shuffle_ps(dvec, dvec, 0), updot(_mm256_sign_epi8(avec0, avec0), _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec0)), Cv[j][0]);
						Cv[j][1] = madd(_mm256_shuffle_ps(dvec, dvec, 85), updot(_mm256_sign_epi8(avec1, avec1), _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec1)), Cv[j][1]);
						Cv[j][2] = madd(_mm256_shuffle_ps(dvec, dvec, 170), updot(_mm256_sign_epi8(avec2, avec2), _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec2)), Cv[j][2]);
						Cv[j][3] = madd(_mm256_shuffle_ps(dvec, dvec, 255), updot(_mm256_sign_epi8(avec3, avec3), _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec3)), Cv[j][3]);
					}
				}

				for (int64_t j = 0; j < RN; ++j)
					for (int64_t i = 0; i < 4; ++i)
						C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
			}
		}

		// Templated functions for gemm of dimensions Mx4
		template<size_t RM> OIML_FORCE_INLINE void gemmMx4(int64_t m0, int64_t m, int64_t n0, int64_t n) {
			int64_t ytiles = (m - m0) / RM;
			int64_t xtiles = (n - n0) / 4;
			int64_t tiles  = xtiles * ytiles;
			int64_t duty   = (tiles + nth - 1) / nth;
			int64_t start  = duty * ith;
			int64_t end	   = start + duty;
			if (end > tiles)
				end = tiles;
			for (int64_t job = start; job < end; ++job) {
				int64_t ii		 = m0 + job / xtiles * RM;
				int64_t jj		 = n0 + job % xtiles * 4;
				__m256 Cv[4][RM] = {};
				for (int64_t l = 0; l < k; ++l) {
					uint64_t b_delta = (( uint64_t )B[ldb * (jj + 3) + l].d << 48) | (( uint64_t )B[ldb * (jj + 2) + l].d << 32) | (( uint64_t )B[ldb * (jj + 1) + l].d << 16) |
						(B[ldb * (jj + 0) + l].d);
					// Convert delta values for four blocks to float values
					__m128 db	  = _mm_cvtph_ps(_mm_set_epi64x(0, b_delta));
					__m256i bvec0 = load(B + ldb * (jj + 0) + l);
					__m256i bvec1 = load(B + ldb * (jj + 1) + l);
					__m256i bvec2 = load(B + ldb * (jj + 2) + l);
					__m256i bvec3 = load(B + ldb * (jj + 3) + l);
					for (int64_t i = 0; i < RM; ++i) {
						__m128 da = _mm_set1_ps(unhalf((A[lda * (ii + i) + l].d)));
						// Computation of product of delta values for four blocks and replicate it across 256 bit lane
						__m256 dvec = _mm256_castps128_ps256(_mm_mul_ps(da, db));
						dvec		= _mm256_permute2f128_ps(dvec, dvec, 0);
						// Computation of dot product and multiplication with appropriate delta value products
						Cv[0][i] = madd(_mm256_shuffle_ps(dvec, dvec, 0),
							updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l), load(A + lda * (ii + i) + l)), _mm256_sign_epi8(bvec0, load(A + lda * (ii + i) + l))), Cv[0][i]);
						Cv[1][i] = madd(_mm256_shuffle_ps(dvec, dvec, 85),
							updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l), load(A + lda * (ii + i) + l)), _mm256_sign_epi8(bvec1, load(A + lda * (ii + i) + l))), Cv[1][i]);
						Cv[2][i] = madd(_mm256_shuffle_ps(dvec, dvec, 170),
							updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l), load(A + lda * (ii + i) + l)), _mm256_sign_epi8(bvec2, load(A + lda * (ii + i) + l))), Cv[2][i]);
						Cv[3][i] = madd(_mm256_shuffle_ps(dvec, dvec, 255),
							updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l), load(A + lda * (ii + i) + l)), _mm256_sign_epi8(bvec3, load(A + lda * (ii + i) + l))), Cv[3][i]);
					}
				}
				for (int64_t j = 0; j < 4; ++j)
					for (int64_t i = 0; i < RM; ++i)
						C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
			}
		}
	#endif

		template<size_t RM, size_t RN> OIML_FORCE_INLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
			int64_t ytiles = (m - m0) / RM;
			int64_t xtiles = (n - n0) / RN;
			int64_t tiles  = xtiles * ytiles;
			int64_t duty   = (tiles + nth - 1) / nth;
			int64_t start  = duty * ith;
			int64_t end	   = start + duty;
			if (end > tiles)
				end = tiles;
			for (int64_t job = start; job < end; ++job) {
				int64_t ii		  = m0 + job / xtiles * RM;
				int64_t jj		  = n0 + job % xtiles * RN;
				__m256 Cv[RN][RM] = {};
				for (int64_t l = 0; l < k; ++l)
					for (int64_t j = 0; j < RN; ++j)
						for (int64_t i = 0; i < RM; ++i) {
	#if defined(__AVX2__)
							__m256 udTmp = updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l), load(A + lda * (ii + i) + l)),
								_mm256_sign_epi8(load(B + ldb * (jj + j) + l), load(A + lda * (ii + i) + l)));
	#else
							__m128i ali0 = load0(A + lda * (ii + i) + l);
							__m128i ali1 = load1(A + lda * (ii + i) + l);
							__m128i blj0 = load0(B + ldb * (jj + j) + l);
							__m128i blj1 = load1(B + ldb * (jj + j) + l);

							__m128i sepAA0 = _mm_sign_epi8(ali0, ali0);
							__m128i sepAA1 = _mm_sign_epi8(ali1, ali1);
							__m128i sepBA0 = _mm_sign_epi8(blj0, ali0);
							__m128i sepBA1 = _mm_sign_epi8(blj1, ali1);

							// updot
							const __m128i oneFill = _mm_set1_epi16(1);
							__m128i mad0		  = _mm_maddubs_epi16(sepAA0, sepBA0);
							__m128i mad1		  = _mm_maddubs_epi16(sepAA1, sepBA1);
							__m256 udTmp		  = _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_madd_epi16(oneFill, mad1), _mm_madd_epi16(oneFill, mad0)));
	#endif
							Cv[j][i] = madd(_mm256_set1_ps(unhalf(A[lda * (ii + i) + l].d) * unhalf(B[ldb * (jj + j) + l].d)), udTmp, Cv[j][i]);
						}
				for (int64_t j = 0; j < RN; ++j)
					for (int64_t i = 0; i < RM; ++i)
						C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
			}
		}

		OIML_FORCE_INLINE static __m256i load(const block_q8_0<oiml_half>* b) {
			return _mm256_loadu_si256(( const __m256i* )b->qs);
		}

		OIML_FORCE_INLINE static __m128i load0(const block_q8_0<oiml_half>* b) {
			return _mm_loadu_si128(( const __m128i* )b->qs);
		}

		OIML_FORCE_INLINE static __m128i load1(const block_q8_0<oiml_half>* b) {
			return _mm_loadu_si128((( const __m128i* )b->qs) + 1);
		}

		OIML_FORCE_INLINE static __m256 updot(__m256i u, __m256i s) {
			__m256i res;
	#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
			res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
	#elif defined(__AVXVNNI__)
			res = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s);
	#else
			res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
	#endif
			return _mm256_cvtepi32_ps(res);
		}

		const TA* const A;
		const TB* const B;
		TC* const C;
		const int64_t k;
		const int64_t lda;
		const int64_t ldb;
		const int64_t ldc;
		const size_t ith;
		const size_t nth;
	};
#endif// __AVX__

	template<size_t KN, size_t k_new, typename D, typename V, typename TA, typename TB, typename TC> class tiny_blas {
	  public:
		static constexpr size_t k{ k_new };
		OIML_FORCE_INLINE tiny_blas(size_t thread_index, size_t thread_count, const TA* tensor01, size_t lda, const TB* tensor02, size_t ldb, TC* tensor03, size_t ldc)
			: thread_index(thread_index), thread_count{ thread_count }, tensor01(tensor01), tensor02(tensor02), tensor03(tensor03), lda(lda), ldb(ldb), ldc(ldc) {};

		template<size_t m, size_t n> OIML_FORCE_INLINE bool matmul() {
			if constexpr (k % KN != 0) {
				return false;
			}
#if VECTOR_REGISTERS == 32
			if (m % 16 == 0 && (m / 16 >= thread_count)) {
				static constexpr size_t SIZE_N = block_size(6, n);
				mnpack<4, 6, 4, m, n, SIZE_N, 12>();
				return true;
			}
			if constexpr (m % 8 == 0) {
				static constexpr size_t SIZE_N = block_size(6, n);
				mnpack<4, 6, 2, m, n, SIZE_N, 12>();
				return true;
			}
			if constexpr (m % 4 == 0) {
				static constexpr size_t SIZE_N = block_size(6, n);
				mnpack<4, 6, 1, m, n, SIZE_N, 12>();
				return true;
			}
#else
			if (m % 16 == 0 && (m / 16 >= thread_count)) {
				static constexpr size_t SIZE_N = block_size(3, n);
				mnpack<4, 3, 4, m, n, SIZE_N, 24>();
				return true;
			}
			if constexpr (m % 8 == 0) {
				static constexpr size_t SIZE_N = block_size(3, n);
				mnpack<4, 3, 2, m, n, SIZE_N, 24>();
				return true;
			}
			if constexpr (m % 4 == 0) {
				static constexpr size_t SIZE_N = block_size(3, n);
				mnpack<4, 3, 1, m, n, SIZE_N, 24>();
				return true;
			}
#endif
			return false;
		}

	  private:
		template<size_t RM, size_t RN, size_t BM, size_t m, size_t n, size_t SIZE_N, size_t BN> OIML_FORCE_INLINE void mnpack() {
			if constexpr (SIZE_N == RN) {
				return gemm<RM, RN, BM, m, n, BN>();
			}
			if constexpr (RN > 1) {
				return mnpack<RM, RN - 1, BM, m, n, SIZE_N, BN>();
			} else {
				OIML_LOG_ERROR("mnpack<%d, %d> bloc size not supported\n", RM, ( size_t )SIZE_N);
				OIML_ASSERT(false);
			}
		}

		template<size_t RM, size_t RN> OIML_FORCE_INLINE void gemm_bloc(size_t ii, size_t jj) {
			D Cv[RN][RM] = {};
			for (size_t l = 0; l < k; l += KN) {
				if constexpr (RM <= RN) {
					V Av[RM];
					for (size_t i = 0; i < RM; ++i) {
						Av[i] = load<V>(tensor01 + lda * (ii + i) + l);
					}
					for (size_t j = 0; j < RN; ++j) {
						V Bv = load<V>(tensor02 + ldb * (jj + j) + l);
						for (size_t i = 0; i < RM; ++i) {
							Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
						}
					}
				} else {
					V Bv[RN];
					for (size_t j = 0; j < RN; ++j) {
						Bv[j] = load<V>(tensor02 + ldb * (jj + j) + l);
					}
					for (size_t i = 0; i < RM; ++i) {
						V Av = load<V>(tensor01 + lda * (ii + i) + l);
						for (size_t j = 0; j < RN; ++j) {
							Cv[j][i] = madd(Av, Bv[j], Cv[j][i]);
						}
					}
				}
			}
			for (size_t j = 0; j < RN; ++j)
				for (size_t i = 0; i < RM; ++i)
					tensor03[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
		}

		template<size_t RM, size_t RN, size_t BM, size_t m, size_t n, size_t BN> OIML_FORCE_INLINE void gemm() {
			static std::atomic<size_t> current_chunk;

			OIML_ASSERT(m % (RM * BM) == 0);
			static constexpr size_t ytiles = m / (RM * BM);
			static constexpr size_t xtiles = (n + RN - 1) / RN;
			static constexpr size_t jj_RN  = (xtiles - (xtiles * RN - n));

			static constexpr size_t NB_BN	= xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
			static constexpr size_t SIZE_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
			static constexpr size_t jj_BN	= (NB_BN - (NB_BN * SIZE_BN - xtiles));
			static constexpr size_t nb_job	= ytiles * NB_BN;

			size_t job = thread_index;
			while (job < nb_job) {
				const size_t ii	 = (job % ytiles) * RM * BM;
				const size_t jb	 = job / ytiles;
				const size_t jr0 = block_pos(jb, jj_BN, SIZE_BN);
				const size_t jrN = block_pos(jb + 1, jj_BN, SIZE_BN);

				const size_t jj0 = block_pos(jr0, jj_RN, RN);
				const size_t jj2 = block_pos(jrN, jj_RN, RN);
				const size_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

				for (size_t bi = 0; bi < BM * RM; bi += RM) {
					size_t jj = jj0;
					for (; jj < jj1; jj += RN) {
						gemm_bloc<RM, RN>(ii + bi, jj);
					}
					if constexpr (RN > 1) {
						for (; jj < jj2; jj += RN - 1) {
							gemm_bloc<RM, RN - 1>(ii + bi, jj);
						}
					}
					OIML_ASSERT(jj == jj2);
				}

				++job;
			}

			return;
		}

		size_t thread_index{};
		size_t thread_count{};
		const TA* const tensor01;
		const TB* const tensor02;
		TC* const tensor03;
		const size_t lda;
		const size_t ldb;
		const size_t ldc;
	};

	template<size_t m, size_t n, size_t k> OIML_FORCE_INLINE static bool oiml_sgemm(size_t thread_index, size_t thread_count, const block_q8_0<oiml_half>* tensor01, int64_t lda,
		const block_q8_0<oiml_half>* tensor02, int64_t ldb, float* tensor03, int64_t ldc) {
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
		tiny_blas_q0_avx<block_q8_0<oiml_half>, block_q8_0<oiml_half>, float> tb{ k, tensor01, lda, tensor02, ldb, tensor03, ldc, thread_index, thread_count };
		tb.matmul(m, n);
		return true;
#elif defined(__ARM_FEATURE_DOTPROD)
		tinyBLAS_Q0_ARM<block_q8_0<oiml_half>> tb{ k, tensor01, lda, tensor02, ldb, tensor03, ldc, static_cast<int32_t>(thread_index), static_cast<int32_t>(thread_count) };
		tb.matmul(m, n);
		return true;

#elif defined(__MMA__)
		if constexpr (n < 8 && n != 4)
			return false;
		if constexpr (m < 8 && m != 4)
			return false;
		tinyBLAS_Q0_PPC<block_q8_0<oiml_half>, block_q8_0<oiml_half>, float> tb{ k, tensor01, lda, tensor02, ldb, tensor03, ldc, thread_index, thread_count };
		tb.matmul(m, n);
		return true;

#else
		return false;
#endif
	}

	OIML_FORCE_INLINE static bool oiml_sgemm(size_t m, size_t n, size_t k, size_t thread_index, size_t thread_count, const void* tensor01, int64_t lda, const void* tensor02,
		int64_t ldb, void* tensor03, int64_t ldc) {
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
		tiny_blas_q0_avx<block_q8_0<oiml_half>, block_q8_0<oiml_half>, float> tb{ static_cast<int64_t>(k), static_cast<const block_q8_0<oiml_half>*>(tensor01), lda,
			static_cast<const block_q8_0<oiml_half>*>(tensor02), ldb, static_cast<float*>(tensor03), ldc, thread_index, thread_count };
		tb.matmul(m, n);
		return true;
#elif defined(__ARM_FEATURE_DOTPROD)
		tinyBLAS_Q0_ARM<block_q8_0<oiml_half>> tb{ static_cast<int64_t>(k), static_cast<const block_q8_0<oiml_half>*>(tensor01), lda,
			static_cast<const block_q8_0<oiml_half>*>(tensor02), ldb, static_cast<float*>(tensor03), ldc, static_cast<int32_t>(thread_index), static_cast<int32_t>(thread_count) };
		tb.matmul(m, n);
		return true;

#elif defined(__MMA__)
		if constexpr (n < 8 && n != 4)
			return false;
		if constexpr (m < 8 && m != 4)
			return false;
		tinyBLAS_Q0_PPC<block_q8_0<oiml_half>, block_q8_0<oiml_half>, float> tb{ k, tensor01, lda, tensor02, ldb, tensor03, ldc, thread_index, thread_count };
		tb.matmul(m, n);
		return true;

#else
		return false;
#endif
	}

	template<size_t m, size_t n, size_t k> OIML_FORCE_INLINE static bool oiml_sgemm(size_t thread_index, size_t thread_count, const float* tensor01, int64_t lda,
		const float* tensor02, int64_t ldb, float* tensor03, int64_t ldc) {
#if defined(__AVX512F__)
		tiny_blas<16, k, __m512, __m512, float, float, float> tb{ params, tensor01, lda, tensor02, ldb, tensor03, ldc };
		return tb.template matmul<m, n>();
#elif defined(__AVX__) || defined(__AVX2__)
		tiny_blas<8, k, __m256, __m256, float, float, float> tb{ thread_index, thread_count, tensor01, lda, tensor02, ldb, tensor03, ldc };
		return tb.template matmul<m, n>();
#elif defined(__ARM_NEON)
		if (n < 4)
			return false;
		tiny_blas<4, k, float32x4_t, float32x4_t, float, float, float> tb{ thread_index, thread_count, tensor01, lda, tensor02, ldb, tensor03, ldc };
		return tb.template matmul<m, n>();
#endif
	}

	template<typename tensor_type01, typename tensor_type02> using temp_tensor_type =
		oiml_get_tensor_type_t<typename tensor_type01::device_type, tensor_type01::type, tensor_type01::op_type, tensor_type02::dims[0], tensor_type02::dims[1]>;
	template<typename tensor_type01, typename tensor_type02> inline thread_local temp_tensor_type<tensor_type01, tensor_type02> temp_tensor{};

	inline static thread_local oiml_dynamic_tensor temp_tensor_dynamic{};

	struct oiml_mat_mul_singlethreaded {
		template<is_dynamic_tensor_base tensor_type01, is_dynamic_tensor_base tensor_type02, is_dynamic_tensor_base tensor_type03> OIML_FORCE_INLINE static void impl(
			const tensor_type01& tensor01, const tensor_type02& tensor02, tensor_type03& tensor03, size_t thread_index = 0, size_t thread_count = 1) {
			size_t ne00 = tensor01.dims[0];
			size_t ne01 = tensor01.dims[1];
			size_t ne02 = tensor01.dims[2];
			size_t ne03 = tensor01.dims[3];
			size_t nb00 = tensor01.byte_strides[0];
			size_t nb01 = tensor01.byte_strides[1];
			size_t nb02 = tensor01.byte_strides[2];
			size_t nb03 = tensor01.byte_strides[3];
			size_t ne10 = tensor02.dims[0];
			size_t ne11 = tensor02.dims[1];
			size_t ne12 = tensor02.dims[2];
			size_t ne13 = tensor02.dims[3];
			size_t nb10 = tensor02.byte_strides[0];
			size_t nb11 = tensor02.byte_strides[1];
			size_t nb12 = tensor02.byte_strides[2];
			size_t nb13 = tensor02.byte_strides[3];
			size_t ne0	= tensor03.dims[0];
			size_t ne1	= tensor03.dims[1];
			size_t ne2	= tensor03.dims[2];
			size_t ne3	= tensor03.dims[3];
			size_t nb0	= tensor03.byte_strides[0];
			size_t nb1	= tensor03.byte_strides[1];
			size_t nb2	= tensor03.byte_strides[2];
			size_t nb3	= tensor03.byte_strides[3];
			size_t r2	= ne12 / ne02;
			size_t r3	= ne13 / ne03;

			if (tensor02.oiml_is_contiguous() && tensor01.type == tensor02.type) {
				for (int64_t i13 = 0; i13 < ne13; i13++) {
					for (int64_t i12 = 0; i12 < ne12; i12++) {
						if (!oiml_sgemm(ne01, ne11, ne00 / tensor01.rep_traits.type_size, thread_index, thread_count,
								( const void* )(( const char* )tensor01.data() + i12 / r2 * nb02 + i13 / r3 * nb03), nb01 / tensor01.rep_traits.type_size,
								( const void* )(( const char* )tensor02.data() + i12 * nb12 + i13 * nb13), nb11 / tensor02.rep_traits.type_size,
								( void* )(( char* )tensor03.data() + i12 * nb2 + i13 * nb3), nb1 / tensor03.rep_traits.type_size)) {
							goto UseGgmlGemm1;
						}
					}
				}
				return;
			}
		UseGgmlGemm1:;
			char* wdata{};
			if (tensor02.type != tensor01.rep_traits.vec_dot_type) {
				temp_tensor_dynamic.update_tensor_properties(tensor01.type, tensor02.dims[0], tensor02.dims[1], tensor02.dims[2], tensor02.dims[3]);
				wdata		= ( char* )temp_tensor_dynamic.data();
				size_t nbw1 = tensor01.get_row_size(ne10);
				size_t nbw2 = nbw1 * ne11;
				size_t nbw3 = nbw2 * ne12;


				for (int64_t i13 = 0; i13 < ne13; ++i13) {
					for (int64_t i12 = 0; i12 < ne12; ++i12) {
						for (int64_t i11 = thread_index; i11 < ne11; i11 += thread_count) {
							from_float_function_dispatcher::impl(cpu_arch_index, tensor_type01::type, ( float* )(( char* )tensor02.data() + i13 * nb13 + i12 * nb12 + i11 * nb11),
								( void* )(wdata + i13 * nbw3 + i12 * nbw2 + i11 * nbw1), ne10);
						}
					}
				}
			}
			if (tensor02.type != tensor01.rep_traits.vec_dot_type) {
				size_t nbw1 = tensor01.get_row_size(ne10);

				for (int64_t i13 = 0; i13 < ne13; i13++) {
					for (int64_t i12 = 0; i12 < ne12; i12++) {
						if (!oiml_sgemm(ne01, ne11, ne00 / tensor01.rep_traits.block_size, thread_index, thread_count,
								( const void* )(( const char* )tensor01.data() + i12 / r2 * nb02 + i13 / r3 * nb03), nb01 / tensor01.rep_traits.type_size,
								( const void* )(( const char* )wdata + (i12 * ne11 + i13 * ne12 * ne11) * nbw1), nbw1 / temp_tensor_dynamic.rep_traits.type_size,
								( void* )(( char* )tensor03.data() + i12 * nb2 + i13 * nb3), nb1 / tensor03.rep_traits.type_size)) {
							goto UseGgmlGemm2;
						}
					}
				}
				return;
			}
		UseGgmlGemm2:;

			int64_t vec_dot_num_rows = tensor01.rep_traits.n_rows;

			int64_t nr0 = ne0;

			int64_t nr1 = ne1 * ne2 * ne3;

			size_t chunk_size = (nr0 == 1 || nr1 == 1) ? 64 : 16;

			int64_t nchunk0_first = (nr0 + chunk_size - 1) / chunk_size;
			int64_t nchunk1_first = (nr1 + chunk_size - 1) / chunk_size;

			int64_t nchunk0 = (nchunk0_first * nchunk1_first < thread_count * 4) ? (nr0 > nr1 ? thread_count : 1) : nchunk0_first;
			int64_t nchunk1 = (nchunk0_first * nchunk1_first < thread_count * 4) ? (nr1 ? 1 : thread_count) : nchunk1_first;

			int64_t dr0			 = (nr0 + nchunk0 - 1) / nchunk0;
			int64_t dr1			 = (nr1 + nchunk1 - 1) / nchunk1;
			size_t current_chunk = thread_index;
			float tmp[32]{};

			while (current_chunk < nchunk0 * nchunk1) {
				const int64_t ith0			 = current_chunk % nchunk0;
				const int64_t ith1			 = current_chunk / nchunk0;
				const int64_t ir0_start		 = dr0 * ith0;
				const int64_t ir0_end		 = MIN(ir0_start + dr0, nr0);
				const int64_t ir1_start		 = dr1 * ith1;
				const int64_t ir1_end		 = MIN(ir1_start + dr1, nr1);
				int64_t num_rows_per_vec_dot = 1;
				size_t nbw1					 = tensor01.get_row_size(ne10);

				const int64_t blck_0 = 16;
				const int64_t blck_1 = 16;

				const size_t src1_col_stride = nbw1;

				for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
					for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
						for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
							const int64_t i13 = (ir1 / (ne12 * ne1));
							const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
							const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

							const int64_t i03 = i13 / r3;
							const int64_t i02 = i12 / r2;

							const int64_t i1 = i11;
							const int64_t i2 = i12;
							const int64_t i3 = i13;

							const char* src0_row = ( const char* )tensor01.data() + (0 + i02 * nb02 + i03 * nb03);
							const char* src1_col = ( const char* )wdata + (i11 + i12 * ne11 + i13 * ne12 * ne11) * nbw1;
							float* dst_col		 = ( float* )(( char* )tensor03.data() + (i1 * nb1 + i2 * nb2 + i3 * nb3));

							for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
								//oiml_vec_dot_dynamic<void, void, void>::impl(ne00, tensor01.rep_traits.vec_dot_type, ( void* )( char* )(&tmp[ir0 - iir0]),
								//( const void* )( const char* )(src0_row + ir0 * nb01), ( const void* )( const char* )src1_col);
							}

							for (size_t cn = 0; cn < num_rows_per_vec_dot; ++cn) {
								memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
							}
						}
					}
				}
				if (thread_count >= nchunk0 * nchunk1) {
					break;
				}
				++current_chunk;
			}
		}

		template<typename tensor_type01, typename tensor_type02, typename tensor_type03> OIML_FORCE_INLINE static void impl(const tensor_type01& tensor01,
			const tensor_type02& tensor02, tensor_type03& tensor03, size_t thread_index = 0, size_t thread_count = 1) {
			static constexpr size_t ne00 = tensor_type01::dims[0];
			static constexpr size_t ne01 = tensor_type01::dims[1];
			static constexpr size_t ne02 = tensor_type01::dims[2];
			static constexpr size_t ne03 = tensor_type01::dims[3];
			static constexpr size_t nb00 = tensor_type01::byte_strides[0];
			static constexpr size_t nb01 = tensor_type01::byte_strides[1];
			static constexpr size_t nb02 = tensor_type01::byte_strides[2];
			static constexpr size_t nb03 = tensor_type01::byte_strides[3];
			static constexpr size_t ne10 = tensor_type02::dims[0];
			static constexpr size_t ne11 = tensor_type02::dims[1];
			static constexpr size_t ne12 = tensor_type02::dims[2];
			static constexpr size_t ne13 = tensor_type02::dims[3];
			static constexpr size_t nb10 = tensor_type02::byte_strides[0];
			static constexpr size_t nb11 = tensor_type02::byte_strides[1];
			static constexpr size_t nb12 = tensor_type02::byte_strides[2];
			static constexpr size_t nb13 = tensor_type02::byte_strides[3];
			static constexpr size_t ne0	 = tensor_type03::dims[0];
			static constexpr size_t ne1	 = tensor_type03::dims[1];
			static constexpr size_t ne2	 = tensor_type03::dims[2];
			static constexpr size_t ne3	 = tensor_type03::dims[3];
			static constexpr size_t nb0	 = tensor_type03::byte_strides[0];
			static constexpr size_t nb1	 = tensor_type03::byte_strides[1];
			static constexpr size_t nb2	 = tensor_type03::byte_strides[2];
			static constexpr size_t nb3	 = tensor_type03::byte_strides[3];
			static constexpr size_t r2	 = ne12 / ne02;
			static constexpr size_t r3	 = ne13 / ne03;

			if constexpr (tensor_type02::oiml_is_contiguous() && std::is_same_v<typename tensor_type01::value_type, typename tensor_type02::value_type>) {
				for (int64_t i13 = 0; i13 < ne13; i13++) {
					for (int64_t i12 = 0; i12 < ne12; i12++) {
						if (!oiml_sgemm<ne01, ne11, ne00 / tensor_type01::type_size>(thread_index, thread_count,
								( const typename tensor_type01::value_type* )( const char* )tensor01.data() + i12 / r2 * nb02 + i13 / r3 * nb03, nb01 / tensor_type01::type_size,
								( const typename tensor_type02::value_type* )( const char* )tensor02.data() + i12 * nb12 + i13 * nb13, nb11 / tensor_type02::type_size,
								( typename tensor_type03::value_type* )( char* )tensor03.data() + i12 * nb2 + i13 * nb3, nb1 / tensor_type03::type_size)) {
							goto UseGgmlGemm1;
						}
					}
				}
				return;
			}
		UseGgmlGemm1:;
			char* wdata{};
			if constexpr (tensor_type02::type != tensor_type01::vec_dot_type) {
				wdata = ( char* )temp_tensor<tensor_type01, tensor_type02>.data();

				static constexpr size_t nbw1 = tensor_type01::get_row_size(ne10);
				static constexpr size_t nbw2 = nbw1 * ne11;
				static constexpr size_t nbw3 = nbw2 * ne12;

				for (int64_t i13 = 0; i13 < ne13; ++i13) {
					for (int64_t i12 = 0; i12 < ne12; ++i12) {
						for (int64_t i11 = thread_index; i11 < ne11; i11 += thread_count) {
							//oiml_from_float::impl(oiml::cpu_arch_index,
							//( float* )(( char* )tensor02.data() + i13 * nb13 + i12 * nb12 + i11 * nb11),
							//								( typename temp_tensor_type<tensor_type01, tensor_type02>::value_type* )( void* )(wdata + i13 * nbw3 + i12 * nbw2 + i11 * nbw1));
						}
					}
				}
			}

			if constexpr (tensor_type02::type != tensor_type01::vec_dot_type) {
				static constexpr size_t get_row_size = tensor_type01::get_row_size(ne10);

				for (int64_t i13 = 0; i13 < ne13; i13++) {
					for (int64_t i12 = 0; i12 < ne12; i12++) {
						if (!oiml_sgemm<ne01, ne11, ne00 / tensor_type01::block_size>(thread_index, thread_count,
								( const typename tensor_type01::value_type* )(( const char* )tensor01.data() + i12 / r2 * nb02 + i13 / r3 * nb03), nb01 / tensor_type01::type_size,
								( const typename temp_tensor_type<tensor_type01, tensor_type02>::value_type* )( const char* )wdata +
									(i12 * ne11 + i13 * ne12 * ne11) * get_row_size,
								get_row_size / temp_tensor_type<tensor_type01, tensor_type02>::type_size,
								( typename tensor_type03::value_type* )(( char* )tensor03.data() + i12 * nb2 + i13 * nb3), nb1 / tensor_type03::type_size)) {
							goto UseGgmlGemm2;
						}
					}
				}
				return;
			}
		UseGgmlGemm2:;

			static constexpr int64_t vec_dot_num_rows = tensor_type01::n_rows;

			static constexpr int64_t nr0 = ne0;

			static constexpr int64_t nr1 = ne1 * ne2 * ne3;

			static constexpr size_t chunk_size = (nr0 == 1 || nr1 == 1) ? 64 : 16;

			static constexpr int64_t nchunk0_first = (nr0 + chunk_size - 1) / chunk_size;
			static constexpr int64_t nchunk1_first = (nr1 + chunk_size - 1) / chunk_size;

			int64_t nchunk0 = (nchunk0_first * nchunk1_first < thread_count * 4) ? (nr0 > nr1 ? thread_count : 1) : nchunk0_first;
			int64_t nchunk1 = (nchunk0_first * nchunk1_first < thread_count * 4) ? (nr1 ? 1 : thread_count) : nchunk1_first;

			int64_t dr0			  = (nr0 + nchunk0 - 1) / nchunk0;
			int64_t dr1			  = (nr1 + nchunk1 - 1) / nchunk1;
			int64_t current_chunk = thread_index;
			float tmp[32];

			while (current_chunk < nchunk0 * nchunk1) {
				const int64_t ith0							  = current_chunk % nchunk0;
				const int64_t ith1							  = current_chunk / nchunk0;
				const int64_t ir0_start						  = dr0 * ith0;
				const int64_t ir0_end						  = MIN(ir0_start + dr0, nr0);
				const int64_t ir1_start						  = dr1 * ith1;
				const int64_t ir1_end						  = MIN(ir1_start + dr1, nr1);
				static constexpr int64_t num_rows_per_vec_dot = 1;
				static constexpr size_t get_row_size		  = tensor_type01::get_row_size(ne10);

				const int64_t blck_0 = 16;
				const int64_t blck_1 = 16;

				const size_t src1_col_stride = get_row_size;

				for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
					for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
						for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
							const int64_t i13 = (ir1 / (ne12 * ne1));
							const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
							const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

							const int64_t i03 = i13 / r3;
							const int64_t i02 = i12 / r2;

							const int64_t i1 = i11;
							const int64_t i2 = i12;
							const int64_t i3 = i13;

							const char* src0_row = ( const char* )tensor01.data() + (0 + i02 * nb02 + i03 * nb03);
							const char* src1_col = ( const char* )wdata + (i11 + i12 * ne11 + i13 * ne12 * ne11) * get_row_size;
							float* dst_col		 = ( float* )(( char* )tensor03.data() + (i1 * nb1 + i2 * nb2 + i3 * nb3));

							for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
								//oiml_vec_dot::impl(cpu_arch_index, tensor_type01::vec_dot_type, ( typename tensor_type03::value_type* )(&tmp[ir0 - iir0]),
								//( typename tensor_type01::value_type* )(src0_row + ir0 * nb01),
								//									( typename temp_tensor_type<tensor_type01, tensor_type02>::value_type* )src1_col);
							}

							for (size_t cn = 0; cn < num_rows_per_vec_dot; ++cn) {
								memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
							}
						}
					}
				}
				if (static_cast<int64_t>(thread_count) >= nchunk0 * nchunk1) {
					break;
				}
				++current_chunk;
			}
		}
	};

	struct oiml_mat_mul_multihreaded {
		template<typename tensor_type01, typename tensor_type02, typename tensor_type03>
		OIML_FORCE_INLINE static void impl(tensor_type01& tensor01, tensor_type02& tensor02, tensor_type03& tensor03) {
			static const size_t thread_count = std::thread::hardware_concurrency();

			auto block_multiply = [&](size_t thread_index) {
				oiml_mat_mul_singlethreaded::impl(tensor01, tensor02, tensor03, thread_index, thread_count);
			};

			std::vector<std::future<void>> futures{};

			for (size_t x = 0; x < thread_count; ++x) {
				futures.emplace_back(thread_pool_val.enqueue(block_multiply, x));
			}

			for (auto& value: futures) {
				value.wait();
			}
		}
	};

	template<enum oiml_backend_device_types backend_type, oiml_op_type op_type, auto... type_properties> struct op_executor;

	template<enum oiml_backend_device_types backend_type, oiml_op_type op_type, auto... type_properties> struct op_executor_dynamic;

	template<> struct op_executor<oiml_backend_device_types::cpu, oiml_op_type::oiml_op_mul_mat> {
		template<typename tensor_type01, typename tensor_type02, typename tensor_type03>
		OIML_FORCE_INLINE static auto impl(const tensor_type01& tensor01, const tensor_type02& tensor02, tensor_type03& tensor03) {
			return impl_single_threaded(tensor01, tensor02, tensor03);
		}

		template<typename tensor_type01, typename tensor_type02, typename tensor_type03>
		OIML_FORCE_INLINE static auto impl_single_threaded(const tensor_type01& tensor01, const tensor_type02& tensor02, tensor_type03& tensor03) {
			oiml_mat_mul_singlethreaded::impl(tensor01, tensor02, tensor03);
		}

		template<typename tensor_type01, typename tensor_type02, typename tensor_type03>
		OIML_FORCE_INLINE static auto impl_multi_threaded(const tensor_type01& tensor01, const tensor_type02& tensor02, tensor_type03& tensor03) {
			oiml_mat_mul_multihreaded::impl(tensor01, tensor02, tensor03);
		}
	};

	template<> struct op_executor<oiml_backend_device_types::cpu, oiml_op_type::oiml_op_load> {
		template<typename tensor_type01, typename value_type01> OIML_FORCE_INLINE static auto impl(tensor_type01& tensor01, value_type01* input) {
			std::memcpy(tensor01.data(), input, tensor01.get_total_bytes());
		}
	};

	template<> struct op_executor<oiml_backend_device_types::cpu, oiml_op_type::oiml_op_store> {
		template<typename tensor_type01> OIML_FORCE_INLINE static auto impl(tensor_type01& tensor01, tensor_type01::value_type* output) {
			std::memcpy(output, tensor01.data(), tensor01.get_total_bytes());
		}
	};

	template<> struct op_executor_dynamic<oiml_backend_device_types::cpu, oiml_op_type::oiml_op_mul_mat> {
		template<typename tensor_type01, typename tensor_type02, typename tensor_type03>
		OIML_FORCE_INLINE static auto impl(const tensor_type01& tensor01, const tensor_type02& tensor02, tensor_type03& tensor03) {
			return impl_single_threaded(tensor01, tensor02, tensor03);
		}

		template<typename tensor_type01, typename tensor_type02, typename tensor_type03>
		OIML_FORCE_INLINE static auto impl_single_threaded(const tensor_type01& tensor01, const tensor_type02& tensor02, tensor_type03& tensor03) {
			oiml_mat_mul_singlethreaded::impl(tensor01, tensor02, tensor03);
		}

		template<typename tensor_type01, typename tensor_type02, typename tensor_type03>
		OIML_FORCE_INLINE static auto impl_multi_threaded(const tensor_type01& tensor01, const tensor_type02& tensor02, tensor_type03& tensor03) {
			oiml_mat_mul_multihreaded::impl(tensor01, tensor02, tensor03);
		}
	};

	template<> struct op_executor_dynamic<oiml_backend_device_types::cpu, oiml_op_type::oiml_op_load> {
		template<typename tensor_type01, typename value_type01> OIML_FORCE_INLINE static auto impl(tensor_type01& tensor01, value_type01* input) {
			std::memcpy(tensor01.data(), input, tensor01.get_total_bytes());
		}
	};

	template<> struct op_executor_dynamic<oiml_backend_device_types::cpu, oiml_op_type::oiml_op_store> {
		template<typename tensor_type01, typename value_type> OIML_FORCE_INLINE static auto impl(tensor_type01& tensor01, value_type* output) {
			std::memcpy(output, tensor01.data(), tensor01.get_total_bytes());
		}
	};
}
