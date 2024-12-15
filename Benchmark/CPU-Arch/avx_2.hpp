#pragma once

#include "detect_isa.hpp"

#if defined(OIML_IS_X86_64)

	#include <immintrin.h>

BNCH_SWT_INLINE void impl_02() {
	__m256 a = _mm256_set1_ps(1.0f);
	__m256 b = _mm256_set1_ps(2.0f);
	__m256 c = _mm256_add_ps(a, b);
	std::cout << "avx_2 Function Executed\n";
}

#endif