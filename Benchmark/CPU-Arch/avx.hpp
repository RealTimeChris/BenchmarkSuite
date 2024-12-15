#pragma once

#include "detect_isa.hpp"

#if defined(OIML_IS_X86_64)

#include <immintrin.h>

BNCH_SWT_INLINE void impl_01() {
	__m128 a = _mm_set1_ps(1.0f);
	__m128 b = _mm_set1_ps(2.0f);
	__m128 c = _mm_add_ps(a, b);
	std::cout << "AVX Function Executed\n";
}

#endif