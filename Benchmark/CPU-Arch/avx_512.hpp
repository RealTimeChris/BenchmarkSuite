#pragma once

#include "detect_isa.hpp"

#if defined(OIML_IS_X86_64)

#include <immintrin.h>

BNCH_SWT_INLINE void impl_03() {
	__m512 a = _mm512_set1_ps(1.0f);
	__m512 b = _mm512_set1_ps(2.0f);
	__m512 c = _mm512_add_ps(a, b);
	std::cout << "avx_512 Function Executed\n";
}

#endif