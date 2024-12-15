#pragma once

#include "detect_isa.hpp"

#if defined(OIML_IS_ARM64)

	#include <arm_neon.h>
	#include <iostream>

BNCH_SWT_INLINE void impl_01() {
	float32x4_t a = vdupq_n_f32(1.0f);
	float32x4_t b = vdupq_n_f32(2.0f);
	float32x4_t c = vaddq_f32(a, b);
	std::cout << "NEON Function Executed\n";
}

#endif