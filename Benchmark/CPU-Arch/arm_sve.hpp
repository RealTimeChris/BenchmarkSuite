#pragma once

#include "detect_isa.hpp"

#if defined(OIML_IS_ARM64)

	#include <arm_sve.h>
	#include <iostream>

BNCH_SWT_INLINE void impl_02() {
	svfloat32_t va = svdup_f32(1.0f);
	svfloat32_t vb = svdup_f32(2.0f);
	svadd_f32_z(svptrue_b32(), va, vb);
	std::cout << "SVE Function Executed\n";
}

BNCH_SWT_INLINE void impl_03() {
	std::cerr << "Erroneous cpu-arch-index. Failing" << std::endl;
	std::abort();
}

#endif