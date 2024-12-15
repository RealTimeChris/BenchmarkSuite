#pragma once

#include <detect_isa.hpp>

#pragma once

#include <cpu_arch.hpp>
#include <arm_neon.hpp>
#include <arm_sve.hpp>
#include <avx_512.hpp>
#include <avx_2.hpp>
#include <avx.hpp>

void impl_test_function_01(size_t index) {
	if (index == 0) {
		std::cout << "Index 0: " << std::endl;
		impl_01();
	} else if (index == 1) {
		std::cout << "Index 1: " << std::endl;
		impl_02();
	} else if (index == 2) {
		std::cout << "Index 2: " << std::endl;
		impl_03();
	}
}
