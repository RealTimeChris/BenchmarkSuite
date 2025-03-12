#pragma once

#include <oiml/common/util_functions.hpp>
#include <oiml-cpu/detect_isa.hpp>
#include <oiml/common/common.hpp>
#include <oiml-cpu/common.hpp>
#include <assert.h>
#include <cassert>

#if defined(OIML_IS_X86_64)

namespace oiml {

	template<> struct function_dispatcher<function_type::from_float, oiml_representation_types::q8_0, 1> {
		OIML_FORCE_INLINE static void impl(const float* __restrict x, block_q8_0<oiml_half>* __restrict y, int64_t k) {
			std::cerr << "Not implemented!" << std::endl;
			std::abort();
		}
	};

	template<> struct function_dispatcher<function_type::to_float, oiml_representation_types::q8_0, 1> {
		OIML_FORCE_INLINE static void impl(const block_q8_0<oiml_half>* __restrict x, float* __restrict y, int64_t k) {
			std::cerr << "Not implemented!" << std::endl;
			std::abort();
		}
	};

	template<> struct function_dispatcher<function_type::vec_dot, oiml_representation_types::q8_0, 1> {
		OIML_FORCE_INLINE static void impl(const block_q8_0<oiml_half>* __restrict x, const block_q8_0<oiml_half>* __restrict y, float* __restrict z, int64_t k) {
			std::cerr << "Not implemented!" << std::endl;
			std::abort();
		}
	};
}
#endif