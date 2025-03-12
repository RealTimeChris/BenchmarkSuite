#pragma once

#include <oiml-cpu/detect_isa.hpp>
#include <oiml-cpu/cpu_traits.hpp>
#include <oiml/common/representation_traits.hpp>

namespace oiml {

	struct to_float_function_dispatcher {
		static void impl(size_t, oiml_representation_types, const void* , float* , size_t);
	};

	struct from_float_function_dispatcher {
		static void impl(size_t, oiml_representation_types, const float* x, void* y, size_t);
	};

	struct from_float_ref_function_dispatcher {
		static void impl(size_t, oiml_representation_types, const float* , void* , size_t);
	};

	struct vec_dot_function_dispatcher {
		static void impl(size_t, oiml_representation_types, const void* , const void* , void* , size_t);
	};

}