#include <oiml-cpu/fallback.hpp>
#include <oiml-cpu/cpu_arch.hpp>
#include <oiml-cpu/arm_neon.hpp>
#include <oiml-cpu/arm_sve.hpp>
#include <oiml-cpu/avx_512.hpp>
#include <oiml-cpu/avx_2.hpp>
#include <oiml-cpu/avx.hpp>

namespace oiml {

	inline static constexpr uint64_t max_index{ [] {
#if defined(OIML_IS_X86_64)
		return 3;
#else
		return 2;
#endif
	}() };

	template<function_type f_type, oiml_representation_types rep_type, template<function_type, oiml_representation_types, size_t> typename dispatcher_type>
	struct global_function_dispatcher {
		template<typename... arg_types, size_t... indices> OIML_FORCE_INLINE static void impl(size_t index, arg_types&... args, std::index_sequence<indices...>) noexcept {
			((indices == index ? (dispatcher_type<f_type, rep_type, indices>::impl(std::forward<arg_types>(args)...), false) : false), ...);
		}
	};

	template<oiml_representation_types type> struct to_float_function_dispatcher_impl;

	template<> struct to_float_function_dispatcher_impl<oiml_representation_types::q8_0> {
		using dst_type	 = float;
		using src_type01 = const block_q8_0<oiml_half>;
		using src_type02 = const block_q8_0<oiml_half>;
		OIML_FORCE_INLINE static void impl(size_t index, const block_q8_0<oiml_half>*   x, float*  y, int64_t k) {
			global_function_dispatcher<function_type::to_float, oiml_representation_types::q8_0, function_dispatcher>::impl<const block_q8_0<oiml_half>* ,
				float* , int64_t>(index, x, y, k, std::make_index_sequence<max_index>{});
		}
	};

	void to_float_function_dispatcher::impl(size_t index, oiml_representation_types type, const void*  x, float*  y, size_t k) {
		switch (type) {
			case oiml_representation_types::q8_0: {
				return to_float_function_dispatcher_impl<oiml_representation_types::q8_0>::impl(index, static_cast<const block_q8_0<oiml_half>* >(x), y, k);
			}
		}
	};

	template<oiml_representation_types type> struct from_float_function_dispatcher_impl;

	template<> struct from_float_function_dispatcher_impl<oiml_representation_types::q8_0> {
		using dst_type	 = float;
		using src_type01 = const block_q8_0<oiml_half>;
		using src_type02 = const block_q8_0<oiml_half>;
		OIML_FORCE_INLINE static void impl(size_t index, const float*  x, block_q8_0<oiml_half>*  y, int64_t k) {
			global_function_dispatcher<function_type::from_float, oiml_representation_types::q8_0, function_dispatcher>::impl<const float* ,
				block_q8_0<oiml_half>* , int64_t>(index, x, y, k, std::make_index_sequence<max_index>{});
		}
	};

	void from_float_function_dispatcher::impl(size_t index, oiml_representation_types type, const float*  x, void*  y, size_t k) {
		switch (type) {
			case oiml_representation_types::q8_0: {
				return from_float_function_dispatcher_impl<oiml_representation_types::q8_0>::impl(index, x, static_cast<block_q8_0<oiml_half>* >(y), k);
			}
		}
	};

	template<oiml_representation_types type> struct vec_dot_function_dispatcher_impl;

	template<> struct vec_dot_function_dispatcher_impl<oiml_representation_types::q8_0> {
		using dst_type	 = float;
		using src_type01 = const block_q8_0<oiml_half>;
		using src_type02 = const block_q8_0<oiml_half>;
		OIML_FORCE_INLINE static void impl(size_t index, const block_q8_0<oiml_half>*  x, const block_q8_0<oiml_half>*  y, float*  z, size_t k) {
			global_function_dispatcher<function_type::vec_dot, oiml_representation_types::q8_0, function_dispatcher>::impl<const block_q8_0<oiml_half>* ,
				const block_q8_0<oiml_half>* , float* , size_t>(index, x, y, z, k, std::make_index_sequence<max_index>{});
		}
	};

	void vec_dot_function_dispatcher::impl(size_t index, oiml_representation_types type, const void*  x, const void*  y, void*  z, size_t k) {
		switch (type) {
			case oiml_representation_types::q8_0: {
				return vec_dot_function_dispatcher_impl<oiml_representation_types::q8_0>::impl(index, static_cast<const block_q8_0<oiml_half>* >(x),
					static_cast<const block_q8_0<oiml_half>* >(y), static_cast<float* >(z), k);
			}
		}
	};

	void from_float_ref_function_dispatcher::impl(size_t index, oiml_representation_types type, const float* , void* , size_t n) {};

}