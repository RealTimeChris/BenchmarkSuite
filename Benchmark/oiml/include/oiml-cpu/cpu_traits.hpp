#pragma once

#include <oiml/common/config.hpp>

namespace oiml {

	enum class oiml_cpu_type {
		x64 = 0,
		avx,
		avx_2,
		avx_512,
		arm_neon,
		arm_sve,
	};

	template<oiml_cpu_type type_new> struct oiml_cpu_traits;

	template<> struct oiml_cpu_traits<oiml_cpu_type::x64> {
		static constexpr uint64_t alignment{ 8 };
	};

	template<> struct oiml_cpu_traits<oiml_cpu_type::avx> {
		static constexpr uint64_t alignment{ 16 };
	};

	template<> struct oiml_cpu_traits<oiml_cpu_type::avx_2> {
		static constexpr uint64_t alignment{ 32 };
	};

	template<> struct oiml_cpu_traits<oiml_cpu_type::avx_512> {
		static constexpr uint64_t alignment{ 64 };
	};

	template<> struct oiml_cpu_traits<oiml_cpu_type::arm_neon> {
		static constexpr uint64_t alignment{ 16 };
	};

	template<> struct oiml_cpu_traits<oiml_cpu_type::arm_sve> {
		static constexpr uint64_t alignment{ 256 };
	};

	inline static constexpr const uint64_t aligments[]{ 8, 16, 32, 64, 16, 256 };

}