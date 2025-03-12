#pragma once

#include <oiml/common/config.hpp>
#include <oiml-cpu/detect_isa.hpp>
#include <oiml/common/tables.hpp>
#include <oiml/common/array.hpp>

#include <array>
#include <bitset>
#include <vector>

enum oiml_backend_device_types { cpu, gpu, OIML_BACKEND_DEVICE_TYPE_ACCEL };

#if defined(OIML_IS_X86_64)

	#include <immintrin.h>

namespace oiml {

	using oiml_simd_int_128 = __m128i;
	using oiml_simd_int_256 = __m256i;
	using oiml_simd_int_512 = __m512i;

	constexpr oiml_array<uint64_t, 3> bitsPerStep{ { 128, 256, 512 } };

#elif defined(OIML_IS_ARM64)

	#include <arm_neon.h>

namespace oiml {

	using oiml_simd_int_128 = uint8x16_t;
	using oiml_simd_int_256 = uint32_t;
	using oiml_simd_int_512 = size_t;

#else

namespace oiml {

	using oiml_simd_int_128 = uint16_t;
	using oiml_simd_int_256 = uint32_t;
	using oiml_simd_int_512 = uint64_t;

#endif

	template<typename value_type>
	concept simd_int_512_type = std::is_same_v<oiml_simd_int_512, std::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept simd_int_256_type = std::is_same_v<oiml_simd_int_256, std::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept simd_int_128_type = std::is_same_v<oiml_simd_int_128, std::remove_cvref_t<value_type>>;

	inline static constexpr uint64_t K_SCALE_SIZE{ 12 };
	inline static constexpr uint64_t Q_SIZE{ 32 };
	inline static constexpr uint64_t QK8_0{ 32 };
	inline static constexpr uint64_t QK_K{ 256 };
	inline static constexpr uint64_t QR8_0{ 1 };
	inline static constexpr uint64_t QI8_0{ (QK8_0 / (4 * QR8_0)) };

	using oiml_half	  = uint16_t;
	using oiml_half2  = uint32_t;
	using oiml_fp16_t = uint16_t;
	using oiml_bf16_t = uint16_t;
	using oiml_float  = float;

	template<typename half_type> struct block_q8_0 {
		half_type d;
		int8_t qs[Q_SIZE];

		OIML_FORCE_INLINE bool operator==(const block_q8_0<oiml_half>& other) const {
			if (d != other.d) {
				return false;
			}
			for (size_t x = 0; x < Q_SIZE; ++x) {
				if (qs[x] != other.qs[x]) {
					return false;
				}
			}
			return true;
		}
	};
	static_assert(sizeof(block_q8_0<oiml_half>) == sizeof(oiml_half) + Q_SIZE, "Wrong q8_0 block size/padding.");
}