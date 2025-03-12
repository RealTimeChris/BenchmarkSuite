#pragma once

#include <oiml/common/util_functions.hpp>
#include <oiml-cpu/detect_isa.hpp>
#include <oiml/common/common.hpp>
#include <oiml-cpu/common.hpp>
#include <assert.h>
#include <cassert>

namespace oiml {

	template<> struct function_dispatcher<function_type::from_float, oiml_representation_types::q8_0, 0> {
		OIML_FORCE_INLINE static void impl(const float* __restrict x, block_q8_0<oiml_half>* __restrict y, int64_t k) {
			assert(k % oiml::Q_SIZE == 0);
			const uint64_t nb = static_cast<uint64_t>(k) / oiml::Q_SIZE;

			for (uint64_t i = 0; i < nb; i++) {
				float amax = 0.0f;

				for (uint64_t j = 0; j < oiml::Q_SIZE; j++) {
					const float v = x[i * oiml::Q_SIZE + j];
					amax		  = std::max(amax, fabsf(v));
				}

				const float d  = amax / ((1 << 7) - 1);
				const float id = d ? 1.0f / d : 0.0f;

				y[i].d = oiml::oiml_fp32_to_fp16(d);

				for (uint64_t j = 0; j < oiml::Q_SIZE; ++j) {
					const float x0 = x[i * oiml::Q_SIZE + j] * id;

					y[i].qs[j] = roundf(x0);
				}
			}
		}
	};

	template<> struct function_dispatcher<function_type::to_float, oiml_representation_types::q8_0, 0> {
		OIML_FORCE_INLINE static void impl(const block_q8_0<oiml_half>* __restrict x, float* __restrict y, int64_t k) {
			std::cerr << "Not implemented!" << std::endl;
			std::abort();
		}
	};

	template<> struct function_dispatcher<function_type::vec_dot, oiml_representation_types::q8_0, 0> {
		OIML_FORCE_INLINE static void impl(const block_q8_0<oiml_half>* __restrict x, const block_q8_0<oiml_half>* __restrict y, float* __restrict z, int64_t k) {
			static constexpr int qk = oiml::Q_SIZE;
			static_assert(qk == 32);

			const uint64_t nb = static_cast<uint64_t>(k) / oiml::Q_SIZE;
			uint64_t ib = 0;
			float sumf	= 0;
			for (; ib < nb; ++ib) {
				int sumi = 0;

				for (int j = 0; j < qk; j++) {
					sumi += x[ib].qs[j] * y[ib].qs[j];
				}

				sumf += sumi * (oiml::oiml_lookup_fp16_to_fp32(x[ib].d) * oiml::oiml_lookup_fp16_to_fp32(y[ib].d));
			}

			*z = sumf;
		}
	};

}