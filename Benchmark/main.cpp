#include <iostream>
#include <vector>
#include <array>
#include <bit>

inline static constexpr float fp32_from_bits(uint32_t w) {
	return std::bit_cast<float>(w);
}

inline static constexpr uint32_t fp32_to_bits(float f) {
	return std::bit_cast<uint32_t>(f);
}

inline static constexpr float oiml_compute_fp16_to_fp32(uint16_t h) {
	const uint32_t w	 = static_cast<uint32_t>(h) << 16;
	const uint32_t sign	 = w & 0x80000000u;
	const uint32_t two_w = w + w;

	constexpr uint32_t exp_offset = 0xE0u << 23;
	constexpr float exp_scale	  = fp32_from_bits(0x7800000u);
	const float normalized_value  = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

	constexpr uint32_t magic_mask  = 126u << 23;
	constexpr float magic_bias	   = 0.5f;
	const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

	constexpr uint32_t denormalized_cutoff = 1u << 27;
	const uint32_t result				   = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
	return fp32_from_bits(result);
}

alignas(64)
inline const std::array<float, (1 << 16)> fp16_to_fp32_table{ [] {
	std::array<float, (1 << 16)> returnValues{};
	for (uint32_t x = 0; x < (1 << 16); ++x) {
		returnValues[x] = oiml_compute_fp16_to_fp32(static_cast<uint16_t>(x));
	}
	return returnValues;
}() };

int main() {
	auto new_value = fp16_to_fp32_table[0];
	std::cout << "CURRENT VALUE: " << new_value << std::endl;
	return 0;
}
