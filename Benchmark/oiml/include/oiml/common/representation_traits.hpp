#pragma once

#include <oiml/common/common.hpp>
#include <oiml/common/array.hpp>

namespace oiml {

	enum class oiml_representation_types {
		float_32,
		float_16,
		brain_float_16,
		q8_0,
		i8_32_f32_1,
		int_8,
		int_32,
		count
	};

	struct oiml_representation_traits_dynamic {
		oiml_representation_types vec_dot_type{};
		oiml_representation_types type{};
		uint64_t block_size{};
		uint64_t type_size{};
		bool is_quantized{};
		uint64_t n_rows{};
	};

	template<oiml_representation_types> struct oiml_representation_traits;

	template<> struct oiml_representation_traits<oiml_representation_types::float_32> {
		using value_type = float;
		using quant_type = float;
		inline static constexpr oiml_representation_types vec_dot_type{ oiml_representation_types::float_32 };
		inline static constexpr oiml_representation_types type{ oiml_representation_types::float_32 };
		inline static constexpr uint64_t type_size{ sizeof(float) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct oiml_representation_traits<oiml_representation_types::float_16> {
		using value_type = oiml_fp16_t;
		using quant_type = oiml_fp16_t;
		inline static constexpr oiml_representation_types vec_dot_type{ oiml_representation_types::float_16 };
		inline static constexpr oiml_representation_types type{ oiml_representation_types::float_16 };
		inline static constexpr uint64_t type_size{ sizeof(oiml_fp16_t) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct oiml_representation_traits<oiml_representation_types::brain_float_16> {
		using value_type = oiml_bf16_t;
		using quant_type = oiml_bf16_t;
		inline static constexpr oiml_representation_types vec_dot_type{ oiml_representation_types::brain_float_16 };
		inline static constexpr oiml_representation_types type{ oiml_representation_types::brain_float_16 };
		inline static constexpr uint64_t type_size{ sizeof(oiml_bf16_t) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct oiml_representation_traits<oiml_representation_types::q8_0> {
		using value_type = block_q8_0<oiml_half>;
		using quant_type = block_q8_0<oiml_half>;
		inline static constexpr oiml_representation_types vec_dot_type{ oiml_representation_types::q8_0 };
		inline static constexpr oiml_representation_types type{ oiml_representation_types::q8_0 };
		inline static constexpr uint64_t type_size{ sizeof(block_q8_0<oiml_half>) };
		inline static constexpr bool is_quantized{ true };
		inline static constexpr uint64_t block_size{ oiml::Q_SIZE };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct oiml_representation_traits<oiml_representation_types::i8_32_f32_1> {
		using value_type = block_q8_0<oiml_half>;
		using quant_type = block_q8_0<oiml_half>;
		inline static constexpr oiml_representation_types vec_dot_type{ oiml_representation_types::q8_0 };
		inline static constexpr oiml_representation_types type{ oiml_representation_types::q8_0 };
		inline static constexpr uint64_t type_size{ sizeof(block_q8_0<oiml_half>) };
		inline static constexpr bool is_quantized{ true };
		inline static constexpr uint64_t block_size{ oiml::Q_SIZE };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<size_t index = 0>
	static constexpr auto get_rep_traits_dynamic(oiml_array<oiml_representation_traits_dynamic, static_cast<size_t>(oiml_representation_types::count)> array_of_traits = {}) {
		if constexpr (index < static_cast<size_t>(oiml_representation_types::count) - 2) {
			using oiml_rep_traits				= oiml_representation_traits<static_cast<oiml_representation_types>(index)>;
			array_of_traits[index].type_size	= oiml_rep_traits::type_size;
			array_of_traits[index].is_quantized = oiml_rep_traits::is_quantized;
			array_of_traits[index].n_rows		= oiml_rep_traits::n_rows;
			array_of_traits[index].type			= oiml_rep_traits::type;
			array_of_traits[index].vec_dot_type = oiml_rep_traits::vec_dot_type;
			array_of_traits[index].block_size	= oiml_rep_traits::block_size;
			return get_rep_traits_dynamic<index + 1>(array_of_traits);
		}
		return array_of_traits;
	}

	static constexpr auto array_of_rep_traits{ get_rep_traits_dynamic() };

	OIML_FORCE_INLINE static constexpr oiml_representation_traits_dynamic get_rep_traits(oiml_representation_types type) {
		return array_of_rep_traits[static_cast<size_t>(type)];
	}
}