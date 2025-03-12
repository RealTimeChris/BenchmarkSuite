#pragma once

#include <oiml/common/representation_traits.hpp>

namespace oiml {

	enum class oiml_data_types {
		uint_8,
		uint_16,
		uint_32,
		uint_64,
		float_16,
		float_32,
		float_64,
	};

	enum class oiml_op_type {
		oiml_op_none = 0,
		oiml_op_load,
		oiml_op_store,
		oiml_op_dup,
		oiml_op_add,
		oiml_op_add1,
		oiml_op_acc,
		oiml_op_sub,
		oiml_op_mul,
		oiml_op_div,
		oiml_op_sqr,
		oiml_op_sqrt,
		oiml_op_log,
		oiml_op_sin,
		oiml_op_cos,
		oiml_op_sum,
		oiml_op_sum_rows,
		oiml_op_mean,
		oiml_op_argmax,
		oiml_op_count_equal,
		oiml_op_repeat,
		oiml_op_repeat_back,
		oiml_op_concat,
		oiml_op_silu_back,
		oiml_op_norm,
		oiml_op_rms_norm,
		oiml_op_rms_norm_back,
		oiml_op_group_norm,
		oiml_op_mul_mat,
		oiml_op_mul_mat_id,
		oiml_op_out_prod,
		oiml_op_scale,
		oiml_op_set,
		oiml_op_cpy,
		oiml_op_cont,
		oiml_op_reshape,
		oiml_op_view,
		oiml_op_permute,
		oiml_op_transpose,
		oiml_op_get_rows,
		oiml_op_get_rows_back,
		oiml_op_diag,
		oiml_op_diag_mask_inf,
		oiml_op_diag_mask_zero,
		oiml_op_soft_max,
		oiml_op_soft_max_back,
		oiml_op_rope,
		oiml_op_rope_back,
		oiml_op_clamp,
		oiml_op_conv_transpose_1d,
		oiml_op_im2col,
		oiml_op_im2col_back,
		oiml_op_conv_transpose_2d,
		oiml_op_pool_1d,
		oiml_op_pool_2d,
		oiml_op_pool_2d_back,
		oiml_op_upscale,
		oiml_op_pad,
		oiml_op_pad_reflect_1d,
		oiml_op_arange,
		oiml_op_timestep_embedding,
		oiml_op_argsort,
		oiml_op_leaky_relu,
		oiml_op_flash_attn_ext,
		oiml_op_flash_attn_back,
		oiml_op_ssm_conv,
		oiml_op_ssm_scan,
		oiml_op_win_part,
		oiml_op_win_unpart,
		oiml_op_get_rel_pos,
		oiml_op_add_rel_pos,
		oiml_op_rwkv_wkv6,
		oiml_op_gated_linear_attn,
		oiml_op_unary,
		oiml_op_map_unary,
		oiml_op_map_binary,
		oiml_op_map_custom1_f32,
		oiml_op_map_custom2_f32,
		oiml_op_map_custom3_f32,
		oiml_op_map_custom1,
		oiml_op_map_custom2,
		oiml_op_map_custom3,
		oiml_op_cross_entropy_loss,
		oiml_op_cross_entropy_loss_back,
		oiml_op_opt_step_adamw
	};

	template<oiml_op_type type_new, typename... tensor_traits> struct oiml_op_traits;

	template<oiml_op_type type_new, typename tensor_traits01, typename tensor_traits02> struct oiml_op_traits<type_new, tensor_traits01, tensor_traits02>;

	template<oiml_op_type type_new, typename tensor_traits> struct oiml_op_traits<type_new, tensor_traits>;

	template<typename tensor_traits_new01, typename tensor_traits_new02, typename tensor_traits_new03>
	struct oiml_op_traits<oiml_op_type::oiml_op_mul_mat, tensor_traits_new01, tensor_traits_new02, tensor_traits_new03> {
		using tensor_traits01 = tensor_traits_new01;
		using tensor_traits02 = tensor_traits_new02;
		using tensor_traits03 = tensor_traits_new03;

		static constexpr oiml_op_type op_type{ oiml_op_type::oiml_op_mul_mat };
		static_assert(tensor_traits01::dims[0] == tensor_traits02::dims[0],
			"Dimension mismatch: tensor_traits01::dims[0] must be equal to tensor_traits02::dims[0] for matrix multiplication.");
		static_assert(tensor_traits02::dims[2] % tensor_traits01::dims[2] == 0,
			"Dimension mismatch: tensor_traits02::dims[2] must be divisible by tensor_traits01::dims[2] for matrix multiplication.");
		static_assert(tensor_traits02::dims[3] % tensor_traits01::dims[3] == 0,
			"Dimension mismatch: tensor_traits02::dims[3] must be divisible by tensor_traits01::dims[3] for matrix multiplication.");
		static_assert(tensor_traits03::dims[0] == tensor_traits01::dims[1],
			"Dimension mismatch: tensor_traits03::dims[0] must be equal to tensor_traits01::dims[1] for matrix multiplication.");
		static_assert(tensor_traits03::dims[1] == tensor_traits02::dims[1],
			"Dimension mismatch: tensor_traits03::dims[1] must be equal to tensor_traits02::dims[1] for matrix multiplication.");
	};

	template<oiml_op_type type_new> struct oiml_op_traits<type_new> {
		template<typename tensor_type01, typename tensor_type02, typename tensor_type03>
		OIML_FORCE_INLINE static bool check_dimensions(tensor_type01& tensor01, tensor_type02& tensor02, tensor_type03& tensor03) {
			if constexpr (type_new == oiml_op_type::oiml_op_mul_mat) {
				return (tensor01.dims[0] == tensor02.dims[0] && tensor02.dims[2] % tensor01.dims[2] == 0 && tensor02.dims[3] % tensor01.dims[3] == 0);
			} else {
				return true;
			}
		}
	};

}