#pragma once

#include <oiml/common/config.hpp>
#include <oiml/common/op_executor.hpp>
#include <oiml/common/ring_buffer.hpp>
#include <oiml/common/tensor.hpp>
#include <oiml/common/common.hpp>

namespace oiml {

	template<::oiml_backend_device_types type_new> struct oiml_backend_device;

	template<> struct oiml_backend_device<oiml_backend_device_types::cpu> {
		static constexpr ::oiml_backend_device_types type{ oiml_backend_device_types::cpu };
		static constexpr size_t alignment{ 64 };

		template<oiml_op_type type_new, not_dynamic_tensor_base... tensor_types> OIML_FORCE_INLINE static auto execute_binary_op(tensor_types&... tensors) {
			static constexpr oiml_op_traits<type_new, tensor_types...> op_traits{};
			return op_executor<oiml_backend_device_types::cpu, type_new>::impl(tensors...);
		}

		template<oiml_op_type type_new, not_dynamic_tensor_base... tensor_types> OIML_FORCE_INLINE static auto execute_unary_op(tensor_types&... tensors) {
			return op_executor<oiml_backend_device_types::cpu, type_new>::impl(tensors...);
		}

		template<oiml_op_type type_new, typename... tensor_types> OIML_FORCE_INLINE static auto execute_binary_op(tensor_types&... tensors) {
			if (oiml_op_traits<type_new>::check_dimensions(tensors...)) {
				op_executor_dynamic<oiml_backend_device_types::cpu, type_new>::impl(tensors...);
			}
		}

		template<oiml_op_type type_new, typename... tensor_types> OIML_FORCE_INLINE static auto execute_unary_op(tensor_types&... tensors) {
			return op_executor_dynamic<oiml_backend_device_types::cpu, type_new>::impl(tensors...);
		}
	};
}
