#pragma once

#include <oiml/common/representation_traits.hpp>
#include <oiml/common/tensor.hpp>

namespace oiml {

	struct oiml_binary_operation_record {
		oiml_array<oiml_dynamic_tensor*, 3> tensor_ptrs{};
		oiml_op_type type{};
	};

	struct oiml_unary_operation_record {
		oiml_array<oiml_dynamic_tensor*, 2> tensor_ptrs{};
		oiml_op_type type{};
	};

	struct oiml_op_graph {};
}