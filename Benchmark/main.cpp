#include <iostream>
#include <vector>
#include <array>
#include <bit>

#include <type_traits>
enum class oiml_op {};
enum class oiml_type {};

template<oiml_op op_type, uint64_t index, oiml_type type01, oiml_type type02, oiml_type type03> struct function_dispatcher_impl;

template<oiml_op op_type, uint64_t index, oiml_type type01, oiml_type type02, oiml_type type03> constexpr auto error_printer() {
	using impl = function_dispatcher_impl<op_type, index, type01, type02, type03>;

	// This will only be triggered when the function is actually called
	if constexpr (!impl::specialized) {
		// This produces a helpful error message with the template parameters included
		static_assert(impl::specialized, "Missing specialization for function_dispatcher_impl with these template parameters");
	}

	return impl{};
}

template<oiml_op op_type, uint64_t index, oiml_type type01, oiml_type type02, oiml_type type03> struct error_printer_instantiation;

// Primary template - deliberately minimal
template<oiml_op op_type, uint64_t index, oiml_type type01, oiml_type type02, oiml_type type03> struct function_dispatcher_impl {
	inline static constexpr bool specialized = false;
	inline static constexpr auto error_printer{ error_printer<op_type, index, type01, type02, type03> };
};

// Example specialization
template<> struct function_dispatcher_impl<oiml_op{}, 0, oiml_type{}, oiml_type{}, oiml_type{}> {
	inline static constexpr bool specialized = true;

	static void execute() {
		// Implementation
	}
};

// This is the function you use to safely access a dispatcher

int main() {
	// This works fine
	auto d1 = error_printer<oiml_op{}, 0, oiml_type{}, oiml_type{}, oiml_type{}>();
	// This would error if uncommented with the template parameters visible in the error:
	return 0;
}