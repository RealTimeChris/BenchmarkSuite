#include <iostream>
#include <type_traits>
#include <cstdint>

// === CONSTEXPR DETECTION TRICK === //
template<typename T>
concept IsConstant = requires {
	[]<T V>() {
	}.template operator()<T>();
};

// === HYBRID WRAPPER === //
template<IsConstant auto value> constexpr auto wrap_constant_or_runtime() {
	std::cout << "[compile-time path]" << std::endl;
	return std::integral_constant<decltype(value), value>{};
}

template<typename T> constexpr auto wrap_constant_or_runtime(T value) {
	std::cout << "[runtime path]" << std::endl;
	return value;
}

// === INTROSPECTIVE TYPE DEMO === //
template<typename T> void print_type_info(const T& value) {
	if constexpr (requires { typename T::value_type; }) {
		std::cout << "Result is std::integral_constant<" << typeid(typename T::value_type).name() << ", " << T::value << ">" << std::endl;
	} else {
		std::cout << "Result is runtime value: " << value << std::endl;
	}
}

// === MAIN === //
int main() {
	constexpr uint64_t static_val = 42;
	uint64_t dynamic_val		  = 1337;

	auto result_static	= wrap_constant_or_runtime(static_val);
	auto result_dynamic = wrap_constant_or_runtime(dynamic_val);

	std::cout << "---- TYPE CHECKS ----" << std::endl;
	print_type_info(result_static);
	print_type_info(result_dynamic);

	return 0;
}
