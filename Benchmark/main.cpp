#include <cstdint>
#include <iostream>
#include <jsonifier/Index.hpp>

struct test_struct {
	std::string val01{};
	std::vector<int32_t> val02{};
	std::vector<std::string> val03{};
};

template<typename index_sequence, auto... values> struct json_entities_impl;

template<typename index_sequence, auto... values> struct json_entities_impl_internal {
	void initialize() const {
		((std::cout << "CURRENT TYPE: " << typeid(values).name() << std::endl), ...);
	}
};

template<size_t... I, auto... values> struct json_entities_impl<std::index_sequence<I...>, values...> {
	using value_type = json_entities_impl_internal<std::index_sequence<I...>,jsonifier_internal::makeJsonEntityAuto<sizeof...(I),I,values>()...>;
};

template<typename value_type> struct core;

template<auto... valuesNew> struct json_entities {
	using value_type = json_entities_impl<std::make_index_sequence<sizeof...(valuesNew)>, valuesNew...>;
};

template<typename value_type_new> struct core {
	using value_type = json_entities<&test_struct::val03, &test_struct::val01, &test_struct::val02>;
};

int main() {
	constexpr core<test_struct>::value_type::value_type ::value_type test{};
	test.initialize();
	return 0;
}
