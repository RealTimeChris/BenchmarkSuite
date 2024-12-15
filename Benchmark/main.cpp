#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include "RandomGenerators.hpp"
#include <thread>

template<typename value_type>
concept convertible_to_string_view = std::convertible_to<value_type, jsonifier::string_view>;

template<typename value_type>
concept is_json_entity = requires(std::remove_cvref_t<value_type> value) {
	{ value.type };
	{ value.memberPtr };
};

template<auto testPtr, jsonifier_internal::json_type type, jsonifier_internal::string_literal name> constexpr auto make_json_entity() {
	return jsonifier_internal::makeJsonEntityAuto<type>(name.view(), testPtr);
}

template<auto testPtr, jsonifier_internal::json_type type> constexpr auto make_json_entity() {
	return jsonifier_internal::makeJsonEntityAuto<type>(jsonifier_internal::getName<testPtr>(), testPtr);
}

template<auto testPtr, jsonifier_internal::json_type type, convertible_to_string_view key_type> constexpr auto make_json_entity(key_type key) {
	return jsonifier_internal::makeJsonEntityAuto<type>(key, testPtr);
}

template<auto testPtr> constexpr auto make_json_entity() {
	return jsonifier_internal::makeJsonEntityAuto(jsonifier_internal::getName<testPtr>(), testPtr);
}

template<auto testPtr, jsonifier_internal::string_literal name> constexpr auto make_json_entity() {
	return jsonifier_internal::makeJsonEntityAuto(name.view(), testPtr);
}

template<auto testPtr, convertible_to_string_view key_type> constexpr auto make_json_entity(key_type key) {
	return jsonifier_internal::makeJsonEntityAuto(key, testPtr);
}

struct test_struct_new {
	int32_t value{};
	std::optional<int32_t> stringValue{};
};

template<size_t index = 0, typename tuple_type> constexpr auto testFunctionNew(tuple_type tuple) {
	if constexpr (index < std::tuple_size_v<tuple_type>) {
		if (!std::is_constant_evaluated()) {
			auto newJsonEntity = jsonifier_internal::get<index>(tuple);
			std::cout << "Name: " << newJsonEntity.name << std::endl;
			std::cout << "Type: " << ( int32_t )newJsonEntity.type << std::endl;
		}
		testFunctionNew<index + 1>(tuple);
	}
}

template<auto... testVals, is_json_entity... Ts> constexpr decltype(auto) testFunction(Ts... values) {
	static_assert(((is_json_entity<decltype(testVals)>) && ...), "All values must have a type that satisfies the is_json_entity concept.");
	jsonifier_internal::tuple testTuple{ std::forward<Ts>(values)... };
	jsonifier_internal::tuple testTuple02{ testVals... };
	auto newerTuple = jsonifier_internal::tupleCat(testTuple02, testTuple);
	testFunctionNew(newerTuple);
	return newerTuple;
}

template<typename tuple_type, typename value_type, size_t currentIndex = 0> constexpr auto& getValueAtTypeIndex(value_type& value, tuple_type& tuple) {
	if constexpr (currentIndex < std::tuple_size_v<std::remove_cvref_t<tuple_type>>) {
		using class_type  = typename std::remove_cvref_t<decltype(jsonifier_internal::get<currentIndex>(tuple))>::class_type;
		using member_type = typename std::remove_cvref_t<decltype(jsonifier_internal::get<currentIndex>(tuple))>::member_type;
		if constexpr (std::is_same_v<std::remove_cvref_t<value_type>, member_type class_type::*>) {
			return jsonifier_internal::get<currentIndex>(tuple);
		}
		return getValueAtTypeIndex<tuple_type, value_type, currentIndex + 1>(value, tuple);
	} else {
		auto& newVal = getValueAtTypeIndex<tuple_type, value_type, currentIndex - 1>(value, tuple);
		return newVal;
	}
}

template<auto... values> struct json_entity_collector;

template<typename tuple_type> struct tuple_mutator {
	tuple_type tuple;
	jsonifier::string_view key{};
	template<typename value_type> constexpr auto operator=(const value_type& value) {
		return json_entity_collector{ value, tuple };
	}
	template<convertible_to_string_view key_type, typename tuple_type_new> constexpr tuple_mutator(tuple_type_new& tupleNew, key_type keyNew) : tuple{ tupleNew }, key{ keyNew } {};
};

template<auto... values> struct json_entity_collector {
	static constexpr jsonifier_internal::tuple newValue{ jsonifier_internal::json_entity{ jsonifier_internal::constructMemberFromPtr<values>() }... };

	template<typename value_type, convertible_to_string_view key_type> static constexpr auto updateValues(value_type value, key_type key) {
		tuple_mutator<std::remove_cvref_t<decltype(newValue)>> newValues{ newValue, key };
		getValueAtTypeIndex(value, newValues.tuple).name = key;
		return newValues;
	}
};

template<class T> struct remove_member_pointer {
	using type = T;
};

template<class C, class T> struct remove_member_pointer<T C::*> {
	using type = T;
};


constexpr auto newFunction(auto newArg) {
	return newArg;
}

template<typename tuple_type> constexpr auto createValueNew(tuple_type&& args) noexcept {
	if constexpr (jsonifier_internal::tuple_size_v<tuple_type> == 1) {
		return jsonifier::scalar_value{ args };
	} else {
		return jsonifier::value{ args };
	}
}

template<> struct jsonifier::core<test_struct_new> {
	static constexpr auto jsonEntity = []() {
		constexpr json_entity_collector<&test_struct_new::stringValue> jsonEntity{};
		constexpr auto newValue = jsonEntity.updateValues(&test_struct_new::stringValue, "TEST22");
		return newValue;
	}();
	static constexpr auto parseValue = createValueNew(jsonEntity.tuple);
};

int main() {
	static constexpr json_entity_collector<&test_struct_new::stringValue> jsonEntity{};
	std::cout << "CURRENT TYPE: " << typeid(decltype(&test_struct_new::stringValue)).name() << std::endl;
	std::cout << "CURRENT TYPE: " << typeid(typename decltype(jsonifier_internal::member_ptr_helper{ &test_struct_new::stringValue })::member_type).name() << std::endl;
	std::cout << "CURRENT TYPE: " << typeid(typename decltype(jsonifier_internal::member_ptr_helper{ &test_struct_new::stringValue })::class_type).name() << std::endl;
	//jsonEntity["TEST22"]			 = &test_struct_new::stringValue;
	static constexpr auto newVal = newFunction(jsonEntity);
	std::cout << "CURRENT NAME: " << jsonifier_internal::get<0>(jsonifier::core<test_struct_new>::parseValue.val).name << std::endl;
	std::cout << "CURRENT SIZE: " << typeid(newVal).name() << std::endl;
	std::cout << "CURRENT TYPE: " << ( int32_t )jsonifier_internal::get<0>(jsonEntity.newValue).type << std::endl;
}
