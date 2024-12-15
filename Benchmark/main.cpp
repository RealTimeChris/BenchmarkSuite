#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include "RandomGenerators.hpp"
#include <thread>

struct test_struct_new_03 {
	std::vector<std::string> firstVal{ 3424 };
};

struct test_struct_new_02 {
	test_struct_new_03 firstVal{ { "3424" } };
	int32_t secondVal{ 3424 };
};

struct test_struct_new_01 {
	test_struct_new_02 firstVal{ { { "3424" } } };
};

struct test_struct_new {
	test_struct_new_01 firstVal{};
	std::vector<test_struct_new_02> testVector{};
};

template<> struct jsonifier::core<test_struct_new_03> {
	static constexpr auto parseValue = createValue<&test_struct_new_03::firstVal>();
};

template<> struct jsonifier::core<test_struct_new_02> {
	static constexpr auto parseValue = createValue<make_json_entity<&test_struct_new_02::firstVal, json_type::object, "TESTING">(), &test_struct_new_02::secondVal>();
};

template<> struct jsonifier::core<test_struct_new_01> {
	static constexpr auto parseValue = createValue<&test_struct_new_01::firstVal>();
};

template<> struct jsonifier::core<test_struct_new> {
	static constexpr auto parseValue = createValue<&test_struct_new::firstVal, &test_struct_new::testVector>();
};

template<typename value_type> constexpr bool areWeInsideRepeated() {
	return (jsonifier::concepts::map_t<value_type> || jsonifier::concepts::vector_t<value_type> || jsonifier::concepts::raw_array_t<value_type>);
}

template<typename value_type, size_t currentIndex = 0> constexpr size_t countTotalDepth(size_t currentDepth) {
	if constexpr (currentIndex < jsonifier_internal::tuple_size_v<typename jsonifier_internal::core_tuple_type<value_type>::core_type>) {
		auto newMember = jsonifier_internal::get<currentIndex>(jsonifier_internal::core_tuple_type<value_type>::coreTupleV);
		using member_type = typename decltype(newMember)::member_type;
		if constexpr (jsonifier::concepts::jsonifier_object_t<member_type>) {
			currentDepth += countTotalDepth<member_type>(0);
		} else if constexpr (jsonifier_internal::has_value_type<member_type>) {
			using member_type_new = typename decltype(newMember)::member_type::value_type;
			currentDepth += countTotalDepth<member_type_new>(currentDepth);
		} else {
			++currentDepth;
		}
		return countTotalDepth<value_type, currentIndex + 1>(currentDepth);
	} else {
		return currentDepth;
	}
}

int32_t main() {
	jsonifier::jsonifier_core parser{};
	std::string newString{};
	parser.serializeJson(test_struct_new{}, newString);
	test_struct_new newData{};
	std::cout << "CURRENT DATA: " << newString << std::endl;
	parser.parseJson<jsonifier::parse_options{ .partialRead = true }>(newData, newString);
	newString.clear();
	parser.serializeJson(newData, newString);
	std::cout << "CURRENT DATA: " << newString << std::endl;
	std::cout << "TOTAL MEMBERS: " << countTotalDepth<test_struct_new>(0) << std::endl;
	bnch_swt::benchmark_stage<"TEST STAGE">::runBenchmark<"TEST", "CYAN">([] {
		return 0ull;
	});
	bnch_swt::benchmark_stage<"TEST STAGE">::printResults();
	return 0;
}