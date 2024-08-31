#include <array>
#include <iostream>
#include <optional>
#include <cstddef>
#include <jsonifier/Index.hpp>
#include "Tests/ConformanceTests.hpp"

struct test_struct {
	int32_t testInt01{};
	int32_t testInt02{};
	int32_t testInt03{};
};

template<> struct jsonifier::core<test_struct> {
	using value_type = test_struct;
	static constexpr decltype(auto) parseValue = createValue<&value_type::testInt01, &value_type::testInt02, &value_type::testInt03>();
};

template<typename tuple_type, size_t currentIndex = 0>
constexpr decltype(auto) generateKeyArrayImpl(const tuple_type& tuple, std::array<jsonifier::string_view, std::tuple_size_v<tuple_type>> keys) {
	if constexpr (currentIndex < std::tuple_size_v<tuple_type>) {
		keys[currentIndex] = std::get<currentIndex>(tuple).view();
		return generateKeyArrayImpl<tuple_type, currentIndex + 1>(tuple, keys);
	} else {
		return keys;
	}
}

template<typename tuple_type>
constexpr decltype(auto) generateKeyArray(const tuple_type& tuple) {
	std::array<jsonifier::string_view, std::tuple_size_v<tuple_type>> keys{};
	return generateKeyArrayImpl(tuple, keys);
}

template<typename value_type> constexpr auto keyArray{ generateKeyArray(jsonifier_internal::coreTupleV<value_type>) };

template<typename value_type> static constexpr decltype(auto) keyStatsVal = jsonifier_internal::keyStatsImpl(jsonifier_internal::tupleReferencesByFirstByte<value_type>);

struct trie_node_base {
	const trie_node_base* next{};
	char data{};
};

template<jsonifier_internal::string_literal string, size_t currentDepth, size_t currentIndex> struct trie_node;

template<jsonifier_internal::string_literal string> constexpr size_t stringLiteralSize = string.size();

template<jsonifier_internal::string_literal string, size_t currentIndex> struct trie_node<string, 128, currentIndex> : public trie_node_base {};

template<jsonifier_internal::string_literal string, size_t currentIndex> struct trie_node<string, 0, currentIndex> : public trie_node_base {
	static constexpr trie_node<string, 1, currentIndex> nextVal{};
	static constexpr auto data{ string[0] };
	constexpr trie_node() noexcept : trie_node_base{ &nextVal, string[0] } {};
};

template<jsonifier_internal::string_literal string, size_t currentDepth, size_t currentIndex> struct trie_node : public trie_node_base {
	static constexpr trie_node<string, currentDepth + 1, currentIndex> nextVal{};
	static constexpr auto index{ currentIndex };
	static constexpr auto data{ string[(currentDepth - 1) >= string.size() ? string.size() : (currentDepth - 1)] };
	constexpr trie_node() noexcept
		: trie_node_base{ (currentDepth - 1) >= string.size() ? nullptr : &nextVal, string[(currentDepth - 1) >= string.size() ? string.size() : (currentDepth - 1)] } {};
};

template<typename value_type, size_t...indices> constexpr auto generateTrieImpl(std::index_sequence<indices...>) {
	return std::make_tuple(trie_node<jsonifier_internal::stringLiteralFromView<keyArray<value_type>[indices].size()>(keyArray<value_type>[indices]), 0, indices>{}...);
}

template<typename value_type> constexpr auto generateTrie() {
	return generateTrieImpl<value_type>(std::make_index_sequence<keyArray<value_type>.size()>{});
}

template<typename tuple_type, size_t currentIndex = 0> void introspectTuple(const tuple_type& tuple) {
	if constexpr (currentIndex < std::tuple_size_v<tuple_type>) {
		const auto* current = std::get<currentIndex>(tuple).next;
		while (current) {
			std::cout << "Data at index 0: " << current->data << std::endl;
			current = current->next;
		}
		return introspectTuple<tuple_type, currentIndex + 1>(tuple);
	}
}

int main() {
	static constexpr jsonifier_internal::string_literal<7> literal = "TETING";
	introspectTuple(generateTrie<test_struct>());
	trie_node<literal, 0, 0> testVal{};
	const auto* current = testVal.next;
	while (current) {
		std::cout << "Data at index 0: " << current->data << std::endl;
		current = current->next;
	}
	return 0;
}