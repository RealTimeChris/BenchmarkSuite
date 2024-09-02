#include <array>
#include <iostream>
#include <optional>
#include <cstddef>
#include <jsonifier/Index.hpp>
#include "Tests/ConformanceTests.hpp"

struct test_struct {
	int32_t testingTw1{};
	int32_t testingTwo{};
	int32_t testingThr{};
};

template<> struct jsonifier::core<test_struct> {
	using value_type = test_struct;
	static constexpr decltype(auto) parseValue = createValue<&value_type::testingTw1, &value_type::testingTwo, &value_type::testingThr>();
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

JSONIFIER_ALWAYS_INLINE static constexpr size_t findLastUniqueColumnIndex(const jsonifier_internal::tuple_references& tupleRefs, size_t maxIndex, size_t startingIndex = 0) noexcept {
	constexpr size_t alphabetSize = 256;
	size_t lastUniqueIndex{ std::numeric_limits<size_t>::max() };
	jsonifier::string_view key{};
	for (size_t index = maxIndex; index > 0; --index) {
		std::array<bool, alphabetSize> seen{};
		bool allDifferent = true;

		for (size_t x = 0; x < tupleRefs.count; ++x) {
			key				 = tupleRefs.rootPtr[x].key;
			if (index >= key.size()) {
				break;
			}
			const char c	 = key[index];
			size_t charIndex = static_cast<const unsigned char>(c);

			if (seen[charIndex]) {
				allDifferent = false;
				break;
			}
			seen[charIndex] = true;
		}

		if (allDifferent) {
			lastUniqueIndex = index;
		}
	}

	return lastUniqueIndex;
}

template<typename tuple_type>
constexpr decltype(auto) generateKeyArray(const tuple_type& tuple) {
	std::array<jsonifier::string_view, std::tuple_size_v<tuple_type>> keys{};
	return generateKeyArrayImpl(tuple, keys);
}

template<typename value_type> constexpr auto keyArray{ generateKeyArray(jsonifier_internal::coreTupleV<value_type>) };

template<typename value_type> constexpr auto keyStatsVal{ jsonifier_internal::keyStatsImpl(jsonifier_internal::tupleReferencesByLength<value_type>) };

template<typename value_type>
constexpr auto lastUniqueIndex{ findLastUniqueColumnIndex(jsonifier_internal::tupleReferencesByLength<value_type>, keyStatsVal<value_type>.minLength) };

template<typename value_type> struct TrieNodeBase {
	TrieNodeBase* children[128]{};
	int isLeaf{};
	char data{};
	virtual bool processIndex(value_type& value, const char*& iter, const char*& end) = 0;
};

template<typename value_type, size_t count, size_t currentIndex>
TrieNodeBase<value_type>* make_trienode(char data, const std::array<jsonifier::string_view, count>& keys, size_t currentDepth = 0, TrieNodeBase<value_type>* node = nullptr);

template<typename value_type, size_t currentIndex = 0> struct TrieNode : public TrieNodeBase<value_type> {

	bool processIndex(value_type& value, const char*& iter, const char*& end) {
		std::cout << "CURRENT INDEX: " << currentIndex << std::endl;
		return true;
	};

	TrieNode() noexcept = default;

	template<size_t count> TrieNode(char data, const std::array<jsonifier::string_view, count>& keys, size_t currentDepth = 0, TrieNodeBase<value_type>* node = nullptr) {
		if (node == nullptr) {
			node = new TrieNode<value_type, currentIndex>{};
		}
		if constexpr (currentIndex < count) {
			auto word					   = keys[currentIndex];
			TrieNodeBase<value_type>* temp = node;
			for (int i = currentDepth; i < word.size(); i++) {
				uint8_t idx = static_cast<uint8_t>(word[i]);
				if (temp->children[idx] == NULL) {
					temp->children[idx] = make_trienode<value_type, count, currentIndex>(data, keys, currentDepth + 1);
				}
				temp = temp->children[idx];
			}
			temp->isLeaf = 1;
			*this		 = *static_cast<TrieNode*>(make_trienode<value_type, count, currentIndex + 1>(data, keys, currentDepth, node));
		} else {
			return;
		}
	}

	TrieNodeBase<value_type>* search_trie(const char* word) {
		TrieNodeBase<value_type>* temp = this;

		for (int i = 0; word[i] != '\0'; i++) {
			uint8_t position = static_cast<uint8_t>(word[i]);
			if (temp->children[position] == NULL) {
				return static_cast<TrieNodeBase<value_type>*>(nullptr);
			}
			temp = temp->children[position];
		}
		if (temp != NULL && temp->isLeaf == 1) {
			return temp;
		}
		return static_cast<TrieNodeBase<value_type>*>(nullptr);
	}
};

template<typename value_type, size_t count, size_t currentIndex>
TrieNodeBase<value_type>* make_trienode(char data, const std::array<jsonifier::string_view, count>& keys, size_t currentDepth, TrieNodeBase<value_type>* node ) {
	if (node == nullptr) {
		node = new TrieNode<value_type, currentIndex>{};
	}	
	if constexpr (currentIndex < count) {
		auto word = keys[currentIndex];
		TrieNodeBase<value_type>* temp = node;
		for (int i = currentDepth; i < word.size(); i++) {
			uint8_t idx = static_cast<uint8_t>(word[i]);
			if (temp->children[idx] == NULL) {
				temp->children[idx] = make_trienode<value_type, count, currentIndex>(data, keys, currentDepth + 1);
			}
			temp = temp->children[idx];
		}
		temp->isLeaf = 1;
		return make_trienode<value_type, count, currentIndex + 1>(data, keys, currentDepth, node);
	} else {
		return node;
	}
}

template<typename value_type> void free_trienode(TrieNodeBase<value_type>* node) {
	for (int i = 0; i < 128; i++) {
		if (node->children[i] != NULL) {
			free_trienode<value_type>(node->children[i]);
		} else {
			continue;
		}
	}
	free(node);
}

int main() {
	static constexpr jsonifier_internal::string_literal<7> literal = "TETING";
	TrieNode<test_struct> trie{ '\0', keyArray<test_struct>, lastUniqueIndex<test_struct> };
	test_struct newVal{};
	const char* valueNew{};
	const char* endNew{};
	jsonifier::string_view string01{ "testingTw1" };
	jsonifier::string_view string02{ "testingTwo" };
	jsonifier::string_view string03{ "testingThr" };
	std::cout << "DID WE FIND IT?: " << lastUniqueIndex<test_struct> << std::endl;
	std::cout << "DID WE FIND IT?: " << keyStatsVal<test_struct>.minLength << std::endl;
	std::cout << "DID WE FIND IT?: " << trie.search_trie(string01.data() + lastUniqueIndex<test_struct>)->processIndex(newVal, valueNew, endNew) << std::endl;
	std::cout << "DID WE FIND IT?: " << trie.search_trie(string02.data() + lastUniqueIndex<test_struct>)->processIndex(newVal, valueNew, endNew) << std::endl;
	std::cout << "DID WE FIND IT?: " << trie.search_trie(string03.data() + lastUniqueIndex<test_struct>)->processIndex(newVal, valueNew, endNew) << std::endl;
	std::cout << "DID WE FIND IT?: " << trie.search_trie("testing05") << std::endl;
	return 0;
}