#include <array>
#include <iostream>
#include <optional>
#include <cstddef>
#include <jsonifier/Index.hpp>

template<typename value_type> struct trie_node_base {
	std::array<trie_node_base<value_type>*, 128> children{};
	const trie_node_base<value_type>* finalNode{};
	bool isActive{ false };
	size_t index{};

	virtual bool processIndex(value_type& value, const char*& iter, const char*& end) const = 0;
};

template<> struct trie_node_base<void> {
	std::array<trie_node_base<void>*, 128> children{};
	const trie_node_base<void>* finalNode{};
	bool isActive{ false };
	size_t index{};

	virtual bool processIndex(std::nullptr_t& value, const char*& iter, const char*& end) const = 0;
};

template<typename member_type, typename class_type> struct member_pointer {
	member_type class_type::*ptr{};
	JSONIFIER_ALWAYS_INLINE constexpr member_pointer(member_type class_type::*p) noexcept : ptr(p){};
	constexpr member_pointer() noexcept = default;
};

template<size_t currentIndex, typename value_type> struct trie_index_node : public trie_node_base<value_type> {
	static constexpr auto index{ currentIndex };
	jsonifier::string_view key{};

	bool processIndex(value_type& value, const char*& iter, const char*& end) const {
		std::cout << "CURRENT INDEX: " << currentIndex << std::endl;
		return true;
	}

	JSONIFIER_ALWAYS_INLINE constexpr trie_index_node() noexcept = default;

	JSONIFIER_ALWAYS_INLINE constexpr trie_index_node(const jsonifier::string_view& keyNew) noexcept : key{ keyNew } {};
};

template<typename value_type, size_t... indices> JSONIFIER_ALWAYS_INLINE constexpr auto createNewTupleImplTrie(std::index_sequence<indices...>) noexcept {
	return std::make_tuple(trie_index_node<indices, value_type>{ std::get<indices>(jsonifier_internal::coreTupleV<value_type>).view() }...);
}

template<typename value_type> JSONIFIER_ALWAYS_INLINE constexpr auto createNewTupleTrie() noexcept {
	constexpr auto& tupleRefs = jsonifier_internal::sortedTupleReferencesByLength<value_type>;
	return createNewTupleImplTrie<value_type>(std::make_index_sequence<tupleRefs.size()>{});
}

template<typename value_type> struct trie_index_node<0, value_type> : public trie_node_base<value_type> {
	jsonifier::string_view key{};
	static constexpr auto index{ 0 };

	bool processIndex(value_type& value, const char*& iter, const char*& end) const {
		return false;
	}

	JSONIFIER_ALWAYS_INLINE constexpr trie_index_node() noexcept = default;

	JSONIFIER_ALWAYS_INLINE constexpr trie_index_node(const jsonifier::string_view& keyNew) noexcept : key{ keyNew } {};
};

template<typename value_type> struct trie {
	std::array<trie_index_node<0, value_type>, 128> nodePlaceHolders{};
	std::array<trie_node_base<value_type>*, 128> nodes{};
	trie_node_base<value_type>* root{};

	template<typename tuple_type> JSONIFIER_ALWAYS_INLINE constexpr trie(tuple_type& strings, size_t uniqueIndex = 0) : nodes{}, root(&nodePlaceHolders[0]) {
		for (uint64_t x = 0; x < 128; ++x) {
			nodes[x] = &nodePlaceHolders[x];
		}
		for (auto& node: nodes) {
			for (auto& child: node->children) {
				child = nullptr;
			}
		}

		insert<0, std::tuple_size_v<tuple_type>>(strings, uniqueIndex, root);
	}

	template<size_t currentIndex, size_t N, typename tuple_type>
	JSONIFIER_ALWAYS_INLINE constexpr void insert(tuple_type& strings, size_t depth, trie_node_base<value_type>* node) {
		if constexpr (currentIndex < N) {
			insertImpl(std::get<currentIndex>(strings), currentIndex, 0, root);
			return insert<currentIndex + 1, N>(strings, depth, root);
		}
	}

	template<typename tuple_elem_type>
	JSONIFIER_ALWAYS_INLINE constexpr void insertImpl(tuple_elem_type& str, size_t index, size_t depth, trie_node_base<value_type>* node) {
		if (depth >= str.key.size()) {
			node->index		  = index;
			node->finalNode = &str;
			return;
		}
		size_t charIndex = static_cast<unsigned char>(str.key[depth]);
		if (node->children[charIndex] == nullptr) {
			size_t nextNodeIndex				= findAvailableNode();
			node->children[charIndex]			= nodes[nextNodeIndex];
			node->children[charIndex]->isActive = true;
		}
		insertImpl(str, index, depth + 1, node->children[charIndex]);
	}

	JSONIFIER_ALWAYS_INLINE constexpr size_t findAvailableNode() const {
		for (size_t i = 0; i < nodes.size(); ++i) {
			bool allChildrenNull = true;
			for (const auto& child: nodes[i]->children) {
				if (child != nullptr) {
					allChildrenNull = false;
					break;
				}
			}
			if (allChildrenNull && !nodes[i]->finalNode) {
				return i;
			}
		}
		return nodes.size();
	}

	JSONIFIER_ALWAYS_INLINE constexpr std::optional<size_t> search(jsonifier::string_view str) const {
		return search(str, 0, root);
	}

	JSONIFIER_ALWAYS_INLINE constexpr std::optional<size_t> search(jsonifier::string_view str, size_t depth, trie_node_base<value_type>* node) const {
		if (depth >= str.size()) {
			if (node->isEndOfWord) {
				return node->index;
			} else {
				return std::nullopt;
			}
		}
		size_t charIndex = static_cast<unsigned char>(str[depth]);
		if (node->children[charIndex] == nullptr) {
			return std::nullopt;
		}
		return search(str, depth + 1, node->children[charIndex]);
	}
};

template<typename value_type> class partial_search {
  public:
	JSONIFIER_ALWAYS_INLINE constexpr partial_search(const trie<value_type>& trie) noexcept : trieVal(trie), currentNode(trie.root) {
	}

	JSONIFIER_ALWAYS_INLINE constexpr auto searchNext(char c) const {
		size_t charIndex = static_cast<unsigned char>(c);

		if (!currentNode) {
			return static_cast<trie_node_base<value_type>*>(nullptr);
		}

		auto nextNode = currentNode->children[charIndex];

		if (!nextNode->isActive) {
			return static_cast<trie_node_base<value_type>*>(nullptr);
		}

		currentNode = nextNode;

		if (currentNode->finalNode || currentNode->isActive) {
			return currentNode;
		}

		return currentNode;
	}

	JSONIFIER_ALWAYS_INLINE constexpr void reset() const noexcept {
		currentNode = trieVal.root;
	}

  private:
	mutable trie_node_base<value_type>* currentNode{};
	mutable trie<value_type> trieVal{};
};

template<typename value_type, size_t... indices> JSONIFIER_ALWAYS_INLINE constexpr auto getArrayOfStringsImpl(std::index_sequence<indices...>) {
	return std::array<jsonifier::string_view, std::tuple_size_v<jsonifier_internal::core_tuple_t<value_type>>>{
		std::get<indices>(jsonifier_internal::coreTupleV<value_type>).view()...
	};
}

template<typename value_type> JSONIFIER_ALWAYS_INLINE constexpr auto getArrayOfStrings() {
	return getArrayOfStringsImpl<value_type>(std::make_index_sequence<std::tuple_size_v<jsonifier_internal::core_tuple_t<value_type>>>{});
}

struct test_struct {
	int32_t testInt01{};
	int32_t testInt02{};
	int32_t testInt03{};
};

template<> struct jsonifier::core<test_struct> {
	using value_type = test_struct;
	static constexpr auto parseValue = createValue<&value_type::testInt01, &value_type::testInt02, &value_type::testInt03>();
};

int main() {
	static constexpr auto strings  = createNewTupleTrie<test_struct>();
	static constexpr auto newTuple = jsonifier_internal::generateInterleavedTuple<0, 4>(std::make_tuple(), "testInt01", &test_struct::testInt01, "testInt02",
		&test_struct::testInt02, "testInt03", &test_struct::testInt03);
	const std::string searchTerm01 = "testInt01";
	const std::string searchTerm02 = "testInt02";
	const std::string searchTerm03 = "testInt03";
	std::vector<std::string> searchTerms{ searchTerm01, searchTerm02, searchTerm03 };
	trie<test_struct> trieNew(strings, 0);// Start from the beginning
	partial_search<test_struct> partialSearch{ trieNew };
	test_struct testData{};
	const char* ptr01{};
	for (auto& value: searchTerms) {
		for (uint64_t x = 0; x < value.size(); ++x) {// Start from the beginning of the string
			auto result = partialSearch.searchNext(value[x]);
			if (result && result->finalNode && x == value.size() - 1) {
				std::cout << "Match found with index: " << result->index << std::endl;
			} else {
							std::cout << "No match found for character '" << value[x] << "'\n";
			}
		}
		partialSearch.reset();
	}
	
	return 0;
}