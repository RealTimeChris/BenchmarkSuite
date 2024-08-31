#include <array>
#include <iostream>
#include <optional>
#include <cstddef>
#include <jsonifier/Index.hpp>

constexpr size_t ALPHABET_SIZE = 128;
constexpr size_t MAX_DEPTH	   = 256;

template<typename value_type, typename iterator> struct trie_node_base {
	std::array<trie_node_base<value_type, iterator>*, 128> children{};
	bool is_end_of_word{ false };
	bool isActive{ false };
	size_t index{};

	virtual bool processIndex(value_type& value, iterator& iter, iterator& end);

	constexpr trie_node_base() noexcept : children{} {
	}
};

template<typename member_type, typename class_type> struct member_pointer {
	member_type class_type::*ptr{};
	JSONIFIER_ALWAYS_INLINE constexpr member_pointer(member_type class_type::*p) noexcept : ptr(p){};
};

template<typename member_type_new, typename value_type, typename iterator> struct data_member : public trie_node_base<value_type, iterator> {
	using member_type = member_type_new;
	using class_type  = value_type;
	member_pointer<member_type, class_type> memberPtr{};
	uint8_t padding[4]{};
	jsonifier::string_view name{};

	JSONIFIER_ALWAYS_INLINE constexpr auto& view() const noexcept {
		return name;
	}

	JSONIFIER_ALWAYS_INLINE constexpr auto& ptr() const noexcept {
		return memberPtr.ptr;
	}

	JSONIFIER_ALWAYS_INLINE constexpr data_member(jsonifier::string_view str, member_type class_type::*ptr) noexcept : memberPtr(ptr), name(str){};
};

template<typename value_type, typename iterator> struct trie {
	std::array<trie_node_base<value_type, iterator>*, 128> nodes;
	trie_node_base<value_type, iterator>*root;

	template<size_t N> constexpr trie(const std::array<jsonifier::string_view, N>& strings, size_t uniqueIndex = 0) : nodes{}, root(nodes[0]) {
		for (auto& node: nodes) {
			for (auto& child: node->children) {
				child = nullptr;
			}
		}

		for (size_t i = 0; i < strings.size(); ++i) {
			insert(strings[i], i, uniqueIndex, root);
		}
	}

	constexpr void insert(jsonifier::string_view str, size_t index, size_t depth, trie_node_base<value_type, iterator>* node) {
		if (depth >= str.size()) {
			node->index			 = index;
			node->is_end_of_word = true;
			return;
		}
		size_t charIndex = static_cast<unsigned char>(str[depth]);
		if (node->children[charIndex] == nullptr) {
			size_t nextNodeIndex				= findAvailableNode();
			node->children[charIndex]			= nodes[nextNodeIndex];
			node->children[charIndex]->isActive = true;
		}
		insert(str, index, depth + 1, node->children[charIndex]);
	}

	constexpr size_t findAvailableNode() const {
		for (size_t i = 0; i < nodes.size(); ++i) {
			bool allChildrenNull = true;
			for (const auto& child: nodes[i]->children) {
				if (child != nullptr) {
					allChildrenNull = false;
					break;
				}
			}
			if (allChildrenNull && !nodes[i]->is_end_of_word) {
				return i;
			}
		}
		return nodes.size();
	}

	constexpr std::optional<size_t> search(jsonifier::string_view str) const {
		return search(str, 0, root);
	}

	constexpr std::optional<size_t> search(jsonifier::string_view str, size_t depth, trie_node_base<value_type, iterator>* node) const {
		if (depth >= str.size()) {
			if (node->is_end_of_word) {
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

template<typename value_type, typename iterator> class partial_search {
  public:
	constexpr partial_search(const trie<value_type, iterator>& trie) noexcept : trie_(trie), currentNode_(trie.root) {
	}

	constexpr auto searchNext(char c) {
		size_t charIndex = c;
		if (currentNode_ == nullptr) {
			return static_cast<trie_node_base<value_type, iterator>*>(nullptr);
		}
		if (currentNode_->children[charIndex] == nullptr) {
			currentNode_ = nullptr;
			return static_cast<trie_node_base<value_type, iterator>*>(nullptr);
		}
		currentNode_ = currentNode_->children[charIndex];
		if (currentNode_->is_end_of_word || currentNode_->isActive) {
			return currentNode_;
		}
		return static_cast<trie_node_base<value_type, iterator>*>(nullptr);
	}

	constexpr void reset() noexcept {
		currentNode_ = trie_.root;
	}

  private:
	const trie<value_type, iterator>& trie_;
	trie_node_base<value_type, iterator>* currentNode_;
};

template<typename... value_types> constexpr auto generateStringArrays(value_types&&... values) {
	std::array<jsonifier::string_view, sizeof...(value_types)> returnValues{ values... };
	return returnValues;
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
	constexpr auto strings = generateStringArrays("cabbling", "cabular", "cabling", "cabled", "cabl");
	static constexpr auto newTuple = jsonifier::core<test_struct>::parseValue.val;
	trie<test_struct, char*> trie(strings, 2);
	partial_search partialSearch{ trie };
	const std::string searchTerm = "cabl";
	for (uint64_t x = 3; x < searchTerm.size();++x) {
		auto result = partialSearch.searchNext(searchTerm[x]);
		if (result && result->is_end_of_word && x == searchTerm.size()-1) {
			std::cout << "Match found with index: " << result->index << std::endl;
		} else {
			std::cout << "No match found for character '" << searchTerm[x ] << "'\n";
		}
	}
	return 0;
}