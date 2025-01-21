/*
	MIT License

	Copyright (c) 2024 RealTimeChris

	Permission is hereby granted, free of charge, to any person obtaining a copy of this
	software and associated documentation files (the "Software"), to deal in the Software
	without restriction, including without limitation the rights to use, copy, modify, merge,
	publish, distribute, sublicense, and/or sell copies of the Software, and to permit
	persons to whom the Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all copies or
	substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	DEALINGS IN THE SOFTWARE.
*/
/// https://github.com/RealTimeChris/BenchmarkSuite
#pragma once

#include <concepts>
#include <cstdint>
#include <variant>

namespace bnch_swt {

	namespace internal {

		template<typename value_type>
		concept has_range = requires(std::remove_cvref_t<value_type> value) {
			{ value.begin() };
			{ value.end() };
		};

		template<typename value_type>
		concept map_subscriptable = requires(std::remove_cvref_t<value_type> value) {
			{ value[typename std::remove_cvref_t<value_type>::key_type{}] } -> std::same_as<const typename std::remove_cvref_t<value_type>::mapped_type&>;
		} || requires(std::remove_cvref_t<value_type> value) {
			{ value[typename std::remove_cvref_t<value_type>::key_type{}] } -> std::same_as<typename std::remove_cvref_t<value_type>::mapped_type&>;
		};

		template<typename value_type>
		concept vector_subscriptable = requires(std::remove_cvref_t<value_type> value) {
			{ value[typename std::remove_cvref_t<value_type>::size_type{}] } -> std::same_as<typename std::remove_cvref_t<value_type>::const_reference>;
		} || requires(std::remove_cvref_t<value_type> value) {
			{ value[typename std::remove_cvref_t<value_type>::size_type{}] } -> std::same_as<typename std::remove_cvref_t<value_type>::reference>;
		};

		template<typename value_type>
		concept has_size = requires(std::remove_cvref_t<value_type> value) {
			{ value.size() } -> std::same_as<typename std::remove_cvref_t<value_type>::size_type>;
		};

		template<typename value_type>
		concept has_empty = requires(std::remove_cvref_t<value_type> value) {
			{ value.empty() } -> std::same_as<bool>;
		};

		template<typename value_type>
		concept variant_t = requires(std::remove_cvref_t<value_type> var) {
			{ var.index() } -> std::same_as<size_t>;
			{ var.valueless_by_exception() } -> std::same_as<bool>;
			{ std::holds_alternative<decltype(std::get<0>(var))>(var) } -> std::same_as<bool>;
			{ std::get<0>(var) } -> std::same_as<decltype(std::get<0>(var))&>;
			{ std::get_if<0>(&var) } -> std::same_as<std::remove_cvref_t<decltype(std::get<0>(var))>*>;
		};

		template<typename value_type>
		concept has_resize = requires(std::remove_cvref_t<value_type> value) { value.resize(typename std::remove_cvref_t<value_type>::size_type{}); };

		template<typename value_type>
		concept has_reserve = requires(std::remove_cvref_t<value_type> value) { value.reserve(typename std::remove_cvref_t<value_type>::size_type{}); };

		template<typename value_type>
		concept has_data = requires(std::remove_cvref_t<value_type> value) {
			{ value.data() } -> std::same_as<typename std::remove_cvref_t<value_type>::const_pointer>;
		} || requires(std::remove_cvref_t<value_type> value) {
			{ value.data() } -> std::same_as<typename std::remove_cvref_t<value_type>::pointer>;
		};

		template<typename value_type>
		concept stateless = std::is_empty_v<std::remove_cvref_t<value_type>>;

		template<typename value_type>
		concept bool_t = std::same_as<std::remove_cvref_t<value_type>, bool> || std::same_as<std::remove_cvref_t<value_type>, std::vector<bool>::reference> ||
			std::same_as<std::remove_cvref_t<value_type>, std::vector<bool>::const_reference>;

		template<typename value_type>
		concept always_null_t = std::same_as<std::remove_cvref_t<value_type>, std::nullptr_t> || std::same_as<std::remove_cvref_t<value_type>, std::monostate> ||
			std::same_as<std::remove_cvref_t<value_type>, std::nullopt_t>;

		template<typename value_type>
		concept pointer_t = (std::is_pointer_v<std::remove_cvref_t<value_type>> ||
								( std::is_null_pointer_v<std::remove_cvref_t<value_type>> && !std::is_array_v<std::remove_cvref_t<value_type>> )) &&
			!always_null_t<value_type>;

		template<typename value_type>
		concept floating_point_t = std::floating_point<std::remove_cvref_t<value_type>> && (std::numeric_limits<std::remove_cvref_t<value_type>>::radix == 2) &&
			std::numeric_limits<std::remove_cvref_t<value_type>>::is_iec559;

		template<typename value_type>
		concept char_t = std::same_as<std::remove_cvref_t<value_type>, char>;

		template<typename value_type>
		concept has_substr = requires(std::remove_cvref_t<value_type> value) {
			{
				value.substr(typename std::remove_cvref_t<value_type>::size_type{}, typename std::remove_cvref_t<value_type>::size_type{})
			} -> std::same_as<std::remove_cvref_t<value_type>>;
		};

		template<typename value_type>
		concept has_find = requires(std::remove_cvref_t<value_type> value) {
			{ value.find(typename std::remove_cvref_t<value_type>::value_type{}) } -> std::same_as<typename std::remove_cvref_t<value_type>::size_type>;
		} || requires(std::remove_cvref_t<value_type> value) {
			{ value.find(typename std::remove_cvref_t<value_type>::key_type{}) } -> std::same_as<typename std::remove_cvref_t<value_type>::iterator>;
		} || requires(std::remove_cvref_t<value_type> value) {
			{ value.find(typename std::remove_cvref_t<value_type>::key_type{}) } -> std::same_as<typename std::remove_cvref_t<value_type>::const_iterator>;
		};

		template<typename value_type>
		concept string_t = has_substr<value_type> && has_data<value_type> && has_size<value_type> && vector_subscriptable<value_type> && has_find<value_type>;

		template<typename value_type>
		concept string_view_t =
			has_substr<value_type> && has_data<value_type> && has_size<value_type> && vector_subscriptable<value_type> && has_find<value_type> && !has_resize<value_type>;

		template<typename value_type>
		concept map_t = map_subscriptable<value_type> && has_range<value_type> && has_size<value_type> && has_find<value_type> && has_empty<value_type>;

		template<typename value_type>
		concept pair_t = requires(std::remove_cvref_t<value_type> value) {
			typename std::remove_cvref_t<value_type>::first_type;
			typename std::remove_cvref_t<value_type>::second_type;
		};

		template<typename value_type>
		concept has_fill = requires(std::remove_cvref_t<value_type> value) {
			{ value.fill(typename std::remove_cvref_t<value_type>::value_type{}) } -> std::same_as<void>;
		};

		template<typename value_type>
		concept has_emplace_back = requires(std::remove_cvref_t<value_type> value) {
			{ value.emplace_back(typename std::remove_cvref_t<value_type>::value_type{}) } -> std::same_as<typename std::remove_cvref_t<value_type>::reference>;
		};

		template<typename value_type>
		concept has_release = requires(std::remove_cvref_t<value_type> value) {
			{ value.release() } -> std::same_as<typename std::remove_cvref_t<value_type>::pointer>;
		};

		template<typename value_type>
		concept has_reset = requires(std::remove_cvref_t<value_type> value) {
			{ value.reset() } -> std::same_as<void>;
		};

		template<typename value_type>
		concept has_get = requires(std::remove_cvref_t<value_type> value) {
			{ value.get() } -> std::same_as<typename std::remove_cvref_t<value_type>::element_type*>;
		};

		template<typename value_type>
		concept copyable = std::copyable<std::remove_cvref_t<value_type>>;

		template<typename value_type>
		concept unique_ptr_t = requires(std::remove_cvref_t<value_type> value) {
			typename std::remove_cvref_t<value_type>::element_type;
			typename std::remove_cvref_t<value_type>::deleter_type;
		} && has_release<value_type> && has_get<value_type>;

		template<typename value_type>
		concept shared_ptr_t = has_reset<value_type> && has_get<value_type> && copyable<value_type>;

		template<typename value_type>
		concept nullable_t = !string_t<value_type> && requires(std::remove_cvref_t<value_type> value) {
			bool(value);
			{ *value };
		};

		template<typename value_type>
		concept null_t = nullable_t<value_type> || always_null_t<value_type>;

		template<typename value_type> constexpr bool hasSizeEqualToZero{ std::tuple_size_v<std::remove_cvref_t<value_type>> == 0 };

		template<typename value_type>
		concept has_get_template = requires(std::remove_cvref_t<value_type> value) {
			{ std::get<0>(value) } -> std::same_as<decltype(std::get<0>(value))&>;
		};

		template<typename value_type>
		concept tuple_t = requires(std::remove_cvref_t<value_type> t) { std::tuple_size<std::remove_cvref_t<value_type>>::value; } &&
			(hasSizeEqualToZero<value_type> || has_get_template<value_type>) && !has_data<value_type>;

		template<typename value_type>
		concept optional_t = requires(std::remove_cvref_t<value_type> opt) {
			{ opt.has_value() } -> std::same_as<bool>;
			{ opt.value() } -> std::same_as<typename std::remove_cvref_t<value_type>::value_type&>;
			{ *opt } -> std::same_as<typename std::remove_cvref_t<value_type>::value_type&>;
			{ opt.reset() } -> std::same_as<void>;
			{ opt.emplace(typename std::remove_cvref_t<value_type>::value_type{}) } -> std::same_as<typename std::remove_cvref_t<value_type>::value_type&>;
		};

		template<typename value_type>
		concept enum_t = std::is_enum_v<std::remove_cvref_t<value_type>>;

		template<typename value_type>
		concept vector_t = vector_subscriptable<value_type> && has_fill<value_type>;

		template<typename value_type>
		concept raw_array_t = ( std::is_array_v<std::remove_cvref_t<value_type>> && !std::is_pointer_v<std::remove_cvref_t<value_type>> ) ||
			(vector_subscriptable<value_type> && !vector_t<value_type> && !has_substr<value_type> && !tuple_t<value_type>);

		template<typename value_type>
		concept integer_t = std::integral<std::remove_cvref_t<value_type>> && !bool_t<value_type> && !std::floating_point<std::remove_cvref_t<value_type>>;

		template<typename value_type>
		concept printable = requires(std::remove_cvref_t<value_type> value) { std::cout << value << std::endl; };

		template<typename value_type>
		concept not_printable = !printable<value_type>;

		template<const auto& function, uint64_t currentIndex = 0, typename variant_type, typename... arg_types>
		constexpr void visit(variant_type&& variant, arg_types&&... args) noexcept {
			if constexpr (currentIndex < std::variant_size_v<std::remove_cvref_t<variant_type>>) {
				variant_type&& variantNew = std::forward<variant_type>(variant);
				if (variantNew.index() == currentIndex) {
					function(std::get<currentIndex>(std::forward<variant_type>(variantNew)), std::forward<arg_types>(args)...);
					return;
				}
				visit<function, currentIndex + 1>(std::forward<variant_type>(variantNew), std::forward<arg_types>(args)...);
			}
		}

		template<typename value_type> void printValue(std::ostream& os, const value_type& value) {
			os << value;
		}

		template<bool_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << std::boolalpha << value;
		}

		template<string_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << "\"" << value << "\"";
		}

		template<string_view_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << "\"" << value << "\"";
		}

		template<vector_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << "[";
			for (size_t x = 0; x < value.size(); ++x) {
				printValue(os, value[x]);
				if (x < value.size() - 1) {
					os << ',';
				}
			}
			os << "]";
		}

		template<map_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << "{";
			size_t index{};
			for (auto iter = value.begin(); iter != value.end(); ++iter) {
				printValue(os, iter->first);
				os << ":";
				printValue(os, iter->second);
				if (index < value.size() - 1) {
					os << ",";
				}
				++index;
			}
			os << "}";
		}

		template<variant_t value_type> void printValue(std::ostream& os, const value_type& value) {
			static constexpr auto lambda = [](auto&& valueNew, auto& osNew) {
				printValue(osNew, valueNew);
			};
			visit<lambda>(value, os);
		}

		template<optional_t value_type> void printValue(std::ostream& os, const value_type& value) {
			if (value.has_value()) {
				printValue(os, value.value());
			}
		}

		template<pair_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << "{";
			printValue(os, std::get<0>(value));
			os << ",";
			printValue(os, std::get<1>(value));
			os << "}";
		}

		template<tuple_t value_type, size_t index> void printValue(std::ostream& os, const value_type& value) {
			if constexpr (index < std::tuple_size_v<std::remove_cvref_t<value_type>>) {
				printValue(os, std::get<index>(value));
				if constexpr (index < std::tuple_size_v<std::remove_cvref_t<value_type>> - 1) {
					os << ",";
				}
				printValue<value_type, index + 1>(os, value);
			}
		}

		template<tuple_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << "{";
			printValue<value_type, 0>(os, value);
			os << "}";
		}

	}

}

template<bnch_swt::internal::vector_t value_type> std::ostream& operator<<(std::ostream& os, const value_type& value) {
	bnch_swt::internal::printValue(os, value);
	return os;
}

template<bnch_swt::internal::map_t value_type> std::ostream& operator<<(std::ostream& os, const value_type& value) {
	bnch_swt::internal::printValue(os, value);
	return os;
}

template<bnch_swt::internal::variant_t value_type> std::ostream& operator<<(std::ostream& os, const value_type& value) {
	bnch_swt::internal::printValue(os, value);
	return os;
}

template<bnch_swt::internal::optional_t value_type> std::ostream& operator<<(std::ostream& os, const value_type& value) {
	bnch_swt::internal::printValue(os, value);
	return os;
}

template<bnch_swt::internal::pair_t value_type> std::ostream& operator<<(std::ostream& os, const value_type& value) {
	bnch_swt::internal::printValue(os, value);
	return os;
}

template<bnch_swt::internal::tuple_t value_type> std::ostream& operator<<(std::ostream& os, const value_type& value) {
	bnch_swt::internal::printValue(os, value);
	return os;
}
