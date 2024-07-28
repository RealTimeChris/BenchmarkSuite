/*
	MIT License

	Copyright (c) 2023 RealTimeChris

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
/// https://github.com/RealTimeChris/jsonifier
/// Feb 20, 2023
#pragma once

#include <jsonifier/NumberUtils.hpp>
#include <jsonifier/StringUtils.hpp>
#include <jsonifier/Parser.hpp>

#include <memory>

namespace jsonifier_internal {

	template<size_t currentIndex, typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, typename iterator>
	JSONIFIER_ALWAYS_INLINE bool indexIntoSubTuple(value_type& value, iterator& iter, iterator& end) {
		if constexpr (currentIndex < uniqueStringLengths<value_type>.size()) {
			return subTuple<derived_type, options, value_type, iterator, currentIndex>.find(value, iter, end);
		} else {
			return false;
		}
	}

	template<typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, typename iterator> using function_type =
		decltype(&indexIntoSubTuple<0, derived_type, options, value_type, iterator>);

	template<typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, typename iterator, size_t... indices>
	constexpr auto generateArrayOfParsePtrsInternal(std::index_sequence<indices...>) {
		return std::array<function_type<derived_type, options, value_type, iterator>, sizeof...(indices)>{
			&indexIntoSubTuple<indices, derived_type, options, value_type, iterator>...
		};
	}

	template<typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, typename iterator> constexpr auto generateArrayOfParsePtrs() {
		return generateArrayOfParsePtrsInternal<derived_type, options, value_type, iterator>(std::make_index_sequence<uniqueStringLengths<value_type>.size()>{});
	}

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::jsonifier_value_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::jsonifier_value_t value_type, jsonifier::concepts::is_double_ptr iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal((*iter) + (64 * x));
			}
			if (**iter == '{') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				return;
			}
			bool isItFirst{ true };
			static constexpr auto memberCount = std::tuple_size_v<jsonifier::concepts::core_t<value_type>>;
			if constexpr (memberCount > 0) {
				while (**iter != '}') [[unlikely]] {
					if (!isItFirst) {
						if (**iter == ',') [[likely]] {
							++iter;
						} else {
							static constexpr auto sourceLocation{ std::source_location::current() };
							return;
						}
					} else {
						isItFirst = false;
					}

					if constexpr (jsonifier::concepts::has_excluded_keys<value_type>) {
						const auto keySize = getKeyLength<options, value_type>(iter, end, options.parserPtr->getErrors());
						jsonifier::string_view key{ *iter + 1, keySize };
						auto& keys = value.jsonifierExcludedKeys;
						if (keys.find(static_cast<typename unwrap_t<decltype(keys)>::key_type>(key)) != keys.end()) {
							skipToNextValue(iter, end);
							continue;
						}
					}
					//static constexpr auto invokeParsePtrs = generateArrayOfParsePtrs<derived_type, options, value_type, iterator>();

					static constexpr auto indexMappings = generateMappings(uniqueStringLengths<value_type>);
					if (!hash_tuple<derived_type, options, value_type, iterator>::find(indexMappings[static_cast<uint64_t>(*(*iter + 1))], value, iter, end)) {
						skipToNextValue(iter, end);
					}
				}
				++iter;
			} else {
				skipToNextValue(iter, end);
			}
		}

		template<jsonifier::concepts::jsonifier_value_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (*iter == '{') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				return;
			}
			bool isItFirst{ true };
			static constexpr auto memberCount = std::tuple_size_v<jsonifier::concepts::core_t<value_type>>;
			if constexpr (memberCount > 0) {
				while (*iter != '}') [[unlikely]] {
					if (!isItFirst) {
						if (*iter == ',') [[likely]] {
							++iter;
						} else {
							static constexpr auto sourceLocation{ std::source_location::current() };
							return;
						}
					} else {
						isItFirst = false;
					}

					if constexpr (jsonifier::concepts::has_excluded_keys<value_type>) {
						const auto keySize = getKeyLength<options, value_type>(iter, end, options.parserPtr->getErrors());
						jsonifier::string_view key{ *iter + 1, keySize };
						auto& keys = value.jsonifierExcludedKeys;
						if (keys.find(static_cast<typename unwrap_t<decltype(keys)>::key_type>(key)) != keys.end()) {
							skipToNextValue(iter, end);
							continue;
						}
					}

					static constexpr auto invokeParsePtrs = generateArrayOfParsePtrs<derived_type, options, value_type, iterator>();

					static constexpr auto indexMappings = generateMappings(uniqueStringLengths<value_type>);
					if (!invokeParsePtrs[indexMappings[static_cast<uint64_t>(*(iter + 1))]](value, iter, end)) {
						skipToNextValue(iter, end);
					}
				}
				++iter;
			} else {
				skipToNextValue(iter, end);
			}
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::jsonifier_scalar_value_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::jsonifier_scalar_value_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			static constexpr auto size{ std::tuple_size_v<jsonifier::concepts::core_t<value_type_new>> };
			if constexpr (size > 0) {
				static constexpr auto newPtr = std::get<0>(jsonifier::concepts::coreV<value_type_new>);
				auto& newMember				 = getMember<newPtr>(value);
				using member_type			 = unwrap_t<decltype(newMember)>;
				parse_impl<derived_type, options, member_type>::impl(newMember, iter, end);
			}
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::array_tuple_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::array_tuple_t value_type, jsonifier::concepts::is_double_ptr iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (**iter == '[') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_Start>(
					getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}
			static constexpr auto n = std::tuple_size_v<value_type_new>;
			parseObjects<options, n>(value, iter, end);
			if (**iter == ']') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_End>(
					getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}
		}

		template<size_t n, size_t indexNew = 0, bool isItFirst = true, jsonifier::concepts::array_tuple_t value_type, jsonifier::concepts::is_double_ptr iterator>
		JSONIFIER_ALWAYS_INLINE static void parseObjects(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			auto& item = std::get<indexNew>(value);

			if constexpr (!isItFirst) {
				if (**iter == ',') [[likely]] {
					++iter;
				} else {
					static constexpr auto sourceLocation{ std::source_location::current() };
					options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Comma>(
						getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
					return;
				}
			}

			parse_impl<derived_type, options, decltype(item)>::impl(item, iter, end);
			if constexpr (indexNew < n - 1) {
				parseObjects<options, n, indexNew + 1, false>(value, iter, end);
			}
		}

		template<jsonifier::concepts::array_tuple_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (*iter == '[') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_Start>(
					getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}
			static constexpr auto n = std::tuple_size_v<value_type_new>;
			parseObjects<options, n>(value, iter, end);
			if (*iter == ']') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_End>(
					getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}
		}

		template<size_t n, size_t indexNew = 0, bool isItFirst = true, jsonifier::concepts::array_tuple_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void parseObjects(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			auto& item = std::get<indexNew>(value);

			if constexpr (!isItFirst) {
				if (*iter == ',') [[likely]] {
					++iter;
				} else {
					static constexpr auto sourceLocation{ std::source_location::current() };
					options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Comma>(
						getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
					return;
				}
			}

			parse_impl<derived_type, options, decltype(item)>::impl(item, iter, end);
			if constexpr (indexNew < n - 1) {
				parseObjects<options, n, indexNew + 1, false>(value, iter, end);
			}
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::map_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::map_t value_type, jsonifier::concepts::is_double_ptr iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (**iter == '{') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Object_Start>(
					getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}

			bool first{ true };
			while (*iter != *end) {
				if (**iter == '}') [[unlikely]] {
					++iter;
					return;
				} else if (first) {
					first = false;
				} else {
					if (**iter == ',') [[likely]] {
						++iter;
					} else {
						static constexpr auto sourceLocation{ std::source_location::current() };
						options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Comma_Or_Object_End>(
							getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
						skipToNextValue(iter, end);
						return;
					}
				}

				static thread_local typename value_type_new::key_type key{};
				parse_impl<derived_type, options, typename value_type_new::key_type>::impl(key, iter, end);

				if (**iter == ':') [[likely]] {
					++iter;
				} else {
					static constexpr auto sourceLocation{ std::source_location::current() };
					options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Colon>(
						getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
					skipToNextValue(iter, end);
					return;
				}
				parse_impl<derived_type, options, typename value_type_new::mapped_type>::impl(value[key], iter, end);
			}
		}

		template<jsonifier::concepts::map_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (*iter == '{') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Object_Start>(
					getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}

			bool first{ true };
			while (*iter != *end) {
				if (*iter == '}') [[unlikely]] {
					++iter;
					return;
				} else if (first) {
					first = false;
				} else {
					if (*iter == ',') [[likely]] {
						++iter;
					} else {
						static constexpr auto sourceLocation{ std::source_location::current() };
						options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Comma_Or_Object_End>(
							getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
						skipToNextValue(iter, end);
						return;
					}
				}

				static thread_local typename value_type_new::key_type key{};
				parse_impl<derived_type, options, typename value_type_new::key_type>::impl(key, iter, end);

				if (*iter == ':') [[likely]] {
					++iter;
				} else {
					static constexpr auto sourceLocation{ std::source_location::current() };
					options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Colon>(
						getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
					skipToNextValue(iter, end);
					return;
				}
				parse_impl<derived_type, options, typename value_type_new::mapped_type>::impl(value[key], iter, end);
			}
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::variant_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::variant_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			static constexpr auto lambda = [&](auto&& valueNew, auto&& value, auto&& buffer, auto&& index) {
				using member_type = decltype(valueNew);
				return parse_impl<derived_type, options, member_type>::impl(value, iter, end);
			};
			visit<lambda>(std::forward<value_type>(value), std::forward<value_type>(value), std::forward<iterator>(iter), std::forward<iterator>(end));
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::optional_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::optional_t value_type, jsonifier::concepts::is_double_ptr iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (**iter != 'n') [[unlikely]] {
				parse_impl<derived_type, options, decltype(*value)>::impl(value.emplace(), iter, end);
			} else {
				++iter;
				return;
			}
		}

		template<jsonifier::concepts::optional_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (*iter != 'n') [[unlikely]] {
				parse_impl<derived_type, options, decltype(*value)>::impl(value.emplace(), iter, end);
			} else {
				iter += 4;
				return;
			}
		}
	};

	JSONIFIER_ALWAYS_INLINE void noop() noexcept {};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::vector_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::vector_t value_type, jsonifier::concepts::is_double_ptr iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (**iter == '[') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_Start>(
					getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}

			if (**iter == ']') [[unlikely]] {
				++iter;
				return;
			}

			const auto n = value.size();
			auto iterNew = value.begin();
			for (size_t i = 0; i < n; ++i) {
				parse_impl<derived_type, options, typename unwrap_t<value_type_new>::value_type>::impl(*(iterNew++), iter, end);

				if (**iter == ',') [[likely]] {
					++iter;
				} else {
					if (**iter == ']') [[likely]] {
						++iter;
						return value.size() == (i + 1) ? noop() : value.resize(i + 1);
					} else {
						static constexpr auto sourceLocation{ std::source_location::current() };
						options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_End>(
							getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
					}
					return;
				}
			}
			while (*iter != *end) {
				parse_impl<derived_type, options, typename unwrap_t<value_type_new>::value_type>::impl(value.emplace_back(), iter, end);

				if (**iter == ',') [[likely]] {
					++iter;
				} else {
					if (**iter == ']') [[likely]] {
						++iter;
					} else {
						static constexpr auto sourceLocation{ std::source_location::current() };
						options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_End>(
							getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
					}
					return;
				}
			}
		}

		template<jsonifier::concepts::vector_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (*iter == '[') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_Start>(
					getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}

			if (*iter == ']') [[unlikely]] {
				++iter;
				return;
			}

			const auto n = value.size();
			auto iterNew = value.begin();
			for (size_t i = 0; i < n; ++i) {
				parse_impl<derived_type, options, typename unwrap_t<value_type_new>::value_type>::impl(*(iterNew++), iter, end);

				if (*iter == ',') [[likely]] {
					++iter;
				} else {
					if (*iter == ']') [[likely]] {
						++iter;
						return value.size() == (i + 1) ? noop() : value.resize(i + 1);
					} else {
						static constexpr auto sourceLocation{ std::source_location::current() };
						options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_End>(
							getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
					}
					return;
				}
			}
			while (*iter != *end) {
				parse_impl<derived_type, options, typename unwrap_t<value_type_new>::value_type>::impl(value.emplace_back(), iter, end);

				if (*iter == ',') [[likely]] {
					++iter;
				} else {
					if (*iter == ']') [[likely]] {
						++iter;
					} else {
						static constexpr auto sourceLocation{ std::source_location::current() };
						options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_End>(
							getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
					}
					return;
				}
			}
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::raw_array_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::raw_array_t value_type, jsonifier::concepts::is_double_ptr iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (**iter == '[') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_Start>(
					getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}

			if (**iter == ']') [[unlikely]] {
				++iter;
				return;
			}

			static constexpr auto n = value.size();
			auto iterNew			= value.begin();
			for (size_t i = 0; i < n; ++i) {
				parse_impl<derived_type, options, decltype(value[0])>::impl(*(iterNew++), iter, end);

				if (**iter == ',') [[likely]] {
					++iter;
				} else {
					if (**iter == ']') [[likely]] {
						++iter;
					} else {
						static constexpr auto sourceLocation{ std::source_location::current() };
						options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_End>(
							getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
					}
					return;
				}
			}
		}

		template<jsonifier::concepts::raw_array_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (*iter == '[') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_Start>(
					getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}

			if (*iter == ']') [[unlikely]] {
				++iter;
				return;
			}

			static constexpr auto n = value.size();
			auto iterNew			= value.begin();
			for (size_t i = 0; i < n; ++i) {
				parse_impl<derived_type, options, decltype(value[0])>::impl(*(iterNew++), iter, end);

				if (*iter == ',') [[likely]] {
					++iter;
				} else {
					if (*iter == ']') [[likely]] {
						++iter;
					} else {
						static constexpr auto sourceLocation{ std::source_location::current() };
						options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_End>(
							getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
					}
					return;
				}
			}
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::string_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::string_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			parseString<options>(value, iter, end, options.parserPtr->getErrors());
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::char_type value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::char_type value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			value = static_cast<value_type_new>(*(*iter + 1));
			++iter;
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::raw_json_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::raw_json_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			auto newPtr = *iter;
			skipToNextValue(iter, end);
			int64_t newSize = getUnderlyingPtr(iter) - newPtr;
			if (newSize > 0) [[likely]] {
				jsonifier::string newString{};
				newString.resize(static_cast<size_t>(newSize));
				std::memcpy(newString.data(), newPtr, static_cast<size_t>(newSize));
				value = newString;
			}
			return;
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::unique_ptr_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::unique_ptr_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			parse_impl<derived_type, options, decltype(*value)>::impl(*value, iter, end);
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::always_null_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::null_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&&, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (!parseNull(iter)) {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Invalid_Null_Value>(
					getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
				return;
			}
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::enum_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::enum_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			size_t newValue{};
			parse_impl<derived_type, options, size_t>::impl(newValue, iter, end);
			value = static_cast<value_type_new>(newValue);
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::num_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::num_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (!parseNumber(value, iter, end)) {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Invalid_Number_Value>(
					getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
				return;
			}
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::bool_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::bool_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			for (uint64_t x = 0; x < sixtyFourBitsPerStep; ++x) {
				jsonifierPrefetchInternal(iter + (64 * x));
			}
			if (!parseBool(value, iter)) {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Invalid_Bool_Value>(
					getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
				return;
			}
		}
	};

}