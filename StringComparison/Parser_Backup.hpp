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
#include <jsonifier/Derailleur.hpp>
#include <jsonifier/Parser.hpp>

#include <memory>

namespace jsonifier_internal {

	template<size_t subTupleIndex, size_t index, typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, typename iterator,
		jsonifier::concepts::uint64_type size_type>
	JSONIFIER_ALWAYS_INLINE void invokeParse(value_type& value, iterator& iter, iterator& end, size_type keySize) noexcept {
		if constexpr (jsonifier::concepts::json_structural_iterator_t<iterator>) {
			++iter;
		} else {
			iter += keySize + 2;
		}
		if (*iter == ':') [[likely]] {
			++iter;
		} else {
			static constexpr auto sourceLocation{ std::source_location::current() };
			options.parserPtr->getErrors().emplace_back(
				error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Colon>(iter - options.rootIter, end - iter, options.rootIter));
			skipToNextValue(iter, end);
			return;
		}
		static constexpr auto ptr = std::get<index>(std::get<subTupleIndex>(jsonifier_internal::final_tuple_static_data<value_type>)).ptr();
		using member_type		  = unwrap_t<decltype(value.*ptr)>;
		parse_impl<derived_type, options, member_type>::impl(value.*ptr, iter, end);
	}

	template<size_t subTupleIndex, size_t index, typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, typename iterator,
		jsonifier::concepts::uint64_type size_type>
	using invoke_parse_function_ptr = const decltype(&invokeParse<subTupleIndex, index, derived_type, options, value_type, iterator, size_type>);

	template<size_t subTupleIndex, typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, typename iterator,
		jsonifier::concepts::uint64_type size_type, size_t... indices>
	JSONIFIER_ALWAYS_INLINE constexpr auto generateArrayOfInvokeParsePtrsInternal(std::index_sequence<indices...>) noexcept {
		return std::array<invoke_parse_function_ptr<subTupleIndex, 0, derived_type, options, value_type, iterator, size_type>, sizeof...(indices)>{
			&invokeParse<subTupleIndex, indices, derived_type, options, value_type, iterator, size_type>...
		};
	}

	template<size_t subTupleIndex, typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, typename iterator,
		jsonifier::concepts::uint64_type size_type>
	JSONIFIER_ALWAYS_INLINE constexpr auto generateArrayOfInvokeParsePtrs() noexcept {
		constexpr auto& subTuple = std::get<subTupleIndex>(final_tuple_static_data<value_type>);
		constexpr auto tupleSize = std::tuple_size_v<unwrap_t<decltype(subTuple)>>;
		using tuple_type		 = unwrap_t<decltype(subTuple)>;
		return generateArrayOfInvokeParsePtrsInternal<subTupleIndex, derived_type, options, value_type, iterator, size_type>(std::make_index_sequence<tupleSize>{});
	}

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::jsonifier_value_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::jsonifier_value_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			if (*iter == '{') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_Start>(
					iter - options.rootIter, end - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}
			bool isItFirst{ true };
			static constexpr auto memberCount = std::tuple_size_v<jsonifier::concepts::core_t<value_type>>;
			if constexpr (memberCount > 0) {
				while (*iter != '}') {
					if (!isItFirst) {
						if (*iter == ',') [[likely]] {
							++iter;
						} else {
							static constexpr auto sourceLocation{ std::source_location::current() };
							options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Comma>(
								iter - options.rootIter, end - options.rootIter, options.rootIter));
							skipToNextValue(iter, end);
							return;
						}
					} else {
						isItFirst = false;
					}

					const auto keySize = getKeyLength<options, value_type>(iter, end, options.parserPtr->getErrors());

					if constexpr (jsonifier::concepts::has_excluded_keys<value_type>) {
						jsonifier::string_view key{ static_cast<const char*>(iter) + 1, keySize };
						auto& keys = value.jsonifierExcludedKeys;
						if (keys.find(static_cast<typename unwrap_t<decltype(keys)>::key_type>(key)) != keys.end()) {
							skipToNextValue(iter, end);
							continue;
						}
					}

					static constexpr auto functionLambda = [](const auto hashSubTupleIndex, auto&& functionLambda, auto&& value, auto&& iter, auto&& end, const auto keySize) {
						static constexpr auto subTupleFunctionPtrArray = generateArrayOfInvokeParsePtrs<hashSubTupleIndex, derived_type, options, value_type, iterator, size_t>();
						using function_type							   = decltype(functionLambda);
						static constexpr function_type function{};
						if (!hash_tuple<function, value_type>::template find<hashSubTupleIndex, subTupleFunctionPtrArray>(static_cast<const char*>(iter) + 1, value, iter, end,
								keySize)) {
							skipToNextValue(iter, end);
							return false;
						} else {
							return true;
						}
						return true;
					};
					hash_tuple<functionLambda, value_type>::find(keySize, functionLambda, value, iter, end, keySize);
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
		template<jsonifier::concepts::array_tuple_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			if (*iter == '[') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_Start>(
					iter - options.rootIter, end - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}
			static constexpr auto n = std::tuple_size_v<value_type_new>;
			parseObjects<options, n>(value, iter, end);
			if (*iter == ']') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_End>(iter - options.rootIter,
					end - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}
		}

		template<size_t n, size_t indexNew = 0, bool isItFirst = true, jsonifier::concepts::array_tuple_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void parseObjects(value_type&& value, iterator& iter, iterator& end) noexcept {
			auto& item = std::get<indexNew>(value);

			if constexpr (!isItFirst) {
				if (*iter == ',') [[likely]] {
					++iter;
				} else {
					static constexpr auto sourceLocation{ std::source_location::current() };
					options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Comma>(iter - options.rootIter,
						end - options.rootIter, options.rootIter));
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
		template<jsonifier::concepts::map_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			if (*iter == '{') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Object_Start>(
					iter - options.rootIter, end - options.rootIter, options.rootIter));
				skipToNextValue(iter, end);
				return;
			}

			bool first{ true };
			while (static_cast<const char*>(iter) != static_cast<const char*>(end)) {
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
							iter - options.rootIter, end - options.rootIter, options.rootIter));
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
					options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Colon>(iter - options.rootIter,
						end - options.rootIter, options.rootIter));
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
			static constexpr auto lambda = [&](auto&& valueNew, auto&& value, auto&& buffer, auto&& index) {
				using member_type = decltype(valueNew);
				return parse_impl<derived_type, options, member_type>::impl(value, iter, end);
			};
			visit<lambda>(std::forward<value_type>(value), std::forward<value_type>(value), std::forward<iterator>(iter), std::forward<iterator>(end));
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::optional_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::optional_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			if (*iter != 'n') [[unlikely]] {
				parse_impl<derived_type, options, decltype(*value)>::impl(value.emplace(), iter, end);
			} else {
				if constexpr (jsonifier::concepts::json_structural_iterator_t<iterator>) {
					++iter;
				} else {
					iter += 4;
				}
				return;
			}
		}
	};

	JSONIFIER_ALWAYS_INLINE void noop() noexcept {};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::vector_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::vector_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			if (*iter == '[') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_Start>(
					iter - options.rootIter, end - options.rootIter, options.rootIter));
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
							iter - options.rootIter, end - options.rootIter, options.rootIter));
					}
					return;
				}
			}
			while (static_cast<const char*>(iter) != static_cast<const char*>(end)) {
				parse_impl<derived_type, options, typename unwrap_t<value_type_new>::value_type>::impl(value.emplace_back(), iter, end);

				if (*iter == ',') [[likely]] {
					++iter;
				} else {
					if (*iter == ']') [[likely]] {
						++iter;
					} else {
						static constexpr auto sourceLocation{ std::source_location::current() };
						options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_End>(
							iter - options.rootIter, end - options.rootIter, options.rootIter));
					}
					return;
				}
			}
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::raw_array_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::raw_array_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			if (*iter == '[') [[likely]] {
				++iter;
			} else {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Array_Start>(
					iter - options.rootIter, end - options.rootIter, options.rootIter));
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
							iter - options.rootIter, end - options.rootIter, options.rootIter));
					}
					return;
				}
			}
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::string_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::string_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			parseString<options>(value, iter, end, options.parserPtr->getErrors());
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::char_type value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::char_type value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			value = static_cast<value_type_new>(*(static_cast<const char*>(iter) + 1));
			++iter;
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::raw_json_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::raw_json_t value_type, typename iterator>
		JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			auto newPtr = static_cast<const char*>(iter);
			skipToNextValue(iter, end);
			int64_t newSize = static_cast<const char*>(iter) - newPtr;
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
			parse_impl<derived_type, options, decltype(*value)>::impl(*value, iter, end);
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::always_null_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::null_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&&, iterator& iter, iterator& end) noexcept {
			if (!parseNull(iter)) {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Invalid_Null_Value>(iter - options.rootIter,
					end - options.rootIter, options.rootIter));
				return;
			}
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::enum_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::enum_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			size_t newValue{};
			parse_impl<derived_type, options, size_t>::impl(newValue, iter, end);
			value = static_cast<value_type_new>(newValue);
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::num_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::num_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			if (!parseNumber(value, iter, end)) {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Invalid_Number_Value>(
					iter - options.rootIter, end - options.rootIter, options.rootIter));
				return;
			}
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, jsonifier::concepts::bool_t value_type_new>
	struct parse_impl<derived_type, options, value_type_new> {
		template<jsonifier::concepts::bool_t value_type, typename iterator> JSONIFIER_ALWAYS_INLINE static void impl(value_type&& value, iterator& iter, iterator& end) noexcept {
			if (!parseBool(value, iter)) {
				static constexpr auto sourceLocation{ std::source_location::current() };
				options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Invalid_Bool_Value>(iter - options.rootIter,
					end - options.rootIter, options.rootIter));
				return;
			}
		}
	};

}