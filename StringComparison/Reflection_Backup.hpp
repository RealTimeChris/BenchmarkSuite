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
/// Feb 3, 2023
#pragma once

#include <jsonifier/TypeEntities.hpp>
#include <jsonifier/StringView.hpp>

namespace jsonifier {

	template<typename char_type> class string_view_base;

}

namespace jsonifier_internal {

	using string_view = jsonifier::string_view_base<char>;

	template<size_t sizeVal> struct string_literal {
		static constexpr auto length{ sizeVal > 0 ? sizeVal - 1 : 0 };

		constexpr string_literal() noexcept = default;

		constexpr string_literal(const char (&str)[sizeVal]) {
			std::copy(str, str + sizeVal, values);
		}

		constexpr size_t size() const {
			return length;
		}

		constexpr const char* data() const {
			return values;
		}

		constexpr jsonifier::string_view view() const {
			return { values, length };
		}

		char values[sizeVal]{};
	};

	template<size_t N> constexpr auto stringLiteralFromView(jsonifier::string_view str) {
		string_literal<N + 1> sl{};
		std::copy_n(str.data(), str.size(), sl.values);
		*(sl.values + N) = '\0';
		return sl;
	}

	template<typename member_type, typename class_type> struct member_pointer {
		member_type class_type::*ptr{};
		constexpr member_pointer(member_type class_type::*p) : ptr(p) {};
	};

	template<typename member_type_new, typename class_type_new> struct data_member {
		using member_type = member_type_new;
		using class_type  = class_type_new;
		member_pointer<member_type, class_type> memberPtr{};
		jsonifier::string_view name{};

		constexpr jsonifier::string_view view() const {
			return name;
		}

		constexpr auto ptr() const {
			return memberPtr.ptr;
		}

		constexpr data_member(jsonifier::string_view str, member_type class_type::*ptr) : memberPtr(ptr), name(str) {};
	};

	template<typename member_type, typename class_type> constexpr auto makeDataMemberAuto(jsonifier::string_view str, member_type class_type::*ptr) {
		return data_member<member_type, class_type>(str, ptr);
	}

	/**
	 * @brief External template variable declaration.
	 *
	 * Declare an external template variable to be defined elsewhere.
	 *
	 * @tparam value_type The type of the external template variable.
	 */
	template<typename value_type> extern const value_type external;

	/**
	 * @brief Struct to remove member pointers.
	 *
	 * Define a struct to remove member pointers from the given type.
	 *
	 * @tparam value_type The type from which to remove member pointers.
	 */
	template<typename member_type> struct remove_member_pointer {
		using type = member_type;
	};

	template<typename value_type, typename member_type> struct remove_member_pointer<member_type value_type::*> {
		using type = value_type;
	};

	template<typename value_type, typename member_type, typename... Args> struct remove_member_pointer<member_type (value_type::*)(Args...)> {
		using type = value_type;
	};

	template<typename value_type> using remove_member_pointer_t = typename remove_member_pointer<value_type>::type;

	template<auto valueNew> struct make_static {
		static constexpr auto value{ valueNew };
	};

#if defined(JSONIFIER_CLANG)
	constexpr auto pretty_function_tail = "]";
#elif defined(JSONIFIER_GNUCXX)
	constexpr auto pretty_function_tail = ";";
#endif

	/**
	 * @brief Get the name of a member pointer.
	 *
	 * function_typetion to extract the name of a member pointer.
	 *
	 * @tparam p The member pointer.
	 * @return The name of the member pointer.
	 */
#if defined(JSONIFIER_MSVC) && !defined(JSONIFIER_CLANG)
	template<typename value_type, auto p> consteval jsonifier::string_view getNameInternal() {
		jsonifier::string_view str = std::source_location::current().function_name();
		str						   = str.substr(str.find("->") + 2);
		return str.substr(0, str.find(">"));
	}
#else
	template<auto p> consteval jsonifier::string_view getNameInternal() {
		jsonifier::string_view str = std::source_location::current().function_name();
		str						   = str.substr(str.find("&") + 1);
		str						   = str.substr(0, str.find(pretty_function_tail));
		return str.substr(str.rfind("::") + 2);
	}
#endif

	template<auto p>
		requires(std::is_member_pointer_v<decltype(p)>)
	constexpr auto getName() {
#if defined(JSONIFIER_MSVC) && !defined(JSONIFIER_CLANG)
		using value_type		 = remove_member_pointer<unwrap_t<decltype(p)>>::type;
		constexpr auto pNew		 = p;
		constexpr auto newString = getNameInternal<value_type, &(external<value_type>.*pNew)>();
#else
		constexpr auto newString = getNameInternal<p>();
#endif
		return newString;
	}

	/**
	 * @brief Get the names of multiple member pointers.
	 *
	 * function_typetion to extract the names of multiple member pointers.
	 *
	 * @tparam args Member pointers.
	 * @return An array of member pointer names.
	 */
	template<auto... args> constexpr decltype(auto) getNames() {
		return std::array<jsonifier::string_view, sizeof...(args)>{ getName<args>()... };
	}

	/**
	 * @brief Internal function to generate an interleaved tuple of member names and values.
	 *
	 * Internal function to generate an interleaved tuple of member names and values.
	 *
	 * @tparam tuple_types Types of tuple elements.
	 * @tparam indices indices of tuple elements.
	 * @param tuple The input tuple.
	 * @param views Array of member names.
	 * @return Interleaved tuple of member names and values.
	 */
	template<typename... tuple_types, size_t... indices> constexpr decltype(auto) generateInterleavedTupleInternal(const std::tuple<tuple_types...>& tuple,
		const std::array<jsonifier::string_view, sizeof...(indices)>& views, std::index_sequence<indices...>) {
		return std::make_tuple(makeDataMemberAuto(views[indices], std::get<indices>(tuple))...);
	}

	/**
	 * @brief Generate an interleaved tuple of member names and values.
	 *
	 * function_typetion to generate an interleaved tuple of member names and values.
	 *
	 * @tparam tuple_types Types of tuple elements.
	 * @param tuple The input tuple.
	 * @param views Array of member names.
	 * @return Interleaved tuple of member names and values.
	 */
	template<typename... tuple_types>
	constexpr decltype(auto) generateInterleavedTuple(const std::tuple<tuple_types...>& tuple, const std::array<jsonifier::string_view, sizeof...(tuple_types)>& views) {
		return generateInterleavedTupleInternal(tuple, views, std::index_sequence_for<tuple_types...>{});
	}

}