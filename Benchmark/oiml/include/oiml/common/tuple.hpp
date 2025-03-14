/*
	MIT License

	Copyright (c) 2025 OpenInfer

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
/// Taken mostly from: https://github.com/codeinred/tuplet
/// Feb 3, 2023
#pragma once

#include <type_traits>
#include <cstddef>
#include <utility>

#if defined(OIML_TUPLET_NO_UNIQUE_ADDRESS) && !OIML_TUPLET_NO_UNIQUE_ADDRESS
	#define OIML_TUPLET_NO_UNIQUE_ADDRESS
#else
	#if _MSVC_LANG >= 202002L && (!defined __clang__)

		#define OIML_TUPLET_HAS_NO_UNIQUE_ADDRESS 1
		#define OIML_TUPLET_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]

	#elif _MSC_VER
		#define OIML_TUPLET_HAS_NO_UNIQUE_ADDRESS 0
		#define OIML_TUPLET_NO_UNIQUE_ADDRESS

	#elif __cplusplus > 201703L && (__has_cpp_attribute(no_unique_address))

		#define OIML_TUPLET_HAS_NO_UNIQUE_ADDRESS 1
		#define OIML_TUPLET_NO_UNIQUE_ADDRESS [[no_unique_address]]

	#else
		#define OIML_TUPLET_HAS_NO_UNIQUE_ADDRESS 0
		#define OIML_TUPLET_NO_UNIQUE_ADDRESS
	#endif
#endif

namespace oiml {

	template<typename tup, typename B> struct forward_as {
		using type = B&&;
	};

	template<typename tup, typename B> struct forward_as<tup&, B> {
		using type = B&;
	};

	template<typename tup, typename B> struct forward_as<tup const&, B> {
		using type = B const&;
	};

	template<typename tup, typename B> using forward_as_t = typename forward_as<tup, B>::type;

	template<typename value_type> using identity_t = value_type;

	template<typename first, typename...> using first_t = first;

	template<typename value_type> using type_t = typename value_type::type;

	template<typename... value_type> struct type_list {};

	template<typename... Ls, typename... Rs> constexpr auto operator+(type_list<Ls...>, type_list<Rs...>) {
		return type_list<Ls..., Rs...>{};
	}

	template<size_t index> using tag = std::integral_constant<size_t, index>;

	template<typename tup> using base_list_t = typename std::decay_t<tup>::base_list;

	template<typename tuple>
	concept base_list_tuple = requires { typename std::decay_t<tuple>::base_list; };

	template<typename value_type>
	concept stateless = std::is_empty_v<std::decay_t<value_type>>;

	template<typename value_type>
	concept indexable = stateless<value_type> || requires(value_type t) { t[tag<0>()]; };

	template<class... bases> struct type_map : bases... {
		using base_list = type_list<bases...>;
		using bases::operator[]...;
		using bases::decl_elem...;
	};

	template<size_t index, typename value_type> struct tuple_elem {
		static value_type decl_elem(tag<index>);
		using type = value_type;

		OIML_TUPLET_NO_UNIQUE_ADDRESS value_type value;

		constexpr decltype(auto) operator[](tag<index>) & {
			return (value);
		}
		constexpr decltype(auto) operator[](tag<index>) const& {
			return (value);
		}
		constexpr decltype(auto) operator[](tag<index>) && {
			return (std::move(*this).value);
		}
	};

	template<typename index_sequence, typename... value_type> struct get_tuple_base;

	template<size_t... index, typename... value_type> struct get_tuple_base<std::index_sequence<index...>, value_type...> {
		using type = type_map<tuple_elem<index, value_type>...>;
	};

	template<typename... value_type> using tuple_base_t = typename get_tuple_base<std::make_index_sequence<sizeof...(value_type)>, value_type...>::type;

	template<typename... value_type> struct tuple : tuple_base_t<value_type...> {
		static constexpr size_t index = sizeof...(value_type);
		using super					  = tuple_base_t<value_type...>;
		using super::operator[];
		using super::decl_elem;
	};

	template<> struct tuple<> : tuple_base_t<> {
		static constexpr size_t index = 0;
		using super					  = tuple_base_t<>;
		using base_list				  = type_list<>;
		using element_list			  = type_list<>;
	};

	template<typename... types> tuple(types&&...) -> tuple<std::remove_cvref_t<types>...>;

	template<size_t index, indexable tup> static constexpr decltype(auto) get(tup&& tupleVal) {
		return static_cast<tup&&>(tupleVal)[tag<index>()];
	}

	template<typename... types> static constexpr auto makeTuple(types&&... args) {
		return tuple<std::remove_cvref_t<types>...>{ { { static_cast<types&&>(args) }... } };
	}

	template<typename value_type, typename... Q> static constexpr auto repeatType(type_list<Q...>) {
		return type_list<first_t<value_type, Q>...>{};
	}

	template<typename... outer> static constexpr auto getOuterBases(type_list<outer...>) {
		return (repeatType<outer>(base_list_t<type_t<outer>>{}) + ...);
	}

	template<typename... inner> static constexpr auto getInnerBases(type_list<inner...>) {
		return (base_list_t<type_t<inner>>{} + ...);
	}

	template<typename value_type, typename... outer, typename... inner> static constexpr auto tupleCatImpl(value_type tupleVal, type_list<outer...>, type_list<inner...>)
		-> tuple<type_t<inner>...> {
		return { { { static_cast<forward_as_t<type_t<outer>&&, inner>>(tupleVal.identity_t<outer>::value).value }... } };
	}

	template<base_list_tuple... value_type> static constexpr auto tupleCat(value_type&&... ts) {
		if constexpr (sizeof...(value_type) == 0) {
			return tuple<>{};
		} else {
#if !defined(OIML_TUPLET_CAT_BY_FORWARDING_TUPLE)
	#if defined(__clang__)
		#define OIML_TUPLET_CAT_BY_FORWARDING_TUPLE 0
	#else
		#define OIML_TUPLET_CAT_BY_FORWARDING_TUPLE 1
	#endif
#endif
#if OIML_TUPLET_CAT_BY_FORWARDING_TUPLE
			using big_tuple = tuple<value_type&&...>;
#else
			using big_tuple = tuple<std::decay_t<value_type>...>;
#endif
			using outer_bases	 = base_list_t<big_tuple>;
			constexpr auto outer = getOuterBases(outer_bases{});
			constexpr auto inner = getInnerBases(outer_bases{});
			return tupleCatImpl(big_tuple{ { { static_cast<value_type&&>(ts) }... } }, outer, inner);
		}
	}

	template<typename... value_type> struct tuple_size;

	template<size_t index, typename... value_type> struct tuple_element;

	template<size_t index, typename... value_type> struct tuple_element<index, tuple<value_type...>> {
		using type = decltype(tuple<std::remove_cvref_t<value_type>...>::decl_elem(tag<index>()));
	};

	template<size_t index, typename tuple_type> using tuple_element_t = typename tuple_element<index, tuple_type>::type;

	template<typename... value_type> struct tuple_size<tuple<value_type...>> : std::integral_constant<size_t, sizeof...(value_type)> {};

	template<typename... value_type> struct tuple_size<std::tuple<value_type...>> : std::integral_constant<size_t, sizeof...(value_type)> {};

	template<typename... value_type> static constexpr auto tuple_size_v = tuple_size<std::remove_cvref_t<value_type>...>::value;
}

namespace std {

	template<typename... value_type> struct tuple_size<oiml::tuple<value_type...>> : std::integral_constant<size_t, sizeof...(value_type)> {};

	template<size_t index, typename... value_type> struct tuple_element<index, oiml::tuple<value_type...>> {
		using type = decltype(oiml::tuple<value_type...>::decl_elem(oiml::tag<index>()));
	};

}