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
/// https://github.com/RealTimeChris/jsonifier
/// Feb 3, 2023
#pragma once

#include <jsonifier/Reflection.hpp>

namespace jsonifier_internal {

	template<typename value_type> static constexpr auto getJsonTypePre() {
		if constexpr (jsonifier::concepts::jsonifier_object_t<value_type> || jsonifier::concepts::map_t<value_type>) {
			return jsonifier::json_type::object;
		} else if constexpr (jsonifier::concepts::raw_array_t<value_type> || jsonifier::concepts::vector_t<value_type>) {
			return jsonifier::json_type::array;
		} else if constexpr (jsonifier::concepts::string_t<value_type> || jsonifier::concepts::string_view_t<value_type>) {
			return jsonifier::json_type::string;
		} else if constexpr (jsonifier::concepts::bool_t<value_type>) {
			return jsonifier::json_type::boolean;
		} else if constexpr (jsonifier::concepts::num_t<value_type> || jsonifier::concepts::enum_t<value_type>) {
			return jsonifier::json_type::number;
		} else if constexpr (jsonifier::concepts::always_null_t<value_type>) {
			return jsonifier::json_type::null;
		} else if constexpr (jsonifier::concepts::accessor_t<value_type>) {
			return jsonifier::json_type::accessor;
		} else {
			return jsonifier::json_type::custom;
		}
	}

	template<jsonifier::json_type type, typename value_type> static constexpr auto setJsonType() {
		if constexpr (type == jsonifier::json_type::object) {
			static_assert(( jsonifier::concepts::jsonifier_object_t<value_type> || jsonifier::concepts::map_t<value_type> || jsonifier::concepts::tuple_t<value_type> ),
				"Sorry, but that is not a valid value of type 'object'.");
		} else if constexpr (type == jsonifier::json_type::array) {
			static_assert(( jsonifier::concepts::vector_t<value_type> || jsonifier::concepts::raw_array_t<value_type> || jsonifier::concepts::tuple_t<value_type> ),
				"Sorry, but that is not a valid value of type 'array'.");
		} else if constexpr (type == jsonifier::json_type::string) {
			static_assert(( jsonifier::concepts::string_t<value_type> || jsonifier::concepts::string_view_t<value_type> ),
				"Sorry, but that is not a valid value of type 'string'.");
		} else if constexpr (type == jsonifier::json_type::number) {
			static_assert(( jsonifier::concepts::num_t<value_type> || jsonifier::concepts::enum_t<value_type> ), "Sorry, but that is not a valid value of type 'number'.");
		} else if constexpr (type == jsonifier::json_type::boolean) {
			static_assert(( jsonifier::concepts::bool_t<value_type> ), "Sorry, but that is not a valid value of type 'boolean'.");
		} else if constexpr (type == jsonifier::json_type::null) {
			static_assert(( jsonifier::concepts::always_null_t<value_type> ), "Sorry, but that is not a valid value of type 'null'.");
		} else if (type == jsonifier::json_type::accessor) {
			static_assert(( jsonifier::concepts::accessor_t<value_type> ), "Sorry, but that is not a valid value of type 'accessor'.");
		}
		return type;
	}

	template<auto... values> struct json_entity_pre;

	template<auto memberPtrNew> struct json_entity_pre<memberPtrNew> {
		using member_type = jsonifier_internal::remove_class_pointer_t<std::remove_cvref_t<decltype(memberPtrNew)>>;
		using class_type  = jsonifier_internal::remove_member_pointer_t<std::remove_cvref_t<decltype(memberPtrNew)>>;
		static constexpr auto nameTemp{ jsonifier_internal::getName<memberPtrNew>() };
		static constexpr jsonifier_internal::string_literal name{ jsonifier_internal::stringLiteralFromView<nameTemp.size()>(nameTemp) };
		static constexpr jsonifier::json_type type{ getJsonTypePre<member_type>() };
		static constexpr auto memberPtr{ memberPtrNew };
	};

	template<auto memberPtrNew, jsonifier::json_type typeNew> struct json_entity_pre<memberPtrNew, typeNew> {
		using member_type = jsonifier_internal::remove_class_pointer_t<std::remove_cvref_t<decltype(memberPtrNew)>>;
		using class_type  = jsonifier_internal::remove_member_pointer_t<std::remove_cvref_t<decltype(memberPtrNew)>>;
		static constexpr auto nameTemp{ jsonifier_internal::getName<memberPtrNew>() };
		static constexpr jsonifier_internal::string_literal name{ jsonifier_internal::stringLiteralFromView<nameTemp.size()>(nameTemp) };
		static constexpr jsonifier::json_type type{ setJsonType<typeNew, member_type>() };
		static constexpr auto memberPtr{ memberPtrNew };
	};

	template<auto memberPtrNew, jsonifier_internal::string_literal nameNew> struct json_entity_pre<memberPtrNew, nameNew> {
		using member_type = jsonifier_internal::remove_class_pointer_t<std::remove_cvref_t<decltype(memberPtrNew)>>;
		using class_type  = jsonifier_internal::remove_member_pointer_t<std::remove_cvref_t<decltype(memberPtrNew)>>;
		static constexpr jsonifier::json_type type{ getJsonTypePre<member_type>() };
		static constexpr jsonifier_internal::string_literal name{ nameNew };
		static constexpr auto memberPtr{ memberPtrNew };
	};

	template<auto memberPtrNew, jsonifier::json_type typeNew, jsonifier_internal::string_literal nameNew> struct json_entity_pre<memberPtrNew, typeNew, nameNew> {
		using member_type = jsonifier_internal::remove_class_pointer_t<std::remove_cvref_t<decltype(memberPtrNew)>>;
		using class_type  = jsonifier_internal::remove_member_pointer_t<std::remove_cvref_t<decltype(memberPtrNew)>>;
		static constexpr jsonifier::json_type type{ setJsonType<typeNew, member_type>() };
		static constexpr jsonifier_internal::string_literal name{ nameNew };
		static constexpr auto memberPtr{ memberPtrNew };
	};

	template<typename value_type>
	concept is_json_entity_pre = requires {
		typename std::remove_cvref_t<value_type>::member_type;
		typename std::remove_cvref_t<value_type>::class_type;
		std::remove_cvref_t<value_type>::memberPtr;
	} && !std::is_member_pointer_v<std::remove_cvref_t<value_type>>;

	template<typename value_type> constexpr bool getForceInline() noexcept {
		if constexpr (has_force_inline<value_type>) {
			return value_type::forceInline;
		} else {
			return false;
		}
	}

	template<size_t maxIndex, size_t index, auto value> constexpr auto createJsonEntityNewAuto() noexcept {
		if constexpr (is_json_entity_pre<decltype(value)>) {
			return value;
		} else {
			return json_entity_pre<value>{};
		}
	}

	template<auto... values, size_t... indices> constexpr auto createValueImpl(std::index_sequence<indices...>) {
		static_assert((convertible_to_json_entity<decltype(values)> && ...), "All arguments passed to createValue must be constructible to a json_entity.");
		return jsonifier_internal::makeTuple(createJsonEntityNewAuto<sizeof...(values), indices, values>()...);
	}

	template<typename current_type, typename containing_type> static constexpr bool isRecursive() {
		if constexpr (std::is_same_v<current_type, containing_type>) {
			return true;
		} else if constexpr (jsonifier::concepts::jsonifier_object_t<current_type>) {
			constexpr auto tuple = jsonifier::core<current_type>::parseValue;
			return []<size_t... Indices>(std::index_sequence<Indices...>) {
				return (isRecursive<typename std::remove_cvref_t<decltype(get<Indices>(tuple))>::member_type, containing_type>() || ...);
			}(std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<decltype(tuple)>>>{});
		} else if constexpr (jsonifier::concepts::shared_ptr_t<current_type> || jsonifier::concepts::unique_ptr_t<current_type> || jsonifier::concepts::pointer_t<current_type>) {
			using pointee_type = std::remove_cvref_t<decltype(*std::declval<current_type>())>;
			return isRecursive<pointee_type, containing_type>();
		} else if constexpr (jsonifier::concepts::map_t<current_type>) {
			using pointee_type = typename std::remove_cvref_t<current_type>::mapped_type;
			return isRecursive<pointee_type, containing_type>();
		} else {
			return false;
		}
	}

	template<auto memberPtrNew, jsonifier::json_type typeNew, jsonifier_internal::string_literal nameNew, size_t indexNew, size_t depthNew, bool forceInlineNew>
	struct json_entity {
		using member_type = jsonifier_internal::remove_class_pointer_t<std::remove_cvref_t<decltype(memberPtrNew)>>;
		using class_type  = jsonifier_internal::remove_member_pointer_t<std::remove_cvref_t<decltype(memberPtrNew)>>;
		static constexpr jsonifier::json_type type{ setJsonType<typeNew, member_type>() };
		static constexpr jsonifier_internal::string_literal name{ nameNew };
		static constexpr bool isRecursive{ isRecursive<member_type, class_type>() };
		static constexpr auto memberPtr{ memberPtrNew };
		static constexpr size_t index{ indexNew };
		static constexpr size_t depth{ depthNew };
		static constexpr bool forceInline{ forceInlineNew ? forceInlineNew : ((index * depth) < forceInlineLimit) };
		static constexpr bool isItLast{ indexNew == jsonifier_internal::tuple_size_v<raw_core_type<class_type>> - 1 };

		JSONIFIER_FORCE_INLINE decltype(auto) operator[](jsonifier_internal::tag<indexNew>) const {
			return *this;
		}
	};

	template<size_t depth, typename... bases> struct json_entities : public bases... {
		using bases::operator[]...;
		static constexpr size_t size{ sizeof...(bases) };

		template<typename value_type, typename context_type> JSONIFIER_FORCE_INLINE void iterateValues(value_type& value, context_type& context) const {
			((static_cast<const bases*>(this)->processIndex(value, context)), ...);
		}
	};

	template<template<auto options, typename> typename json_entity_type_final, auto options, typename json_entity_pre, size_t depth, size_t index> struct construct_json_entity {
		using json_entity_pre_type = std::remove_cvref_t<json_entity_pre>;
		using type				   = json_entity_type_final<options,
							json_entity<json_entity_pre_type::memberPtr, json_entity_pre_type::type, json_entity_pre_type::name, index, depth,
								getForceInline<jsonifier::core<std::remove_cvref_t<typename json_entity_pre_type::class_type>>>()>>;
	};

	template<template<auto options, typename> typename json_entity_type_final, auto options, typename value_type, size_t depth, typename index_sequence> struct get_json_entities;

	template<template<auto options, typename> typename json_entity_type_final, auto options, typename value_type, size_t depth, size_t... I>
	struct get_json_entities<json_entity_type_final, options, value_type, depth, std::index_sequence<I...>> {
		using type = json_entities<depth,
			typename construct_json_entity<json_entity_type_final, options, decltype(jsonifier_internal::get<I>(jsonifier::core<std::remove_cvref_t<value_type>>::parseValue)),
				depth, I>::type...>;
	};

	template<typename value_type> using raw_core_type = std::remove_cvref_t<decltype(jsonifier::core<value_type>::parseValue)>;

	template<template<auto options, typename> typename json_entity_type_final, auto options, typename value_type, size_t depth> using json_entities_t =
		typename get_json_entities<json_entity_type_final, options, value_type, depth,
			jsonifier_internal::tag_range<jsonifier_internal::tuple_size_v<raw_core_type<value_type>>>>::type;

}

namespace jsonifier_internal {

	template<size_t depth, typename... bases> struct tuple_size<jsonifier_internal::json_entities<depth, bases...>> : public std::integral_constant<size_t, sizeof...(bases)> {};

};

namespace jsonifier {

	template<auto testPtr, jsonifier::json_type type, jsonifier_internal::string_literal nameNew> constexpr auto createJsonEntityNew() {
		using member_type = jsonifier_internal::remove_class_pointer_t<decltype(testPtr)>;
		return jsonifier_internal::json_entity_pre<testPtr, jsonifier_internal::setJsonType<type, member_type>(), nameNew>{};
	}

	template<auto testPtr, jsonifier::json_type type> constexpr auto createJsonEntityNew() {
		using member_type = jsonifier_internal::remove_class_pointer_t<decltype(testPtr)>;
		return jsonifier_internal::json_entity_pre<testPtr, jsonifier_internal::setJsonType<type, member_type>()>{};
	}

	template<auto testPtr, jsonifier_internal::string_literal nameNew> constexpr auto createJsonEntityNew() {
		return jsonifier_internal::json_entity_pre<testPtr, nameNew>{};
	}

	template<auto... values> constexpr auto createValueNew() noexcept {
		static_assert((jsonifier_internal::convertible_to_json_entity<decltype(values)> && ...), "All arguments passed to createValue must be constructible to a json_entity.");
		return jsonifier_internal::createValueImpl<values...>(std::make_index_sequence<sizeof...(values)>{});
	}

}

namespace std {
	template<size_t depth, typename... bases> struct tuple_size<jsonifier_internal::json_entities<depth, bases...>> : public std::integral_constant<size_t, sizeof...(bases)> {};
}