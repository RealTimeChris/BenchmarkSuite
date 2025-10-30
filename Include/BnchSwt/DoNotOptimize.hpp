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
/// https://github.com/RealTimeChris/benchmarksuite
/// Sep 1, 2024
#pragma once

#include <BnchSwt/Config.hpp>

namespace bnch_swt::internal {

	template<typename value_type, typename... arg_types>
	concept invocable = std::is_invocable_v<std::remove_cvref_t<value_type>, arg_types...>;

	template<typename value_type, typename... arg_types>
	concept not_invocable = !invocable<value_type, arg_types...>;

	template<typename value_type, typename... arg_types>
	concept invocable_void = invocable<value_type, arg_types...> && std::is_void_v<std::invoke_result_t<value_type, arg_types...>>;

	template<typename value_type, typename... arg_types>
	concept invocable_not_void = invocable<value_type, arg_types...> && !std::is_void_v<std::invoke_result_t<value_type, arg_types...>>;

#if BNCH_SWT_COMPILER_MSVC
	#pragma optimize("", off)
	BNCH_SWT_INLINE void doNotOptimize(const void* value) {
		( void )value;
	};
	#pragma optimize("", on)
#else
	BNCH_SWT_INLINE void doNotOptimize(const void* value) {
	#if defined(BNCH_SWT_COMPILER_CLANG)
		asm volatile("" : "+r,m"(value) : : "memory");
	#elif defined(BNCH_SWT_COMPILER_GNU)
		asm volatile("" : "+m,r"(value) : : "memory");
	#endif
	}
#endif
}

namespace bnch_swt {

	template<internal::not_invocable value_type> BNCH_SWT_INLINE void doNotOptimizeAway(value_type&& value) {
		auto* valuePtr = &value;
		internal::doNotOptimize(valuePtr);
	}

	template<internal::invocable_void function_type, typename... arg_types> BNCH_SWT_INLINE void doNotOptimizeAway(function_type&& value, arg_types&&... args) {
		std::forward<function_type>(value)(std::forward<arg_types>(args)...);
		internal::doNotOptimize(value);
	}

	template<internal::invocable_not_void function_type, typename... arg_types> BNCH_SWT_INLINE auto doNotOptimizeAway(function_type&& value, arg_types&&... args) {
		auto resultVal = std::forward<function_type>(value)(std::forward<arg_types>(args)...);
		internal::doNotOptimize(&resultVal);
		return resultVal;
	}

}
