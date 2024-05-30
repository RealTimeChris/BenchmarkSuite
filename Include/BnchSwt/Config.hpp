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

#include <cstdint>
#include <chrono>

#if defined(__clang__) || (defined(__GNUC__) && defined(__llvm__)) || (defined(__APPLE__) && defined(__clang__))
	#define BNCH_SWT_CLANG 1
#elif defined(_MSC_VER)
	#define BNCH_SWT_MSVC 1
	#pragma warning(disable : 4820)
	#pragma warning(disable : 4371)
	#pragma warning(disable : 4710)
	#pragma warning(disable : 4711)
#elif defined(__GNUC__) && !defined(__clang__)
	#define BNCH_SWT_GNUCXX 1
#endif

#if defined(__has_builtin)
	#define BNCH_SWT_HAS_BUILTIN(x) __has_builtin(x)
#else
	#define BNCH_SWT_HAS_BUILTIN(x) 0
#endif

#define BNCH_SWT_GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)

#if defined(macintosh) || defined(Macintosh) || (defined(__APPLE__) && defined(__MACH__))
	#define BNCH_SWT_MAC 1
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__gnu_linux__)
	#define BNCH_SWT_LINUX 1
#elif defined(WIN32) || defined(_WIN32) || defined(_WIN64)
	#define BNCH_SWT_WIN 1
#endif

#if defined(BNCH_SWT_GNUCXX) || defined(BNCH_SWT_CLANG)
	#define JSONIFIER_ASSUME(x) \
		do { \
			if (!(x)) \
				__builtin_unreachable(); \
		} while (0)
#elif defined(BNCH_SWT_MSVC)
	#include <intrin.h>
	#define JSONIFIER_ASSUME(x) __assume(x)
#else
	#define JSONIFIER_ASSUME(x) (( void )0)
#endif

#if defined(BNCH_SWT_MSVC)
	#define BNCH_SWT_NO_INLINE __declspec(noinline)
	#define BNCH_SWT_FLATTEN [[msvc::flatten]] inline 
	#define BNCH_SWT_ALWAYS_INLINE [[msvc::forceinline]] inline 
	#define BNCH_SWT_INLINE inline
#elif defined(BNCH_SWT_CLANG)
	#define BNCH_SWT_NO_INLINE __attribute__((__noinline__))
	#define BNCH_SWT_FLATTEN inline __attribute__((flatten))
	#define BNCH_SWT_ALWAYS_INLINE inline __attribute__((always_inline))
	#define BNCH_SWT_INLINE inline
#elif defined(BNCH_SWT_GNUCXX)
	#define BNCH_SWT_NO_INLINE __attribute__((noinline))
	#define BNCH_SWT_FLATTEN inline __attribute__((flatten))
	#define BNCH_SWT_ALWAYS_INLINE inline __attribute__((always_inline))
	#define BNCH_SWT_INLINE inline
#else
	#define BNCH_SWT_FLATTEN inline
	#define BNCH_SWT_NO_INLINE
	#define BNCH_SWT_ALWAYS_INLINE inline
	#define BNCH_SWT_INLINE inline
#endif

using clock_type = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>; 