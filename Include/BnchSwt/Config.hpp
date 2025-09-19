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

#if (defined(__x86_64__) || defined(_M_AMD64)) && !defined(_M_ARM64EC)
	#define BNCH_SWT_IS_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
	#define BNCH_SWT_IS_ARM64 1
#endif

#if defined(macintosh) || defined(Macintosh) || (defined(__APPLE__) && defined(__MACH__))
	#define BNCH_SWT_MAC 1
#elif defined(WIN32) || defined(_WIN32) || defined(_WIN64)
	#define BNCH_SWT_WIN 1
#elif defined(__ANDROID__)
	#define BNCH_SWT_ANDROID 1
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__gnu_linux__)
	#define BNCH_SWT_LINUX 1
#endif

#if defined(BNCH_SWT_FORCE_INLINE)
	#if defined(BNCH_SWT_MSVC)
		#define BNCH_SWT_INLINE [[msvc::forceinline]] inline 
	#elif defined(BNCH_SWT_CLANG)
		#define BNCH_SWT_INLINE inline __attribute__((always_inline))
	#elif defined(BNCH_SWT_GNUCXX)
		#define BNCH_SWT_INLINE inline __attribute__((always_inline))
	#else
		#define BNCH_SWT_INLINE inline
	#endif
#else
	#if defined(BNCH_SWT_MSVC)
		#define BNCH_SWT_INLINE [[msvc::noinline]]
	#elif defined(BNCH_SWT_CLANG)
		#define BNCH_SWT_INLINE inline 
	#elif defined(BNCH_SWT_GNUCXX)
		#define BNCH_SWT_INLINE inline 
	#else
		#define BNCH_SWT_INLINE inline
	#endif
#endif


#define BNCH_SWT_ALIGN alignas(64)

using clock_type = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>; 
