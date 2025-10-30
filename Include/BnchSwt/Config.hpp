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

#if defined(macintosh) || defined(Macintosh) || (defined(__APPLE__) && defined(__MACH__))
	#define BNCH_SWT_PLATFORM_MAC 1
#elif defined(WIN32) || defined(_WIN32) || defined(_WIN64)
	#define BNCH_SWT_PLATFORM_WINDOWS 1
#elif defined(__ANDROID__)
	#define BNCH_SWT_ANDROID 1
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__gnu_linux__)
	#define BNCH_SWT_PLATFORM_LINUX 1
#endif

#define BNCH_SWT_ALIGN alignas(64)

using clock_type = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>; 
