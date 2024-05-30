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
/// Dec 6, 2024
#pragma once

#include <BnchSwt/Config.hpp>

#if defined(BNCH_SWT_WIN)

	#include <cerrno>// for errno
	#include <cstring>// for memset
	#include <stdexcept>

	#include <iostream>
	#include <vector>

namespace bnch_swt {

	BNCH_SWT_ALWAYS_INLINE size_t rdtsc() {
		return __rdtsc();
	}

	template<typename event_count> struct event_collector_type {
		template<typename function_type, typename... arg_types> BNCH_SWT_ALWAYS_INLINE event_count start(function_type&& function, arg_types&&... args) {
			event_count count{};
			const auto startClock		 = clock_type::now();
			volatile uint64_t cycleStart = rdtsc();
			count.bytesProcessedVal.emplace(std::forward<function_type>(function)(std::forward<arg_types>(args)...));
			volatile uint64_t cycleEnd = rdtsc();
			const auto endClock		   = clock_type::now();
			count.cyclesVal.emplace(cycleEnd - cycleStart);
			count.elapsed = endClock - startClock;
			return count;
		}
	};

}
#endif