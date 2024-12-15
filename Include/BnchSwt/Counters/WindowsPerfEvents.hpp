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

	#include <intrin.h>

	#include <stdexcept>
	#include <iostream>
	#include <cstring>
	#include <cerrno>
	#include <vector>

namespace bnch_swt::internal {

	template<typename event_count, size_t count> struct event_collector_type : public std::vector<event_count> {
		size_t currentIndex{};

		BNCH_SWT_INLINE event_collector_type() : std::vector<event_count>{ count } {};

		template<typename function_type, typename... arg_types> BNCH_SWT_INLINE void start(function_type&& function, arg_types&&... args) {
			volatile uint64_t cycleStart = __rdtsc();
			const auto startClock		 = clock_type::now();
			std::vector<event_count>::operator[](currentIndex)
				.bytesProcessedVal.emplace(static_cast<size_t>(std::forward<function_type>(function)(std::forward<arg_types>(args)...)));
			const auto endClock		   = clock_type::now();
			volatile uint64_t cycleEnd = __rdtsc();
			std::vector<event_count>::operator[](currentIndex).cyclesVal.emplace(cycleEnd - cycleStart);
			std::vector<event_count>::operator[](currentIndex).elapsed = endClock - startClock;
			++currentIndex;
			return;
		}
	};

}
#endif