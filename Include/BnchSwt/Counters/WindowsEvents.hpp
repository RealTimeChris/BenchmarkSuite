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
#pragma once

#include <BnchSwt/Config.hpp>
#include <cerrno>// for errno
#include <cstring>// for memset
#include <stdexcept>

#include <iostream>
#include <vector>

enum class event_counter_types { CPU_CYCLES = 0 };

namespace bnch_swt {

	BNCH_SWT_ALWAYS_INLINE uint64_t rdtsc() {
		return __rdtsc();
	}

	template<typename event_count> struct event_collector_type {
		std::chrono::time_point<std::chrono::steady_clock> start_clock{};
		volatile uint64_t cycleStart{};
		volatile uint64_t cycleEnd{};
		event_count count{};

		BNCH_SWT_INLINE event_collector_type() {
		}

		BNCH_SWT_INLINE bool has_events() {
			return false;
		}

		BNCH_SWT_INLINE void start() {
			start_clock = std::chrono::steady_clock::now();
			cycleStart	= rdtsc();
		}

		BNCH_SWT_INLINE event_count& end() {
			cycleEnd			 = rdtsc();
			const auto end_clock = std::chrono::steady_clock::now();
			count.cpuCycles.emplace(cycleEnd - cycleStart);
			count.elapsed = end_clock - start_clock;
			return count;
		}
	};

}
