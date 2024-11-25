#pragma once

#include <BnchSwt/Config.hpp>
#include <cerrno>// for errno
#include <cstring>// for memset
#include <stdexcept>

#include <iostream>
#include <vector>

enum class event_counter_types { CPU_CYCLES = 0 };

namespace bnch_swt {

	BNCH_SWT_ALWAYS_INLINE size_t rdtsc() {
		return __rdtsc();
	}

	template<typename event_count> struct event_collector_type {
		std::chrono::time_point<std::chrono::steady_clock> start_clock{};
		uint64_t cycleStart{};
		uint64_t cycleEnd{};
		event_count count{};

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
