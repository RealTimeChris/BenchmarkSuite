#pragma once

#include <BnchSwt/Config.hpp>
#include <cerrno>// for errno
#include <cstring>// for memset
#include <stdexcept>

#include <iostream>
#include <vector>

enum class event_counter_types { CPU_CYCLES = 0 };

namespace bnch_swt {

	BNCH_SWT_INLINE size_t rdtsc() {
		return __rdtsc();
	}

	template<typename event_count> struct event_collector_type {
		template<typename function_type, typename... arg_types> BNCH_SWT_ALWAYS_INLINE event_count start(function_type&& function, arg_types&&... args) {
			event_count count{};
			const auto startClock		 = std::chrono::steady_clock::now();
			volatile uint64_t cycleStart = rdtsc();
			count.bytesProcessed.emplace(std::forward<function_type>(function)(std::forward<arg_types>(args)...));
			volatile uint64_t cycleEnd = rdtsc();
			const auto endClock		   = std::chrono::steady_clock::now();
			count.cpuCycles.emplace(cycleEnd - cycleStart);
			count.elapsed = endClock - startClock;
			return count;
		}
	};

}
