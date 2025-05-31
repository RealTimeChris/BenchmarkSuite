// BnchSwt/Counters/AndroidCounters.hpp
#pragma once

#include <BnchSwt/Config.hpp>
#include <vector>
#include <chrono>

#if defined(BNCH_SWT_ANDROID)

namespace bnch_swt::internal {

	template<typename event_count_t, size_t count_t> struct event_collector_type : public std::vector<event_count_t> {
		size_t currentIndex{};

		BNCH_SWT_INLINE event_collector_type() : std::vector<event_count_t>{ count_t } {};

		template<typename function_type, typename... arg_types> BNCH_SWT_INLINE void run(function_type&& function, arg_types&&... args) {
			const auto startClock = std::chrono::high_resolution_clock::now();
			std::vector<event_count_t>::operator[](currentIndex)
				.bytesProcessedVal.emplace(static_cast<size_t>(std::forward<function_type>(function)(std::forward<arg_types>(args)...)));
			const auto endClock = std::chrono::high_resolution_clock::now();
			std::vector<event_count_t>::operator[](currentIndex).elapsed = endClock - startClock;
			++currentIndex;
			return;
		}
	};

}

#endif