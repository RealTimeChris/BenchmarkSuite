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
// Sampled mostly from https://github.com/fastfloat/fast_float
#pragma once

#include <BnchSwt/Config.hpp>

#if defined(BNCH_SWT_LINUX)

	#include <linux/perf_event.h>
	#include <asm/unistd.h>
	#include <sys/ioctl.h>
	#include <unistd.h>
	#include <cstring>
	#include <vector>

namespace bnch_swt::internal {

	BNCH_SWT_INLINE uint64_t rdtsc() {
	#if defined(__x86_64__)
		uint32_t a, d;
		__asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
		return static_cast<unsigned long>(a) | (static_cast<unsigned long>(d) << 32);
	#elif defined(__i386__)
		uint64_t x;
		__asm__ volatile("rdtsc" : "=A"(x));
		return x;
	#else
		return 0;
	#endif
	}

	class linux_events {
	  protected:
		std::vector<uint64_t> temp_result_vec{};
		std::vector<uint64_t> ids{};
		perf_event_attr attribs{};
		uint64_t num_events{};
		bool working{};
		int32_t fd{};

	  public:
		BNCH_SWT_INLINE explicit linux_events(std::vector<int32_t> config_vec) : working(true) {
			memset(&attribs, 0, sizeof(attribs));
			attribs.type		   = PERF_TYPE_HARDWARE;
			attribs.size		   = sizeof(attribs);
			attribs.disabled	   = 1;
			attribs.exclude_kernel = 1;
			attribs.exclude_hv	   = 1;

			attribs.sample_period	  = 0;
			attribs.read_format		  = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
			const int32_t pid		  = 0;
			const int32_t cpu		  = -1;
			const unsigned long flags = 0;

			int32_t group = -1;
			num_events	  = config_vec.size();
			ids.resize(config_vec.size());
			uint32_t i = 0;
			for (auto config: config_vec) {
				attribs.config = static_cast<long long unsigned int>(config);
				int32_t _fd	   = static_cast<int32_t>(syscall(__NR_perf_event_open, &attribs, pid, cpu, group, flags));
				if (_fd == -1) {
					reportError("perf_event_open");
				}
				ioctl(_fd, PERF_EVENT_IOC_ID, &ids[i++]);
				if (group == -1) {
					group = _fd;
					fd	  = _fd;
				}
			}

			temp_result_vec.resize(num_events * 2 + 1);
		}

		BNCH_SWT_INLINE ~linux_events() {
			if (fd != -1) {
				close(fd);
			}
		}

		BNCH_SWT_INLINE void run() {
			if (fd != -1) {
				if (ioctl(fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) == -1) {
					reportError("ioctl(PERF_EVENT_IOC_RESET)");
				}

				if (ioctl(fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == -1) {
					reportError("ioctl(PERF_EVENT_IOC_ENABLE)");
				}
			}
		}

		BNCH_SWT_INLINE void end(std::vector<uint64_t>& results) {
			if (fd != -1) {
				if (ioctl(fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) == -1) {
					reportError("ioctl(PERF_EVENT_IOC_DISABLE)");
				}

				if (read(fd, temp_result_vec.data(), temp_result_vec.size() * 8) == -1) {
					reportError("read");
				}
			}

			for (uint32_t i = 1; i < temp_result_vec.size(); i += 2) {
				results[i / 2] = temp_result_vec[i];
			}
			for (uint32_t i = 2; i < temp_result_vec.size(); i += 2) {
				if (ids[i / 2 - 1] != temp_result_vec[i]) {
					reportError("event mismatch");
				}
			}
		}

		bool isWorking() {
			return working;
		}

	  protected:
		BNCH_SWT_INLINE void reportError(const std::string&) {
			working = false;
		}
	};

	template<typename event_count, uint64_t count> struct event_collector_type : public linux_events, public std::vector<event_count> {
		std::vector<uint64_t> results{};
		uint64_t currentIndex{};
		BNCH_SWT_INLINE event_collector_type()
			: linux_events{ std::vector<int32_t>{ PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, PERF_COUNT_HW_BRANCH_MISSES,
				  PERF_COUNT_HW_CACHE_REFERENCES, PERF_COUNT_HW_CACHE_MISSES } },
			  std::vector<event_count>{ count } {}

		BNCH_SWT_INLINE bool hasEvents() {
			return linux_events::isWorking();
		}

		template<typename function_type, typename... arg_types> BNCH_SWT_INLINE void run(arg_types&&... args) {
			if (hasEvents()) {
				linux_events::run();
			}
			uint64_t result;
			const auto startClock		 = clock_type::now();
			volatile uint64_t cycleStart = rdtsc();
			result						 = static_cast<uint64_t>(function_type::impl(std::forward<arg_types>(args)...));
			volatile uint64_t cycleEnd	 = rdtsc();
			const auto endClock			 = clock_type::now();
			std::vector<event_count>::operator[](currentIndex).cyclesVal.emplace(cycleEnd - cycleStart);
			std::vector<event_count>::operator[](currentIndex).elapsed = endClock - startClock;
			std::vector<event_count>::operator[](currentIndex).bytesProcessedVal.emplace(result);
			if (hasEvents()) {
				if (results.size() != linux_events::temp_result_vec.size()) {
					results.resize(linux_events::temp_result_vec.size());
				}
				linux_events::end(results);
				std::vector<event_count>::operator[](currentIndex).instructionsVal.emplace(results[1]);
				std::vector<event_count>::operator[](currentIndex).branchesVal.emplace(results[2]);
				std::vector<event_count>::operator[](currentIndex).branchMissesVal.emplace(results[3]);
				std::vector<event_count>::operator[](currentIndex).cacheReferencesVal.emplace(results[4]);
				std::vector<event_count>::operator[](currentIndex).cacheMissesVal.emplace(results[5]);
			}
			++currentIndex;
			return;
		}

		template<typename function_type, typename... arg_types> BNCH_SWT_INLINE void run(function_type&& function, arg_types&&... args) {
			if (hasEvents()) {
				linux_events::run();
			}
			uint64_t result;
			const auto startClock		 = clock_type::now();
			volatile uint64_t cycleStart = rdtsc();
			result						 = static_cast<uint64_t>(std::forward<function_type>(function)(std::forward<arg_types>(args)...));
			volatile uint64_t cycleEnd	 = rdtsc();
			const auto endClock			 = clock_type::now();
			std::vector<event_count>::operator[](currentIndex).cyclesVal.emplace(cycleEnd - cycleStart);
			std::vector<event_count>::operator[](currentIndex).elapsed = endClock - startClock;
			std::vector<event_count>::operator[](currentIndex).bytesProcessedVal.emplace(result);
			if (hasEvents()) {
				if (results.size() != linux_events::temp_result_vec.size()) {
					results.resize(linux_events::temp_result_vec.size());
				}
				linux_events::end(results);
				std::vector<event_count>::operator[](currentIndex).instructionsVal.emplace(results[1]);
				std::vector<event_count>::operator[](currentIndex).branchesVal.emplace(results[2]);
				std::vector<event_count>::operator[](currentIndex).branchMissesVal.emplace(results[3]);
				std::vector<event_count>::operator[](currentIndex).cacheReferencesVal.emplace(results[4]);
				std::vector<event_count>::operator[](currentIndex).cacheMissesVal.emplace(results[5]);
			}
			++currentIndex;
			return;
		}
	};
}

#endif
