#pragma once
// Sampled mostly from https://github.com/fastfloat/fast_float

#include <BnchSwt/Config.hpp>

#include <asm/unistd.h>// for __NR_perf_event_open
#include <linux/perf_event.h>// for perf event constants
#include <sys/ioctl.h>// for ioctl
#include <unistd.h>// for syscall

#include <cerrno>// for errno
#include <cstring>// for memset
#include <stdexcept>

#include <iostream>
#include <vector>

namespace bnch_swt {

	template<int32_t TYPE = PERF_TYPE_HARDWARE> class linux_events {
		int32_t fd;
		bool working;
		perf_event_attr attribs{};
		size_t num_events{};
		std::vector<uint64_t> temp_result_vec{};
		std::vector<uint64_t> ids{};

	  public:
		explicit linux_events(std::vector<int32_t> config_vec) : fd(0), working(true) {
			memset(&attribs, 0, sizeof(attribs));
			attribs.type		   = TYPE;
			attribs.size		   = sizeof(attribs);
			attribs.disabled	   = 1;
			attribs.exclude_kernel = 1;
			attribs.exclude_hv	   = 1;

			attribs.sample_period	  = 0;
			attribs.read_format		  = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
			const int32_t pid		  = 0;// the current process
			const int32_t cpu		  = -1;// all CPUs
			const unsigned long flags = 0;

			int32_t group = -1;// no group
			num_events	  = config_vec.size();
			ids.resize(config_vec.size());
			uint32_t i = 0;
			for (auto config: config_vec) {
				attribs.config = config;
				int32_t _fd	   = static_cast<int32_t>(syscall(__NR_perf_event_open, &attribs, pid, cpu, group, flags));
				if (_fd == -1) {
					report_error("perf_event_open");
				}
				ioctl(_fd, PERF_EVENT_IOC_ID, &ids[i++]);
				if (group == -1) {
					group = _fd;
					fd	  = _fd;
				}
			}

			temp_result_vec.resize(num_events * 2 + 1);
		}

		~linux_events() {
			if (fd != -1) {
				close(fd);
			}
		}

		inline void start() {
			if (fd != -1) {
				if (ioctl(fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) == -1) {
					report_error("ioctl(PERF_EVENT_IOC_RESET)");
				}

				if (ioctl(fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == -1) {
					report_error("ioctl(PERF_EVENT_IOC_ENABLE)");
				}
			}
		}

		inline void end(std::vector<uint64_t>& results) {
			if (fd != -1) {
				if (ioctl(fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) == -1) {
					report_error("ioctl(PERF_EVENT_IOC_DISABLE)");
				}

				if (read(fd, temp_result_vec.data(), temp_result_vec.size() * 8) == -1) {
					report_error("read");
				}
			}
			// our actual results are in slots 1,3,5, ... of this structure
			for (uint32_t i = 1; i < temp_result_vec.size(); i += 2) {
				results[i / 2] = temp_result_vec[i];
			}
			for (uint32_t i = 2; i < temp_result_vec.size(); i += 2) {
				if (ids[i / 2 - 1] != temp_result_vec[i]) {
					report_error("event mismatch");
				}
			}
		}

		bool is_working() {
			return working;
		}

	  private:
		void report_error(const std::string&) {
			working = false;
		}
	};

	template<typename event_count> struct event_collector_type {
		event_count count{};
		std::chrono::time_point<std::chrono::steady_clock> start_clock{};
		std::vector<uint64_t> results{};

		linux_events<PERF_TYPE_HARDWARE> linuxEvents;
		event_collector_type()
			: linuxEvents(std::vector<int32_t>{ PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS,
				  PERF_COUNT_HW_BRANCH_INSTRUCTIONS,// Retired branch instructions
				  PERF_COUNT_HW_BRANCH_MISSES }) {
		}
		bool has_events() {
			return linuxEvents.is_working();
		}

		inline void start() {
			linuxEvents.start();
			start_clock = std::chrono::steady_clock::now();
		}

		inline event_count& end() {
			const auto end_clock = std::chrono::steady_clock::now();
			linuxEvents.end(results);
			if (results.size() >= 4) {
				count.cpuCycles.emplace(results[0]);
				count.instructionsVal.emplace(results[1]);
				count.branchesVal.emplace(results[2]);
				count.missedBranches.emplace(results[3]);
			}
			count.elapsed = end_clock - start_clock;
			return count;
		}
	};

}