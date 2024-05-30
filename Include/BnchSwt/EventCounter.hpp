
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

#if !defined(_MSC_VER)
	#include <dirent.h>
#endif

#include <BnchSwt/Config.hpp>

#include <cinttypes>
#include <optional>
#include <cstring>
#include <chrono>
#include <vector>
#include <cctype>

#include <BnchSwt/Counters/WindowsEvents.hpp>
#include <BnchSwt/Counters/LinuxPerfEvents.hpp>
#include <BnchSwt/Counters/AppleArmEvents.hpp>

namespace bnch_swt {

	struct event_count {
		std::optional<uint64_t> instructionsVal{};
		std::optional<uint64_t> bytesProcessed{};
		std::optional<uint64_t> missedBranches{};
		std::chrono::duration<double> elapsed{};
		std::optional<uint64_t> branchesVal{};
		std::optional<uint64_t> cpuCycles{};

		BNCH_SWT_INLINE event_count() : elapsed(0) {
		}

		BNCH_SWT_INLINE double elapsed_sec() const {
			return std::chrono::duration<double>(elapsed).count();
		}

		BNCH_SWT_INLINE double elapsed_ns() const {
			return std::chrono::duration<double, std::nano>(elapsed).count();
		}

		BNCH_SWT_INLINE bool cycles(double& cyclesNew) const {
			if (cpuCycles.has_value()) {
				cyclesNew = static_cast<double>(cpuCycles.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_INLINE double instructions(double& instructionsNew) const {
			if (instructionsVal.has_value()) {
				instructionsNew = static_cast<double>(instructionsVal.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_INLINE double branches(double& branchesNew) const {
			if (branchesVal.has_value()) {
				branchesNew = static_cast<double>(branchesVal.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_INLINE double missed_branches(double& missedBranchesNew) const {
			if (missedBranches.has_value()) {
				missedBranchesNew = static_cast<double>(missedBranches.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_INLINE event_count& operator=(const event_count& other) {
			this->missedBranches  = other.missedBranches;
			this->bytesProcessed  = other.bytesProcessed;
			this->instructionsVal = other.instructionsVal;
			this->branchesVal	  = other.branchesVal;
			this->cpuCycles		  = other.cpuCycles;
			this->elapsed		  = other.elapsed;
			return *this;
		}

		BNCH_SWT_INLINE event_count(const event_count& other) {
			*this = other;
		}

		BNCH_SWT_INLINE event_count operator+(const event_count& other) const {
			event_count countNew{};
			if (cpuCycles.has_value() && other.cpuCycles.has_value()) {
				countNew.cpuCycles = cpuCycles.value() + other.cpuCycles.value();
			}
			if (cpuCycles.has_value() && other.cpuCycles.has_value()) {
				countNew.cpuCycles = cpuCycles.value() + other.cpuCycles.value();
			}
			if (cpuCycles.has_value() && other.cpuCycles.has_value()) {
				countNew.cpuCycles = cpuCycles.value() + other.cpuCycles.value();
			}
			if (cpuCycles.has_value() && other.cpuCycles.has_value()) {
				countNew.cpuCycles = cpuCycles.value() + other.cpuCycles.value();
			}
			countNew.elapsed = elapsed + other.elapsed;
			return countNew;
		}

		BNCH_SWT_INLINE event_count& operator+=(const event_count& other) {
			*this = *this + other;
			return *this;
		}
	};

	using event_collector = event_collector_type<event_count>;

}
