#pragma once

#include <cctype>
#ifndef _MSC_VER
	#include <dirent.h>
#endif

#include <BnchSwt/Config.hpp>

#include <cinttypes>
#include <optional>
#include <cstring>
#include <chrono>
#include <vector>

#if defined(WIN32) || defined(WIN64)
	#include <BnchSwt/Counters/WindowsEvents.hpp>
#endif

#if defined(__linux__)
	#include <BnchSwt/Counters/LinuxPerfEvents.hpp>
	#include <libgen.h>
#endif

#if __APPLE__ && __aarch64__
	#include <BnchSwt/Counters/AppleArmEvents.hpp>
#endif

namespace bnch_swt {

	struct event_count {
		std::optional<uint64_t> instructionsVal{};
		std::optional<uint64_t> missedBranches{};
		std::chrono::duration<double> elapsed{};
		std::optional<uint64_t> branchesVal{};
		std::optional<uint64_t> cpuCycles{};

		BNCH_SWT_ALWAYS_INLINE event_count() : elapsed(0) {
		}

		BNCH_SWT_ALWAYS_INLINE double elapsed_sec() const {
			return std::chrono::duration<double>(elapsed).count();
		}

		BNCH_SWT_ALWAYS_INLINE double elapsed_ns() const {
			return std::chrono::duration<double, std::nano>(elapsed).count();
		}

		BNCH_SWT_ALWAYS_INLINE bool cycles(double& cyclesNew) const {
			if (cpuCycles.has_value()) {
				cyclesNew = cpuCycles.value();
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_ALWAYS_INLINE double instructions(double& instructionsNew) const {
			if (instructionsVal.has_value()) {
				instructionsNew = instructionsVal.value();
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_ALWAYS_INLINE double branches(double& branchesNew) const {
			if (branchesVal.has_value()) {
				branchesNew = branchesVal.value();
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_ALWAYS_INLINE double missed_branches(double& missedBranchesNew) const {
			if (missedBranches.has_value()) {
				missedBranchesNew = missedBranches.value();
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_ALWAYS_INLINE event_count& operator=(const event_count& other) {
			this->missedBranches  = other.missedBranches;
			this->instructionsVal = other.instructionsVal;
			this->branchesVal	  = other.branchesVal;
			this->cpuCycles		  = other.cpuCycles;
			this->elapsed		  = other.elapsed;
			return *this;
		}

		BNCH_SWT_ALWAYS_INLINE event_count operator+(const event_count& other) const {
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

		BNCH_SWT_ALWAYS_INLINE event_count& operator+=(const event_count& other) {
			*this = *this + other;
			return *this;
		}
	};

	using event_collector = event_collector_type<event_count>;

}
