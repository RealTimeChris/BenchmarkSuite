#pragma once

#include <cctype>
#ifndef _MSC_VER
	#include <dirent.h>
#endif
#include <cinttypes>

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
		std::chrono::duration<double> elapsed;
		std::optional<uint64_t> missedBranches{};
		std::optional<uint64_t> instructionsVal{};
		std::optional<uint64_t> cpuCycles{};
		std::optional<uint64_t> branchesVal{};
		event_count() : elapsed(0) {
		}
		event_count(const std::chrono::duration<double> _elapsed, const std::vector<uint64_t> _event_counts) : elapsed(_elapsed) {
		}
		event_count(const event_count& other)
			: elapsed(other.elapsed), missedBranches(other.missedBranches), instructionsVal{ other.instructionsVal }, cpuCycles{ other.cpuCycles },
			  branchesVal{ other.branchesVal } {
		}

		double elapsed_sec() const {
			return std::chrono::duration<double>(elapsed).count();
		}

		double elapsed_ns() const {
			return std::chrono::duration<double, std::nano>(elapsed).count();
		}

		bool cycles(double& cyclesNew) const {
			if (cpuCycles.has_value()) {
				cyclesNew = cpuCycles.value();
				return true;
			} else {
				return false;
			}
		}

		double instructions(double& instructionsNew) const {
			if (instructionsVal.has_value()) {
				instructionsNew = instructionsVal.value();
				return true;
			} else {
				return false;
			}
		}

		double branches(double& branchesNew) const {
			if (branchesVal.has_value()) {
				branchesNew = branchesVal.value();
				return true;
			} else {
				return false;
			}
		}

		double missed_branches(double& missedBranchesNew) const {
			if (missedBranches.has_value()) {
				missedBranchesNew = missedBranches.value();
				return true;
			} else {
				return false;
			}
		}

		event_count& operator=(const event_count& other) {
			this->missedBranches = other.missedBranches;
			this->instructionsVal = other.instructionsVal;
			this->branchesVal	 = other.branchesVal;
			this->cpuCycles		 = other.cpuCycles;
			this->elapsed	   = other.elapsed;
			return *this;
		}
		event_count operator+(const event_count& other) const {
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

		void operator+=(const event_count& other) {
			*this = *this + other;
		}
	};

	struct event_aggregate {
		bool has_events	   = false;
		int32_t iterations = 0;
		event_count total{};
		event_count best{};
		event_count worst{};

		event_aggregate() = default;

		void operator<<(const event_count& other) {
			if (iterations == 0 || other.elapsed < best.elapsed) {
				best = other;
			}
			if (iterations == 0 || other.elapsed > worst.elapsed) {
				worst = other;
			}
			iterations++;
			total += other;
		}

		double elapsed_sec() const {
			return total.elapsed_sec() / iterations;
		}
		double elapsed_ns() const {
			return total.elapsed_ns() / iterations;
		}
		bool cycles(double& cyclesNew) const {
			if (total.cycles(cyclesNew)) {
				cyclesNew = cyclesNew / iterations;
				return true;
			} else {
				return false;
			}
		}
		bool instructions(double& instructionsNew) const {
			if (total.instructions(instructionsNew)) {
				instructionsNew = instructionsNew / iterations;
				return true;
			} else {
				return false;
			}
		}
		bool branches(double& branchesNew) const {
			if (total.branches(branchesNew)) {
				branchesNew = branchesNew / iterations;
				return true;
			} else {
				return false;
			}
		}
		bool missed_branches(double& missedBranchesNew) const {
			if (total.missed_branches(missedBranchesNew)) {
				missedBranchesNew = missedBranchesNew / iterations;
				return true;
			} else {
				return false;
			}
		}
	};

	using event_collector = event_collector_type<event_count>;

}
