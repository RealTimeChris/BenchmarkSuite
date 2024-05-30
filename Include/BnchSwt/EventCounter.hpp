
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

#include <BnchSwt/Counters/WindowsPerfEvents.hpp>
#include <BnchSwt/Counters/LinuxPerfEvents.hpp>
#include <BnchSwt/Counters/AppleArmPerfEvents.hpp>

namespace bnch_swt {

	struct event_count {
		template<typename value_type> friend struct event_collector_type;

		BNCH_SWT_ALWAYS_INLINE event_count() noexcept = default; 

		BNCH_SWT_ALWAYS_INLINE double elapsedNs() const noexcept {
			return std::chrono::duration<double, std::nano>(elapsed).count();
		}

		BNCH_SWT_ALWAYS_INLINE bool cycles(double& cyclesNew) const {
			if (cyclesVal.has_value()) {
				cyclesNew = static_cast<double>(cyclesVal.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_ALWAYS_INLINE bool instructions(double& instructionsNew) const noexcept {
			if (instructionsVal.has_value()) {
				instructionsNew = static_cast<double>(instructionsVal.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_ALWAYS_INLINE bool branches(double& branchesNew) const noexcept {
			if (branchesVal.has_value()) {
				branchesNew = static_cast<double>(branchesVal.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_ALWAYS_INLINE bool missedBranches(double& missedBranchesNew) const noexcept {
			if (missedBranchesVal.has_value()) {
				missedBranchesNew = static_cast<double>(missedBranchesVal.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_ALWAYS_INLINE bool bytesProcessed(double& bytesProcessedNew) const noexcept {
			if (bytesProcessedVal.has_value()) {
				bytesProcessedNew = static_cast<double>(bytesProcessedVal.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_ALWAYS_INLINE event_count& operator=(const event_count& other) noexcept {
			if (other.instructionsVal.has_value()) {
				instructionsVal.emplace(other.instructionsVal.value());
			}
			if (other.bytesProcessedVal.has_value()) {
				bytesProcessedVal.emplace(other.bytesProcessedVal.value());
			}
			if (other.missedBranchesVal.has_value()) {
				missedBranchesVal.emplace(other.missedBranchesVal.value());
			}
			if (other.missedBranchesVal.has_value()) {
				missedBranchesVal.emplace(other.missedBranchesVal.value());
			}
			if (other.branchesVal.has_value()) {
				branchesVal.emplace(other.branchesVal.value());
			}
			if (other.cyclesVal.has_value()) {
				cyclesVal.emplace(other.cyclesVal.value());
			}
			this->elapsed		  = other.elapsed;
			return *this;
		}

		BNCH_SWT_ALWAYS_INLINE event_count(const event_count& other) noexcept {
			*this = other;
		}

		BNCH_SWT_ALWAYS_INLINE event_count operator+(const event_count& other) const noexcept {
			event_count countNew{};
			if (instructionsVal.has_value() && other.instructionsVal.has_value()) {
				countNew.instructionsVal.emplace(instructionsVal.value() + other.instructionsVal.value());
			}
			if (bytesProcessedVal.has_value() && other.bytesProcessedVal.has_value()) {
				countNew.bytesProcessedVal.emplace(bytesProcessedVal.value() + other.bytesProcessedVal.value());
			}
			if (branchesVal.has_value() && other.branchesVal.has_value()) {
				countNew.branchesVal.emplace(branchesVal.value() + other.branchesVal.value());
			}
			if (missedBranchesVal.has_value() && other.missedBranchesVal.has_value()) {
				countNew.missedBranchesVal.emplace(missedBranchesVal.value() + other.missedBranchesVal.value());
			}
			if (cyclesVal.has_value() && other.cyclesVal.has_value()) {
				countNew.cyclesVal.emplace(cyclesVal.value() + other.cyclesVal.value());
			}
			countNew.elapsed = elapsed + other.elapsed;
			return countNew;
		}

		BNCH_SWT_ALWAYS_INLINE event_count& operator+=(const event_count& other) {
			*this = *this + other;
			return *this;
		}

	  protected:
		std::optional<uint64_t> missedBranchesVal{};
		std::optional<uint64_t> bytesProcessedVal{};
		std::optional<uint64_t> instructionsVal{};
		std::chrono::duration<double> elapsed{};
		std::optional<uint64_t> branchesVal{};
		std::optional<uint64_t> cyclesVal{};
	};

	using event_collector = event_collector_type<event_count>;

}
