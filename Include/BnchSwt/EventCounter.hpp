// BnchSwt/EventCounter.hpp
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

#include <BnchSwt/Config.hpp>
#include <BnchSwt/Counters/AppleArmPerfEvents.hpp>
#include <BnchSwt/Counters/WindowsPerfEvents.hpp>
#include <BnchSwt/Counters/LinuxPerfEvents.hpp>
#include <BnchSwt/Counters/AndriodEvents.hpp>
#include <optional>
#include <chrono>

namespace bnch_swt::internal {

	struct event_count {
		template<typename value_type, size_t count> friend struct event_collector_type;

		BNCH_SWT_INLINE event_count() noexcept = default;

		BNCH_SWT_INLINE double elapsedNs() const noexcept {
			return std::chrono::duration<double, std::nano>(elapsed).count();
		}

		BNCH_SWT_INLINE bool bytesProcessed(uint64_t& bytesProcessedNew) const noexcept {
			if (bytesProcessedVal.has_value()) {
				bytesProcessedNew = bytesProcessedVal.value();
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_INLINE bool cycles(double& cyclesNew) const {
			if (cyclesVal.has_value()) {
				cyclesNew = static_cast<double>(cyclesVal.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_INLINE bool instructions(double& instructionsNew) const noexcept {
			if (instructionsVal.has_value()) {
				instructionsNew = static_cast<double>(instructionsVal.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_INLINE bool branches(double& branchesNew) const noexcept {
			if (branchesVal.has_value()) {
				branchesNew = static_cast<double>(branchesVal.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_INLINE bool branchMisses(double& branchMissesNew) const noexcept {
			if (branchMissesVal.has_value()) {
				branchMissesNew = static_cast<double>(branchMissesVal.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_INLINE bool cacheMisses(double& cacheMissesNew) const noexcept {
			if (cacheMissesVal.has_value()) {
				cacheMissesNew = static_cast<double>(cacheMissesVal.value());
				return true;
			} else {
				return false;
			}
		}

		BNCH_SWT_INLINE bool cacheReferences(double& cacheReferencesNew) const noexcept {
			if (cacheReferencesVal.has_value()) {
				cacheReferencesNew = static_cast<double>(cacheReferencesVal.value());
				return true;
			} else {
				return false;
			}
		}

	  protected:
		std::optional<uint64_t> cacheReferencesVal{};
		std::optional<uint64_t> bytesProcessedVal{};
		std::optional<uint64_t> branchMissesVal{};
		std::optional<uint64_t> instructionsVal{};
		std::optional<uint64_t> cacheMissesVal{};
		std::chrono::duration<double> elapsed{};
		std::optional<uint64_t> branchesVal{};
		std::optional<uint64_t> cyclesVal{};
	};

	template<size_t count> using event_collector = event_collector_type<event_count, count>;

}