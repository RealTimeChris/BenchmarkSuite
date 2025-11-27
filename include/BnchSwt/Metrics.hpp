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
/// https://github.com/RealTimeChris/BenchmarkSuite
#pragma once

#include <BnchSwt/EventCounter.hpp>
#include <cstdint>
#include <span>

namespace bnch_swt {	

	BNCH_SWT_HOST double calculateThroughputMBps(double nanoseconds, double bytesProcessed) {
		constexpr double bytesPerMB		= 1024.0 * 1024.0;
		constexpr double nanosPerSecond = 1e9;
		double megabytes				= bytesProcessed / bytesPerMB;
		double seconds					= nanoseconds / nanosPerSecond;
		if (seconds == 0.0) {
			return 0.0;
		}
		return megabytes / seconds;
	}

	BNCH_SWT_HOST double calculateUnitsPs(double nanoseconds, double bytesProcessed) {
		return (bytesProcessed * 1000000000.0) / nanoseconds;
	}

	template<benchmark_types benchmark_type = benchmark_types::cpu> struct performance_metrics {
		double throughputPercentageDeviation{ std::numeric_limits<double>::max() };
		std::optional<double> cacheReferencesPerExecution{};
		std::optional<uint64_t> measuredIterationCount{};
		std::optional<double> instructionsPerExecution{};
		std::optional<double> branchMissesPerExecution{};
		std::optional<double> cacheMissesPerExecution{};
		std::optional<uint64_t> totalIterationCount{};
		std::optional<double> instructionsPerCycle{};
		std::optional<double> branchesPerExecution{};
		std::optional<double> instructionsPerByte{};
		std::optional<double> cyclesPerExecution{};
		std::optional<double> cyclesPerByte{};
		std::optional<double> frequencyGHz{};
		double throughputMbPerSec{};
		uint64_t bytesProcessed{};
		std::string name{};
		double timeInNs{};

		BNCH_SWT_HOST bool operator>(const performance_metrics& other) const {
			return throughputMbPerSec > other.throughputMbPerSec;
		}

		template<string_literal benchmarkNameNew, bool mbps = true>
		BNCH_SWT_HOST static performance_metrics collectMetrics(std::span<internal::event_count<benchmark_type>>&& eventsNewer, size_t totalIterationCount) {
			static constexpr string_literal benchmarkName{ benchmarkNameNew };
			performance_metrics metrics{};
			metrics.name = benchmarkName.operator std::string();
			metrics.measuredIterationCount.emplace(eventsNewer.size());
			metrics.totalIterationCount.emplace(totalIterationCount);
			double throughPut{};
			double throughPutTotal{};
			double throughPutAvg{};
			double throughPutMin{ std::numeric_limits<double>::max() };
			uint64_t bytesProcessed{};
			uint64_t bytesProcessedTotal{};
			uint64_t bytesProcessedAvg{};
			double ns{};
			double nsTotal{};
			double nsAvg{};
			double cycles{};
			double cyclesTotal{};
			double cyclesAvg{};
			double instructions{};
			double instructionsTotal{};
			double instructionsAvg{};
			double branches{};
			double branchesTotal{};
			double branchesAvg{};
			double branchMisses{};
			double branchMissesTotal{};
			double branchMissesAvg{};
			double cacheReferences{};
			double cacheReferencesTotal{};
			double cacheReferencesAvg{};
			double cacheMisses{};
			double cacheMissesTotal{};
			double cacheMissesAvg{};
			for (const internal::event_count<benchmark_type>& e: eventsNewer) {
				ns = e.elapsedNs();
				nsTotal += ns;

				if (e.bytesProcessed(bytesProcessed)) {
					bytesProcessedTotal += bytesProcessed;
					if constexpr (mbps) {
						throughPut = calculateThroughputMBps(ns, static_cast<double>(bytesProcessed));
					} else {
						throughPut = calculateUnitsPs(ns, static_cast<double>(bytesProcessed));
					}
					throughPutTotal += throughPut;
					throughPutMin = throughPut < throughPutMin ? throughPut : throughPutMin;
				}

				if (e.cycles(cycles)) {
					cyclesTotal += cycles;
				}

				if (e.instructions(instructions)) {
					instructionsTotal += instructions;
				}

				if (e.branches(branches)) {
					branchesTotal += branches;
				}

				if (e.branchMisses(branchMisses)) {
					branchMissesTotal += branchMisses;
				}

				if (e.cacheReferences(cacheReferences)) {
					cacheReferencesTotal += cacheReferences;
				}

				if (e.cacheMisses(cacheMisses)) {
					cacheMissesTotal += cacheMisses;
				}
			}
			if (eventsNewer.size() > 0) {
				bytesProcessedAvg  = bytesProcessedTotal / eventsNewer.size();
				nsAvg			   = nsTotal / static_cast<double>(eventsNewer.size());
				throughPutAvg	   = throughPutTotal / static_cast<double>(eventsNewer.size());
				cyclesAvg		   = cyclesTotal / static_cast<double>(eventsNewer.size());
				instructionsAvg	   = instructionsTotal / static_cast<double>(eventsNewer.size());
				branchesAvg		   = branchesTotal / static_cast<double>(eventsNewer.size());
				branchMissesAvg	   = branchMissesTotal / static_cast<double>(eventsNewer.size());
				cacheReferencesAvg = cacheReferencesTotal / static_cast<double>(eventsNewer.size());
				cacheMissesAvg	   = cacheMissesTotal / static_cast<double>(eventsNewer.size());
				metrics.timeInNs   = nsAvg;
			} else {
				return {};
			}

			constexpr double epsilon = 1e-6;
			if (std::abs(nsAvg) > epsilon) {
				metrics.bytesProcessed				  = bytesProcessedAvg;
				metrics.throughputMbPerSec			  = throughPutAvg;
				metrics.throughputPercentageDeviation = ((throughPutAvg - throughPutMin) * 100.0) / throughPutAvg;
			}
			if (std::abs(cyclesAvg) > epsilon) {
				if (metrics.bytesProcessed > 0) {
					metrics.cyclesPerByte.emplace(cyclesAvg / static_cast<double>(metrics.bytesProcessed));
				}
				metrics.cyclesPerExecution.emplace(cyclesTotal / static_cast<double>(eventsNewer.size()));
				metrics.frequencyGHz.emplace(cyclesAvg / nsAvg);
			}
			if (std::abs(instructionsAvg) > epsilon) {
				if (metrics.bytesProcessed > 0) {
					metrics.instructionsPerByte.emplace(instructionsAvg / static_cast<double>(metrics.bytesProcessed));
				}
				if (std::abs(cyclesAvg) > epsilon) {
					metrics.instructionsPerCycle.emplace(instructionsAvg / cyclesAvg);
				}
				metrics.instructionsPerExecution.emplace(instructionsTotal / static_cast<double>(eventsNewer.size()));
			}
			if (std::abs(branchesAvg) > epsilon) {
				metrics.branchMissesPerExecution.emplace(branchMissesAvg / static_cast<double>(eventsNewer.size()));
				metrics.branchesPerExecution.emplace(branchesAvg / static_cast<double>(eventsNewer.size()));
			}
			if (std::abs(cacheMissesAvg) > epsilon) {
				metrics.cacheMissesPerExecution.emplace(cacheMissesAvg / static_cast<double>(eventsNewer.size()));
			}
			if (std::abs(cacheReferencesAvg) > epsilon) {
				metrics.cacheReferencesPerExecution.emplace(cacheReferencesAvg / static_cast<double>(eventsNewer.size()));
			}
			return metrics;
		}
	};
}