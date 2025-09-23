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

#if !defined(NOMINMAX)
	#define NOMINMAX
#endif

#include <BnchSwt/RandomGenerators.hpp>
#include <BnchSwt/StringLiteral.hpp>
#include <BnchSwt/DoNotOptimize.hpp>
#include <BnchSwt/EventCounter.hpp>
#include <BnchSwt/CacheClearer.hpp>
#include <BnchSwt/FileLoader.hpp>
#include <BnchSwt/Printable.hpp>
#include <BnchSwt/Metrics.hpp>
#include <BnchSwt/Config.hpp>
#include <unordered_map>

namespace bnch_swt {

	template<string_literal stageNameNew, uint64_t maxExecutionCount = 200, uint64_t measuredIterationCount = 25, bool clearCpuCacheBetweenEachIteration = false,
		string_literal metricNameNew = string_literal<1>{}>
	struct benchmark_stage {
		static_assert(maxExecutionCount % measuredIterationCount == 0, "Sorry, but please enter a maxExecutionCount that is divisible by measuredIterationCount.");
		//static_assert(maxExecutionCount > 1, "Sorry, but please enter a maxExecutionCount that is greater than 1.");
		inline static thread_local std::unordered_map<std::string_view, performance_metrics> results{};
		static constexpr bool useNonMbpsMetric{ metricNameNew.size() == 0 };

		BNCH_SWT_INLINE static void printResults(bool showComparison = true, bool showMetrics = true) {
			std::vector<performance_metrics> resultsNew{};
			for (const auto& [key, value]: results) {
				resultsNew.emplace_back(value);
			}
			if (resultsNew.size() > 0) {
				std::sort(resultsNew.begin(), resultsNew.end(), std::greater<performance_metrics>{});
				std::cout << "Performance Metrics for: " << stageNameNew.operator std::string_view() << std::endl;
				if (showMetrics) {
					for (const auto& value: resultsNew) {
						std::cout << "Metrics for: " << value.name << std::endl;
						std::cout << std::fixed << std::setprecision(2);

						static constexpr auto printMetric = []<typename value_type>(const std::string_view& label, value_type&& valueNew) {
							if constexpr (internal::optional_t<value_type>) {
								if (valueNew.has_value()) {
									std::cout << std::left << std::setw(60ull) << label << ": " << valueNew.value() << std::endl;
								}
							} else {
								std::cout << std::left << std::setw(60ull) << label << ": " << valueNew << std::endl;
							}
						};
						std::string instructionCount{};
						std::string throughPutString{};
						std::string cycleCount{};
						std::string metricName{};
						if constexpr (metricNameNew.size() > 0) {
							throughPutString = "Throughput (" + metricNameNew.operator std::string() + "/s)";
							metricName		 = metricNameNew.operator std::string() + "s Processed";
							cycleCount		 = "Cycles per " + metricNameNew.operator std::string();
							instructionCount = "Instructions per " + metricNameNew.operator std::string();
						} else {
							throughPutString = "Throughput (MB/s)";
							metricName		 = "Bytes Processed";
							cycleCount		 = "Cycles per Byte";
							instructionCount = "Instructions per Byte";
						}
						printMetric("Total Iterations", maxExecutionCount);
						printMetric("Measured Iterations", value.measuredIterationCount);
						printMetric("Total Iterations to Stabilization Window", maxExecutionCount - value.totalIterationCount.value());
						printMetric(metricName, value.bytesProcessed);
						printMetric("Nanoseconds per Execution", value.timeInNs);
						printMetric("Frequency (GHz)", value.frequencyGHz);
						printMetric(throughPutString, value.throughputMbPerSec);
						printMetric("Throughput Percentage Deviation (+/-%)", value.throughputPercentageDeviation);
						printMetric("Cycles per Execution", value.cyclesPerExecution);
						printMetric(cycleCount, value.cyclesPerByte);
						printMetric("Instructions per Execution", value.instructionsPerExecution);
						printMetric("Instructions per Cycle", value.instructionsPerCycle);
						printMetric(instructionCount, value.instructionsPerByte);
						printMetric("Branches per Execution", value.branchesPerExecution);
						printMetric("Branch Misses per Execution", value.branchMissesPerExecution);
						printMetric("Cache References per Execution", value.cacheReferencesPerExecution);
						printMetric("Cache Misses per Execution", value.cacheMissesPerExecution);
						std::cout << "----------------------------------------" << std::endl;
					}
				}

				if (showComparison) {
					double difference{};
					for (uint64_t x = 0; x < resultsNew.size() - 1; ++x) {
						difference = ((resultsNew[x].throughputMbPerSec - resultsNew[x + 1].throughputMbPerSec) / resultsNew[x + 1].throughputMbPerSec) * 100.0;
						std::cout << "Library " << resultsNew[x].name << ", is faster than library: " << resultsNew[x + 1].name << ", by roughly: " << difference << "%."
								  << std::endl;
					}
				}
			} else {
				std::cout << "Not enough data for benchmark stage: " << stageNameNew.operator std::string_view() << std::endl;
			}
		}

		template<string_literal subjectNameNew, typename function_type, internal::not_invocable... arg_types>
		BNCH_SWT_INLINE static performance_metrics runBenchmark(arg_types&&... args) {
			static constexpr string_literal subjectName{ subjectNameNew };
			static_assert(std::convertible_to<std::invoke_result_t<decltype(function_type::impl), arg_types...>, uint64_t>,
				"Sorry, but the lambda passed to runBenchmark() must return a uint64_t, reflecting the number of bytes processed!");
			internal::event_collector<maxExecutionCount> events{};
			internal::cache_clearer cacheClearer{};
			performance_metrics lowestResults{};
			performance_metrics resultsTemp{};
			uint64_t currentGlobalIndex{ measuredIterationCount };
			for (uint64_t x = 0; x < maxExecutionCount; ++x) {
				if constexpr (clearCpuCacheBetweenEachIteration) {
					cacheClearer.evictCaches();
				}
				events.template run<function_type>(std::forward<arg_types>(args)...);
			}
			std::span<internal::event_count> newPtr{ static_cast<std::vector<internal::event_count>&>(events) };
			static constexpr uint64_t finalMeasuredIterationCount{ maxExecutionCount - measuredIterationCount > 0 ? maxExecutionCount - measuredIterationCount : 1 };
			for (uint64_t x = 0; x < finalMeasuredIterationCount; ++x, ++currentGlobalIndex) {
				resultsTemp	  = collectMetrics<subjectName, useNonMbpsMetric>(newPtr.subspan(x, measuredIterationCount), currentGlobalIndex);
				lowestResults = resultsTemp.throughputPercentageDeviation < lowestResults.throughputPercentageDeviation ? resultsTemp : lowestResults;
			}
			results[subjectName.operator std::string_view()] = lowestResults;
			return results[subjectName.operator std::string_view()];
		}

		template<string_literal subjectNameNew, typename function_type, internal::not_invocable... arg_types>
		BNCH_SWT_INLINE static performance_metrics runBenchmark(function_type&& functionNew, arg_types&&... args) {
			static constexpr string_literal subjectName{ subjectNameNew };
			static_assert(std::convertible_to<std::invoke_result_t<function_type, arg_types...>, uint64_t>,
				"Sorry, but the lambda passed to runBenchmark() must return a uint64_t, reflecting the number of bytes processed!");
			std::remove_cvref_t<function_type> functionNewer{ std::forward<function_type>(functionNew) };
			internal::event_collector<maxExecutionCount> events{};
			internal::cache_clearer cacheClearer{};
			performance_metrics lowestResults{};
			performance_metrics resultsTemp{};
			uint64_t currentGlobalIndex{ measuredIterationCount };
			for (uint64_t x = 0; x < maxExecutionCount; ++x) {
				if constexpr (clearCpuCacheBetweenEachIteration) {
					cacheClearer.evictCaches();
				}
				events.run(functionNewer, std::forward<arg_types>(args)...);
			}
			std::span<internal::event_count> newPtr{ static_cast<std::vector<internal::event_count>&>(events) };
			static constexpr uint64_t finalMeasuredIterationCount{ maxExecutionCount - measuredIterationCount > 0 ? maxExecutionCount - measuredIterationCount : 1 };
			for (uint64_t x = 0; x < finalMeasuredIterationCount; ++x, ++currentGlobalIndex) {
				resultsTemp	  = collectMetrics<subjectName, useNonMbpsMetric>(newPtr.subspan(x, measuredIterationCount), currentGlobalIndex);
				lowestResults = resultsTemp.throughputPercentageDeviation < lowestResults.throughputPercentageDeviation ? resultsTemp : lowestResults;
			}
			results[subjectName.operator std::string_view()] = lowestResults;
			return results[subjectName.operator std::string_view()];
		}
		static constexpr uint64_t finalMeasuredIterationCount{ maxExecutionCount - measuredIterationCount > 0 ? maxExecutionCount - measuredIterationCount : 1 };

		template<string_literal subjectNameNew, typename prep_function_type, typename function_type, internal::not_invocable... arg_types>
		BNCH_SWT_INLINE static performance_metrics runBenchmarkWithPrep(arg_types&&... args) {
			static constexpr string_literal subjectName{ subjectNameNew };
			//static_assert(std::convertible_to<std::invoke_result_t<decltype(function_type::impl), arg_types...>, uint64_t>,
			//"Sorry, but the lambda passed to runBenchmark() must return a uint64_t, reflecting the number of bytes processed!");
			internal::event_collector<maxExecutionCount> events{};
			internal::cache_clearer cacheClearer{};
			uint64_t currentGlobalIndex{ measuredIterationCount };
			for (uint64_t x = 0; x < maxExecutionCount; ++x) {
				prep_function_type::impl(std::forward<arg_types>(args)...);
				if constexpr (clearCpuCacheBetweenEachIteration) {
					cacheClearer.evictCaches();
				}
				events.template run<function_type>(std::forward<arg_types>(args)...);
			}
			std::span<internal::event_count> newPtr{ static_cast<std::vector<internal::event_count>&>(events) };
			results[subjectName.operator std::string_view()] = sort_results<subjectName>(currentGlobalIndex, newPtr);
			return results[subjectName.operator std::string_view()];
		}

		template<string_literal subjectNameNew> BNCH_SWT_INLINE static auto sort_results(uint64_t currentGlobalIndex, auto& newPtr) {
			performance_metrics lowestResults{};
			performance_metrics resultsTemp{};
			if constexpr (measuredIterationCount == 1) {
				std::vector<performance_metrics> vector{};
				for (uint64_t x = 0; x < finalMeasuredIterationCount; ++x, ++currentGlobalIndex) {
					vector.emplace_back(collectMetrics<subjectNameNew, useNonMbpsMetric>(newPtr.subspan(x, measuredIterationCount), currentGlobalIndex));
				}
				std::sort(vector.data(), vector.data() + vector.size(), std::less<performance_metrics>{});
				return vector[0];
			} else {
				for (uint64_t x = 0; x < finalMeasuredIterationCount; ++x, ++currentGlobalIndex) {
					resultsTemp	  = collectMetrics<subjectNameNew, useNonMbpsMetric>(newPtr.subspan(x, measuredIterationCount), currentGlobalIndex);
					lowestResults = resultsTemp.throughputPercentageDeviation < lowestResults.throughputPercentageDeviation ? resultsTemp : lowestResults;
				}
			}
			return lowestResults;
		}

		template<string_literal subjectNameNew, typename prep_function_type, typename function_type, internal::not_invocable... arg_types>
		BNCH_SWT_INLINE static performance_metrics runBenchmarkWithPrep(prep_function_type&& prepFunctionNew, function_type&& functionNew, arg_types&&... args) {
			static constexpr string_literal subjectName{ subjectNameNew };
			static_assert(std::convertible_to<std::invoke_result_t<function_type, arg_types...>, uint64_t>,
				"Sorry, but the lambda passed to runBenchmarkWithPrep() must return a uint64_t, reflecting the number of bytes processed!");
			std::remove_cvref_t<prep_function_type> prepFunctionNewer{ std::forward<prep_function_type>(prepFunctionNew) };
			std::remove_cvref_t<function_type> functionNewer{ std::forward<function_type>(functionNew) };
			internal::event_collector<maxExecutionCount> events{};
			internal::cache_clearer cacheClearer{};
			uint64_t currentGlobalIndex{ measuredIterationCount };
			for (uint64_t x = 0; x < maxExecutionCount; ++x) {
				prepFunctionNewer();
				if constexpr (clearCpuCacheBetweenEachIteration) {
					cacheClearer.evictCaches();
				}
				events.run(functionNewer, std::forward<arg_types>(args)...);
			}
			std::span<internal::event_count> newPtr{ static_cast<std::vector<internal::event_count>&>(events) };
			results[subjectName.operator std::string_view()] = sort_results<subjectName>(currentGlobalIndex, newPtr);
			return results[subjectName.operator std::string_view()];
		}
	};

}
