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

#include <BnchSwt/StringLiteral.hpp>
#include <BnchSwt/EventCounter.hpp>
#include <BnchSwt/DoNotOptimize.hpp>
#include <BnchSwt/CacheClearer.hpp>
#include <BnchSwt/Printable.hpp>
#include <BnchSwt/Config.hpp>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <random>
#include <vector>
#include <map>
#include <set>

namespace bnch_swt {

	class file_loader {
	  public:
		BNCH_SWT_INLINE constexpr file_loader(){};

		BNCH_SWT_INLINE static std::string loadFile(const std::string& filePath) {
			std::string directory{ filePath.substr(0, filePath.find_last_of("/") + 1) };
			if (!std::filesystem::exists(directory)) {
				std::filesystem::create_directories(directory);
			}

			if (!std::filesystem::exists(static_cast<std::string>(filePath))) {
				std::ofstream createFile(filePath.data());
				createFile.close();
			}
			std::string fileContents{};
			std::ifstream theStream(filePath.data(), std::ios::binary | std::ios::in);
			std::stringstream inputStream{};
			inputStream << theStream.rdbuf();
			fileContents = inputStream.str();
			theStream.close();
			return fileContents;
		}

		BNCH_SWT_INLINE static void saveFile(const std::string& fileToSave, const std::string& filePath, bool retry = true) {
			std::ofstream theStream(filePath.data(), std::ios::binary | std::ios::out | std::ios::trunc);
			theStream.write(fileToSave.data(), static_cast<int64_t>(fileToSave.size()));
			if (theStream.is_open()) {
				std::cout << "File succesfully written to: " << filePath << std::endl;
			} else {
				std::string directory{ filePath.substr(0, filePath.find_last_of("/") + 1) };
				if (!std::filesystem::exists(directory) && retry) {
					std::filesystem::create_directories(directory);
					return saveFile(fileToSave, filePath, false);
				}
				std::cerr << "File failed to be written to: " << filePath << std::endl;
			}
			theStream.close();
		}
	};

#if defined(small)
	#undef small
#endif

	struct performance_metrics {
		std::optional<double> instructionsPercentageDeviation{};
		std::optional<double> cyclesPercentageDeviation{};
		std::optional<double> instructionsPerExecution{};
		std::optional<double> branchMissesPerExecution{};
		std::optional<uint64_t> measuredIterationCount{};
		std::optional<uint64_t> totalIterationCount{};
		std::optional<double> instructionsPerCycle{};
		std::optional<double> branchesPerExecution{};
		std::optional<double> instructionsPerByte{};
		std::optional<double> cyclesPerExecution{};
		double throughputPercentageDeviation{};
		std::optional<double> cyclesPerByte{};
		std::optional<double> frequencyGHz{};
		double throughputMbPerSec{};
		double bytesProcessed{};
		std::string name{};
		double timeInNs{};

		BNCH_SWT_ALWAYS_INLINE bool operator>(const performance_metrics& other) const {
			return throughputMbPerSec > other.throughputMbPerSec;
		}
	};

	BNCH_SWT_ALWAYS_INLINE static std::string urlEncode(std::string value) {
		std::ostringstream escaped;
		escaped.fill('0');
		escaped << std::hex;

		for (char c: value) {
			if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
				escaped << c;
			} else if (c == ':') {
				escaped << '%' << std::setw(2) << int32_t(( unsigned char )' ');
			} else {
				escaped << '%' << std::setw(2) << int32_t(( unsigned char )c);
			}
		}

		return escaped.str();
	}

	template<string_literal stageNameNew, uint64_t maxExecutionCount = 100> struct benchmark_stage {
		static constexpr auto addAmount{ maxExecutionCount % 2 == 0 ? 0 : 1 };
		inline static std::unordered_map<std::string, performance_metrics> results{};
		inline static std::vector<event_count> measuredEventsNew{ [] {
			std::vector<event_count> returnValues{};
			returnValues.resize((maxExecutionCount + addAmount) / 2);
			return returnValues;
		}() };
		inline static std::vector<event_count> eventsNew{ [] {
			std::vector<event_count> returnValues{};
			returnValues.resize(maxExecutionCount + addAmount);
			return returnValues;
		}() };

		BNCH_SWT_ALWAYS_INLINE static void warmupThread() {
			auto start	  = clock_type::now();
			auto end	  = clock_type::now();
			auto duration = end - start;
			while (std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(duration) < std::chrono::duration<float, std::milli>{ 1000 }) {
				cache_clearer::evictL1Cache();
				end		 = clock_type::now();
				duration = end - start;
			}
		}

		BNCH_SWT_ALWAYS_INLINE static void printResults(bool showComparison = true) {
			std::vector<performance_metrics> resultsNew{};
			for (auto& [key, value]: results) {
				resultsNew.emplace_back(value);
			}
			if (resultsNew.size() > 0) {
				std::sort(resultsNew.begin(), resultsNew.end(), std::greater<performance_metrics>{});
				std::cout << "Performance Metrics for: " << stageNameNew.view() << "\n";
				for (auto& value: resultsNew) {
					std::cout << "Metrics for: " << value.name << "\n";
					std::cout << std::fixed << std::setprecision(2);

					auto printMetric = [](const std::string& label, const auto& value) {
						if constexpr (optional_t<decltype(value)>) {
							if (value.has_value()) {
								std::cout << std::left << std::setw(40) << label << ": " << value.value() << "\n";
							}
						} else {
							std::cout << std::left << std::setw(40) << label << ": " << value << "\n";
						}
					};

					printMetric("Bytes Processed", value.bytesProcessed);
					printMetric("Throughput (MB/s)", value.throughputMbPerSec);
					printMetric("Throughput Percentage Deviation (+/-%)", value.throughputPercentageDeviation);
					printMetric("Instructions per Execution", value.instructionsPerExecution);
					printMetric("Instructions Percentage Deviation (+/-%)", value.instructionsPercentageDeviation);
					printMetric("Instructions per Cycle", value.instructionsPerCycle);
					printMetric("Instructions per Byte", value.instructionsPerByte);
					printMetric("Branches per Execution", value.branchesPerExecution);
					printMetric("Branch Misses per Execution", value.branchMissesPerExecution);
					printMetric("Cycles per Execution", value.cyclesPerExecution);
					printMetric("Cycles Percentage Deviation (+/-%)", value.cyclesPercentageDeviation);
					printMetric("Cycles per Byte", value.cyclesPerByte);
					printMetric("Frequency (GHz)", value.frequencyGHz);
					printMetric("Measured Iterations", value.measuredIterationCount);
					printMetric("Total Iterations", value.totalIterationCount);

					std::cout << "----------------------------------------\n";
				}

				if (showComparison) {
					double difference{};
					for (uint64_t x = 0; x < resultsNew.size() - 1; ++x) {
						difference = ((resultsNew[x].throughputMbPerSec - resultsNew[x + 1].throughputMbPerSec) / resultsNew[x + 1].throughputMbPerSec) * 100.0f;
						std::cout << "Library " << resultsNew[x].name << ", is faster than library: " << resultsNew[x + 1].name << ", by roughly: " << difference << "%."
								  << std::endl;
					}
				}
			} else {
				std::cout << "Not enough data for benchmark stage: " << stageNameNew.view() << "\n";
			}
		}

#if defined(NDEBUG)
		static constexpr double thresholdStart{ 4.5f };
		static constexpr double threshold{ 4.0f };
#else
		static constexpr double thresholdStart{ 25.0f };
		static constexpr double threshold{ 20.0f };
#endif
		template<string_literal subjectNameNew, string_literal colorNew, typename prep_function_type, typename execution_function_type, not_invocable... arg_types>
		BNCH_SWT_ALWAYS_INLINE static const performance_metrics& runBenchmarkWithPrep(prep_function_type&& prepFunctionNew, execution_function_type&& functionNew,
			arg_types&&... args) {
			static constexpr string_literal subjectName{ subjectNameNew };
			static constexpr string_literal color{ colorNew };
			std::string subjectNameNewer{ subjectName.data(), subjectName.size() };
			event_collector collector{};
			std::string colorName{ color.data(), color.size() };

			auto prepLambda = [=]() mutable {
				return std::forward<prep_function_type>(prepFunctionNew)();
			};
			auto executionLambda = [=]() mutable {
				return std::forward<execution_function_type>(functionNew)(std::forward<arg_types>(args)...);
			};
			std::vector<event_count> aggregate{};
			double variation{ thresholdStart };
			performance_metrics resultsTemp{};
			uint64_t i{};
			warmupThread();
			while ((variation > threshold && i < maxExecutionCount + addAmount)) {
				cache_clearer::evictL1Cache();
				prepLambda();
				eventsNew[i] = collector.start(executionLambda);
				if ((i % (maxExecutionCount / 2) == 0 && i > 0) || maxExecutionCount < 2) {
					resultsTemp = collectMetrics<subjectName>(i, maxExecutionCount / 2);
					variation	= resultsTemp.throughputPercentageDeviation;
				}
				++i;
			}
			results[static_cast<std::string>(subjectName)] = resultsTemp;
			return results[static_cast<std::string>(subjectName)];
		}

		template<string_literal subjectNameNew, string_literal colorNew, typename function_type, typename... arg_types>
		BNCH_SWT_ALWAYS_INLINE static const performance_metrics& runBenchmark(function_type&& functionNew, arg_types&&... args) {
			static constexpr string_literal subjectName{ subjectNameNew };
			static constexpr string_literal color{ colorNew };
			std::string subjectNameNewer{ subjectName.data(), subjectName.size() };
			event_collector collector{};
			std::string colorName{ color.data(), color.size() };
			auto executionLambda = [=]() mutable {
				return std::forward<function_type>(functionNew)(std::forward<arg_types>(args)...);
			};
			std::vector<event_count> aggregate{};
			double variation{ thresholdStart };
			performance_metrics resultsTemp{};
			uint64_t i{};
			warmupThread();
			while ((variation > threshold && i < maxExecutionCount + addAmount)) {
				cache_clearer::evictL1Cache();
				eventsNew[i] = collector.start(executionLambda);
				if ((i % (maxExecutionCount / 2) == 0 && i > 0) || maxExecutionCount < 2) {
					resultsTemp = collectMetrics<subjectName>(i, maxExecutionCount / 2);
					variation	= resultsTemp.throughputPercentageDeviation;
				}
				++i;
			}
			results[static_cast<std::string>(subjectName)] = resultsTemp;
			return results[static_cast<std::string>(subjectName)];
		}

		template<string_literal benchmarkNameNew> BNCH_SWT_ALWAYS_INLINE static performance_metrics collectMetrics(uint64_t offset, uint64_t measuredIterationCount) {
			static constexpr string_literal benchmarkName{ benchmarkNameNew };
			performance_metrics metrics{};
			metrics.name = static_cast<std::string>(benchmarkName.data());
			metrics.measuredIterationCount.emplace(measuredIterationCount);
			metrics.totalIterationCount.emplace(offset);
			eventsNew[offset].bytesProcessed(metrics.bytesProcessed);
			double volumeMB{ static_cast<double>(metrics.bytesProcessed) / (1024. * 1024.) };
			double averageNs{ 0 };
			double minNs{ std::numeric_limits<double>::max() };
			double cyclesMin{ std::numeric_limits<double>::max() };
			double instructionsMin{ std::numeric_limits<double>::max() };
			double cyclesAvg{ 0 };
			double instructionsAvg{ 0 };
			double branchesMin{ 0 };
			double branchesAvg{ 0 };
			double missedBranchesMin{ std::numeric_limits<double>::max() };
			double missedBranchesAvg{ 0 };
			double cycles{};
			double instructions{};
			double branches{};
			double missedBranchesVal{};
			std::copy(eventsNew.begin() + static_cast<int64_t>(offset - measuredIterationCount), eventsNew.begin() + static_cast<int64_t>(offset), measuredEventsNew.begin());
			for (const event_count& e: measuredEventsNew) {
				double ns = e.elapsedNs();
				averageNs += ns;
				minNs = minNs < ns ? minNs : ns;

				e.cycles(cycles);
				cyclesAvg += cycles;
				cyclesMin = cyclesMin < cycles ? cyclesMin : cycles;

				e.instructions(instructions);
				instructionsAvg += instructions;
				instructionsMin = instructionsMin < instructions ? instructionsMin : instructions;

				e.branches(branches);
				branchesAvg += branches;
				branchesMin = branchesMin < branches ? branchesMin : branches;

				e.missedBranches(missedBranchesVal);
				missedBranchesAvg += missedBranchesVal;
				missedBranchesMin = missedBranchesMin < missedBranchesVal ? missedBranchesMin : missedBranchesVal;
			}
			metrics.timeInNs = averageNs;
			cyclesAvg /= static_cast<double>(measuredIterationCount);
			instructionsAvg /= static_cast<double>(measuredIterationCount);
			averageNs /= static_cast<double>(measuredIterationCount);
			branchesAvg /= static_cast<double>(measuredIterationCount);
			metrics.throughputMbPerSec			  = volumeMB * 1000000000 / averageNs;
			metrics.throughputPercentageDeviation = (averageNs - minNs) * 100.0 / averageNs;
			if (instructionsAvg != 0.0f) {
				metrics.instructionsPerByte.emplace(instructionsAvg / metrics.bytesProcessed);
				metrics.instructionsPerCycle.emplace(instructionsAvg / cyclesAvg);
				metrics.instructionsPerExecution.emplace(instructionsAvg / static_cast<double>(measuredIterationCount));
				metrics.instructionsPercentageDeviation.emplace((instructionsAvg - instructionsMin) * 100.0 / instructionsAvg);
			}
			if (cyclesAvg != 0.0f) {
				metrics.cyclesPerByte.emplace(cyclesAvg / metrics.bytesProcessed);
				metrics.cyclesPerExecution.emplace(cyclesAvg / static_cast<double>(measuredIterationCount));
				metrics.cyclesPercentageDeviation.emplace((cyclesAvg - cyclesMin) * 100.0 / cyclesAvg);
				metrics.frequencyGHz.emplace(cyclesMin / minNs);
			}
			if (branchesAvg != 0.0f) {
				metrics.branchMissesPerExecution.emplace(missedBranchesAvg / static_cast<double>(measuredIterationCount));
				metrics.branchesPerExecution.emplace(branchesAvg / static_cast<double>(measuredIterationCount));
			}
			return metrics;
		}
	};

}
