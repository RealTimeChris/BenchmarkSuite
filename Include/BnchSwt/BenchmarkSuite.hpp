// BenchmarkSuite.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <BnchSwt/Config.hpp>
#include <jsonifier/Index.hpp>
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

#if defined(BNCH_SWT_MAC)
	#include <mach/mach_time.h>
	#include <sys/sysctl.h>
#endif

template<typename value_type, size_t size> std::ostream& operator<<(std::ostream& os, const std::array<value_type, size>& values) {
	os << "[";
	for (uint64_t x = 0; x < size; ++x) {
		os << values[x];
		if (x < size - 1) {
			os << ",";
		}
	}
	os << "]";
	return os;
}

namespace bnch_swt {

	class file_loader {
	  public:
		constexpr BNCH_SWT_INLINE file_loader(){};

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

	inline thread_local jsonifier::jsonifier_core parser{};

	template<typename function_type, typename... arg_types> struct return_type_helper {
		using type = std::invoke_result_t<function_type, arg_types...>;
	};

	template<typename value_type, typename... arg_types>
	concept invocable = std::is_invocable_v<std::remove_cvref_t<value_type>, arg_types...>;

	template<typename value_type, typename... arg_types>
	concept not_invocable = !std::is_invocable_v<std::remove_cvref_t<value_type>, arg_types...>;

	template<typename value_type, typename... arg_types>
	concept invocable_void = invocable<value_type, arg_types...> && std::is_void_v<typename return_type_helper<value_type, arg_types...>::type>;

	template<typename value_type, typename... arg_types>
	concept invocable_not_void = invocable<value_type, arg_types...> && !std::is_void_v<typename return_type_helper<value_type, arg_types...>::type>;

	using clock_type = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;

	static void const volatile* volatile globalForceEscapePointer;

	void useCharPointer(char const volatile* const v) {
		globalForceEscapePointer = reinterpret_cast<void const volatile*>(v);
	}

#if defined(BNCH_SWT_MSVC)
	#define doNotOptimize(value) \
		useCharPointer(&reinterpret_cast<char const volatile&>(value)); \
		_ReadWriteBarrier();
#elif defined(BNCH_SWT_CLANG)
	#define doNotOptimize(value) asm volatile("" : "+r,m"(value) : : "memory");
#else
	#define doNotOptimize(value) asm volatile("" : "+m,r"(value) : : "memory");
#endif

	template<not_invocable value_type> BNCH_SWT_INLINE void doNotOptimizeAway(value_type&& value) {
		const auto* valuePtr = &value;
		doNotOptimize(valuePtr)
	}

	template<invocable_void function_type, typename... arg_types> BNCH_SWT_INLINE void doNotOptimizeAway(function_type&& value, arg_types&&... args) {
		std::forward<function_type>(value)(std::forward<arg_types>(args)...);
		doNotOptimize(value);
	}

	template<invocable_not_void function_type, typename... arg_types> BNCH_SWT_INLINE void doNotOptimizeAway(function_type&& value, arg_types&&... args) {
		auto resultVal = std::forward<function_type>(value)(std::forward<arg_types>(args)...);
		doNotOptimize(resultVal);
	}

	double getCpuFrequency();

#if defined(BNCH_SWT_MSVC)

	#define rdtsc() __rdtsc()

#else

	__inline__ size_t rdtsc() {
	#if defined(__x86_64__)
		uint32_t a, d;
		__asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
		return ( unsigned long )a | (( unsigned long )d << 32);
	#elif defined(__i386__)
		size_t x;
		__asm__ volatile("rdtsc" : "=A"(x));
		return x;
	#else
		mach_timebase_info_data_t timebase_info;
		mach_timebase_info(&timebase_info);
		size_t nanoseconds = mach_absolute_time() * timebase_info.numer / timebase_info.denom;

		double seconds	  = static_cast<double>(nanoseconds) / 1e9;
		double cpu_cycles = seconds * getCpuFrequency();
		return cpu_cycles;
	#endif
	}
#endif

#if defined(BNCH_SWT_WIN)
	#include <Windows.h>
	#include <Pdh.h>
	#pragma comment(lib, "Pdh.lib")

	double getCpuFrequency() {
		LARGE_INTEGER qwWait, qwStart, qwCurrent;
		QueryPerformanceCounter(&qwStart);
		QueryPerformanceFrequency(&qwWait);
		qwWait.QuadPart >>= 5;
		unsigned __int64 Start = __rdtsc();
		do {
			QueryPerformanceCounter(&qwCurrent);
		} while (qwCurrent.QuadPart - qwStart.QuadPart < qwWait.QuadPart);
		return static_cast<double>((__rdtsc() - Start) << 5) / 1000000.0;
	}

#elif defined(BNCH_SWT_LINUX)
	#include <fstream>

	double getCpuFrequency() {
		std::ifstream file("/proc/cpuinfo");
		if (!file.is_open()) {
			std::cout << "Error opening /proc/cpuinfo" << std::endl;
			return 0.0;
		}

		std::string line;
		double frequency = 0.0;
		while (std::getline(file, line)) {
			if (line.find("cpu MHz") != std::string::npos) {
				size_t pos = line.find(":");
				if (pos != std::string::npos) {
					frequency = std::stod(line.substr(pos + 1));
					break;
				}
			}
		}
		file.close();

		return frequency;
	}

#elif defined(BNCH_SWT_MAC)

	double getCpuFrequency() {
		volatile int32_t counter{};
		for (int32_t i = 0; i < 1000000000; ++i) {
			++counter;
		}
		counter	   = 0;
		auto start = clock_type::now();
		for (int32_t i = 0; i < 1000000000; ++i) {
			++counter;
		}
		auto end		 = clock_type::now();
		auto newDuration = std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(end - start);
		return (static_cast<double>(counter) * newDuration.count() / 1000000000.0f) / 1e6;
	}

#else

	double getCpuFrequency() {
		return 1000.0;
	}

#endif

#if defined(small)
	#undef small
#endif

	template<typename function_type, typename... arg_types> BNCH_SWT_INLINE double collectCycles(function_type&& function, arg_types&&... args) {
		volatile size_t start{}, end{};
		start = rdtsc();
		std::forward<function_type>(function)(std::forward<arg_types>(args)...);
		end = rdtsc();
		return static_cast<double>(end - start);
	}

	BNCH_SWT_INLINE double cyclesToTime(double cycles, double frequencyMHz) {
		double frequencyHz	   = frequencyMHz * 1e6;
		double timeNanoseconds = (cycles * 1e9) / frequencyHz;

		return timeNanoseconds;
	}

	template<typename function_type> BNCH_SWT_INLINE double collectTime(function_type&& function, double cpuFrequency) {
#if defined(BNCH_SWT_MAC)
		auto startTime = clock_type::now();
		std::forward<function_type>(function)();
		auto endTime	= clock_type::now();
		double duration = std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(endTime - startTime).count();
#else
		auto duration = collectCycles(std::forward<function_type>(function));
		duration	  = cyclesToTime(duration, cpuFrequency);
#endif
		return duration;
	}

	BNCH_SWT_INLINE double calcMedian(double* data, size_t length) {
		std::sort(data, data + length);
		const size_t midIdx = length / 2;
		if (length % 2 == 1) {
			return data[midIdx];
		}
		return (data[midIdx - 1] + data[midIdx]) / 2.0f;
	}

	BNCH_SWT_INLINE double calcMean(double* v, size_t length) {
		double mean = 0;

		for (uint32_t i = 0; i < length; ++i) {
			mean += v[i];
		}

		mean /= static_cast<double>(length);

		return mean;
	}

	BNCH_SWT_INLINE double calcStdv(double* v, size_t length, double mean) {
		double stdv = 0;

		for (uint32_t i = 0; i < length; ++i) {
			double x = v[i] - mean;

			stdv += x * x;
		}

		stdv = std::sqrt(stdv / (static_cast<double>(length) + 1));

		return stdv;
	}

	BNCH_SWT_INLINE double roundToDecimalPlaces(double value, int32_t decimalPlaces) {
		const double scale = std::pow(10.0, decimalPlaces);
		return std::round(value * scale) / scale;
	}

	BNCH_SWT_INLINE void removeOutliers(double* temp, double* v, size_t& length) {
		if (length == 0) {
			return;
		}
		const double m			 = calcMean(v, length);
		const double sd			 = calcStdv(v, length, m);
		const double lower_bound = m - (3 * sd);
		const double upper_bound = m + (3 * sd);
		size_t currentIndex{};
		for (size_t x = 0; x < length; ++x) {
			if (v[x] >= lower_bound && v[x] <= upper_bound) {
				temp[currentIndex] = v[x];
				++currentIndex;
			}
		}
		length = currentIndex;
		std::copy(temp, temp + currentIndex, v);
	}

	BNCH_SWT_INLINE bool checkForValidLt(double valueToCheck, double valueToCheckAgainst) {
		return std::isfinite(valueToCheck) && valueToCheck < valueToCheckAgainst;
	}

	enum class bench_state { starting = 0, running = 1, complete_success = 2, complete_failure = 3 };

	enum class result_type { unset = 0, cycles = 1, time = 2 };

	struct bench_options {
		result_type type{ result_type::cycles };
		size_t totalIterationCountCap{ 300 };
		size_t maxDurationCount{ 100 };
		size_t minDurationCount{ 30 };
		size_t maxExecutionCount{ 4 };
		size_t targetCvDenom{ 100 };
		size_t maxEpochCount{ 4 };
	};

	struct benchmark_result_final {
		std::string benchmarkColor{};
		std::string benchmarkName{};
		std::string libraryName{};
		size_t iterationCount{};
		bench_state state{};
		result_type type{};
		double median{};
		double cv{};
	};

	BNCH_SWT_ALWAYS_INLINE void evictCache() {
		const size_t cache_size = 64 * 1024;
		char* memory			= new char[cache_size];

		for (size_t i = 0; i < cache_size; ++i) {
			memory[i] = static_cast<char>(i);
		}

		delete[] memory;
	}

	template<typename function_type, bench_options optionsNew> struct benchmark_subject {
		BNCH_SWT_INLINE benchmark_subject& operator=(benchmark_subject&&) noexcept		= default;
		BNCH_SWT_INLINE benchmark_subject(benchmark_subject&&) noexcept					= default;
		BNCH_SWT_INLINE benchmark_subject& operator=(const benchmark_subject&) noexcept = default;
		BNCH_SWT_INLINE benchmark_subject(const benchmark_subject&) noexcept			= default;

		BNCH_SWT_INLINE benchmark_subject(const std::string& subjectName, const std::string& subjectColor, function_type&& functionNew)
			: benchmarkColor{ subjectColor }, subjectName{ subjectName }, function{ functionNew } {
			tempDurations.resize(options.totalIterationCountCap);
			durations.resize(options.totalIterationCountCap);
		}

		template<typename... arg_types> BNCH_SWT_INLINE benchmark_result_final executeEpoch(arg_types&&... args) {
			double cpuFrequency{};
			if constexpr (optionsNew.type == result_type::time) {
				cpuFrequency = getCpuFrequency();
			}
			bench_state state{ bench_state::running };
			benchmark_result_final returnValues{};
			size_t currentDurationCount{};
			size_t warmupDurationCount{};
			double currentMedian{};
			double currentMean{};
			double currentStdv{};
			double currentCv{};
			evictCache();
			while (warmupDurationCount < options.maxDurationCount && totalDurationCount + warmupDurationCount < options.totalIterationCountCap) {
				if constexpr (optionsNew.type == result_type::cycles) {
					evictCache();
					collectCycles(function, std::forward<arg_types>(args)...);
				} else {
					evictCache();
					collectTime(function, cpuFrequency, std::forward<arg_types>(args)...);
				}
				++warmupDurationCount;
			}
			totalDurationCount += warmupDurationCount;
			currentDurationCount = 0;
			while (currentDurationCount < options.maxDurationCount && totalDurationCount + currentDurationCount + warmupDurationCount < options.totalIterationCountCap) {
				if constexpr (optionsNew.type == result_type::cycles) {
					evictCache();
					durations[currentDurationCount] = collectCycles(function, std::forward<arg_types>(args)...);
				} else {
					evictCache();
					durations[currentDurationCount] = collectTime(function, cpuFrequency, std::forward<arg_types>(args)...);
				}
				++currentDurationCount;
			}
			bool greaterOrEqual{ currentDurationCount >= options.minDurationCount };
			if (greaterOrEqual) {
				removeOutliers(tempDurations.data(), durations.data(), currentDurationCount);
				greaterOrEqual = currentDurationCount >= options.minDurationCount;
				if (greaterOrEqual) {
					currentMean = calcMean(durations.data() + currentDurationCount - options.minDurationCount, options.minDurationCount);
					currentStdv = calcStdv(durations.data() + currentDurationCount - options.minDurationCount, options.minDurationCount, currentMean);
				}
			}
			totalDurationCount += currentDurationCount;
			++currentEpochCount;
			currentCv = currentStdv / currentMean;
			if (checkForValidLt(currentCv, targetCv)) {
				currentMedian = calcMedian(durations.data() + currentDurationCount - options.minDurationCount, options.minDurationCount);
				state		  = bench_state::complete_success;
			} else if (std::isfinite(currentCv)) {
				options.maxDurationCount = (options.maxDurationCount * (60.0f * currentCv)) > (options.maxDurationCount * 5) ? (options.maxDurationCount * 5)
																															 : (options.maxDurationCount * (60.0f * currentCv));
				targetCv += 0.01f;
			} else {
				options.maxDurationCount = (options.maxDurationCount * 2);
			}
			if (totalDurationCount + options.maxDurationCount >= options.totalIterationCountCap) {
				options.maxDurationCount = options.totalIterationCountCap - totalDurationCount;
			}
			if (totalDurationCount >= options.totalIterationCountCap || (currentEpochCount >= options.maxEpochCount && state != bench_state::complete_success)) {
				currentMedian = calcMedian(durations.data(), options.minDurationCount);
				state		  = bench_state::complete_failure;
			}
			returnValues.iterationCount = totalDurationCount;
			returnValues.type			= optionsNew.type;
			returnValues.benchmarkColor = benchmarkColor;
			returnValues.median			= currentMedian;
			returnValues.type			= options.type;
			returnValues.libraryName	= subjectName;
			returnValues.cv				= currentCv;
			returnValues.state			= state;
			return returnValues;
		}

	  protected:
		double targetCv{ 1.0f / static_cast<double>(optionsNew.targetCvDenom) };
		std::vector<double> tempDurations{};
		std::vector<double> durations{};
		bench_options options{ optionsNew };
		std::string benchmarkColor{};
		size_t totalDurationCount{};
		size_t currentEpochCount{};
		std::string subjectName{};
		function_type function{};
	};

	struct average_cycle_count {
		std::string libraryName{};
		uint64_t totalCount{};
		double totalCycles{};
	};

	template<typename value_type> bool contains(value_type* values, const value_type& valueToCheckFor, size_t length) {
		for (size_t x = 0; x < length; ++x) {
			if (values[x] == valueToCheckFor) {
				return true;
			}
		}
		return false;
	}

	template<jsonifier_internal::string_literal stageNameNew, bench_options options = bench_options{}> struct benchmark_stage {
		inline static std::unordered_map<std::string, benchmark_result_final> results{};

		BNCH_SWT_INLINE static void printResults(bool verbose = true, bool printDetails = false) {
			static constexpr jsonifier_internal::string_literal stageName{ stageNameNew };
			std::map<std::string, std::vector<benchmark_result_final>> groupedResults{};
			std::set<std::pair<std::string, std::string>> printedResults{};
			std::map<std::string, uint64_t> indices{};
			std::vector<average_cycle_count> resultCycles{};
			for (const auto& [key, value]: results) {
				groupedResults[value.benchmarkName].push_back(value);
			}


			for (auto& [key, value]: groupedResults) {
				std::sort(value.data(), value.data() + value.size(), [](auto& lhs, auto rhs) {
					return lhs.median < rhs.median;
				});
			}
			for (const auto& [benchmarkName, benchmarkResults]: groupedResults) {
				if (verbose) {
					std::cout << "Benchmark Name: " << benchmarkName << std::endl;
				}

				for (const auto& value: benchmarkResults) {
					std::string resultType = (value.type == result_type::cycles) ? "Cycles" : "Time";
					std::string status	   = (value.state == bench_state::complete_success) ? "Success: " : "Failure: ";

					if (!indices.contains(value.libraryName)) {
						indices[value.libraryName] = resultCycles.size();
						resultCycles.emplace_back(average_cycle_count{ value.libraryName, uint64_t{}, double{} });
					}
					resultCycles[indices[value.libraryName]].totalCycles += value.median;
					++resultCycles[indices[value.libraryName]].totalCount;
					if (verbose && printDetails) {
						std::cout << status << "Library Name: " << value.libraryName << ", Result " << resultType << ": " << roundToDecimalPlaces(value.median, 2)
								  << ", Iterations: " << jsonifier::toString(value.iterationCount)
								  << ", Coefficient of Variance: " << jsonifier::toString(roundToDecimalPlaces(value.cv, 6)) << std::endl;
					}
					printedResults.insert({ benchmarkName, value.libraryName });
				}
			}

			if (resultCycles.size() >= 2) {
				std::sort(resultCycles.begin(), resultCycles.end(), [](const auto& lhs, const auto& rhs) {
					return lhs.totalCycles > rhs.totalCycles;
				});
				for (size_t x = resultCycles.size(); x > 1; --x) {
					double totalPercentage =
						((resultCycles.data() + x - 2)->totalCycles - ((resultCycles.data() + x - 1)->totalCycles)) / (resultCycles.data() + x - 1)->totalCycles;
					totalPercentage *= 100.0;
					std::cout << "Library: " << (resultCycles.data() + x - 1)->libraryName << " is faster than " << (resultCycles.data() + x - 2)->libraryName
							  << " by roughly: " << totalPercentage << "%." << std::endl;
				}
			} else {
				std::cout << "Not enough data to compare library performance." << std::endl;
			}
		}

		template<jsonifier_internal::string_literal filePath> BNCH_SWT_INLINE static void writeJsonData() {
			auto stringToWrite = parser.serializeJson(results);
			file_loader::saveFile(static_cast<std::string>(stringToWrite), filePath);
		}

		template<jsonifier_internal::string_literal filePath> BNCH_SWT_INLINE static void writeMarkdownData(std::string repoPath) {
			std::string markdownContent = generateMarkdown(repoPath);
			file_loader fileLoader{ filePath };
			file_loader::saveFile(static_cast<std::string>(markdownContent), filePath);
		}

		template<jsonifier_internal::string_literal filePath> BNCH_SWT_INLINE static std::string writeCsvData() {
			std::string newString{ "benchmarkName,median,benchmarkColor" };
			int32_t x{};
			for (auto [key, value]: results) {
				newString += "\n";
				newString += value.benchmarkName + ",";
				newString += jsonifier::toString(value.median) + ",";
				newString += static_cast<std::string>(value.benchmarkColor);
				if (x < results.size() - 1) {
					newString += ",";
				}
				++x;
			}
			file_loader::saveFile(static_cast<std::string>(newString), filePath);
			return {};
		}

		template<jsonifier_internal::string_literal benchmarkNameNew, jsonifier_internal::string_literal subjectNameNew, jsonifier_internal::string_literal colorNew,
			typename function_type, typename... arg_types>
		BNCH_SWT_INLINE static benchmark_result_final runBenchmark(function_type&& functionNew, arg_types&&... args) {
			static constexpr jsonifier_internal::string_literal benchmarkName{ benchmarkNameNew };
			static constexpr jsonifier_internal::string_literal subjectName{ subjectNameNew };
			static constexpr jsonifier_internal::string_literal color{ colorNew };
			std::string subjectNameNewer{ subjectName.data(), subjectName.size() };
			std::string colorName{ color.data(), color.size() };
			benchmark_subject<function_type, options> benchmarkSubject{ subjectNameNewer, colorName, std::forward<function_type>(functionNew) };
			auto executionLambda = [=]() mutable {
				return benchmarkSubject.executeEpoch(std::forward<arg_types>(args)...);
			};
			size_t currentExecutionCount{};
			while (currentExecutionCount < options.maxExecutionCount) {
				++currentExecutionCount;
				benchmark_result_final resultsNew = executionLambda();
				resultsNew.benchmarkName		  = static_cast<std::string>(benchmarkName.view());
				if (resultsNew.state == bench_state::complete_success || resultsNew.state == bench_state::complete_failure) {
					results[static_cast<std::string>(benchmarkName.view() + subjectName.view())] = resultsNew;
					return std::move(resultsNew);
				} else if (currentExecutionCount == options.maxExecutionCount) {
					results[static_cast<std::string>(benchmarkName.view() + subjectName.view())] = resultsNew;
					return resultsNew;
				}
			}
			return {};
		}

	  protected:
		BNCH_SWT_INLINE static std::string urlEncode(std::string value) {
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

		BNCH_SWT_INLINE static size_t collectUniqueLibraryCount() {
			std::unordered_set<std::string> uniqueLibraries{};
			for (auto& [key, value]: results) {
				uniqueLibraries.emplace(value.libraryName);
			}
			return uniqueLibraries.size();
		}

		BNCH_SWT_INLINE static std::string generateMarkdown(std::string repoPath) {
			std::ostringstream markdownStream{};
			std::vector<benchmark_result_final> resultsNew{};
			jsonifier_internal::string_literal stageName{ stageNameNew };
			size_t uniqueLibraryCount{ collectUniqueLibraryCount() };
			markdownStream << "# Benchmark Results: " + stageName.view() + "\n\n";
			std::string currentTestName{};
			size_t currentIndex{};
			/*
			for (auto [key, value]: results) {
			for (size_t i = 0; i < results.size() / uniqueLibraryCount; ++i) {
				const auto& result = results[i];
				if (currentTestName != result.benchmarkName || i == results.size() - 1) {
					resultsNew.emplace_back(results[i]);
				}
				currentTestName = result.benchmarkName;
			}
			for (auto& value: resultsNew) {
				std::string encodedBenchmarkName = urlEncode(value.benchmarkName);
				markdownStream << "### " << currentIndex + 1 << ". " << value.benchmarkName << "\n";
				markdownStream << "<p align=\"left\"><img src=\"" << repoPath << encodedBenchmarkName << "_results.jpg?raw=true\" width=\"400\"/></p>\n\n";
				++currentIndex;
			}
			*/
			return markdownStream.str();
		}
	};

	template<jsonifier_internal::string_literal filePath> struct data_holder {
		static constexpr file_loader file{};
	};

}

namespace jsonifier {

	template<> struct core<bnch_swt::benchmark_result_final> {
		using value_type = bnch_swt::benchmark_result_final;
		static constexpr auto parseValue =
			createValue<&value_type::benchmarkName, &value_type::median, &value_type::benchmarkColor, &value_type::iterationCount, &value_type::libraryName>();
	};
}