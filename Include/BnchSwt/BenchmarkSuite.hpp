// BenchmarkSuite.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <BnchSwt/EventCounter.hpp>
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

	BNCH_SWT_INLINE double cyclesToTime(double cycles, double frequencyMHz) {
		double frequencyHz	   = frequencyMHz * 1e6;
		double timeNanoseconds = (cycles * 1e9) / frequencyHz;

		return timeNanoseconds;
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

	BNCH_SWT_INLINE void evictCache() {
		const size_t cache_size = 64 * 1024;
		char* memory			= new char[cache_size];

		for (size_t i = 0; i < cache_size; ++i) {
			memory[i] = static_cast<char>(i);
		}

		delete[] memory;
	}

	template<typename value_type> bool contains(value_type* values, const value_type& valueToCheckFor, size_t length) {
		for (size_t x = 0; x < length; ++x) {
			if (values[x] == valueToCheckFor) {
				return true;
			}
		}
		return false;
	}

	struct performance_metrics {
		std::optional<double> instructionsPerExecution{};
		std::optional<double> branchMissesPerExecution{};
		std::optional<double> instructionsVariation{};
		std::optional<double> instructionsPerCycle{};
		std::optional<double> branchesPerExecution{};
		std::optional<double> instructionsPerByte{};
		std::optional<double> throughputVariation{};
		std::optional<double> throughputMbPerSec{};
		std::optional<double> cyclesPerExecution{};
		std::optional<double> cyclesVariation{};
		std::optional<double> cyclesPerByte{};
		std::optional<double> frequencyGHz{};
		uint64_t iterationCount{};
		uint64_t timeInns{};
		std::string name{};
	};

	template<jsonifier_internal::string_literal stageNameNew, uint64_t maxExecutionCount = 100> struct benchmark_stage {
		inline static std::unordered_map<std::string, performance_metrics> results{};

		BNCH_SWT_INLINE static void printResults(bool verbose = true, bool printDetails = false) {
			static constexpr jsonifier_internal::string_literal stageName{ stageNameNew };
			std::cout << "Performance Metrics for: " << stageName.view() << "\n";
			for (auto& [key, value]: results) {
				std::cout << "Metrics for: " << value.name << "\n";
				std::cout << std::fixed << std::setprecision(2);

				auto printMetric = [](const std::string& label, const std::optional<double>& value) {
					if (value.has_value()) {
						std::cout << std::left << std::setw(30) << label << ": " << value.value() << "\n";
					}
				};

				printMetric("Throughput (MB/s)", value.throughputMbPerSec);
				printMetric("Throughput Variation (+/-%)", value.throughputVariation);
				printMetric("Instructions per Execution", value.instructionsPerExecution);
				printMetric("Instructions Variation (+/-%)", value.instructionsVariation);
				printMetric("Instructions per Cycle", value.instructionsPerCycle);
				printMetric("Instructions per Byte", value.instructionsPerByte);
				printMetric("Branches per Execution", value.branchesPerExecution);
				printMetric("Branch Misses per Execution", value.branchMissesPerExecution);
				printMetric("Cycles per Execution", value.cyclesPerExecution);
				printMetric("Cycles Variation (+/-%)", value.cyclesVariation);
				printMetric("Cycles per Byte", value.cyclesPerByte);
				printMetric("Frequency (GHz)", value.frequencyGHz);

				std::cout << "----------------------------------------\n";
			}
		}

		template<jsonifier_internal::string_literal filePath> BNCH_SWT_INLINE static void writeJsonData() {
			auto stringToWrite = parser.serializeJson(results);
			file_loader::saveFile(static_cast<std::string>(stringToWrite), filePath);
		}

		template<jsonifier_internal::string_literal benchmarkNameNew, jsonifier_internal::string_literal subjectNameNew, jsonifier_internal::string_literal colorNew,
			typename function_type, typename... arg_types>
		BNCH_SWT_INLINE static performance_metrics runBenchmark(function_type&& functionNew, arg_types&&... args) {
			static constexpr jsonifier_internal::string_literal benchmarkName{ benchmarkNameNew };
			static constexpr jsonifier_internal::string_literal subjectName{ subjectNameNew };
			static constexpr jsonifier_internal::string_literal color{ colorNew };
			std::string subjectNameNewer{ subjectName.data(), subjectName.size() };
			event_collector collector{};
			std::string colorName{ color.data(), color.size() };
			auto executionLambda = [=]() mutable {
				return std::forward<function_type>(functionNew)(std::forward<arg_types>(args)...);
			};
			size_t currentExecutionCount{};
			std::vector<event_count> aggregate{};
			bool printed_bug = false;
			uint64_t bytesCollected{};
			for (size_t i = 0; i < maxExecutionCount; i++) {
				collector.start();
				bytesCollected = executionLambda();
				aggregate.emplace_back(collector.end());
			}
			results[static_cast<std::string>(subjectName)] = collectMetrics<subjectName>(bytesCollected, aggregate);
			return results[static_cast<std::string>(subjectName)];
		}

		template<jsonifier_internal::string_literal benchmarkNameNew>
		static performance_metrics collectMetrics(uint64_t bytesProcessed, std::vector<event_count> events) {
			performance_metrics metrics{};
			metrics.name	= static_cast<std::string>(benchmarkNameNew.data());
			metrics.iterationCount = maxExecutionCount;
			double volumeMB = bytesProcessed / (1024. * 1024.);
			double average_ns{ 0 };
			double min_ns{ DBL_MAX };
			double cycles_min{ DBL_MAX };
			double instructions_min{ DBL_MAX };
			double cycles_avg{ 0 };
			double instructions_avg{ 0 };
			double branches_min{ 0 };
			double branches_avg{ 0 };
			double missedBranches_min{ 0 };
			double missedBranches_avg{ 0 };
			bool doWeHaveCounters{};
			double cycles{};
			double instructions{};
			double branches{};
			double missedBranches{};
			for (const event_count& e: events) {
				double ns = e.elapsed_ns();
				average_ns += ns;
				min_ns = min_ns < ns ? min_ns : ns;

				e.cycles(cycles);
				cycles_avg += cycles;
				cycles_min = cycles_min < cycles ? cycles_min : cycles;

				if (e.instructions(instructions)) {
					doWeHaveCounters = true;
				}
				instructions_avg += instructions;
				instructions_min = instructions_min < instructions ? instructions_min : instructions;

				e.branches(branches);
				branches_avg += branches;
				branches_min = branches_min < branches ? branches_min : branches;

				e.missed_branches(missedBranches);
				missedBranches_avg += missedBranches;
				missedBranches_min = missedBranches_min < missedBranches ? missedBranches_min : missedBranches;
			}
			metrics.timeInns = average_ns;
			cycles_avg /= events.size();
			instructions_avg /= events.size();
			average_ns /= events.size();
			branches_avg /= events.size();
			metrics.throughputMbPerSec.emplace(volumeMB * 1000000000 / min_ns);
			metrics.throughputVariation.emplace((average_ns - min_ns) * 100.0 / average_ns);
			if (doWeHaveCounters) {
				metrics.instructionsPerByte.emplace(instructions_avg / bytesProcessed);
				metrics.frequencyGHz.emplace(cycles_min / min_ns);
				metrics.instructionsPerCycle.emplace(instructions_avg / cycles_avg);
				metrics.instructionsPerExecution.emplace(instructions_avg / maxExecutionCount);
				metrics.instructionsVariation.emplace((instructions_avg - instructions_min) * 100.0 / instructions_avg);
				metrics.branchMissesPerExecution.emplace(missedBranches_avg / maxExecutionCount);
				metrics.branchesPerExecution.emplace(branches_avg / maxExecutionCount);
				metrics.cyclesPerByte.emplace(cycles_avg / bytesProcessed);
				metrics.cyclesPerExecution.emplace(cycles_avg / maxExecutionCount);
				metrics.cyclesVariation.emplace((cycles_avg - cycles_min) * 100.0 / cycles_avg);
			}
			return metrics;
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
				uniqueLibraries.emplace(value.name);
			}
			return uniqueLibraries.size();
		}

		BNCH_SWT_INLINE static std::string generateMarkdown(std::string repoPath) {
			return {};
		}
	};

	template<jsonifier_internal::string_literal filePath> struct data_holder {
		static constexpr file_loader file{};
	};

}

namespace jsonifier {

	template<> struct core<bnch_swt::performance_metrics> {
		using value_type = bnch_swt::performance_metrics;
		static constexpr auto parseValue = createValue<&value_type::branchesPerExecution, &value_type::branchMissesPerExecution, &value_type::cyclesPerByte,
			&value_type::cyclesPerExecution, &value_type::cyclesVariation>();
	};
}