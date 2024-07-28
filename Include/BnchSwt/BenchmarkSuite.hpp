// BenchmarkSuite.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <BnchSwt/StringLiteral.hpp>
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

namespace bnch_swt {

	template<typename value_type>
	concept printable = requires(value_type value) { std::cout << value << std::endl; };

	template<printable value_type> BNCH_SWT_ALWAYS_INLINE std::ostream& operator<<(std::ostream& os, const jsonifier::vector<value_type>& vector) {
		os << '[';
		for (size_t x = 0; x < vector.size(); ++x) {
			os << vector[x];
			if (x < vector.size() - 1) {
				os << ',';
			}
		}
		os << ']' << std::endl;
		return os;
	}

	class file_loader {
	  public:
		BNCH_SWT_ALWAYS_INLINE file_loader(jsonifier::string_view filePathNew) {
			filePath = filePathNew;
			jsonifier::string directory{ filePathNew.substr(0, filePathNew.findLastOf("/") + 1) };
			if (!std::filesystem::exists(directory.operator std::basic_string_view<char, std::char_traits<char>>())) {
				std::filesystem::create_directories(directory.operator std::basic_string_view<char, std::char_traits<char>>());
			}

			if (!std::filesystem::exists(filePath.operator std::basic_string_view<char, std::char_traits<char>>())) {
				std::ofstream createFile(filePath.data());
				createFile.close();
			}

			std::ifstream theStream(filePath.data(), std::ios::binary | std::ios::in);
			std::stringstream inputStream{};
			inputStream << theStream.rdbuf();
			fileContents = inputStream.str();
			theStream.close();
		}

		BNCH_SWT_ALWAYS_INLINE void saveFile(jsonifier::string_view fileToSave) {
			std::ofstream theStream(filePath.data(), std::ios::binary | std::ios::out | std::ios::trunc);
			theStream.write(fileToSave.data(), static_cast<int64_t>(fileToSave.size()));
			if (theStream.is_open()) {
				std::cout << "File succesfully written to: " << filePath << std::endl;
			} else {
				std::cerr << "File failed to be written to: " << filePath << std::endl;
			}
			theStream.close();
		}

		BNCH_SWT_ALWAYS_INLINE operator jsonifier::string&() {
			return fileContents;
		}

	  protected:
		jsonifier::string fileContents{};
		jsonifier::string filePath{};
	};

	inline thread_local jsonifier::jsonifier_core parser{};

	template<typename value_type> using unwrap_t = jsonifier_internal::unwrap_t<value_type>;

	template<typename function_type, typename... arg_types> struct return_type_helper {
		using type = std::invoke_result_t<function_type, arg_types...>;
	};

	template<typename value_type, typename... arg_types>
	concept invocable = std::is_invocable_v<unwrap_t<value_type>, arg_types...>;

	template<typename value_type, typename... arg_types>
	concept not_invocable = !std::is_invocable_v<unwrap_t<value_type>, arg_types...>;

	template<typename value_type, typename... arg_types>
	concept invocable_void = invocable<value_type, arg_types...> && std::is_void_v<typename return_type_helper<value_type, arg_types...>::type>;

	template<typename value_type, typename... arg_types>
	concept invocable_not_void = invocable<value_type, arg_types...> && !std::is_void_v<typename return_type_helper<value_type, arg_types...>::type>;

	using clock_type = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;

	double getCpuFrequency();

#if defined(BNCH_SWT_MSVC)

	#define rdtsc() __rdtsc()

#else

	#if defined(BNCH_SWT_MAC)
		#include <mach/mach_time.h>
	#endif

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

		double seconds = static_cast<double>(nanoseconds) / 1e9;
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
	#include <sys/sysctl.h>

	#if !defined(CPU_FREQUENCY_MHZ)

	std::string exec(const char* cmd) {
		std::array<char, 128> buffer;
		std::string result;
		std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);

		if (!pipe) {
			throw std::runtime_error("Failed to open pipe for command execution.");
		}

		while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
			result += buffer.data();
		}

		if (result.empty()) {
			//throw std::runtime_error("Command executed but returned no output.");
		}

		return result;
	}

	double getCpuFrequency() {
		const char* cmd = "sysctl -n hw.cpufrequency";
		std::string output;

		try {
			output = exec(cmd);
		} catch (const std::runtime_error& e) {
			std::cerr << "Error executing command: " << e.what() << std::endl;
			return -1.0;
		}

		if (output.empty()) {
			//std::cerr << "Error: Output from sysctl command is empty." << std::endl;
			return -1.0;
		}

		try {
			double frequency = std::stod(output);
			if (frequency <= 0) {
				throw std::runtime_error("Frequency is non-positive.");
			}
			return frequency / 1e6;
		} catch (const std::invalid_argument& e) {
			std::cerr << "Error converting output to double: " << e.what() << std::endl;
			return -1.0;
		} catch (const std::out_of_range& e) {
			std::cerr << "Error: Value out of range: " << e.what() << std::endl;
			return -1.0;
		} catch (const std::runtime_error& e) {
			std::cerr << "Runtime error: " << e.what() << std::endl;
			return -1.0;
		}
	}

	#else

	double getCpuFrequency() {
		volatile int32_t counter{};
		for (int32_t i = 0; i < 1000000000; ++i) {
			++counter;
		}
		counter = 0;
		auto start = clock_type::now();
		for (int32_t i = 0; i < 1000000000; ++i) {
			++counter;
		}
		auto end = clock_type::now();
		auto newDuration = std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(end - start);
		return (static_cast<double>(counter) * newDuration.count() / 1000000000.0f) / 1e6;
	}

	#endif

#else

	double getCpuFrequency() {
		return 0.0;
	}

#endif

#if defined(small)
	#undef small
#endif

	template<typename function_type> BNCH_SWT_NO_INLINE double collectCycles(function_type&& function) {
		volatile size_t start{}, end{};
		start = rdtsc();
		std::forward<function_type>(function)();
		end = rdtsc();
		return static_cast<double>(end - start);
	}

	BNCH_SWT_ALWAYS_INLINE double cyclesToTime(double cycles, double frequencyMHz) {
		double frequencyHz	   = frequencyMHz * 1e6;
		double timeNanoseconds = (cycles * 1e9) / frequencyHz;

		return timeNanoseconds;
	}

	template<typename function_type> BNCH_SWT_ALWAYS_INLINE double collectTime(function_type&& function, double cpuFrequency) {
#if defined(BNCH_SWT_MAC)
		auto startTime = clock_type::now();
		std::forward<function_type>(function)();
		auto endTime	= clock_type::now();
		double duration = std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(endTime - startTime).count();
#else
		auto duration = collectCycles(std::forward<function_type>(function));
		duration = cyclesToTime(duration, cpuFrequency);
#endif
		return duration;
	}

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

	template<not_invocable value_type> BNCH_SWT_ALWAYS_INLINE void doNotOptimizeAway(value_type&& value) {
		const auto* valuePtr = &value;
		doNotOptimize(valuePtr)
	}

	template<invocable_void function_type, typename... arg_types> BNCH_SWT_ALWAYS_INLINE void doNotOptimizeAway(function_type&& value, arg_types&&... args) {
		std::forward<function_type>(value)(std::forward<arg_types>(args)...);
		doNotOptimize(value);
	}

	template<invocable_not_void function_type, typename... arg_types> BNCH_SWT_ALWAYS_INLINE void doNotOptimizeAway(function_type&& value, arg_types&&... args) {
		auto resultVal = std::forward<function_type>(value)(std::forward<arg_types>(args)...);
		doNotOptimize(resultVal);
	}

	BNCH_SWT_ALWAYS_INLINE double calcMedian(double* data, size_t length) {
		std::sort(data, data + length);
		auto midIdx = length / 2;
		if (length % 2 == 1) {
			return data[midIdx];
		}
		return (data[midIdx - 1] + data[midIdx]) / 2.0f;
	}

	BNCH_SWT_ALWAYS_INLINE double calcMean(double* v, size_t length) {
		double mean = 0;

		for (uint32_t i = 0; i < length; ++i) {
			mean += v[i];
		}

		mean /= length;

		return mean;
	}

	BNCH_SWT_ALWAYS_INLINE double calcStdv(double* v, size_t length, double mean) {
		double stdv = 0;

		for (uint32_t i = 0; i < length; ++i) {
			double x = v[i] - mean;

			stdv += x * x;
		}

		stdv = std::sqrt(stdv / (length + 1));

		return stdv;
	}

	BNCH_SWT_ALWAYS_INLINE double roundToDecimalPlaces(double value, int32_t decimalPlaces) {
		double scale = std::pow(10.0, decimalPlaces);
		return std::round(value * scale) / scale;
	}

	BNCH_SWT_ALWAYS_INLINE void removeOutliers(double* temp, double* v, size_t& length) {
		if (length == 0) {
			return;
		}
		double m		   = calcMean(v, length);
		double sd		   = calcStdv(v, length, m);
		double lower_bound = m - (3 * sd);
		double upper_bound = m + (3 * sd);
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

	BNCH_SWT_ALWAYS_INLINE bool checkForValidLt(double valueToCheck, double valueToCheckAgainst) {
		return std::isfinite(valueToCheck) && valueToCheck < valueToCheckAgainst;
	}

	enum class bench_state { starting = 0, running = 1, complete_success = 2, complete_failure = 3 };

	enum class result_type { unset = 0, cycles = 1, time = 2 };

	struct bench_options {
		const size_t totalIterationCountCap{ 1000 };
		result_type type{ result_type::cycles };
		const size_t maxExecutionCount{ 4 };
		size_t maxDurationCount{ 50 };
		size_t minDurationCount{ 30 };
		size_t targetCvDenom{ 100 };
		size_t maxEpochCount{ 4 };
	};

	struct benchmark_result_final_parse {
		jsonifier::string benchmarkColor{};
		jsonifier::string benchmarkName{};
		jsonifier::string libraryName{};
		size_t iterationCount{};
		bench_state state{};
		result_type type{};
		double median{};
		double cv{};
	};

	template<typename function_type, bench_options optionsNew> struct benchmark_subject {
		BNCH_SWT_ALWAYS_INLINE benchmark_subject(const jsonifier::string_view& subjectNameNew, const jsonifier::string_view& subjectColor, function_type&& functionNew)
			: function{ functionNew }, subjectName{ subjectNameNew }, benchmarkColor{ subjectColor } {
			tempDurations.resize(options.maxDurationCount);
			durations.resize(options.maxDurationCount);
		}

		BNCH_SWT_ALWAYS_INLINE benchmark_result_final_parse executeEpoch() {
			double cpuFrequency{};
			if constexpr (optionsNew.type == result_type::time) {
				cpuFrequency = getCpuFrequency();
			}
			benchmark_result_final_parse returnValues{};
			size_t currentDurationCount{};
			double currentMedian{};
			double currentMean{};
			double currentStdv{};
			bench_state state{ bench_state::running };
			double currentCv{};
			while (currentDurationCount < options.minDurationCount) {
				if constexpr (optionsNew.type == result_type::cycles) {
					collectCycles(function);
				} else {
					collectTime(function, cpuFrequency);
				}
				++currentDurationCount;
			}
			currentDurationCount = 0;
			while (currentDurationCount < options.maxDurationCount) {
				if constexpr (optionsNew.type == result_type::cycles) {
					durations[currentDurationCount] = collectCycles(function);
				} else {
					durations[currentDurationCount] = collectTime(function, cpuFrequency);
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
				options.maxDurationCount = (options.maxDurationCount * (60.0f * currentCv)) > (options.maxDurationCount * 10) ? (options.maxDurationCount * 10)
																															  : (options.maxDurationCount * (60.0f * currentCv));
				durations.resize(options.maxDurationCount);
				tempDurations.resize(options.maxDurationCount);
				targetCv += 0.01f;
			} else {
				options.maxDurationCount = (options.maxDurationCount * 2);
				durations.resize(options.maxDurationCount);
				tempDurations.resize(options.maxDurationCount);
			}
			if (totalDurationCount + options.maxDurationCount >= options.totalIterationCountCap) {
				options.maxDurationCount = options.totalIterationCountCap - totalDurationCount;
			}
			if (totalDurationCount >= options.totalIterationCountCap && (currentEpochCount >= options.maxEpochCount && state != bench_state::complete_success)) {
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
		jsonifier::vector<double> tempDurations{};
		jsonifier::string_view benchmarkColor{};
		jsonifier::vector<double> durations{};
		jsonifier::string_view subjectName{};
		bench_options options{ optionsNew };
		size_t totalDurationCount{};
		size_t currentEpochCount{};
		function_type function{};
	};

	template<string_literal stageNameNew, bench_options options> struct benchmark_stage {
		inline static jsonifier::vector<benchmark_result_final_parse> results{};

		BNCH_SWT_ALWAYS_INLINE static void printResults() {
			for (auto& value: results) {
				jsonifier::string resultType{};
				if (value.type == result_type::cycles) {
					resultType = "Cycles";
				} else {
					resultType = "Time";
				}
				std::cout << "Benchmark Name: " << value.benchmarkName << ", Library Name: " << value.libraryName << ", Result " + resultType + ": "
						  << roundToDecimalPlaces(value.median, 2) << ", Iterations: " << jsonifier::toString(value.iterationCount)
						  << ", Coefficient of Variance: " << jsonifier::toString(roundToDecimalPlaces(value.cv, 6)) << std::endl;
			}
			return;
		}

		BNCH_SWT_ALWAYS_INLINE static void writeJsonData(jsonifier::string_view filePath) {
			auto stringToWrite = parser.serializeJson(results);
			file_loader fileLoader{ filePath };
			fileLoader.saveFile(static_cast<jsonifier::string>(stringToWrite));
		}

		BNCH_SWT_ALWAYS_INLINE static void writeMarkdownData(jsonifier::string_view filePath, jsonifier::string_view repoPath) {
			jsonifier::string markdownContent = generateMarkdown(repoPath);
			file_loader fileLoader{ filePath };
			fileLoader.saveFile(static_cast<jsonifier::string>(markdownContent));
		}

		BNCH_SWT_ALWAYS_INLINE static jsonifier::string writeCsvData(jsonifier::string_view filePath) {
			jsonifier::string newString{ "benchmarkName,median,benchmarkColor" };
			for (size_t x = 0; x < results.size(); ++x) {
				newString += "\n";
				newString += results[x].benchmarkName + ",";
				newString += jsonifier::toString(results[x].median) + ",";
				newString += static_cast<jsonifier::string>(results[x].benchmarkColor);
				if (x < results.size() - 1) {
					newString += ",";
				}
			}
			file_loader fileLoader{ filePath };
			fileLoader.saveFile(static_cast<jsonifier::string>(newString));
			return {};
		}

		template<string_literal benchmarkName, string_literal subjectName, string_literal color, typename function_type>
		BNCH_SWT_ALWAYS_INLINE static benchmark_result_final_parse runBenchmark(function_type&& functionNew) {
			benchmark_subject<function_type, options> benchmarkSubject{ subjectName.operator jsonifier::string_view(), color.operator jsonifier::string_view(),
				std::forward<function_type>(functionNew) };
			auto executionLambda = [=]() mutable {
				return benchmarkSubject.executeEpoch();
			};
			size_t currentExecutionCount{};
			while (currentExecutionCount < options.maxExecutionCount) {
				++currentExecutionCount;
				benchmark_result_final_parse resultsNew = executionLambda();
				resultsNew.benchmarkName				= benchmarkName;
				if (resultsNew.state == bench_state::complete_success) {
					results.emplace_back(resultsNew);
					return resultsNew;
				}
			}
			return {};
		}

	  protected:

		BNCH_SWT_ALWAYS_INLINE static jsonifier::string urlEncode(jsonifier::string_view value) {
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

		BNCH_SWT_ALWAYS_INLINE static size_t collectUniqueLibraryCount() {
			std::unordered_set<jsonifier::string> uniqueLibraries{};
			for (auto& value: results) {
				uniqueLibraries.emplace(value.libraryName);
			}
			return uniqueLibraries.size();
		}

		BNCH_SWT_ALWAYS_INLINE static jsonifier::string generateMarkdown(jsonifier::string_view repoPath) {
			std::ostringstream markdownStream{};
			jsonifier::vector<benchmark_result_final_parse> resultsNew{};
			size_t uniqueLibraryCount{ collectUniqueLibraryCount() };
			markdownStream << "# Benchmark Results: " + stageNameNew.operator jsonifier::string_view() + "\n\n";
			jsonifier::string currentTestName{};
			size_t currentIndex{};
			for (size_t i = 0; i < results.size() / uniqueLibraryCount; ++i) {
				const auto& result = results[i];
				if (currentTestName != result.benchmarkName || i == results.size() - 1) {
					resultsNew.emplace_back(results[i]);
				}
				currentTestName = result.benchmarkName;
			}
			for (auto& value: resultsNew) {
				jsonifier::string encodedBenchmarkName = urlEncode(value.benchmarkName);
				markdownStream << "### " << currentIndex + 1 << ". " << value.benchmarkName << "\n";
				markdownStream << "<p align=\"left\"><img src=\"" << repoPath << encodedBenchmarkName << "_results.jpg?raw=true\" width=\"400\"/></p>\n\n";
				++currentIndex;
			}

			return markdownStream.str();
		}
	};

}

namespace jsonifier {

	template<> struct core<bnch_swt::benchmark_result_final_parse> {
		using value_type = bnch_swt::benchmark_result_final_parse;
		static constexpr auto parseValue =
			createValue<&value_type::benchmarkName, &value_type::median, &value_type::benchmarkColor, &value_type::iterationCount, &value_type::libraryName>();
	};
}