// BenchmarkSuite.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <BnchSwt/StringLiteral.hpp>
#include <jsonifier/Index.hpp>
#include <BnchSwt/Config.hpp>
#include <unordered_set>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <vector>

namespace bnch_swt {

	namespace fs = std::filesystem;

	class file_loader {
	  public:
		file_loader(jsonifier::string_view filePathNew) {
			filePath = filePathNew;
			jsonifier::string directory{ filePathNew.substr(0, filePathNew.findLastOf("/") + 1) };
			if (!fs::exists(directory.operator std::basic_string_view<char, std::char_traits<char>>())) {
				std::filesystem::create_directories(directory.operator std::basic_string_view<char, std::char_traits<char>>());
			}

			if (!fs::exists(filePath.operator std::basic_string_view<char, std::char_traits<char>>())) {
				std::ofstream createFile(filePath.data());
				createFile.close();
			}

			std::ifstream theStream(filePath.data(), std::ios::binary | std::ios::in);
			std::stringstream inputStream{};
			inputStream << theStream.rdbuf();
			fileContents = inputStream.str();
			theStream.close();
		}

		void saveFile(jsonifier::string_view fileToSave) {
			std::ofstream theStream(filePath.data(), std::ios::binary | std::ios::out | std::ios::trunc);
			theStream.write(fileToSave.data(), static_cast<int64_t>(fileToSave.size()));
			if (theStream.is_open()) {
				std::cout << "File succesfully written to: " << filePath << std::endl;
			} else {
				std::cerr << "File failed to be written to: " << filePath << std::endl;
			}
			theStream.close();
		}

		operator jsonifier::string&() {
			return fileContents;
		}

	  protected:
		jsonifier::string fileContents{};
		jsonifier::string filePath{};
	};

	inline thread_local jsonifier::jsonifier_core parser{};

	static void const volatile* volatile globalForceEscapePointer;

	void useCharPointer(char const volatile* const v) {
		globalForceEscapePointer = reinterpret_cast<void const volatile*>(v);
	}

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

	double getCpuFrequency();

#if defined(BNCH_SWT_MSVC)

	#define rdtsc(x) __rdtsc()

#else

	#if defined(BNCH_SWT_MAC)
		#include <mach/mach_time.h>
	#endif

	__inline__ uint64_t rdtsc(double cpuFrequency = 0.0f) {
		( void )cpuFrequency;
	#ifdef __x86_64__
		uint32_t a, d;
		__asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
		return ( unsigned long )a | (( unsigned long )d << 32);
	#elif defined(__i386__)
		uint64_t x;
		__asm__ volatile("rdtsc" : "=A"(x));
		return x;
	#else
		mach_timebase_info_data_t timebase_info;
		mach_timebase_info(&timebase_info);
		uint64_t nanoseconds = mach_absolute_time() * timebase_info.numer / timebase_info.denom;

		double seconds	  = static_cast<double>(nanoseconds) / 1e9;
		double cpu_cycles = seconds * cpuFrequency;
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
		counter	   = 0;
		auto start = std::chrono::high_resolution_clock::now();
		for (int32_t i = 0; i < 1000000000; ++i) {
			++counter;
		}
		auto end		 = std::chrono::high_resolution_clock::now();
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

	template<typename function_type> BNCH_SWT_NO_INLINE double collectCycles(function_type&& function, double cpuFrequency) {
		volatile uint64_t start{}, end{};
		start = rdtsc(cpuFrequency);
		function();
		end = rdtsc(cpuFrequency);
		return static_cast<double>(end - start);
	}

	BNCH_SWT_INLINE double cyclesToTime(double cycles, double frequencyMHz) {
		double frequencyHz	   = frequencyMHz * 1e6;
		double timeNanoseconds = (cycles * 1e9) / frequencyHz;

		return timeNanoseconds;
	}

	template<typename function_type> BNCH_SWT_INLINE double collectTime(function_type&& function, double cpuFrequency) {
#if defined(BNCH_SWT_MAC)
		auto startTime = std::chrono::high_resolution_clock::now();
		function();
		auto endTime	= std::chrono::high_resolution_clock::now();
		double duration = std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(endTime - startTime).count();
#else
		auto duration = collectCycles(function, cpuFrequency);
		duration	  = cyclesToTime(duration, cpuFrequency);
#endif
		return duration;
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

	BNCH_SWT_INLINE double calcMedian(jsonifier::vector<double>& data) {
		std::sort(data.begin(), data.end());
		auto midIdx = data.size() / 2;
		if (data.size() % 2 == 1) {
			return data[midIdx];
		}
		return (data[midIdx - 1] + data[midIdx]) / 2.0f;
	}

	BNCH_SWT_INLINE double calcMean(jsonifier::vector<double>& v, double a, double b) {
		double mean = 0;

		for (uint32_t i = static_cast<uint32_t>(a); i <= static_cast<uint32_t>(b); i++) {
			mean += v[i];
		}

		mean /= (b - a + 1);

		return mean;
	}

	BNCH_SWT_INLINE double calcStdv(jsonifier::vector<double>& v, double a, double b) {
		double mean = calcMean(v, a, b);

		double stdv = 0;

		for (uint32_t i = static_cast<uint32_t>(a); i <= static_cast<uint32_t>(b); i++) {
			double x = v[i] - mean;

			stdv += x * x;
		}

		stdv = sqrt(stdv / (b - a + 1));

		return stdv;
	}

	BNCH_SWT_INLINE double roundToDecimalPlaces(double value, int32_t decimalPlaces) {
		double scale = std::pow(10.0, decimalPlaces);
		return std::round(value * scale) / scale;
	}

	BNCH_SWT_INLINE bool containsOutlier(jsonifier::vector<double>& v, size_t len) {
		double mean = 0;

		for (size_t i = 0; i < len; i++) {
			mean += v[i];
		}

		mean /= double(len);

		double stdv = 0;

		for (size_t i = 0; i < len; i++) {
			double x = v[i] - mean;
			stdv += x * x;
		}

		stdv = sqrt(stdv / double(len));

		double cutoff = mean + stdv * 3;

		return v[len - 1] > cutoff;
	}

	BNCH_SWT_INLINE void removeOutliers(jsonifier::vector<double>& v) {
		std::sort(v.begin(), v.end());

		size_t len = 0;

		for (size_t x = 0x40000000; x; x = x >> 1) {
			if ((len | x) >= v.size())
				continue;

			if (!containsOutlier(v, len | x)) {
				len |= x;
			}
		}

		v.resize(len);
	}

	struct mape_return_values {
		double median{};
		double mape{};
	};

	BNCH_SWT_INLINE static mape_return_values medianAbsolutePercentError(const jsonifier::vector<double>& data) {
		mape_return_values returnValues{};
		jsonifier::vector<double> dataNew{ data };
		if (dataNew.empty()) {
			return {};
		}
		returnValues.median = calcMedian(dataNew);
		for (double& x: dataNew) {
			x = (x - returnValues.median) / x;
			if (x < 0.0f) {
				x = -x;
			}
		}
		returnValues.mape = calcMedian(dataNew);
		return returnValues;
	}

	BNCH_SWT_INLINE bool checkDoubleForValidLt(double valueToCheck, double valueToCheckAgainst) {
		return std::isfinite(valueToCheck) && valueToCheck < valueToCheckAgainst;
	}

	BNCH_SWT_INLINE void determineStabilityParameters(uint64_t maxIterationCount, double& stabilityThreshold, uint64_t& stabilityWindow) {
		if (maxIterationCount < 100) {
			stabilityThreshold = 0.05;
			stabilityWindow	   = 3;
		} else if (maxIterationCount < 1000) {
			stabilityThreshold = 0.005;
			stabilityWindow	   = 5;
		} else {
			stabilityThreshold = 0.001;
			stabilityWindow	   = 10;
		}
	}

	struct benchmark_results {
		double medianAbsolutePercentageError{};
		uint64_t currentIterationCount{};
		double resultValue{};
	};

	template<uint64_t maxIterationCount, uint64_t minIterationCount, typename FunctionType>
	BNCH_SWT_INLINE benchmark_results collectMapeCycles(FunctionType&& lambda, double stabilityThreshold, uint64_t stabilityWindow = 5) {
		jsonifier::vector<double> durations{};
		durations.reserve(maxIterationCount);
		for (uint64_t i = 0; i < minIterationCount; ++i) {
			lambda();
		}
		auto cpuFrequency			   = getCpuFrequency();
		uint64_t currentIterationCount = 0;
		while (currentIterationCount < maxIterationCount) {
			auto duration = static_cast<double>(collectCycles(lambda, cpuFrequency));
			durations.emplace_back(duration);

			if (currentIterationCount >= minIterationCount) {
				removeOutliers(durations);
				mape_return_values mapeValues = medianAbsolutePercentError(durations);

				if (checkDoubleForValidLt(mapeValues.mape, stabilityThreshold)) {
					if (currentIterationCount >= stabilityWindow) {
						mape_return_values recentMapeValues{};
						bool stable = true;
						for (uint64_t i = 0; i < stabilityWindow; ++i) {
							if (currentIterationCount < stabilityWindow) {
								stable = false;
								break;
							}
							jsonifier::vector<double> recentDurations{ durations.end() - stabilityWindow, durations.end() };
							removeOutliers(recentDurations);
							recentMapeValues = medianAbsolutePercentError(recentDurations);
							if (!checkDoubleForValidLt(recentMapeValues.mape, stabilityThreshold)) {
								stable = false;
								break;
							} else {
							}
						}
						if (stable) {
							recentMapeValues.mape = recentMapeValues.mape * 100;
							return { recentMapeValues.mape, currentIterationCount, recentMapeValues.median };
						}
					}
				}
			}
			++currentIterationCount;
		}
		auto newMapeValues = medianAbsolutePercentError(durations);
		newMapeValues.mape = newMapeValues.mape * 100;
		return { newMapeValues.mape, currentIterationCount, newMapeValues.median };
	}

	template<uint64_t maxIterationCount, uint64_t minIterationCount, typename function_type>
	inline benchmark_results collectMape(const function_type& lambda, double stabilityThreshold, uint64_t stabilityWindow = 5) {
		jsonifier::vector<double> durations{};

		for (uint64_t i = 0; i < minIterationCount; ++i) {
			lambda();
		}

		auto cpuFrequency			   = getCpuFrequency();
		uint64_t currentIterationCount = 0;

		while (currentIterationCount < maxIterationCount) {
			auto duration = collectTime(lambda, cpuFrequency);
			durations.emplace_back(duration);

			if (currentIterationCount >= minIterationCount) {
				removeOutliers(durations);
				mape_return_values mapeValues = medianAbsolutePercentError(durations);

				if (checkDoubleForValidLt(mapeValues.mape, stabilityThreshold)) {
					if (currentIterationCount >= stabilityWindow) {
						mape_return_values recentMapeValues{};
						bool stable = true;
						for (uint64_t i = 0; i < stabilityWindow; ++i) {
							if (currentIterationCount < stabilityWindow) {
								stable = false;
								break;
							}
							jsonifier::vector<double> recentDurations{ durations.end() - stabilityWindow, durations.end() };
							removeOutliers(recentDurations);
							recentMapeValues = medianAbsolutePercentError(recentDurations);
							if (!checkDoubleForValidLt(recentMapeValues.mape, stabilityThreshold)) {
								stable = false;
								break;
							} else {
							}
						}
						if (stable) {
							recentMapeValues.mape = recentMapeValues.mape * 100;
							return { recentMapeValues.mape, currentIterationCount, recentMapeValues.median };
						}
					}
				}
			}
			++currentIterationCount;
		}
		auto newMapeValues = medianAbsolutePercentError(durations);
		newMapeValues.mape = newMapeValues.mape * 100;
		return { newMapeValues.mape, currentIterationCount, newMapeValues.median };
	}

	template<typename return_type> struct benchmark_result {
		return_type returnValue{};
		uint64_t resultSize{};
		double resultValue{};
		BNCH_SWT_INLINE bool operator<(const benchmark_result& other) const {
			return resultValue < other.resultValue;
		}
		BNCH_SWT_INLINE benchmark_result operator+(const benchmark_result& other) const {
			return benchmark_result{ .returnValue = returnValue, .resultValue = resultValue + other.resultValue };
		}

		template<typename value_type> BNCH_SWT_INLINE benchmark_result operator/(const value_type& other) const {
			return benchmark_result{ .returnValue = returnValue, .resultValue = resultValue / other };
		}
	};

	template<> struct benchmark_result<void> {
		uint64_t resultSize{};
		double resultValue{};
		BNCH_SWT_INLINE bool operator<(const benchmark_result& other) const {
			return resultValue < other.resultValue;
		}
		BNCH_SWT_INLINE benchmark_result operator+(const benchmark_result& other) const {
			return benchmark_result{ .resultValue = resultValue + other.resultValue };
		}

		template<typename value_type> BNCH_SWT_INLINE benchmark_result operator/(const value_type& other) const {
			return benchmark_result{ .resultValue = resultValue / other };
		}
	};

	enum class result_type { unset = 0, cycles = 1, time = 2 };

	struct benchmark_result_final {
		jsonifier::string_view benchmarkColor{};
		jsonifier::string_view benchmarkName{};
		double medianAbsolutePercentageError{};
		jsonifier::string_view libraryName{};
		uint64_t iterationCount{};
		double resultValue{};
		result_type type{};

		BNCH_SWT_INLINE benchmark_result_final() noexcept = default;

		BNCH_SWT_INLINE benchmark_result_final& operator=(benchmark_result<void>&& other) {
			resultValue = other.resultValue;
			return *this;
		}

		BNCH_SWT_INLINE benchmark_result_final(benchmark_result<void>&& other) {
			*this = std::move(other);
		}

		BNCH_SWT_INLINE benchmark_result_final& operator=(const benchmark_result<void>& other) {
			resultValue = other.resultValue;
			return *this;
		}

		BNCH_SWT_INLINE benchmark_result_final(const benchmark_result<void>& other) {
			*this = other;
		}

		template<typename value_type> BNCH_SWT_INLINE benchmark_result_final& operator=(benchmark_result<value_type>&& other) {
			resultValue = other.resultValue;
			return *this;
		}

		template<typename value_type> BNCH_SWT_INLINE benchmark_result_final(benchmark_result<value_type>&& other) {
			*this = std::move(other);
		}

		template<typename value_type> BNCH_SWT_INLINE benchmark_result_final& operator=(const benchmark_result<value_type>& other) {
			resultValue = other.resultValue;
			return *this;
		}

		template<typename value_type> BNCH_SWT_INLINE benchmark_result_final(const benchmark_result<value_type>& other) {
			*this = other;
		}

		BNCH_SWT_INLINE bool operator<(const benchmark_result_final& other) const {
			return resultValue < other.resultValue;
		}

		BNCH_SWT_INLINE benchmark_result_final operator+(const benchmark_result_final& other) const {
			benchmark_result_final returnValues{};
			returnValues.medianAbsolutePercentageError = other.medianAbsolutePercentageError;
			returnValues.benchmarkColor				   = other.benchmarkColor;
			returnValues.benchmarkName				   = other.benchmarkName;
			returnValues.resultValue				   = resultValue + other.resultValue;
			returnValues.type						   = other.type;
			return returnValues;
		}

		template<typename value_type> BNCH_SWT_INLINE benchmark_result_final operator/(value_type other) const {
			benchmark_result_final returnValues{};
			returnValues.medianAbsolutePercentageError = medianAbsolutePercentageError;
			returnValues.benchmarkColor				   = benchmarkColor;
			returnValues.benchmarkName				   = benchmarkName;
			returnValues.resultValue				   = resultValue / other;
			return returnValues;
		}
	};

	struct benchmark_result_final_parse {
		double medianAbsolutePercentageError{};
		jsonifier::string_view libraryName{};
		jsonifier::string benchmarkColor{};
		jsonifier::string benchmarkName{};
		uint64_t iterationCount{};
		double resultValue{};

		BNCH_SWT_INLINE benchmark_result_final_parse() noexcept = default;

		BNCH_SWT_INLINE benchmark_result_final_parse& operator=(benchmark_result_final&& other) {
			benchmarkColor				  = static_cast<jsonifier::string>(other.benchmarkColor);
			benchmarkName				  = static_cast<jsonifier::string>(other.benchmarkName);
			medianAbsolutePercentageError = other.medianAbsolutePercentageError;
			iterationCount				  = other.iterationCount;
			libraryName					  = other.libraryName;
			resultValue					  = other.resultValue;
			return *this;
		}

		BNCH_SWT_INLINE benchmark_result_final_parse(benchmark_result_final&& other) {
			*this = std::move(other);
		}

		BNCH_SWT_INLINE benchmark_result_final_parse& operator=(const benchmark_result_final& other) {
			benchmarkColor				  = static_cast<jsonifier::string>(other.benchmarkColor);
			benchmarkName				  = static_cast<jsonifier::string>(other.benchmarkName);
			medianAbsolutePercentageError = other.medianAbsolutePercentageError;
			iterationCount				  = other.iterationCount;
			libraryName					  = other.libraryName;
			resultValue					  = other.resultValue;
			return *this;
		}

		BNCH_SWT_INLINE benchmark_result_final_parse(const benchmark_result_final& other) {
			*this = other;
		}
	};

	template<string_literal str> struct benchmark_suite_results_stored {
		constexpr benchmark_suite_results_stored() noexcept = default;
		mutable jsonifier::vector<benchmark_result_final> results{};
		mutable jsonifier::string_view benchmarkSuiteName{ str.operator jsonifier::string_view() };
	};

	struct benchmark_suite_results {
		BNCH_SWT_INLINE benchmark_suite_results() noexcept = default;
		BNCH_SWT_INLINE benchmark_suite_results(auto& values) {
			for (uint64_t x = 0; x < values.results.size(); ++x) {
				results.emplace_back(values.results[x]);
			}
			benchmarkSuiteName = values.benchmarkSuiteName;
		}
		jsonifier::vector<benchmark_result_final_parse> results{};
		jsonifier::string_view benchmarkSuiteName{};
	};

	template<typename value_type> std::ostream& operator<<(std::ostream& os, const jsonifier::vector<value_type>& vector) {
		os << '[';
		for (uint64_t x = 0; x < vector.size(); ++x) {
			os << vector[x];
			if (x < vector.size() - 1) {
				os << ',';
			}
		}
		os << ']' << std::endl;
		return os;
	}

	template<string_literal benchmarkSuite> struct benchmark_suite {
		inline static benchmark_suite_results_stored<benchmarkSuite> results{};
		constexpr benchmark_suite() noexcept {};
		BNCH_SWT_INLINE static void printResults() {
			benchmark_suite_results newValues{ results };
			for (auto& value: results.results) {
				jsonifier::string resultType{};
				if (value.type == result_type::cycles) {
					resultType = "Cycles";
				} else {
					resultType = "Time";
				}
				std::cout << "Benchmark Name: " << value.benchmarkName << ", Library Name: " << value.libraryName
						  << ", Mape: " << roundToDecimalPlaces(value.medianAbsolutePercentageError, 4) << ", Result " + resultType + ": "
						  << roundToDecimalPlaces(value.resultValue, 2) << std::endl;
			}
			return;
		}

		BNCH_SWT_INLINE static void writeJsonData(jsonifier::string_view filePath) {
			benchmark_suite_results newValues{ results };
			auto stringToWrite = parser.serializeJson(newValues);
			file_loader fileLoader{ filePath };
			fileLoader.saveFile(static_cast<jsonifier::string>(stringToWrite));
		}

		BNCH_SWT_INLINE static jsonifier::string urlEncode(jsonifier::string_view value) {
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

		BNCH_SWT_INLINE static uint64_t collectUniqueLibraryCount() {
			std::unordered_set<jsonifier::string> uniqueLibraries{};
			for (auto& value: results.results) {
				uniqueLibraries.emplace(value.libraryName);
			}
			return uniqueLibraries.size();
		}

		BNCH_SWT_INLINE static jsonifier::string generateMarkdown(jsonifier::string_view repoPath) {
			std::ostringstream markdownStream{};
			jsonifier::vector<benchmark_result_final_parse> resultsNew{};
			benchmark_suite_results newValues{ results };
			uint64_t uniqueLibraryCount{ collectUniqueLibraryCount() };
			markdownStream << "# Benchmark Results: " + benchmarkSuite.operator jsonifier::string_view() + "\n\n";
			jsonifier::string currentTestName{};
			uint64_t currentIndex{};
			for (size_t i = 0; i < newValues.results.size() / uniqueLibraryCount; ++i) {
				const auto& result = newValues.results[i];
				if (currentTestName != result.benchmarkName || i == newValues.results.size() - 1) {
					resultsNew.emplace_back(newValues.results[i]);
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

		BNCH_SWT_INLINE static void writeMarkdownToFile(jsonifier::string_view filePath, jsonifier::string_view repoPath) {
			jsonifier::string markdownContent = generateMarkdown(repoPath);
			file_loader fileLoader{ filePath };
			fileLoader.saveFile(static_cast<jsonifier::string>(markdownContent));
		}

		BNCH_SWT_INLINE static jsonifier::string writeCsvData(jsonifier::string_view filePath) {
			benchmark_suite_results newValues{ results };
			jsonifier::string newString{ "benchmarkName,medianAbsolutePercentageError,resultValue,benchmarkColor" };
			for (uint64_t x = 0; x < newValues.results.size(); ++x) {
				newString += "\n";
				newString += newValues.results[x].benchmarkName + ",";
				newString += jsonifier::toString(newValues.results[x].medianAbsolutePercentageError) + ",";
				newString += jsonifier::toString(newValues.results[x].resultValue) + ",";
				newString += static_cast<jsonifier::string>(newValues.results[x].benchmarkColor);
				if (x < newValues.results.size() - 1) {
					newString += ",";
				}
			}
			file_loader fileLoader{ filePath };
			fileLoader.saveFile(static_cast<jsonifier::string>(newString));
			return {};
		}

		template<string_literal benchmarkName, string_literal libraryName, string_literal benchmarkColor, int64_t maxIterationCount, invocable_void function_type,
			typename... arg_types>
		static BNCH_SWT_INLINE auto benchmark(function_type&& function, arg_types&&... args) {
			static constexpr int64_t minIterationCount = maxIterationCount / 5;

			using function_type_final = jsonifier_internal::unwrap_t<function_type>;
			auto functionNew		  = [=] {
				 return function(args...);
			};

			double stabilityThreshold{};
			uint64_t stabilityWindow{};
			determineStabilityParameters(maxIterationCount, stabilityThreshold, stabilityWindow);
			for (uint64_t x = 0; x < minIterationCount; ++x) {
				functionNew();
			}
			benchmark_results results = collectMape<maxIterationCount, minIterationCount>(functionNew, stabilityThreshold, stabilityWindow);
			benchmark_result_final resultsReal{};
			resultsReal.type						  = result_type::time;
			resultsReal.medianAbsolutePercentageError = results.medianAbsolutePercentageError;
			resultsReal.resultValue					  = results.resultValue;
			resultsReal.benchmarkName				  = benchmarkName.operator jsonifier::string_view();
			resultsReal.libraryName					  = libraryName;
			resultsReal.benchmarkColor				  = benchmarkColor.operator jsonifier::string_view();
			resultsReal.iterationCount				  = results.currentIterationCount;
			benchmark_suite<benchmarkSuite>::results.results.emplace_back(resultsReal);
			return results;
		}

		template<string_literal benchmarkName, string_literal libraryName, string_literal benchmarkColor, int64_t maxIterationCount, invocable_not_void function_type,
			typename... arg_types>
		static BNCH_SWT_INLINE auto benchmark(function_type&& function, arg_types&&... args) {
			static constexpr int64_t minIterationCount = maxIterationCount / 5;

			using function_type_final = jsonifier_internal::unwrap_t<function_type>;
			auto functionNew		  = [=] {
				 return function(args...);
			};

			double stabilityThreshold{};
			uint64_t stabilityWindow{};
			determineStabilityParameters(maxIterationCount, stabilityThreshold, stabilityWindow);
			for (uint64_t x = 0; x < minIterationCount; ++x) {
				functionNew();
			}
			benchmark_results results = collectMape<maxIterationCount, minIterationCount>(functionNew, stabilityThreshold, stabilityWindow);
			benchmark_result_final resultsReal{};
			resultsReal.type						  = result_type::time;
			resultsReal.medianAbsolutePercentageError = results.medianAbsolutePercentageError;
			resultsReal.resultValue					  = results.resultValue;
			resultsReal.libraryName					  = libraryName;
			resultsReal.benchmarkName				  = benchmarkName.operator jsonifier::string_view();
			resultsReal.benchmarkColor				  = benchmarkColor.operator jsonifier::string_view();
			resultsReal.iterationCount				  = results.currentIterationCount;
			benchmark_suite<benchmarkSuite>::results.results.emplace_back(resultsReal);
			return results;
		}

		template<string_literal benchmarkName, string_literal libraryName, string_literal benchmarkColor, int64_t maxIterationCount, invocable_void function_type,
			typename... arg_types>
		static BNCH_SWT_INLINE auto benchmarkCycles(function_type&& function, arg_types&&... args) {
			static constexpr int64_t minIterationCount = maxIterationCount / 5;

			using function_type_final = jsonifier_internal::unwrap_t<function_type>;
			auto functionNew		  = [=] {
				 return function(args...);
			};

			double stabilityThreshold{};
			uint64_t stabilityWindow{};
			determineStabilityParameters(maxIterationCount, stabilityThreshold, stabilityWindow);
			for (uint64_t x = 0; x < minIterationCount; ++x) {
				functionNew();
			}
			benchmark_results results = collectMapeCycles<maxIterationCount, minIterationCount>(functionNew, stabilityThreshold, stabilityWindow);
			benchmark_result_final resultsReal{};
			resultsReal.type						  = result_type::cycles;
			resultsReal.medianAbsolutePercentageError = results.medianAbsolutePercentageError;
			resultsReal.resultValue					  = results.resultValue;
			resultsReal.benchmarkName				  = benchmarkName.operator jsonifier::string_view();
			resultsReal.libraryName					  = libraryName;
			resultsReal.benchmarkColor				  = benchmarkColor.operator jsonifier::string_view();
			resultsReal.iterationCount				  = results.currentIterationCount;
			benchmark_suite<benchmarkSuite>::results.results.emplace_back(resultsReal);
			return results;
		}

		template<string_literal benchmarkName, string_literal libraryName, string_literal benchmarkColor, int64_t maxIterationCount, invocable_not_void function_type,
			typename... arg_types>
		static BNCH_SWT_INLINE auto benchmarkCycles(function_type&& function, arg_types&&... args) {
			static constexpr int64_t minIterationCount = maxIterationCount / 5;

			using function_type_final = jsonifier_internal::unwrap_t<function_type>;
			auto functionNew		  = [=] {
				 return function(args...);
			};

			double stabilityThreshold{};
			uint64_t stabilityWindow{};
			determineStabilityParameters(maxIterationCount, stabilityThreshold, stabilityWindow);
			for (uint64_t x = 0; x < minIterationCount; ++x) {
				functionNew();
			}
			benchmark_results results = collectMapeCycles<maxIterationCount, minIterationCount>(functionNew, stabilityThreshold, stabilityWindow);
			benchmark_result_final resultsReal{};
			resultsReal.type						  = result_type::cycles;
			resultsReal.medianAbsolutePercentageError = results.medianAbsolutePercentageError;
			resultsReal.resultValue					  = results.resultValue;
			resultsReal.libraryName					  = libraryName;
			resultsReal.benchmarkName				  = benchmarkName.operator jsonifier::string_view();
			resultsReal.benchmarkColor				  = benchmarkColor.operator jsonifier::string_view();
			resultsReal.iterationCount				  = results.currentIterationCount;
			benchmark_suite<benchmarkSuite>::results.results.emplace_back(resultsReal);
			return results;
		}
	};

}

namespace jsonifier {

	template<> struct core<bnch_swt::benchmark_result_final_parse> {
		using value_type				 = bnch_swt::benchmark_result_final_parse;
		static constexpr auto parseValue = createValue<&value_type::benchmarkName, &value_type::medianAbsolutePercentageError, &value_type::resultValue,
			&value_type::benchmarkColor, &value_type::iterationCount, &value_type::libraryName>();
	};

	template<> struct core<bnch_swt::benchmark_result_final> {
		using value_type				 = bnch_swt::benchmark_result_final;
		static constexpr auto parseValue = createValue<&value_type::benchmarkName, &value_type::medianAbsolutePercentageError, &value_type::resultValue,
			&value_type::benchmarkColor, &value_type::iterationCount>();
	};

	template<> struct core<bnch_swt::benchmark_suite_results> {
		using value_type				 = bnch_swt::benchmark_suite_results;
		static constexpr auto parseValue = createValue<&value_type::results, &value_type::benchmarkSuiteName>();
	};
}