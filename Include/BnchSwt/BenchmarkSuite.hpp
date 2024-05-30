// BenchmarkSuite.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <BnchSwt/Config.hpp>
#include <jsonifier/Index.hpp>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>

namespace bnch_swt {

	namespace fs = std::filesystem;

	class file_loader {
	  public:
		file_loader(const std::string& filePathNew) {
			filePath = filePathNew;
			std::string directory{ filePathNew.substr(0, filePathNew.find_last_of("/") + 1) };
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

		void saveFile(const std::string& fileToSave) {
			std::ofstream theStream(filePath.data(), std::ios::binary | std::ios::out | std::ios::trunc);
			theStream.write(fileToSave.data(), fileToSave.size());
			theStream.close();
		}

		operator std::string&() {
			return fileContents;
		}

	  protected:
		std::string fileContents{};
		std::string filePath{};
	};

	inline thread_local jsonifier::jsonifier_core parser{};

	template<uint64_t sizeVal> struct string_literal {
		static constexpr uint64_t length{ sizeVal > 0 ? sizeVal - 1 : 0 };

		constexpr string_literal() noexcept = default;

		constexpr string_literal(const char (&str)[sizeVal]) {
			std::copy(str, str + length, values);
		}

		constexpr const char* data() const {
			return values;
		}

		constexpr auto size() const {
			return length;
		}

		constexpr operator std::string() const {
			return { values, length };
		}

		constexpr operator jsonifier::string_view() const {
			return { values, length };
		}

		char values[sizeVal]{};
	};

	template<size_t N> constexpr auto stringLiteralFromView(jsonifier::string_view str) {
		string_literal<N + 1> sl{};
		std::copy_n(str.data(), str.size(), sl.values);
		*(sl.values + N) = '\0';
		return sl;
	}

	BNCH_SWT_INLINE double roundToDecimalPlaces(double value, int32_t decimalPlaces) {
		double scale = std::pow(10.0, decimalPlaces);
		return std::round(value * scale) / scale;
	}

	constexpr std::size_t countDigits(uint32_t number) {
		std::size_t count = 0;
		do {
			++count;
			number /= 10;
		} while (number != 0);
		return count;
	}

	template<uint32_t number> constexpr string_literal<countDigits(number) + 1> toStringLiteral() {
		constexpr std::size_t num_digits = countDigits(number);
		char buffer[num_digits + 1]{};
		char* ptr		  = buffer + num_digits;
		*ptr			  = '\0';
		uint32_t temp = number;
		do {
			*--ptr = '0' + (temp % 10);
			temp /= 10;
		} while (temp != 0);
		return string_literal<countDigits(number) + 1>{ buffer };
	}

	template<auto valueNew> struct make_static {
		static constexpr auto value{ valueNew };
	};

	template<uint32_t number> constexpr jsonifier::string_view toStringView() {
		constexpr auto& lit = jsonifier_internal::make_static<toStringLiteral<number>()>::value;
		return jsonifier::string_view(lit.value.data(), lit.value.size() - 1);
	}

	template<bnch_swt::string_literal string01, bnch_swt::string_literal string02> constexpr auto joinLiterals() {
		char returnValue[string01.size() + string02.size() + 1]{};
		std::copy(string01.values, string01.values + string01.size(), returnValue);
		std::copy(string02.values, string02.values + string02.size(), returnValue + string01.size());
		return bnch_swt::string_literal{ returnValue };
	}

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

#if defined(BNCH_SWT_MSVC)

	#define rdtsc() __rdtsc()

#else

	__inline__ uint64_t rdtsc() {
	#ifdef __x86_64__
		uint32_t a, d;
		__asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
		return ( unsigned long )a | (( unsigned long )d << 32);
	#elif defined(__i386__)
		uint64_t x;
		__asm__ volatile("rdtsc" : "=A"(x));
		return x;
	#else
		#define NO_CYCLE_COUNTER
		return 0;
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
		return ((__rdtsc() - Start) << 5) / 1000000.0;
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
			throw std::runtime_error("Command executed but returned no output.");
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
			std::cerr << "Error: Output from sysctl command is empty." << std::endl;
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
		return static_cast<double>(counter) * newDuration.count() / 1000000000.0f;
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

	template<typename function_type> BNCH_SWT_NEVER_INLINE uint64_t collectCycles(function_type&& function) {
		volatile uint64_t start{}, end{};
		start = rdtsc();
		function();
		end = rdtsc();
		return end - start;
	}

	BNCH_SWT_INLINE double cyclesToTime(double cycles, double frequencyMHz) {
		double frequencyHz	   = frequencyMHz * 1e6;
		double timeNanoseconds = (cycles * 1e9) / frequencyHz;

		return timeNanoseconds;
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

		for (int32_t i = a; i <= b; i++) {
			mean += v[i];
		}

		mean /= (b - a + 1);

		return mean;
	}

	BNCH_SWT_INLINE double calcStdv(jsonifier::vector<double>& v, double a, double b) {
		double mean = calcMean(v, a, b);

		double stdv = 0;

		for (int32_t i = a; i <= b; i++) {
			double x = v[i] - mean;

			stdv += x * x;
		}

		stdv = sqrt(stdv / (b - a + 1));

		return stdv;
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
			stabilityThreshold = 0.01;
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

		uint64_t currentIterationCount = 0;
		while (currentIterationCount < maxIterationCount) {
			auto duration = static_cast<double>(collectCycles(lambda));
			durations.emplace_back(duration);

			if (currentIterationCount >= minIterationCount) {
				removeOutliers(durations);
				mape_return_values mapeValues = medianAbsolutePercentError(durations);

				if (checkDoubleForValidLt(mapeValues.mape, stabilityThreshold)) {
					if (currentIterationCount >= stabilityWindow) {
						mape_return_values recentMAPEValues{};
						bool stable = true;
						for (uint64_t i = 0; i < stabilityWindow; ++i) {
							if (currentIterationCount < stabilityWindow) {
								stable = false;
								break;
							}
							jsonifier::vector<double> recentDurations{ durations.end() - stabilityWindow, durations.end() };
							removeOutliers(recentDurations);
							recentMAPEValues = medianAbsolutePercentError(recentDurations);
							if (!checkDoubleForValidLt(recentMAPEValues.mape, stabilityThreshold)) {
								stable = false;
								break;
							} else {
							}
						}
						if (stable) {
							recentMAPEValues.mape *= 100;
							return { recentMAPEValues.mape, currentIterationCount, recentMAPEValues.median };
						}
					}
				}
			}
			++currentIterationCount;
		}
		auto newMapeValues = medianAbsolutePercentError(durations);
		newMapeValues.mape *= 100;
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
			auto duration = collectCycles(lambda);
			durations.emplace_back(cyclesToTime(duration, cpuFrequency));

			if (currentIterationCount >= minIterationCount) {
				removeOutliers(durations);
				mape_return_values mapeValues = medianAbsolutePercentError(durations);

				if (checkDoubleForValidLt(mapeValues.mape, stabilityThreshold)) {
					if (currentIterationCount >= stabilityWindow) {
						mape_return_values recentMAPEValues{}; 
						bool stable = true;
						for (uint64_t i = 0; i < stabilityWindow; ++i) {
							if (currentIterationCount < stabilityWindow) {
								stable = false;
								break;
							}
							jsonifier::vector<double> recentDurations{ durations.end() - stabilityWindow, durations.end() };
							removeOutliers(recentDurations);
							recentMAPEValues = medianAbsolutePercentError(recentDurations);
							if (!checkDoubleForValidLt(recentMAPEValues.mape, stabilityThreshold)) {
								stable = false;
								break;
							} else{

							}
						}
						if (stable) {
							recentMAPEValues.mape *= 100;
							return { recentMAPEValues.mape, currentIterationCount, recentMAPEValues.median };
						}
					}
				}
			}
			++currentIterationCount;
		}
		auto newMapeValues = medianAbsolutePercentError(durations);
		newMapeValues.mape *= 100;
		return { newMapeValues.mape, currentIterationCount, newMapeValues.median };
	}

	template<uint64_t iterationCount, typename function_type> BNCH_SWT_INLINE benchmark_results benchmark(function_type&& function) {
		static constexpr int64_t warmupCount	   = iterationCount;
		static constexpr int64_t minIterationCount = iterationCount / 10;
		using function_type_final				   = std::decay_t<function_type>;
		function_type_final functionNew{ std::forward<function_type>(function) };

		jsonifier::vector<function_type_final> warmupLambdas{};
		for (uint64_t x = 0; x < warmupCount; ++x) {
			warmupLambdas.emplace_back(functionNew);
		}

		jsonifier::vector<function_type_final> lambdas{};
		for (uint64_t x = 0; x < iterationCount; ++x) {
			lambdas.emplace_back(functionNew);
		}

		int64_t currentIterationCount = 0;
		jsonifier::vector<double> durations{};

		while (currentIterationCount < warmupCount) {
			auto startTime = std::chrono::high_resolution_clock::now();
			doNotOptimizeAway(std::move(warmupLambdas[currentIterationCount]));
			auto endTime = std::chrono::high_resolution_clock::now();
			++currentIterationCount;
		}

		currentIterationCount = 0;
		return collectMape<iterationCount, minIterationCount>(functionNew);
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

	struct benchmark_result_final {
		jsonifier::string_view benchmarkColor{};
		jsonifier::string_view benchmarkName{};
		double medianAbsolutePercentageError{};
		jsonifier::string_view libraryName{};
		uint64_t iterationCount{};
		double resultValue{};

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
			returnValues.resultValue					   = resultValue + other.resultValue;
			return returnValues;
		}

		BNCH_SWT_INLINE benchmark_result_final operator/(const uint64_t& other) const {
			benchmark_result_final returnValues{};
			returnValues.medianAbsolutePercentageError = medianAbsolutePercentageError;
			returnValues.benchmarkColor				   = benchmarkColor;
			returnValues.benchmarkName				   = benchmarkName;
			returnValues.resultValue					   = resultValue / other;
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
				std::cout << "Benchmark Name: " << value.benchmarkName << ", Library Name: " << value.libraryName
						  << ", MAPE: " << roundToDecimalPlaces(value.medianAbsolutePercentageError, 4) << ", Result Time: " << roundToDecimalPlaces(value.resultValue, 2)
						  << std::endl;
			}
			return;
		}

		BNCH_SWT_INLINE static std::string writeJsonData(const std::string& filePath) {
			benchmark_suite_results newValues{ results };
			auto stringToWrite = parser.serializeJson(newValues);
			file_loader fileLoader{ filePath };
			fileLoader.saveFile(static_cast<std::string>(stringToWrite));
			return {};
		}

		BNCH_SWT_INLINE static std::string urlEncode(const jsonifier::string& value) {
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

		BNCH_SWT_INLINE static std::string generateMarkdown(const jsonifier::vector<benchmark_result_final_parse>& results) {
			std::ostringstream markdownStream{};
			jsonifier::vector<benchmark_result_final_parse> resultsNew{};

			markdownStream << "# Benchmark Results: " + benchmarkSuite.operator jsonifier::string_view() + "\n\n";
			std::string currentTestName{};
			uint64_t currentIndex{};
			for (size_t i = 0; i < results.size() / 3; ++i) {
				const auto& result = results[i];
				if (currentTestName != result.benchmarkName || i == results.size() - 1) {
					resultsNew.emplace_back(results[i]);
				}
				currentTestName = result.benchmarkName;
			}
			for (auto& value: resultsNew) {
				std::string encodedBenchmarkName = urlEncode(value.benchmarkName);
				markdownStream << "### " << currentIndex + 1 << ". " << value.benchmarkName << "\n";
				markdownStream << "<p align=\"left\"><img src=\"https://github.com/RealTimeChris/BenchmarkSuite/blob/main/Graphs/" << encodedBenchmarkName
							   << "_results.jpg?raw=true\" width=\"400\"/></p>\n\n";
				++currentIndex;
			}

			return markdownStream.str();
		}

		BNCH_SWT_INLINE static void writeMarkdownToFile(const std::string& filename) {
			benchmark_suite_results newValues{ results };
			std::string markdownContent = generateMarkdown(newValues.results);
			std::ofstream file(filename);
			if (file.is_open()) {
				file << markdownContent;
				file.close();
			} else {
				std::cerr << "Unable to open file for writing: " << filename << std::endl;
			}
		}

		BNCH_SWT_INLINE static std::string writeCsvData(const std::string& filePath) {
			benchmark_suite_results newValues{ results };
			std::string newString{ "benchmarkName,medianAbsolutePercentageError,resultValue,benchmarkColor" };
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
			fileLoader.saveFile(static_cast<std::string>(newString));
			return {};
		}

		template<string_literal benchmarkName, string_literal libraryName, string_literal benchmarkColor, int64_t maxIterationCount, invocable_void function_type,
			typename... arg_types>
		static BNCH_SWT_INLINE auto benchmark(function_type&& function, arg_types&&... args) {
			static constexpr int64_t minIterationCount = maxIterationCount / 10;
			;
			static constexpr int64_t warmupCount = minIterationCount;
			using function_type_final			 = jsonifier_internal::unwrap_t<function_type>;
			auto functionNew					 = [=] {
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
			static constexpr int64_t minIterationCount = maxIterationCount / 10;
			;
			static constexpr int64_t warmupCount = minIterationCount;
			using function_type_final			 = jsonifier_internal::unwrap_t<function_type>;
			auto functionNew					 = [=] {
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
			static constexpr int64_t minIterationCount = maxIterationCount / 10;
			static constexpr int64_t warmupCount	   = minIterationCount;
			using function_type_final				   = jsonifier_internal::unwrap_t<function_type>;
			auto functionNew						   = [=] {
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
			static constexpr int64_t minIterationCount = maxIterationCount / 10;
			;
			static constexpr int64_t warmupCount = minIterationCount;
			using function_type_final			 = jsonifier_internal::unwrap_t<function_type>;
			auto functionNew					 = [=] {
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
		static constexpr auto parseValue = createValue<&value_type::benchmarkName, &value_type::medianAbsolutePercentageError, &value_type::resultValue, &value_type::benchmarkColor,
			&value_type::iterationCount, &value_type::libraryName>();
	};

	template<> struct core<bnch_swt::benchmark_result_final> {
		using value_type				 = bnch_swt::benchmark_result_final;
		static constexpr auto parseValue = createValue<&value_type::benchmarkName, &value_type::medianAbsolutePercentageError, &value_type::resultValue, &value_type::benchmarkColor,
			&value_type::iterationCount>();
	};

	template<> struct core<bnch_swt::benchmark_suite_results> {
		using value_type				 = bnch_swt::benchmark_suite_results;
		static constexpr auto parseValue = createValue<&value_type::results, &value_type::benchmarkSuiteName>();
	};
}