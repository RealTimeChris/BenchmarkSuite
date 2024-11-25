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

template<typename value_type>
concept printable = requires(std::remove_cvref_t<value_type> value) { std::cout << value << std::endl; };

template<typename value_type>
concept not_printable = !printable<value_type>;

template<not_printable value_type> void printValue(std::ostream& os, const value_type& value) {
	std::cout << "Not a printable value, of type: " << typeid(value_type).name() << std::endl;
}

template<printable value_type> void printValue(std::ostream& os, const value_type& value) {
	if constexpr (jsonifier::concepts::string_t<value_type>) {
		os << "\"" << value << "\"";
	} else if constexpr (jsonifier::concepts::bool_t<value_type>) {
		os << std::boolalpha << value << std::noboolalpha << std::endl;
	} else {
		os << value;
	}
	return;
}

template<typename vector_type>
concept printable_vector = jsonifier::concepts::vector_t<vector_type> && printable<typename std::remove_cvref_t<vector_type>::value_type>;

template<printable_vector vector_type> std::ostream& operator<<(std::ostream& os, const vector_type& values) {
	os << "[";
	for (uint64_t x = 0; x < values.size(); ++x) {
		printValue(os, values[x]);
		if (x < values.size() - 1) {
			os << ",";
		}
	}
	os << "]";
	return os;
}

template<typename map_type>
concept printable_map =
	jsonifier::concepts::map_t<map_type> && printable<typename std::remove_cvref_t<map_type>::key_type> && printable<typename std::remove_cvref_t<map_type>::mapped_type>;

template<printable_map map_type> std::ostream& operator<<(std::ostream& os, const map_type& values) {
	os << "{";
	uint64_t index{};
	for (auto iter = values.begin(); iter != values.end(); ++iter, ++index) {
		printValue(os, iter->first);
		os << ":";
		printValue(os, iter->second);
		if (index < values.size() - 1) {
			os << ",";
		}
	}
	os << "}";
	return os;
}

template<typename pair_type>
concept printable_pair =
	jsonifier::concepts::pair_t<pair_type> && printable<typename std::remove_cvref_t<pair_type>::first_type> && printable<typename std::remove_cvref_t<pair_type>::second_type>;

template<printable_pair pair_type> std::ostream& operator<<(std::ostream& os, const pair_type& values) {
	os << "{";
	printValue(os, std::get<0>(values));
	os << ",";
	printValue(os, std::get<1>(values));
	os << "}";
	return os;
}

template<typename tuple_type, std::uint64_t... Indices> constexpr bool check_printable_helper_tuple(std::index_sequence<Indices...>) {
	return (printable<std::tuple_element_t<Indices, tuple_type>> && ...);
}

template<typename tuple_type> constexpr bool all_printable_in_tuple() {
	constexpr uint64_t tuple_size = std::tuple_size_v<tuple_type>;
	if constexpr (tuple_size == 0) {
		return true;
	} else {
		return check_printable_helper_tuple<tuple_type>(std::make_index_sequence<tuple_size>{});
	}
}

template<typename tuple_type>
concept printable_tuple = jsonifier::concepts::tuple_t<tuple_type> && all_printable_in_tuple<tuple_type>();

template<printable_tuple tuple_type, uint64_t currentIndex = 0> void printTuple(std::ostream& os, tuple_type& value) {
	if constexpr (currentIndex < std::tuple_size_v<tuple_type>) {
		printValue(os, std::get<currentIndex>(value));
		if constexpr (currentIndex < std::tuple_size_v<tuple_type> - 1) {
			os << ",";
		}
		printTuple<tuple_type, currentIndex + 1>(os, value);
	}
}

template<printable_tuple tuple_type> std::ostream& operator<<(std::ostream& os, const tuple_type& values) {
	os << "[";
	printTuple(os, values);
	os << "]";
	return os;
}

template<typename variant_type, size_t... Indices> constexpr bool check_printable_helper(std::index_sequence<Indices...>) {
	return (printable<std::variant_alternative_t<Indices, variant_type>> && ...);
}

template<typename variant_type> constexpr bool all_printable_in_variant() {
	constexpr size_t variant_size = std::variant_size_v<variant_type>;
	return check_printable_helper<variant_type>(std::make_index_sequence<variant_size>{});
}

template<typename variant_type>
concept printable_variant = jsonifier::concepts::variant_t<variant_type> && all_printable_in_variant<variant_type>();

template<printable_variant variant_type> void printVariant(std::ostream& os, const variant_type& value) {
	std::visit(
		[&os]<typename value_type>(const value_type& x) {
			printValue(os, x);
		},
		value);
}

template<printable_variant variant_type> std::ostream& operator<<(std::ostream& os, const variant_type& value) {
	os << "{";
	printVariant(os, value);
	os << "}";
	return os;
}

namespace bnch_swt {

	class file_loader {
	  public:
		constexpr BNCH_SWT_INLINE file_loader() {};

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
				uint64_t pos = line.find(":");
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
		int32_t counter{};
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

	BNCH_SWT_INLINE void evictCache() {
		const uint64_t cache_size = 64 * 1024;
		char* memory			  = new char[cache_size];

		for (uint64_t i = 0; i < cache_size; ++i) {
			memory[i] = static_cast<char>(i);
		}

		delete[] memory;
	}

	template<typename value_type> bool contains(value_type* values, const value_type& valueToCheckFor, uint64_t length) {
		for (uint64_t x = 0; x < length; ++x) {
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
		std::string name{};
		double timeInns{};

		BNCH_SWT_ALWAYS_INLINE bool operator>(const performance_metrics& other) const {
			if (!throughputMbPerSec.has_value() && other.throughputMbPerSec.has_value()) {
				return false;
			} else if (throughputMbPerSec.has_value() && !other.throughputMbPerSec.has_value()) {
				return true;
			} else if (throughputMbPerSec.has_value() && other.throughputMbPerSec.has_value()) {
				return throughputMbPerSec.value() > other.throughputMbPerSec.value();
			} else {
				return false;
			}
		}
	};

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

	template<jsonifier_internal::string_literal stageNameNew, uint64_t maxExecutionCount = 100> struct benchmark_stage {
		inline static std::unordered_map<std::string, performance_metrics> results{};

		BNCH_SWT_INLINE static void printResults(bool showComparison = true) {
			static constexpr jsonifier_internal::string_literal stageName{ stageNameNew };
			std::vector<performance_metrics> resultsNew{};
			for (auto& [key, value]: results) {
				resultsNew.emplace_back(value);
			}
			std::sort(resultsNew.begin(), resultsNew.end(), std::greater<performance_metrics>{});
			std::cout << "Performance Metrics for: " << stageName.view() << "\n";
			for (auto& value: resultsNew) {
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

			if (showComparison) {
				double difference{};
				for (uint64_t x = 0; x < resultsNew.size() - 1; ++x) {
					difference = ((resultsNew[x].throughputMbPerSec.value() - resultsNew[x + 1].throughputMbPerSec.value()) / resultsNew[x].throughputMbPerSec.value()) * 100.0f;
					std::cout << "Library " << resultsNew[x].name << ", is faster than library: " << resultsNew[x + 1].name << ", by roughly: " << difference << "%." << std::endl;
				}
			}
		}

		template<jsonifier_internal::string_literal filePath> BNCH_SWT_INLINE static void writeJsonData() {
			auto stringToWrite = parser.serializeJson(results);
			file_loader::saveFile(static_cast<std::string>(stringToWrite), filePath);
		}

#if defined(NDEBUG)
		static constexpr double thresholdStart{ 5.5f };
		static constexpr double threshold{ 5.0f };
#else
		static constexpr double thresholdStart{ 25.0f };
		static constexpr double threshold{ 20.0f };
#endif

		template<jsonifier_internal::string_literal benchmarkNameNew, jsonifier_internal::string_literal subjectNameNew, jsonifier_internal::string_literal colorNew,
			typename function_type, typename... arg_types>
		BNCH_SWT_INLINE static const performance_metrics& runBenchmark(function_type&& functionNew, arg_types&&... args) {
			static constexpr jsonifier_internal::string_literal benchmarkName{ benchmarkNameNew };
			static constexpr jsonifier_internal::string_literal subjectName{ subjectNameNew };
			static constexpr jsonifier_internal::string_literal color{ colorNew };
			std::string subjectNameNewer{ subjectName.data(), subjectName.size() };
			event_collector collector{};
			std::string colorName{ color.data(), color.size() };
			auto executionLambda = [=]() mutable {
				return std::forward<function_type>(functionNew)(std::forward<arg_types>(args)...);
			};
			std::vector<event_count> aggregate{};
			uint64_t bytesCollected{};
			double variation{ thresholdStart };
			performance_metrics resultsTemp{};
			uint64_t i{};
			while ((variation > threshold && i < maxExecutionCount) || (i < (maxExecutionCount / 2))) {
				++i;
				evictCache();
				collector.start();
				bytesCollected = executionLambda();
				aggregate.emplace_back(collector.end());
				resultsTemp = collectMetrics<subjectName>(i, bytesCollected, aggregate);
				if (resultsTemp.throughputVariation.has_value()) {
					variation = resultsTemp.throughputVariation.value();
				}
			}
			results[static_cast<std::string>(subjectName)] = collectMetrics<subjectName>(i, bytesCollected, aggregate);
			return results[static_cast<std::string>(subjectName)];
		}

		template<jsonifier_internal::string_literal benchmarkNameNew>
		static performance_metrics collectMetrics(uint64_t iterationCount, uint64_t bytesProcessed, std::vector<event_count> eventsNew) {
			performance_metrics metrics{};
			metrics.name		   = static_cast<std::string>(benchmarkNameNew.data());
			metrics.iterationCount = iterationCount;
			double volumeMB		   = bytesProcessed / (1024. * 1024.);
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
			double cycles{};
			double instructions{};
			double branches{};
			double missedBranches{};
			for (const event_count& e: eventsNew) {
				double ns = e.elapsed_ns();
				average_ns += ns;
				min_ns = min_ns < ns ? min_ns : ns;

				e.cycles(cycles);
				cycles_avg += cycles;
				cycles_min = cycles_min < cycles ? cycles_min : cycles;

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
			cycles_avg /= eventsNew.size();
			instructions_avg /= eventsNew.size();
			average_ns /= eventsNew.size();
			branches_avg /= eventsNew.size();
			metrics.throughputMbPerSec.emplace(volumeMB * 1000000000 / average_ns);
			metrics.throughputVariation.emplace((average_ns - min_ns) * 100.0 / average_ns);
			if (instructions_avg != 0.0f) {
				metrics.instructionsPerByte.emplace(instructions_avg / bytesProcessed);
				metrics.instructionsPerCycle.emplace(instructions_avg / cycles_avg);
				metrics.instructionsPerExecution.emplace(instructions_avg / iterationCount);
				metrics.instructionsVariation.emplace((instructions_avg - instructions_min) * 100.0 / instructions_avg);
			}
			if (cycles_avg != 0.0f) {
				metrics.cyclesPerByte.emplace(cycles_avg / bytesProcessed);
				metrics.cyclesPerExecution.emplace(cycles_avg / iterationCount);
				metrics.cyclesVariation.emplace((cycles_avg - cycles_min) * 100.0 / cycles_avg);
				metrics.frequencyGHz.emplace(cycles_min / min_ns);
			}
			if (branches_avg != 0.0f) {
				metrics.branchMissesPerExecution.emplace(missedBranches_avg / iterationCount);
				metrics.branchesPerExecution.emplace(branches_avg / iterationCount);
			}
			return metrics;
		}
	};

}

namespace jsonifier {
	template<> struct core<bnch_swt::performance_metrics> {
		using value_type				 = bnch_swt::performance_metrics;
		static constexpr auto parseValue = createValue<&value_type::branchesPerExecution, &value_type::branchMissesPerExecution, &value_type::cyclesPerByte,
			&value_type::cyclesPerExecution, &value_type::cyclesVariation>();
	};
}