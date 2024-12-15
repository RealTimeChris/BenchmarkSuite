#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <array>
#include <utility>

struct test_struct_virtual_base {
	constexpr test_struct_virtual_base() noexcept {};
	BNCH_SWT_ALWAYS_INLINE virtual size_t highOverheadFunction() const = 0;
	constexpr virtual ~test_struct_virtual_base() noexcept {};
};

template<size_t index> struct test_struct_virtual : public test_struct_virtual_base {
	BNCH_SWT_ALWAYS_INLINE size_t highOverheadFunction() const override {
		size_t currentSize{};
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(1, 100);

		std::vector<std::string> data;

		for (size_t i = 0; i < index * 100; ++i) {
			int randomNum = dis(gen);
			double result = 0.0;
			for (int j = 1; j <= 100; ++j) {
				result += std::sin(randomNum * j) * std::cos(j);
			}
			std::string resultStr = std::to_string(result);

			std::string suffix = "_random" + std::to_string(dis(gen));
			resultStr += suffix;
			currentSize += resultStr.size();
			data.push_back(resultStr);
			std::sort(data.begin(), data.end());
		}
		return currentSize;
	}
};

struct test_struct_function_ptrs {
	template<size_t index> BNCH_SWT_ALWAYS_INLINE size_t highOverheadFunction() const {
		size_t currentSize{};
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(1, 100);

		std::vector<std::string> data;

		for (size_t i = 0; i < index * 100; ++i) {
			int randomNum = dis(gen);
			double result = 0.0;
			for (int j = 1; j <= 100; ++j) {
				result += std::sin(randomNum * j) * std::cos(j);
			}
			std::string resultStr = std::to_string(result);

			std::string suffix = "_random" + std::to_string(dis(gen));
			resultStr += suffix;
			currentSize += resultStr.size();
			data.push_back(resultStr);
			std::sort(data.begin(), data.end());
		}
		return currentSize;
	}
};

using function_type = size_t (test_struct_function_ptrs::*)() const;

template<size_t... indices> static constexpr std::array<function_type, sizeof...(indices)> generateFunctionPtrs(std::index_sequence<indices...>) {
	return { ( &test_struct_function_ptrs::template highOverheadFunction<indices> )... };
}

static constexpr auto functionPtrs = generateFunctionPtrs(std::make_index_sequence<16>{});

int main() {
	bnch_swt::benchmark_stage<"function-ptrs-vs-virtual">::runBenchmark<"virtual", "cyan">([] {
		size_t currentSize{};
		static constexpr test_struct_virtual<15> testVal{};
		for (size_t x = 0; x < 2; ++x) {
			currentSize += testVal.highOverheadFunction();
		}
		return currentSize;
	});

	bnch_swt::benchmark_stage<"function-ptrs-vs-virtual">::runBenchmark<"function-ptrs", "cyan">([] {
		size_t currentSize{};
		static constexpr test_struct_function_ptrs testVal{};
		for (size_t x = 0; x < 2; ++x) {
			currentSize += (testVal.*functionPtrs[15])();
		}
		return currentSize;
	});

	bnch_swt::benchmark_stage<"function-ptrs-vs-virtual">::printResults();
	return 0;
}
