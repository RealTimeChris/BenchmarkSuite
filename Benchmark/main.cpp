#include <BnchSwt/BenchmarkSuite.hpp>
#include <nihilus-incl/index.hpp>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>
#include <source_location>
#include <array>
#include <string>
#include <queue>
#include <latch>
#include <bit>
static constexpr uint64_t total_iterations{ 200 };
static constexpr uint64_t measured_iterations{ 10 };

struct std_find_first_not_of {
	BNCH_SWT_INLINE static uint64_t impl(uint64_t& current_index, std::vector<std::vector<std::string>>& values_to_test_01, std::vector<std::vector<uint64_t>>& values_tested01,
		uint64_t byte_count) {
		std::vector<std::string>& current_iteration_in = values_to_test_01[current_index];
		std::vector<uint64_t>& current_iteration_out   = values_tested01[current_index];
		for (uint64_t x = 0; x < current_iteration_in.size(); ++x) {
			current_iteration_out[x] = current_iteration_in[x].find_first_not_of(" \t\n\r");
			bnch_swt::doNotOptimizeAway(current_iteration_out[x]);
		}
		++current_index;
		return byte_count;
	};
};

struct nihilus_find_first_not_of {
	BNCH_SWT_INLINE static uint64_t impl(uint64_t& current_index, std::vector<std::vector<nihilus::rt_string>>& values_to_test_01,
		std::vector<std::vector<uint64_t>>& values_tested01, uint64_t byte_count) {
		std::vector<nihilus::rt_string>& current_iteration_in = values_to_test_01[current_index];
		std::vector<uint64_t>& current_iteration_out		  = values_tested01[current_index];
		for (uint64_t x = 0; x < current_iteration_in.size(); ++x) {
			current_iteration_out[x] = current_iteration_in[x].find_first_non_whitespace();
			bnch_swt::doNotOptimizeAway(current_iteration_out[x]);
		}
		++current_index;
		return byte_count;
	};
};

struct std_comparison {
	BNCH_SWT_INLINE static uint64_t impl(uint64_t& current_index, std::vector<std::vector<std::string>>& values_to_test_01,
		std::vector<std::vector<std::string>>& values_to_test_02, std::vector<std::vector<uint64_t>>& values_tested01, uint64_t byte_count) {
		std::vector<std::string>& current_iteration_in_01 = values_to_test_01[current_index];
		std::vector<std::string>& current_iteration_in_02 = values_to_test_02[current_index];
		std::vector<uint64_t>& current_iteration_out	  = values_tested01[current_index];
		for (uint64_t x = 0; x < current_iteration_in_01.size(); ++x) {
			auto result = current_iteration_in_01[x] == current_iteration_in_02[x];
			bnch_swt::doNotOptimizeAway(result);
		}
		++current_index;
		return byte_count;
	};
};

struct nihilus_comparison {
	BNCH_SWT_INLINE static uint64_t impl(uint64_t& current_index, std::vector<std::vector<nihilus::rt_string>>& values_to_test_01,
		std::vector<std::vector<nihilus::rt_string>>& values_to_test_02, std::vector<std::vector<uint64_t>>& values_tested01, uint64_t byte_count) {
		std::vector<nihilus::rt_string>& current_iteration_in_01 = values_to_test_01[current_index];
		std::vector<nihilus::rt_string>& current_iteration_in_02 = values_to_test_02[current_index];
		std::vector<uint64_t>& current_iteration_out			 = values_tested01[current_index];
		for (uint64_t x = 0; x < current_iteration_in_01.size(); ++x) {
			auto result = current_iteration_in_01[x] == current_iteration_in_02[x];
			bnch_swt::doNotOptimizeAway(result);
		}
		++current_index;
		return byte_count;
	};
};

struct std_find_last_not_of {
	BNCH_SWT_INLINE static uint64_t impl(uint64_t& current_index, std::vector<std::vector<std::string>>& values_to_test_01, std::vector<std::vector<uint64_t>>& values_tested01,
		uint64_t byte_count) {
		std::vector<std::string>& current_iteration_in = values_to_test_01[current_index];
		std::vector<uint64_t>& current_iteration_out   = values_tested01[current_index];
		for (uint64_t x = 0; x < current_iteration_in.size(); ++x) {
			current_iteration_out[x] = current_iteration_in[x].find_last_not_of(" \t\n\r");
			bnch_swt::doNotOptimizeAway(current_iteration_out[x]);
		}
		++current_index;
		return byte_count;
	};
};

struct nihilus_find_last_not_of {
	BNCH_SWT_INLINE static uint64_t impl(uint64_t& current_index, std::vector<std::vector<nihilus::rt_string>>& values_to_test_01,
		std::vector<std::vector<uint64_t>>& values_tested01, uint64_t byte_count) {
		std::vector<nihilus::rt_string>& current_iteration_in = values_to_test_01[current_index];
		std::vector<uint64_t>& current_iteration_out		  = values_tested01[current_index];
		for (uint64_t x = 0; x < current_iteration_in.size(); ++x) {
			current_iteration_out[x] = current_iteration_in[x].find_last_non_whitespace();
			bnch_swt::doNotOptimizeAway(current_iteration_out[x]);
		}
		++current_index;
		return byte_count;
	};
};


template<uint64_t max_string_length, uint64_t strings_to_check> BNCH_SWT_INLINE void find_first_not_of_test() {
	std::vector<std::vector<std::string>> values_to_test_01{};
	std::vector<std::vector<nihilus::rt_string>> values_to_test_02{};
	values_to_test_01.resize(total_iterations);
	values_to_test_02.resize(total_iterations);
	std::vector<std::vector<uint64_t>> values_tested01{};
	std::vector<std::vector<uint64_t>> values_tested02{};
	std::vector<std::vector<uint64_t>> values_tested03{};
	values_tested01.resize(total_iterations);
	values_tested02.resize(total_iterations);
	values_tested03.resize(total_iterations);
	for (uint64_t x = 0; x < total_iterations; ++x) {
		for (uint64_t y = 0; y < strings_to_check; ++y) {
			auto new_string = bnch_swt::random_generator::template generateValue<std::string>(max_string_length);
			values_to_test_01[x].emplace_back(new_string);
			values_to_test_02[x].emplace_back(nihilus::rt_string{ new_string });
			values_tested01[x].emplace_back(new_string.find_first_not_of(" \r\t\n"));
			values_tested02[x].emplace_back();
			values_tested03[x].emplace_back();
		}
	}
	uint64_t byte_count{ [&] {
		uint64_t return_value{};
		for (auto& value: values_to_test_01) {
			for (auto& new_value: value) {
				return_value += new_value.size();
			}
		}
		return return_value;
	}() };
	uint64_t current_index{};
	static constexpr bnch_swt::string_literal test_name{ "first_not_of-" + bnch_swt::internal::template toStringLiteral<max_string_length>() };
	bnch_swt::benchmark_stage<test_name, total_iterations, measured_iterations>::template runBenchmark<"std::string::find_first_not_of", std_find_first_not_of>(current_index,
		values_to_test_01, values_tested02, byte_count);
	current_index = 0;
	bnch_swt::benchmark_stage<test_name, total_iterations, measured_iterations>::template runBenchmark<"nihilus::whitespace_search::find_first_not_of", nihilus_find_first_not_of>(
		current_index, values_to_test_02, values_tested03, byte_count);
	for (uint64_t x = 0; x < total_iterations; ++x) {
		for (uint64_t y = 0; y < strings_to_check; ++y) {
			if (values_tested01[x][y] != values_tested03[x][y]) {
				std::cout << "Failed to get the correct value at index: [" << x << "," << y << "]" << std::endl;
				std::cout << "Correct Value: " << values_tested01[x][y] << std::endl;
				std::cout << "Found Value: " << values_tested03[x][y] << std::endl;
				std::cout << "For String:\n " << values_to_test_01[x][y] << std::endl;
				std::cout << "For OTHER String:\n " << values_to_test_02[x][y].operator std::basic_string_view<char, std::char_traits<char>>() << std::endl;
			}
		}
	}

	bnch_swt::benchmark_stage<test_name, total_iterations, measured_iterations>::printResults();
}

template<uint64_t max_string_length, uint64_t strings_to_check> BNCH_SWT_INLINE void find_last_not_of_test() {
	std::vector<std::vector<std::string>> values_to_test_01{};
	std::vector<std::vector<nihilus::rt_string>> values_to_test_02{};
	values_to_test_01.resize(total_iterations);
	values_to_test_02.resize(total_iterations);
	std::vector<std::vector<uint64_t>> values_tested01{};
	std::vector<std::vector<uint64_t>> values_tested02{};
	std::vector<std::vector<uint64_t>> values_tested03{};
	values_tested01.resize(total_iterations);
	values_tested02.resize(total_iterations);
	values_tested03.resize(total_iterations);
	for (uint64_t x = 0; x < total_iterations; ++x) {
		for (uint64_t y = 0; y < strings_to_check; ++y) {
			auto new_string = bnch_swt::random_generator::template generateValue<std::string>(max_string_length);
			values_to_test_01[x].emplace_back(new_string);
			values_to_test_02[x].emplace_back(nihilus::rt_string{ new_string });
			values_tested01[x].emplace_back(new_string.find_last_not_of(" \r\t\n"));
			values_tested02[x].emplace_back();
			values_tested03[x].emplace_back();
		}
	}
	uint64_t byte_count{ [&] {
		uint64_t return_value{};
		for (auto& value: values_to_test_01) {
			for (auto& new_value: value) {
				return_value += new_value.size();
			}
		}
		return return_value;
	}() };
	uint64_t current_index{};
	static constexpr bnch_swt::string_literal test_name{ "last_not_of-" + bnch_swt::internal::template toStringLiteral<max_string_length>() };
	bnch_swt::benchmark_stage<test_name, total_iterations, measured_iterations>::template runBenchmark<"std::string::find_last_not_of", std_find_last_not_of>(current_index,
		values_to_test_01, values_tested02, byte_count);
	current_index = 0;
	bnch_swt::benchmark_stage<test_name, total_iterations, measured_iterations>::template runBenchmark<"nihilus::whitespace_search::find_last_not_of", nihilus_find_last_not_of>(
		current_index, values_to_test_02, values_tested03, byte_count);
	for (uint64_t x = 0; x < total_iterations; ++x) {
		for (uint64_t y = 0; y < strings_to_check; ++y) {
			if (values_tested01[x][y] != values_tested03[x][y]) {
				std::cout << "Failed to get the correct value at index: [" << x << "," << y << "]" << std::endl;
				std::cout << "Correct Value: " << values_tested01[x][y] << std::endl;
				std::cout << "Found Value: " << values_tested03[x][y] << std::endl;
				std::cout << "For String:\n " << values_to_test_01[x][y] << std::endl;
				std::cout << "For OTHER String:\n " << values_to_test_02[x][y].operator std::basic_string_view<char, std::char_traits<char>>() << std::endl;
			}
		}
	}

	bnch_swt::benchmark_stage<test_name, total_iterations, measured_iterations>::printResults();
}

template<uint64_t max_string_length, uint64_t strings_to_check> BNCH_SWT_INLINE void comparison_test() {
	std::vector<std::vector<std::string>> values_to_test_01{};
	std::vector<std::vector<std::string>> values_to_test_02{};
	std::vector<std::vector<nihilus::rt_string>> values_to_test_03{};
	std::vector<std::vector<nihilus::rt_string>> values_to_test_04{};
	values_to_test_01.resize(total_iterations);
	values_to_test_02.resize(total_iterations);
	values_to_test_03.resize(total_iterations);
	values_to_test_04.resize(total_iterations);
	std::vector<std::vector<uint64_t>> values_tested01{};
	std::vector<std::vector<uint64_t>> values_tested02{};
	std::vector<std::vector<uint64_t>> values_tested03{};
	values_tested01.resize(total_iterations);
	values_tested02.resize(total_iterations);
	values_tested03.resize(total_iterations);
	for (uint64_t x = 0; x < total_iterations; ++x) {
		for (uint64_t y = 0; y < strings_to_check; ++y) {
			auto new_string = bnch_swt::random_generator::template generateValue<std::string>(max_string_length);
			values_to_test_01[x].emplace_back(new_string);
			values_to_test_03[x].emplace_back(nihilus::rt_string{ new_string });
			new_string = bnch_swt::random_generator::template generateValue<std::string>(max_string_length);
			values_to_test_02[x].emplace_back(new_string);
			values_to_test_04[x].emplace_back(nihilus::rt_string{ new_string });
			values_tested01[x].emplace_back();
			values_tested02[x].emplace_back();
			values_tested03[x].emplace_back();
		}
	}
	uint64_t byte_count{ [&] {
		uint64_t return_value{};
		for (auto& value: values_to_test_01) {
			for (auto& new_value: value) {
				return_value += new_value.size();
			}
		}
		return return_value;
	}() };
	uint64_t current_index{};
	static constexpr bnch_swt::string_literal test_name{ "comparison-" + bnch_swt::internal::template toStringLiteral<max_string_length>() };
	bnch_swt::benchmark_stage<test_name, total_iterations, measured_iterations>::template runBenchmark<"std::string::operator==", std_comparison>(current_index, values_to_test_01,
		values_to_test_02, values_tested02, byte_count);
	current_index = 0;
	bnch_swt::benchmark_stage<test_name, total_iterations, measured_iterations>::template runBenchmark<"nihilus::rt_string::operator==", nihilus_comparison>(current_index,
		values_to_test_03, values_to_test_04, values_tested03, byte_count);
	for (uint64_t x = 0; x < total_iterations; ++x) {
		for (uint64_t y = 0; y < strings_to_check; ++y) {
			if (values_tested01[x][y] != values_tested03[x][y]) {
				std::cout << "Failed to get the correct value at index: [" << x << "," << y << "]" << std::endl;
				std::cout << "Correct Value: " << values_tested01[x][y] << std::endl;
				std::cout << "Found Value: " << values_tested03[x][y] << std::endl;
				std::cout << "For String:\n " << values_to_test_01[x][y] << std::endl;
				std::cout << "For OTHER String:\n " << values_to_test_02[x][y].operator std::basic_string_view<char, std::char_traits<char>>() << std::endl;
			}
		}
	}

	bnch_swt::benchmark_stage<test_name, total_iterations, measured_iterations>::printResults();
}


int main() {
	comparison_test<1, 100>();
	comparison_test<2, 100>();
	comparison_test<4, 100>();
	comparison_test<8, 100>();
	comparison_test<16, 100>();
	comparison_test<32, 100>();
	comparison_test<64, 100>();
	comparison_test<128, 100>();
	comparison_test<256, 100>();
	comparison_test<512, 100>();
	comparison_test<1024, 100>();
	find_first_not_of_test<1, 100>();
	find_first_not_of_test<2, 100>();
	find_first_not_of_test<4, 100>();
	find_first_not_of_test<8, 100>();
	find_first_not_of_test<16, 100>();
	find_first_not_of_test<32, 100>();
	find_first_not_of_test<64, 100>();
	find_first_not_of_test<128, 100>();
	find_first_not_of_test<256, 100>();
	find_first_not_of_test<512, 100>();
	find_first_not_of_test<1024, 100>();
	/**/
	return 0;
}
