#include <BnchSwt/BenchmarkSuite.hpp>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>
#include <source_location>
#include <array>
#include <queue>
#include <latch>
#include <bit>

void mul_mat_scalar_f32(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
	for (size_t i = 0; i < M; ++i) {
		for (size_t j = 0; j < N; ++j) {
			float sum = 0.0f;
			for (size_t k = 0; k < K; ++k) {
				sum += A[i * K + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
}

enum class testing {
	val_01,
	val_02,
	count,
};

template<typename value_type>
concept has_count = requires() { std::remove_cvref_t<value_type>::count; };

namespace testing_real {
#if defined(BNCH_SWT_MSVC) 
	constexpr auto pretty_function_tail = ">(void)";
#else
	constexpr auto pretty_function_tail = "]";
#endif

	template<has_count auto current_index> constexpr std::string_view get_enum_name() {
		std::string_view str = std::source_location::current().function_name();
		str					 = str.substr(str.find("&") + 1);
		str					 = str.substr(0, str.find(pretty_function_tail));
		return str.substr(str.rfind("::") + 2);
	}

	template<has_count current_type, uint64_t current_index = 0>
	constexpr std::array<std::string_view, static_cast<uint64_t>(current_type::count)> get_enum_names(
		std::array<std::string_view, static_cast<uint64_t>(current_type::count)> values = {}) {
		if constexpr (current_index < static_cast<uint64_t>(current_type::count)) {
			values[current_index] = get_enum_name<static_cast<current_type>(current_index)>();
			return get_enum_names<current_type, current_index + 1>(values);
		} else {
			return values;
		}
	}
}

template<auto current_index, auto enum_count> BNCH_SWT_INLINE std::string print_enum_value(auto enum_val) {
	if constexpr (static_cast<uint64_t>(current_index) < static_cast<uint64_t>(enum_count)) {
		if (static_cast<uint64_t>(current_index) == static_cast<uint64_t>(enum_val)) {
			constexpr std::string_view string{ get_enum_name<current_index, enum_count>() };
			return static_cast<std::string>(string);
		} else {
			return print_enum_value<static_cast<decltype(enum_count)>(static_cast<uint64_t>(current_index) + 1), enum_count>(enum_val);
		}
	} else {
		return {};
	}
};

int main() {
	static constexpr auto new_name = testing_real::get_enum_names<testing>();
	static constexpr size_t time_to_wait{ 10 };
	std::cout << "CUYRRENT VALUE: " << new_name << std::endl;

	struct test_struct_no_pause {
		BNCH_SWT_INLINE static uint64_t impl() {
			auto start = std::chrono::high_resolution_clock::now();
			auto end   = std::chrono::high_resolution_clock::now();
			while ((end - start).count() < time_to_wait) {
				end = std::chrono::high_resolution_clock::now();
			}
			return 200000ull;
		};
	};

	struct test_struct_pause {
		BNCH_SWT_INLINE static uint64_t impl() {
			auto start = std::chrono::high_resolution_clock::now();
			auto end   = std::chrono::high_resolution_clock::now();
			while ((end - start).count() < time_to_wait) {
				std::this_thread::yield();
				end = std::chrono::high_resolution_clock::now();
			}
			return 200000ull;
		};
	};

	bnch_swt::benchmark_stage<"test_stage", 2, 1>::runBenchmark<"no-yield", test_struct_no_pause>();
	bnch_swt::benchmark_stage<"test_stage", 2, 1>::runBenchmark<"yield", test_struct_pause>();

	bnch_swt::benchmark_stage<"test_stage", 2, 1>::printResults();
	return 0;
}
