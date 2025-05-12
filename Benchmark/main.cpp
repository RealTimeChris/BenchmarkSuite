#include <BnchSwt/BenchmarkSuite.hpp>

struct test_non_type_01 {
	uint64_t test_array{};
};

struct test_non_type_02 {
	uint64_t test_array{};
};

struct test_non_type_03 {
	uint64_t test_array{};
};

template<uint64_t arg> BNCH_SWT_INLINE static constexpr double fp32_from_bits_fn(uint64_t w) {
	uint64_t test_val_new01{ static_cast<uint64_t>(rand()) };
	uint64_t val_new{};
	for (size_t x = 0; x < sizeof(arg); ++x) {
		val_new += arg + rand();
	}
	return std::bit_cast<double>(val_new + w);
}

template<uint64_t arg> struct fp32_from_bits {
	BNCH_SWT_INLINE static constexpr double impl(uint32_t w) {
		uint64_t test_val_new01{ static_cast<uint64_t>(rand()) };
		uint64_t val_new{};
		for (size_t x = 0; x < sizeof(arg); ++x) {
			val_new += arg + rand();
		}
		return std::bit_cast<double>(val_new + w);
	}
};

template<uint64_t arg_new> struct fp32_from_bits_static {
	static constexpr auto arg{ arg_new };
	BNCH_SWT_INLINE static constexpr double impl(uint32_t w) {
		uint64_t val_new{};
		for (size_t x = 0; x < sizeof(arg); ++x) {
			val_new += arg + rand();
		}
		return std::bit_cast<double>(val_new + w);
	}
};

int main() {
	srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	uint64_t test_val_new01{ static_cast<uint64_t>(rand()) };
	uint64_t test_val_new02{ static_cast<uint64_t>(rand()) };
	uint64_t test_val_new03{ static_cast<uint64_t>(rand()) };
	static constexpr uint64_t test_val_01{ 23225343 };
	static constexpr uint64_t test_val_02{ 234533223 };
	static constexpr uint64_t test_val_03{ 2323223 };
	auto value01 = fp32_from_bits_fn<test_val_01>(test_val_new01);
	auto value02 = fp32_from_bits<test_val_02>::impl(test_val_new02);
	auto value03 = fp32_from_bits_static<test_val_03>::impl(test_val_new03);
	std::cout << "VAL01: " << value01 << std::endl;
	std::cout << "VAL02: " << value02 << std::endl;
	std::cout << "VAL03: " << value03 << std::endl;
	return 0;
}
