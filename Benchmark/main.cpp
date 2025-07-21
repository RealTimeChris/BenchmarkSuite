#include <BnchSwt/BenchmarkSuite.hpp>

struct test_struct {
	int32_t value{};
	BNCH_SWT_INLINE uint64_t operator()(uint64_t test_value) {
		return test_value * test_value;
	}
};

BNCH_SWT_INLINE static uint64_t test_function(uint64_t test_value) {
	return test_value * test_value;
}

int main() {
	srand(static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
	uint64_t rand_val{ static_cast<uint64_t>(rand()) };
	test_struct test_val{};
	bnch_swt::doNotOptimizeAway(test_val(rand_val));

	rand_val = static_cast<uint64_t>(rand());
	bnch_swt::doNotOptimizeAway(test_function(rand_val));
	return 0;
}
