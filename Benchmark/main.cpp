#include <BnchSwt/Index.hpp>

struct test_struct {
	mutable std::array<uint32_t, 32> value{};
	//      ^^^^^^^ THIS IS ESSENTIAL
};

template<typename value_type> consteval value_type get_value() {
	return value_type{};
}

int32_t main() {
	static constinit auto new_value = get_value<test_struct>();

	std::cout << "CURRENT VALUES: " << new_value.value[0] << std::endl;

	// Without mutable, this next line would be a compile error:
	// new_value.value[0] = 42;  // ERROR: cannot modify constexpr

	// With mutable, this works:
	new_value.value[0] = 42;// ✅

	bnch_swt::doNotOptimizeAway(new_value);
	return 0;
}
