#include <BnchSwt/Config.hpp>
#include <array>
#include <iostream>

struct dimensions {
	size_t w{};
	size_t x{};
	size_t y{};
	size_t z{};
};

template<typename test_type> struct traits_test {
	static constexpr size_t dimension01{ test_type::dimension01 };
};

struct test_struct_base {
	std::array<size_t, 4> dimensions{};
};

template<size_t w, size_t x, size_t y, size_t z> struct test_struct : public test_struct_base {
	static constexpr size_t dimension01{ w};
	BNCH_SWT_INLINE void print_size() {
		std::cout << "w: " << w<< std::endl;
		std::cout << "x: " << x << std::endl;
		std::cout << "y: " << y << std::endl;
		std::cout << "z: " << z << std::endl;
	}
};

static constexpr auto generate_dimensions() {
	std::array<dimensions, 15 * 15 * 1 * 1> result{};
	size_t current_index{};
	for (size_t w = 1; w <= 16384; w *= 2) {
		for (size_t x = 1; x <= 16384; x *= 2) {
			for (size_t y = 1; y <= 1; y *= 2) {
				for (size_t z = 1; z <= 1; z *= 2) {
					result[current_index] = { w, x, y, z };
					++current_index;
				}
			}
		}
	}
	return result;
}

constexpr auto dims = generate_dimensions();

template<typename test_type> BNCH_SWT_INLINE void test_function(test_type test_value) {
	static constexpr traits_test<test_type> traits_test_val{};
	test_value.print_size();
}

template<size_t... indices> BNCH_SWT_INLINE void test_function_wrapper(test_struct_base* test_value, size_t index, std::index_sequence<indices...>) {
	((indices == index && (test_function(*static_cast<test_struct<dims[indices].w, dims[indices].x, dims[indices].y, dims[indices].z>*>(test_value)), 0)), ...);
}

int main() {
	srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	auto index = static_cast<size_t>((static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * static_cast<double>(dims.size()));
	test_struct_base test_struct{};
	test_function_wrapper(&test_struct, index, std::make_index_sequence<dims.size()>{});
	return 0;
}