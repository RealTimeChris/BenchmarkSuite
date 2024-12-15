#include <cpu_arch.hpp>
#include <unordered_map>

int main() {
	std::unordered_map<std::string, std::string> test_map{};
	test_map.reserve(354);
	impl_test_function_01(oiml::cpu_arch_index);
	return 0;
}
