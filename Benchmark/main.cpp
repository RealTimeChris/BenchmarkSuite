#include <immintrin.h>

#if defined(_MSC_VER)
	#define JSONIFIER_DECL_ALIGN(x) __declspec(align(x))
#else
	#define JSONIFIER_DECL_ALIGN(x) __attribute__((aligned(x)))
#endif

enum class avx_type {
	m128 = 0,
	m256 = 1,
	m512 = 2,
};

using jsonifier_simd_int_128 = JSONIFIER_DECL_ALIGN(16) __m128i;

template<avx_type type> struct JSONIFIER_DECL_ALIGN(16) aligned_struct_type {};

template<> struct JSONIFIER_DECL_ALIGN(16) aligned_struct_type<avx_type::m128> {
	using type = __m128i;
};

using value_type = aligned_struct_type<avx_type::m128>;

int main() {
	return 0;
}
