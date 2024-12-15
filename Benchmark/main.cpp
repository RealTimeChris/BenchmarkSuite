#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include <glaze/glaze.hpp>
#include "RandomGenerators.hpp"

#if defined(JSONIFIER_MAC)
template<jsonifier::simd_int_type simd_int_type_new> JSONIFIER_INLINE static simd_int_type_new gatherValues(const void* str) noexcept {
	return vld1q_u8(static_cast<const uint8_t*>(str));
}

template<jsonifier::simd_int_type simd_int_type_new> JSONIFIER_INLINE static simd_int_type_new gatherValuesU(const void* str) noexcept {
	return vld1q_u8(static_cast<const uint8_t*>(str));
}
#else
template<jsonifier::simd_int_type simd_int_type_new> JSONIFIER_INLINE static simd_int_type_new gatherValues(const void* str) noexcept {
	return _mm256_load_si256(static_cast<const simd_int_type_new*>(str));
}
template<jsonifier::simd_int_type simd_int_type_new> JSONIFIER_INLINE static simd_int_type_new gatherValuesU(const void* str) noexcept {
	return _mm256_loadu_si256(static_cast<const simd_int_type_new*>(str));
}
#endif

static constexpr uint64_t maxIterations{ 100 };

template<uint64_t lengthNew, uint64_t countNew, bnch_swt::string_literal testNameNew> BNCH_SWT_INLINE void testFunction() {
	static constexpr bnch_swt::string_literal testName{ testNameNew };
	std::vector<std::vector<std::string>> stringsToLoad{ maxIterations };
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < countNew; ++y) {
			stringsToLoad[x].emplace_back(bnch_swt::random_generator::generateValue<std::string>(lengthNew));
		}
	}
	size_t currentIndex{};
	bnch_swt::benchmark_stage<testName, maxIterations, 20>::template runBenchmark<"unaligned", "CYAN">([&] {
		size_t bytesProcessed{};
		char valuesToLoad[lengthNew * countNew];
		char* ptr{ valuesToLoad };
		for (size_t x = 0; x < countNew; ++x) {
			std::memcpy(ptr, stringsToLoad[currentIndex][x].data(), lengthNew);
			ptr += lengthNew;
		}
		for (size_t y = 0; y < countNew; ++y) {
			const jsonifier::jsonifier_simd_int_t values01 = gatherValuesU<jsonifier::jsonifier_simd_int_t>(valuesToLoad + (y * lengthNew));
			const jsonifier::jsonifier_simd_int_t values02 = gatherValuesU<jsonifier::jsonifier_simd_int_t>(valuesToLoad + (y * lengthNew) + jsonifier::bytesPerStep);
			const jsonifier::jsonifier_simd_int_t values03 = gatherValuesU<jsonifier::jsonifier_simd_int_t>(valuesToLoad + (y * lengthNew) + (jsonifier::bytesPerStep * 2));
			const jsonifier::jsonifier_simd_int_t values04 = gatherValuesU<jsonifier::jsonifier_simd_int_t>(valuesToLoad + (y * lengthNew) + (jsonifier::bytesPerStep * 3));
			bnch_swt::doNotOptimizeAway(values01);
			bnch_swt::doNotOptimizeAway(values02);
			bnch_swt::doNotOptimizeAway(values03);
			bnch_swt::doNotOptimizeAway(values04);
			bytesProcessed += stringsToLoad[currentIndex][y].size();
		}
		++currentIndex;
		return bytesProcessed;
	});
	currentIndex = 0;
	bnch_swt::benchmark_stage<testName, maxIterations, 20>::template runBenchmark<"aligned", "CYAN">([&] {
		size_t bytesProcessed{};
		JSONIFIER_ALIGN(jsonifier::bytesPerStep) char valuesToLoad[lengthNew * countNew];
		char* ptr{ valuesToLoad };
		for (size_t x = 0; x < countNew; ++x) {
			std::memcpy(ptr, stringsToLoad[currentIndex][x].data(), lengthNew);
			ptr += lengthNew;
		}
		for (size_t y = 0; y < countNew; ++y) {
			const jsonifier::jsonifier_simd_int_t values01 = gatherValues<jsonifier::jsonifier_simd_int_t>(valuesToLoad + (y * lengthNew));
			const jsonifier::jsonifier_simd_int_t values02 = gatherValues<jsonifier::jsonifier_simd_int_t>(valuesToLoad + (y * lengthNew) + jsonifier::bytesPerStep);
			const jsonifier::jsonifier_simd_int_t values03 = gatherValues<jsonifier::jsonifier_simd_int_t>(valuesToLoad + (y * lengthNew) + (jsonifier::bytesPerStep * 2));
			const jsonifier::jsonifier_simd_int_t values04 = gatherValues<jsonifier::jsonifier_simd_int_t>(valuesToLoad + (y * lengthNew) + (jsonifier::bytesPerStep * 3));
			bnch_swt::doNotOptimizeAway(values01);
			bnch_swt::doNotOptimizeAway(values02);
			bnch_swt::doNotOptimizeAway(values03);
			bnch_swt::doNotOptimizeAway(values04);
			bytesProcessed += stringsToLoad[currentIndex][y].size();
		}
		++currentIndex;
		return bytesProcessed;
	});

	bnch_swt::benchmark_stage<testName, maxIterations, 20>::printResults(true, true);
}

int main() {
	testFunction<jsonifier::bytesPerStep * 4, 10000, "aligned-vs-aligned-loads">();
	return 0;
}