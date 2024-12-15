#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include <glaze/glaze.hpp>
#include "RandomGenerators.hpp"

static constexpr auto maxIterations{ 400 };

template<typename derived_type> struct parse_context_partial {
	parse_context_partial() noexcept = default;

	JSONIFIER_INLINE bool getState() const {
		return remainingMemberCount > 0 && (currentArrayDepth > 0 || currentObjectDepth > 0);
	}

	derived_type* parserPtr{};
	int64_t remainingMemberCount{};
	int64_t currentObjectDepth{};
	int64_t currentArrayDepth{};
	const char* rootIter{};
	const char* endIter{};
	const char* iter{};
};

template<typename value_type> JSONIFIER_INLINE consteval value_type constEval(value_type value) {
	return value;
}

template<typename derived_type> struct parse_context {
	constexpr parse_context() noexcept = default;

	JSONIFIER_INLINE bool getState() const {
		return remainingMemberCount > 0 && (currentArrayDepth > 0 || currentObjectDepth > 0);
	}

	mutable derived_type* parserPtr{};
	mutable int64_t remainingMemberCount{};
	mutable int64_t currentObjectDepth{};
	mutable int64_t currentArrayDepth{};
	mutable const char* rootIter{};
	mutable const char* endIter{};
	mutable const char* iter{};
};

template<uint64_t count, bnch_swt::string_literal testNameNew> BNCH_SWT_INLINE void testFunction() {
	static constexpr bnch_swt::string_literal testName{ testNameNew };
	bnch_swt::benchmark_stage<testName>::template runBenchmark<"non-consteval", "CYAN">([&] {
		uint64_t bytesProcessed{};
		bool testVal{};
		for (uint64_t x = 0ull; x < count; ++x) {
			uint8_t context[16]{ count + 1, count + 1, count + 1, count + 1, count + 1, count + 1, count + 1, count + 1, count + 1, count + 1, count + 1, count + 1, count + 1,
				count + 1, count + 1, count + 1 };
			context[0] += x;
			bnch_swt::doNotOptimizeAway(context);
			bytesProcessed += context[0];
		}
		return bytesProcessed;
	});

	bnch_swt::benchmark_stage<testName>::template runBenchmark<"consteval", "CYAN">([&] {
		uint64_t bytesProcessed{};
		bool testVal{};
		for (uint64_t x = 0ull; x < count; ++x) {
			uint8_t context[16]{ constEval(count + 1), constEval(count + 1), constEval(count + 1), constEval(count + 1), constEval(count + 1), constEval(count + 1),
				constEval(count + 1), constEval(count + 1), constEval(count + 1), constEval(count + 1), constEval(count + 1), constEval(count + 1), constEval(count + 1),
				constEval(count + 1), constEval(count + 1), constEval(count + 1) };
			context[0] += x;
			bnch_swt::doNotOptimizeAway(context);
			bytesProcessed += context[0];
		}
		return bytesProcessed;
	});

	bnch_swt::benchmark_stage<testName>::printResults(true, true);
}

int main() {
	testFunction<32, "consteval-vs-non-consteval">();
	return 0;
}