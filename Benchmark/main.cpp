#include <cstdint>
#include <iostream>
#include <jsonifier/Index.hpp>

constexpr uint64_t parseTwoDigits(const char* str) {
	return static_cast<uint64_t>((str[0ull] - '0') * 10 + (str[1ull] - '0'));
}

constexpr jsonifier_internal::array<jsonifier::string_view, 12> dates{ "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };

constexpr jsonifier_internal::array<uint64_t, 12> daysPerMonth{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

constexpr size_t getMonthIndex(const char* str) {
	jsonifier::string_view month{ str, 3 };
	for (size_t x = 0; x < 12; ++x) {
		if (month == dates[x]) {
			return x;
		}
	}
	return std::numeric_limits<uint64_t>::max();
}

constexpr size_t pow10Table[]{ 1ull, 10ull, 100ull, 1000ull, 10000ull, 100000ull, 1000000ull, 10000000ull, 100000000ull, 1000000000ull, 10000000000ull, 100000000000ull };

template<jsonifier_internal::string_literal timeNew> constexpr bool isValidFormat() {
	return timeNew[2] == ':' && timeNew[5] == ':';
}

constexpr uint64_t accunulateDaysPerYear(size_t monthIndex = 12) {
	uint64_t days{};
	for (size_t x = 0; x < monthIndex; ++x) {
		days += daysPerMonth[x];
	}
	return days;
}

constexpr uint64_t accunulateDays(size_t currentYear) {
	uint64_t days{};
	for (size_t x = 1970; x <= currentYear; ++x) {
		if (x % 4 == 0) {
			days += accunulateDaysPerYear() + 1;
		} else {
			days += accunulateDaysPerYear();
		}
	}
	return days;
}

constexpr uint64_t parse4Digits(const char* str) {
	size_t val{ jsonifier_internal::readBitsCt<uint32_t>(str) - 0x30303030 };
	constexpr auto multiplier{ pow10Table[4] };
	constexpr size_t mask = 0x000000FF000000FF;
	constexpr size_t mul1 = (100ULL << 32ULL);
	constexpr size_t mul2 = 1ULL << 32ULL;
	val					  = (val * 10ULL) + (val >> 8ULL);
	val					  = (((val & mask) * mul1) + (((val >> 16ULL) & mask) * mul2)) >> 32ULL;
	return val;
};

template<jsonifier_internal::string_literal timeNew> constexpr uint64_t timeToNanoseconds() {
	static_assert(isValidFormat<timeNew>(), "Invalid time format");
	constexpr const char* newPtr  = timeNew.data();
	constexpr const char* newPtr3 = timeNew.data() + 3;
	constexpr const char* newPtr6 = timeNew.data() + 6;
	constexpr uint64_t hours	  = parseTwoDigits(newPtr);
	constexpr uint64_t minutes = parseTwoDigits(newPtr3);
	constexpr uint64_t seconds = parseTwoDigits(newPtr6);
	static_assert(hours <= 24 || minutes <= 60 || seconds <= 60, "Time values out of range");
	constexpr uint64_t nanosPerSecond = 1000000000;
	constexpr uint64_t nanosPerMinute = 60 * nanosPerSecond;
	constexpr uint64_t nanosPerHour	= 60 * nanosPerMinute;
	return (hours * nanosPerHour) + (minutes * nanosPerMinute) + (seconds * nanosPerSecond);
}

template<jsonifier_internal::string_literal dateNew> constexpr size_t dateToNanoseconds() {
	constexpr const char* newPtr = &dateNew[0];
	constexpr uint64_t years{ parse4Digits(newPtr + 7) - 1 };
	constexpr uint64_t months{ getMonthIndex(newPtr) };
	constexpr uint64_t days{ parseTwoDigits(newPtr + 4) };
	constexpr uint64_t daysNew{ accunulateDays(years) + accunulateDaysPerYear(months) + days };
	constexpr uint64_t nanosPerSecond = 1'000'000'000;
	constexpr uint64_t nanosPerMinute = 60 * nanosPerSecond;
	constexpr uint64_t nanosPerHour	= 60 * nanosPerMinute;
	constexpr uint64_t nanosPerDay	= 24 * nanosPerHour;
	return static_cast<size_t>((daysNew * nanosPerDay) + timeToNanoseconds<__TIME__>());
}

int main() {
	// Total nanoseconds
	std::cout << "CURRENT NANOS: " << dateToNanoseconds<__DATE__>() << std::endl;
	//static_assert(nanos == 32'345'000'000'000, "Nanoseconds calculation failed");

	return 0;
}
