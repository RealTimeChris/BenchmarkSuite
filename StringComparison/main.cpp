#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Conformance.hpp"
#include "Tests/Simdjson.hpp"
#include "Tests/Jsonifier.hpp"
#include "Tests/Common.hpp"
#include "Tests/Uint.hpp"
#include "Tests/Float.hpp"
#include "Tests/RoundTrip.hpp"
#include "Tests/Int.hpp"
#include "Tests/String.hpp"
#include <glaze/glaze.hpp>

template<size_t digitCount> struct int_serializer {};

template<> struct int_serializer<1> {
	JSONIFIER_ALWAYS_INLINE static void impl(std::uint32_t value, char* string) noexcept {
		uint32_t ascii_digit = value + 0x30;
		*string				 = static_cast<char>(ascii_digit);
	}
};

template<> struct int_serializer<2> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint8_t tens = (value / 10);
		uint8_t ones = value % 10;

		uint64_t packedDigits = ((tens) | uint64_t(ones << 8)) + 0x3030;

		std::memcpy(string, &packedDigits, sizeof(packedDigits));
	}
};

template<> struct int_serializer<3> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint8_t hundreds	  = (value / 100) % 10;
		uint8_t tens		  = (value / 10) % 10;
		uint8_t ones		  = value % 10;
		uint64_t packedDigits = ((hundreds) | (uint64_t(tens) << 8) | (uint64_t(ones) << 16)) + 0x303030;
		std::memcpy(string, &packedDigits, sizeof(packedDigits));
	}
};

template<> struct int_serializer<4> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint8_t thousands	  = (value / 1000) % 10;
		uint8_t hundreds	  = (value / 100) % 10;
		uint8_t tens		  = (value / 10) % 10;
		uint8_t ones		  = value % 10;
		uint64_t packedDigits = (thousands | (uint64_t(hundreds) << 8) | (uint64_t(tens) << 16) | (uint64_t(ones) << 24)) + 0x30303030;
		std::memcpy(string, &packedDigits, sizeof(packedDigits));
	}
};

template<> struct int_serializer<5> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint8_t tenThousands  = (value / 10000) % 10;
		uint8_t thousands	  = (value / 1000) % 10;
		uint8_t hundreds	  = (value / 100) % 10;
		uint8_t tens		  = (value / 10) % 10;
		uint8_t ones		  = value % 10;
		uint64_t packedDigits = (tenThousands | (uint64_t(thousands) << 8) | (uint64_t(hundreds) << 16) | (uint64_t(tens) << 24) | (uint64_t(ones) << 32)) + 0x3030303030;
		std::memcpy(string, &packedDigits, sizeof(packedDigits));
	}
};

template<> struct int_serializer<6> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits = (hundredThousands | (tenThousands << 8) | (thousands << 16) | (hundreds << 24) | (tens << 32) | (ones << 40)) + 0x303030303030;
		std::memcpy(string, &packedDigits, sizeof(packedDigits));
	}
};

template<> struct int_serializer<7> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits = (millions | (hundredThousands << 8) | (tenThousands << 16) | (thousands << 24) | (hundreds << 32) | (tens << 40) | (ones << 48)) + 0x30303030303030;
		std::memcpy(string, &packedDigits, sizeof(packedDigits));
	}
};

template<> struct int_serializer<8> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t tenMillions	  = (value / 10000000) % 10;
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits =
			(tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) | (ones << 56)) +
			0x3030303030303030ull;
		std::memcpy(string, &packedDigits, sizeof(packedDigits));
	}
};

template<> struct int_serializer<9> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t tenMillions	  = (value / 10000000) % 10;
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits =
			(tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) | (ones << 56)) +
			0x3030303030303030ull;

		std::memcpy(string + 1, &packedDigits, sizeof(packedDigits));
		ones		 = (value / 100000000) % 10;
		packedDigits = ones + 0x30;
		string[0]	 = static_cast<char>(packedDigits);
	}
};

template<> struct int_serializer<10> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t tenMillions	  = (value / 10000000) % 10;
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits =
			(tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) | (ones << 56)) +
			0x3030303030303030ull;

		std::memcpy(string + 2, &packedDigits, sizeof(packedDigits));
		tens		 = (value / 1000000000) % 10;
		ones		 = (value / 100000000) % 10;
		packedDigits = tens | (ones << 8) + 0x3030ull;
		std::memcpy(string, &packedDigits, 2);
	}
};

template<> struct int_serializer<11> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t tenMillions	  = (value / 10000000) % 10;
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits =
			(tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) | (ones << 56)) +
			0x3030303030303030ull;

		std::memcpy(string + 3, &packedDigits, sizeof(packedDigits));
		hundreds	 = (value / 10000000000) % 10;
		tens		 = (value / 1000000000) % 10;
		ones		 = (value / 100000000) % 10;
		packedDigits = (hundreds) | (tens << 8) | (ones << 16) + 0x303030ull;
		std::memcpy(string, &packedDigits, 3);
	}
};

template<> struct int_serializer<12> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t tenMillions	  = (value / 10000000) % 10;
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits =
			(tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) | (ones << 56)) +
			0x3030303030303030ull;

		std::memcpy(string + 4, &packedDigits, sizeof(packedDigits));
		thousands	 = (value / 100000000000) % 10;
		hundreds	 = (value / 10000000000) % 10;
		tens		 = (value / 1000000000) % 10;
		ones		 = (value / 100000000) % 10;
		packedDigits = thousands | (hundreds << 8) | (tens << 16) | (ones << 24) + 0x30303030ull;
		std::memcpy(string, &packedDigits, 4);
	}
};

template<> struct int_serializer<13> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t tenMillions	  = (value / 10000000) % 10;
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits =
			(tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) | (ones << 56)) +
			0x3030303030303030ull;

		std::memcpy(string + 5, &packedDigits, sizeof(packedDigits));
		tenThousands = (value / 1000000000000) % 10;
		thousands	 = (value / 100000000000) % 10;
		hundreds	 = (value / 10000000000) % 10;
		tens		 = (value / 1000000000) % 10;
		ones		 = (value / 100000000) % 10;
		packedDigits = tenThousands | (thousands << 8) | (hundreds << 16) | (tens << 24) | (ones << 32) + 0x3030303030ull;
		std::memcpy(string, &packedDigits, 5);
	}
};

template<> struct int_serializer<14> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t tenMillions	  = (value / 10000000) % 10;
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits =
			(tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) | (ones << 56)) +
			0x3030303030303030ull;

		std::memcpy(string + 6, &packedDigits, sizeof(packedDigits));
		hundredThousands = (value / 10000000000000) % 10;
		tenThousands	 = (value / 1000000000000) % 10;
		thousands		 = (value / 100000000000) % 10;
		hundreds		 = (value / 10000000000) % 10;
		tens			 = (value / 1000000000) % 10;
		ones			 = (value / 100000000) % 10;
		packedDigits	 = hundredThousands | (tenThousands << 8) | (thousands << 16) | (hundreds << 24) | (tens << 32) | (ones << 40) + 0x303030303030ull;
		std::memcpy(string, &packedDigits, 6);
	}
};

template<> struct int_serializer<15> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t tenMillions	  = (value / 10000000) % 10;
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits =
			(tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) | (ones << 56)) +
			0x3030303030303030ull;

		std::memcpy(string + 7, &packedDigits, sizeof(packedDigits));
		millions		 = (value / 100000000000000) % 10;
		hundredThousands = (value / 10000000000000) % 10;
		tenThousands	 = (value / 1000000000000) % 10;
		thousands		 = (value / 100000000000) % 10;
		hundreds		 = (value / 10000000000) % 10;
		tens			 = (value / 1000000000) % 10;
		ones			 = (value / 100000000) % 10;
		packedDigits	 = millions | (hundredThousands << 8) | (tenThousands << 16) | (thousands << 24) | (hundreds << 32) | (tens << 40) | (ones << 48) + 0x30303030303030ull;
		std::memcpy(string, &packedDigits, 7);
	}
};

template<> struct int_serializer<16> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint8_t tenMillions		 = (value / 10000000) % 10;
		uint8_t millions		 = (value / 1000000) % 10;
		uint8_t hundredThousands = (value / 100000) % 10;
		uint8_t tenThousands	 = (value / 10000) % 10;
		uint8_t thousands		 = (value / 1000) % 10;
		uint8_t hundreds		 = (value / 100) % 10;
		uint8_t tens			 = (value / 10) % 10;
		uint8_t ones			 = value % 10;

		uint64_t packedDigits =
			(tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) | (ones << 56)) +
			0x3030303030303030ull;

		std::memcpy(string + 8, &packedDigits, sizeof(packedDigits));
		tenMillions		 = (value / 1000000000000000) % 10;
		millions		 = (value / 100000000000000) % 10;
		hundredThousands = (value / 10000000000000) % 10;
		tenThousands	 = (value / 1000000000000) % 10;
		thousands		 = (value / 100000000000) % 10;
		hundreds		 = (value / 10000000000) % 10;
		tens			 = (value / 1000000000) % 10;
		ones			 = (value / 100000000) % 10;
		packedDigits	 = tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) |
			(ones << 56) + 0x3030303030303030ull;
		std::memcpy(string, &packedDigits, 8);
	}
};

template<> struct int_serializer<17> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t tenMillions	  = (value / 10000000) % 10;
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits =
			(tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) | (ones << 56)) +
			0x3030303030303030ull;

		std::memcpy(string + 9, &packedDigits, sizeof(packedDigits));
		tenMillions		 = (value / 1000000000000000) % 10;
		millions		 = (value / 100000000000000) % 10;
		hundredThousands = (value / 10000000000000) % 10;
		tenThousands	 = (value / 1000000000000) % 10;
		thousands		 = (value / 100000000000) % 10;
		hundreds		 = (value / 10000000000) % 10;
		tens			 = (value / 1000000000) % 10;
		ones			 = (value / 100000000) % 10;
		packedDigits	 = tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) |
			(ones << 56) + 0x3030303030303030ull;
		std::memcpy(string + 1, &packedDigits, 8);
		ones		 = (value / 10000000000000000) % 10;
		packedDigits = ones + 0x3030303030303030ull;
		string[0]	 = static_cast<char>(packedDigits);
	}
};

template<> struct int_serializer<18> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t tenMillions	  = (value / 10000000) % 10;
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits =
			(tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) | (ones << 56)) +
			0x3030303030303030ull;

		std::memcpy(string + 10, &packedDigits, sizeof(packedDigits));
		tenMillions		 = (value / 1000000000000000) % 10;
		millions		 = (value / 100000000000000) % 10;
		hundredThousands = (value / 10000000000000) % 10;
		tenThousands	 = (value / 1000000000000) % 10;
		thousands		 = (value / 100000000000) % 10;
		hundreds		 = (value / 10000000000) % 10;
		tens			 = (value / 1000000000) % 10;
		ones			 = (value / 100000000) % 10;
		packedDigits	 = tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) |
			(ones << 56) + 0x3030303030303030ull;
		std::memcpy(string + 2, &packedDigits, 8);
		tens		 = (value / 100000000000000000) % 10;
		ones		 = (value / 10000000000000000) % 10;
		packedDigits = tens | (ones << 8) + 0x3030303030303030ull;
		std::memcpy(string, &packedDigits, 2);
	}
};

template<> struct int_serializer<19> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t tenMillions	  = (value / 10000000) % 10;
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits =
			(tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) | (ones << 56)) +
			0x3030303030303030ull;

		std::memcpy(string + 11, &packedDigits, sizeof(packedDigits));
		tenMillions		 = (value / 1000000000000000) % 10;
		millions		 = (value / 100000000000000) % 10;
		hundredThousands = (value / 10000000000000) % 10;
		tenThousands	 = (value / 1000000000000) % 10;
		thousands		 = (value / 100000000000) % 10;
		hundreds		 = (value / 10000000000) % 10;
		tens			 = (value / 1000000000) % 10;
		ones			 = (value / 100000000) % 10;
		packedDigits	 = tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) |
			(ones << 56) + 0x3030303030303030ull;
		std::memcpy(string + 3, &packedDigits, 8);
		hundreds	 = (value / 1000000000000000000) % 10;
		tens		 = (value / 100000000000000000) % 10;
		ones		 = (value / 10000000000000000) % 10;
		packedDigits = hundreds | (tens << 8) | (ones << 16) + 0x3030303030303030ull;
		std::memcpy(string, &packedDigits, 3);
	}
};

template<> struct int_serializer<20> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t tenMillions	  = (value / 10000000) % 10;
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint8_t thousands		  = (value / 1000) % 10;
		uint8_t hundreds		  = (value / 100) % 10;
		uint8_t tens			  = (value / 10) % 10;
		uint8_t ones			  = value % 10;

		uint64_t packedDigits =
			(tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) | (ones << 56)) +
			0x3030303030303030ull;

		std::memcpy(string + 12, &packedDigits, sizeof(packedDigits));
		tenMillions		 = (value / 1000000000000000) % 10;
		millions		 = (value / 100000000000000) % 10;
		hundredThousands = (value / 10000000000000) % 10;
		tenThousands	 = (value / 1000000000000) % 10;
		thousands		 = (value / 100000000000) % 10;
		hundreds		 = (value / 10000000000) % 10;
		tens			 = (value / 1000000000) % 10;
		ones			 = (value / 100000000) % 10;
		packedDigits	 = tenMillions | (millions << 8) | (hundredThousands << 16) | (tenThousands << 24) | (thousands << 32) | (hundreds << 40) | (tens << 48) |
			(ones << 56) + 0x3030303030303030ull;
		std::memcpy(string + 4, &packedDigits, 8);
		thousands	 = (value / 10000000000000000000) % 10;
		hundreds	 = (value / 1000000000000000000) % 10;
		tens		 = (value / 100000000000000000) % 10;
		ones		 = (value / 10000000000000000) % 10;
		packedDigits = thousands | (hundreds << 8) | (tens << 16) | (ones << 24) + 0x30303030ull;
		std::memcpy(string, &packedDigits, 4);
	}
};

JSONIFIER_ALWAYS_INLINE static void serializeFunctionImpl(uint64_t value, char* string) {
	uint64_t digitCount{ jsonifier_internal::fastDigitCount(value) };
	switch (digitCount) {
		case 0: {
			break;
		}
		case 1: {
			int_serializer<1>::impl(value, string);
			break;
		}
		case 2: {
			int_serializer<2>::impl(value, string);
			break;
		}
		case 3: {
			int_serializer<3>::impl(value, string);
			break;
		}
		case 4: {
			int_serializer<4>::impl(value, string);
			break;
		}
		case 5: {
			int_serializer<5>::impl(value, string);
			break;
		}
		case 6: {
			int_serializer<6>::impl(value, string);
			break;
		}
		case 7: {
			int_serializer<7>::impl(value, string);
			break;
		}
		case 8: {
			int_serializer<8>::impl(value, string);
			break;
		}
		case 9: {
			int_serializer<9>::impl(value, string);
			break;
		}
		case 10: {
			int_serializer<10>::impl(value, string);
			break;
		}
		case 11: {
			int_serializer<11>::impl(value, string);
			break;
		}
		case 12: {
			int_serializer<12>::impl(value, string);
			break;
		}
		case 13: {
			int_serializer<13>::impl(value, string);
			break;
		}
		case 14: {
			int_serializer<14>::impl(value, string);
			break;
		}
		case 15: {
			int_serializer<15>::impl(value, string);
			break;
		}
		case 16: {
			int_serializer<16>::impl(value, string);
			break;
		}
		case 17: {
			int_serializer<17>::impl(value, string);
			break;
		}
		case 18: {
			int_serializer<18>::impl(value, string);
			break;
		}
		case 19: {
			int_serializer<19>::impl(value, string);
			break;
		}
		case 20: {
			int_serializer<20>::impl(value, string);
			break;
		}
	}
}

JSONIFIER_ALWAYS_INLINE static void serializeFunction(std::integral auto value, char* string) {
	auto newValue = static_cast<uint64_t>(value < 0 ? (*string = '-', ++string, value * -1) : value);
	serializeFunctionImpl(newValue, string);
}

template<typename data_structure, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew>
JSONIFIER_ALWAYS_INLINE void runForLengthSerialize(const int64_t size) {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	using value_type = std::conditional_t<std::signed_integral<data_structure>, int64_t, uint64_t>;
	jsonifier::jsonifier_core parser{};
	std::vector<data_structure> rawValues{};
	std::vector<std::string> results01{};
	std::vector<std::string> results02{};
	for (size_t x = 0; x < 300 * 17; ++x) {
		rawValues.emplace_back(test_generator::generateValue<data_structure>(size));
		std::string newString{};
		newString.resize(21);
		results01.emplace_back(newString);
		results02.emplace_back(newString);
	}
	size_t currentIndex{};
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Glaze-Function", "dodgerblue">(
		[=, &results01]() mutable {
			for (size_t x = 0; x < 16; ++x) {
				auto* iter = results01[currentIndex].data();
				jsonifier_internal::toChars(iter, static_cast<value_type>(rawValues[currentIndex]));
				bnch_swt::doNotOptimizeAway(results01);
				++currentIndex;
			}
		});
	currentIndex = 0;
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Jsonifier-Function", "dodgerblue">(
		[=, &results02]() mutable {
			for (size_t x = 0; x < 16; ++x) {
				auto* iter = results02[currentIndex].data();
				serializeFunction(rawValues[currentIndex], iter);
				bnch_swt::doNotOptimizeAway(results02);
				++currentIndex;
			}
		});
	for (size_t x = 0; x < results01.size(); ++x) {
		if (results01[x] != results02[x]) {
			std::cout << "NOT EQUAL AT INDEX: " << x << std::endl;
			std::cout << "INTENDED: " << results01[x] << std::endl;
			std::cout << "ACTUAL: " << results02[x] << std::endl;
		}
	}
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

class int_128 {
  public:
	uint64_t high;
	uint64_t low;

	int_128(uint64_t low = 0, uint64_t high = 0) : high(high), low(low) {
	}

	int_128 operator+(const int_128& other) const {
		uint64_t new_low  = low + other.low;
		uint64_t carry	  = (new_low < low);// Check if overflow occurred in low part
		uint64_t new_high = high + other.high + carry;
		return int_128(new_low, new_high);
	}

	 int_128 operator-(const int_128& other) const {
		uint64_t new_low  = low - other.low;
		uint64_t borrow	  = (low < other.low);// Check if borrow is needed
		uint64_t new_high = high - other.high - borrow;
		return int_128(new_low, new_high);
	}

	// Left shift operator
	int_128 operator<<(int shift) const {
		if (shift >= 64) {
			// Shift beyond 64 bits: low becomes 0, high shifts to low
			return int_128(0, low << (shift - 64));
		} else if (shift > 0) {
			// Regular shift within the 64-bit boundary
			return int_128(low << shift, (high << shift) | (low >> (64 - shift)));
		}
		return *this;
	}

	// Right shift operator
	int_128 operator>>(int shift) const {
		if (shift >= 64) {
			// Shift beyond 64 bits: high becomes 0, low shifts from high
			return int_128(high >> (shift - 64), 0);
		} else if (shift > 0) {
			// Regular shift within the 64-bit boundary
			return int_128((low >> shift) | (high << (64 - shift)), high >> shift);
		}
		return *this;
	}

	// Support ostream output
	friend std::ostream& operator<<(std::ostream& os, const int_128& value) {
		return os << "LOW: " << std::bitset<64>{ value.low } << ", HIGH: " << std::bitset<64>{ value.high } << std::endl;
	}
};

constexpr std::uint64_t log2New(std::uint64_t x) {
	std::uint64_t result = 0;
	while (x >>= 1) {
		++result;
	}
	return result;
}

constexpr std::uint64_t log10New(std::uint64_t x) {
	std::uint64_t result = 0;
	while (x >= 10) {
		x /= 10;
		++result;
	}
	return result;
}

constexpr std::uint64_t powNew(std::uint64_t base, std::uint64_t exp) {
	std::uint64_t result = 1;
	while (exp > 0) {
		if (exp & 1) {
			result *= base;
		}
		base *= base;
		exp >>= 1;
	}
	return result;
}

namespace cpp23 {
	std::uint64_t log2(std::uint64_t x) {
		std::cout << "LOG 2 INPUT: " << x << std::endl;
		std::cout << "LOG 2 OUTPUT: " << __builtin_log2(x) << std::endl;
		return __builtin_log2(x);
	}
	auto log10(std::uint64_t x) {
		std::cout << "LOG 10 INPUT: " << x << std::endl;
		std::cout << "LOG 10 OUTPUT: " << __builtin_log10(x) << std::endl;
		std::cout << "BUILTIN CEIL INPUT: " << __builtin_log10(x) << std::endl;
		std::cout << "BUILTIN CEIL OUTPUT: " << __builtin_ceil(__builtin_log10(x)) << std::endl;
		return long(__builtin_ceil(__builtin_log10(x)));
	}
	std::uint64_t pow(std::uint64_t x, std::uint64_t e) {
		std::cout << "POW INPUT: " << x << std::endl;
		std::cout << "POW OUTPUT: " << __builtin_pow(x, e) << std::endl;
		return __builtin_pow(x, e);
	}
}

	static std::array<std::uint64_t, 32> table = []() {
		std::array<std::uint64_t, 32> table;
		for (unsigned long i = 1; i < 33; i++) {
			const unsigned smallest = cpp23::pow(2, i - 1);
			table[i - 1]			= (i < 31 ? (cpp23::pow(2, 32) - cpp23::pow(10, cpp23::log10(smallest))) : 0) + (cpp23::log10(smallest) << 32);
		}
		return table;
	}();
/*
int digit_count(uint32_t x) {
	static constexpr std::array<std::uint64_t, 32> table = []() {
		std::array<std::uint64_t, 32> table;
		for (unsigned long i = 1; i < 33; i++) {
			const unsigned smallest = pow_custom(2, i - 1);
			table[i - 1]			= (i < 31 ? (pow_custom(2, 32) - pow_custom(10, log10_custom(smallest))) : 0) + (log10_custom(smallest) << 32);
		}
		return table;
	}();

	// Example of accessing the table for digit count.
	return table[log2_custom(x)];
}
*/

int main() {


	// Output the table
	for (int i = 0; i < 32; ++i) {
		std::cout << table[i] << ", ";
	} /*
	std::cout << (int_128{ 1 } << 64) + (0 << 64) << std::endl;
	std::cout << std::ceil(log10(1)) << std::endl;
	std::cout << (int_128{ 1 } << 64) - 10 + (int_128{ 1 } << 64) << std::endl;
	std::cout << std::ceil(log10(2)) << std::endl;
	std::cout << (int_128{ 1 } << 64) - 10 + (int_128{ 1 } << 64) << std::endl;
	std::cout << std::ceil(log10(4)) << std::endl;
	std::cout << (int_128{ 1 } << 64) - 10 + (int_128{ 1 } << 64) << std::endl;
	std::cout << std::ceil(log10(8)) << std::endl;
	std::cout << (int_128{ 1 } << 64) - 100 + (2 << 64) << std::endl;
	std::cout << std::ceil(log10(16)) << std::endl;//2
	std::cout << (int_128{ 1 } << 64) - 100 + (2 << 64) << std::endl;
	std::cout << std::ceil(log10(32)) << std::endl;//2
	std::cout << (int_128{ 1 } << 64) - 100 + (2 << 64) << std::endl;
	std::cout << std::ceil(log10(64)) << std::endl;//2
	std::cout << (int_128{ 1 } << 64) - 1000 + (3 << 64) << std::endl;
	std::cout << std::ceil(log10(128)) << std::endl;//3
	std::cout << (int_128{ 1 } << 64) - 1000 + (3 << 64) << std::endl;
	std::cout << std::ceil(log10(256)) << std::endl;//3
	std::cout << (int_128{ 1 } << 64) - 1000 + (3 << 64) << std::endl;
	std::cout << std::ceil(log10(512)) << std::endl;//3
	std::cout << (int_128{ 1 } << 64) - 10000 + (4 << 64) << std::endl;
	std::cout << std::ceil(log10(1024)) << std::endl;//4
	std::cout << (int_128{ 1 } << 64) - 10000 + (4 << 64) << std::endl;
	std::cout << std::ceil(log10(2048)) << std::endl;//4
	std::cout << (int_128{ 1 } << 64) - 10000 + (4 << 64) << std::endl;
	std::cout << std::ceil(log10(4096)) << std::endl;//4
	std::cout << (int_128{ 1 } << 64) - 10000 + (4 << 64) << std::endl;
	std::cout << std::ceil(log10(8192)) << std::endl;//4
	std::cout << (int_128{ 1 } << 64) - 100000 + (5 << 64) << std::endl;
	std::cout << std::ceil(log10(16384)) << std::endl;//5
	std::cout << (int_128{ 1 } << 64) - 100000 + (5 << 64) << std::endl;
	std::cout << std::ceil(log10(32768)) << std::endl;//5
	std::cout << (int_128{ 1 } << 64) - 100000 + (5 << 64) << std::endl;
	std::cout << std::ceil(log10(65536)) << std::endl;//5
	std::cout << (int_128{ 1 } << 64) - 1000000 + (6 << 64) << std::endl;
	std::cout << std::ceil(log10(131072)) << std::endl;//6
	std::cout << (int_128{ 1 } << 64) - 1000000 + (6 << 64) << std::endl;
	std::cout << std::ceil(log10(262144)) << std::endl;//6
	std::cout << (int_128{ 1 } << 64) - 1000000 + (6 << 64) << std::endl;
	std::cout << std::ceil(log10(524288)) << std::endl;//6
	std::cout << (int_128{ 1 } << 64) - 10000000 + (7 << 64) << std::endl;
	std::cout << std::ceil(log10(1048576)) << std::endl;//7
	std::cout << (int_128{ 1 } << 64) - 10000000 + (7 << 64) << std::endl;
	std::cout << std::ceil(log10(2097152)) << std::endl;//7
	std::cout << (int_128{ 1 } << 64) - 10000000 + (7 << 64) << std::endl;
	std::cout << std::ceil(log10(4194304)) << std::endl;//7
	std::cout << (int_128{ 1 } << 64) - 10000000 + (7 << 64) << std::endl;
	std::cout << std::ceil(log10(8388608)) << std::endl;//7
	std::cout << (int_128{ 1 } << 64) - 100000000 + (8 << 64) << std::endl;
	std::cout << std::ceil(log10(16777216)) << std::endl;//8
	std::cout << (int_128{ 1 } << 64) - 100000000 + (8 << 64) << std::endl;
	std::cout << std::ceil(log10(33554432)) << std::endl;//8
	std::cout << (int_128{ 1 } << 64) - 100000000 + (8 << 64) << std::endl;
	std::cout << std::ceil(log10(67108864)) << std::endl;//8
	std::cout << (int_128{ 1 } << 64) - 1000000000 + (9 << 64) << std::endl;
	std::cout << std::ceil(log10(134217728)) << std::endl;//9
	std::cout << (int_128{ 1 } << 64) - 1000000000 + (9 << 64) << std::endl;
	std::cout << std::ceil(log10(268435456)) << std::endl;//9
	std::cout << (int_128{ 1 } << 64) - 1000000000 + (9 << 64) << std::endl;
	// ceil(log10(536870912)) = 9
	int_128 num(0x00, 1234567890);
	std::cout << num << std::endl;

	int_128 shifted_right = num >> 64;
	std::cout << shifted_right << std::endl;

	int_128 shifted_left = num << 64;
	std::cout << shifted_left << std::endl;

	int_128 num02(1234567890, 0x00);
	std::cout << num02 << std::endl;

	shifted_right = num02 >> 64;
	std::cout << shifted_right << std::endl;

	shifted_left = num02 << 64;
	std::cout << shifted_left << std::endl;

	//int_128 shifted_right = num >> 70;// Shift by more than 64 bits
	//std::cout << shifted_right << std::endl;	
	for (auto value: table) {
		std::cout << value << ", ";
	}
	conformance_tests::conformanceTests();
	for (auto& value: table) {
		std::cout << value << ", " << std::endl;
	}
	uint64_t value{ 1 };
	std::cout << "LEADING ZEROS: " << fastDigitCount(value) << std::endl;
	value = 12;
	std::cout << "LEADING ZEROS: " << fastDigitCount(value) << std::endl;
	value = 123;
	std::cout << "LEADING ZEROS: " << fastDigitCount(value) << std::endl;
	value = 1234;
	std::cout << "LEADING ZEROS: " << fastDigitCount(value) << std::endl;
	value = 12345;
	std::cout << "LEADING ZEROS: " << fastDigitCount(value) << std::endl;
	value = 123456;
	std::cout << "LEADING ZEROS: " << fastDigitCount(value) << std::endl;
	value = 1234567;
	std::cout << "LEADING ZEROS: " << fastDigitCount(value) << std::endl;
	std::string newString{};
	newString.resize(21);
	serializeFunction(value, newString.data());
	value = 12;
	std::cout << "CURRENT DIGITS (REAL): " << newString << std::endl;
	serializeFunction(value, newString.data());
	value = 123;
	std::cout << "CURRENT DIGITS (REAL): " << newString << std::endl;
	serializeFunction(value, newString.data());
	value = 1234;
	std::cout << "CURRENT DIGITS (REAL): " << newString << std::endl;
	serializeFunction(value, newString.data());
	value = 12345;
	std::cout << "CURRENT DIGITS (REAL): " << newString << std::endl;
	serializeFunction(value, newString.data());
	value = 123456;
	std::cout << "CURRENT DIGITS (REAL): " << newString << std::endl;
	serializeFunction(value, newString.data());
	value = 1234567;
	std::cout << "CURRENT DIGITS (REAL): " << newString << std::endl;
	serializeFunction(value, newString.data());
	value = 12345678;
	std::cout << "CURRENT DIGITS (REAL): " << newString << std::endl;
	std::cout << "CURRENT DIGIT COUNT: " << jsonifier_internal::fastDigitCount(12) << std::endl;
	std::cout << "CURRENT DIGIT COUNT: " << jsonifier_internal::fastDigitCount(123) << std::endl;
	std::cout << "CURRENT DIGIT COUNT: " << jsonifier_internal::fastDigitCount(1234) << std::endl;
	std::cout << "CURRENT DIGIT COUNT: " << jsonifier_internal::fastDigitCount(12345) << std::endl;
	std::cout << "CURRENT DIGIT COUNT: " << jsonifier_internal::fastDigitCount(123456) << std::endl;
	std::cout << "CURRENT DIGIT COUNT: " << jsonifier_internal::fastDigitCount(1234567) << std::endl;
	//runForLengthSerialize<uint32_t, "serialize-uint64_t", "serialize-uint64_t">(1);
	//runForLengthSerialize<int32_t, "serialize-int64_t", "serialize-int64_t">(2);*/
	return 0;
}