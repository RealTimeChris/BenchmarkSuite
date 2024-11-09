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
		uint64_t tens = (value / 10);
		uint64_t ones = value % 10;

		uint64_t packedDigits = ((tens) | (ones << 8)) + 0x3030;

		std::memcpy(string, &packedDigits, sizeof(packedDigits));
	}
};

template<> struct int_serializer<3> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t hundreds	  = (value / 100) % 10;
		uint64_t tens		  = (value / 10) % 10;
		uint64_t ones		  = value % 10;
		uint64_t packedDigits = ((hundreds) | (tens << 8) | (ones << 16)) + 0x303030;
		std::memcpy(string, &packedDigits, sizeof(packedDigits));
	}
};

template<> struct int_serializer<4> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t thousands	  = (value / 1000) % 10;
		uint64_t hundreds	  = (value / 100) % 10;
		uint64_t tens		  = (value / 10) % 10;
		uint64_t ones		  = value % 10;
		uint64_t packedDigits = (thousands | (hundreds << 8) | (tens << 16) | (ones << 24)) + 0x30303030;
		std::memcpy(string, &packedDigits, sizeof(packedDigits));
	}
};

template<> struct int_serializer<5> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t tenThousands = (value / 10000) % 10;
		uint64_t thousands	  = (value / 1000) % 10;
		uint64_t hundreds	  = (value / 100) % 10;
		uint64_t tens		  = (value / 10) % 10;
		uint64_t ones		  = value % 10;
		uint64_t packedDigits = (tenThousands | (thousands << 8) | (hundreds << 16) | (tens << 24) | (ones << 32)) + 0x3030303030;
		std::memcpy(string, &packedDigits, sizeof(packedDigits));
	}
};

template<> struct int_serializer<6> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

		uint64_t packedDigits = (hundredThousands | (tenThousands << 8) | (thousands << 16) | (hundreds << 24) | (tens << 32) | (ones << 40)) + 0x303030303030;
		std::memcpy(string, &packedDigits, sizeof(packedDigits));
	}
};

template<> struct int_serializer<7> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t value, char* string) {
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
		uint64_t tenMillions	  = (value / 10000000) % 10;
		uint64_t millions		  = (value / 1000000) % 10;
		uint64_t hundredThousands = (value / 100000) % 10;
		uint64_t tenThousands	  = (value / 10000) % 10;
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
		uint64_t thousands		  = (value / 1000) % 10;
		uint64_t hundreds		  = (value / 100) % 10;
		uint64_t tens			  = (value / 10) % 10;
		uint64_t ones			  = value % 10;

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
			return;
		}
		case 1: {
			int_serializer<1>::impl(value, string);
			return;
		}
		case 2: {
			int_serializer<2>::impl(value, string);
			return;
		}
		case 3: {
			int_serializer<3>::impl(value, string);
			return;
		}
		case 4: {
			int_serializer<4>::impl(value, string);
			return;
		}
		case 5: {
			int_serializer<5>::impl(value, string);
			return;
		}
		case 6: {
			int_serializer<6>::impl(value, string);
			return;
		}
		case 7: {
			int_serializer<7>::impl(value, string);
			return;
		}
		case 8: {
			int_serializer<8>::impl(value, string);
			return;
		}
		case 9: {
			int_serializer<9>::impl(value, string);
			return;
		}
		case 10: {
			int_serializer<10>::impl(value, string);
			return;
		}
		case 11: {
			int_serializer<11>::impl(value, string);
			return;
		}
		case 12: {
			int_serializer<12>::impl(value, string);
			return;
		}
		case 13: {
			int_serializer<13>::impl(value, string);
			return;
		}
		case 14: {
			int_serializer<14>::impl(value, string);
			return;
		}
		case 15: {
			int_serializer<15>::impl(value, string);
			return;
		}
		case 16: {
			int_serializer<16>::impl(value, string);
			return;
		}
		case 17: {
			int_serializer<17>::impl(value, string);
			return;
		}
		case 18: {
			int_serializer<18>::impl(value, string);
			return;
		}
		case 19: {
			int_serializer<19>::impl(value, string);
			return;
		}
		case 20: {
			int_serializer<20>::impl(value, string);
			return;
		}
	}
}

JSONIFIER_ALWAYS_INLINE static void serializeFunction(std::integral auto value, char* string) {
	auto newValue = static_cast<uint64_t>(value < 0 ? (*string = '-', ++string, value * -1) : value);
	serializeFunctionImpl(newValue, string);
}

int main() {
	std::string newString{};
	newString.resize(21);
	int64_t value{ 1 };
	serializeFunction(value, newString.data());
	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	std::cout << "CURRENT DIGITS: " << newString << std::endl;
	value = -12;
	////newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -123;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -1234;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -12345;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -123456;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -1234567;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -12345678;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -123456789;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -1234567890;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -12345678901;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -123456789012;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -1234567890123;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -12345678901234;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -123456789012345;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -1234567890123456;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -12345678901234567;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -123456789012345678;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -1234567890123456789;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	value = -12345678901234567890;
	//newString = "";
	serializeFunction(value, newString.data());
	std::cout << "CURRENT DIGITS: " << newString << std::endl;

	//std::cout << "CURRENT DIGITS: " << digit_count_new(value) << std::endl;
	//conformance_tests::conformanceTests();
	return 0;
}