// BenchmarkSuite.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <BnchSwt/Config.hpp>
#include <jsonifier/Index.hpp>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <vector>

namespace bnch_swt {	

	template<size_t sizeVal> struct string_literal {
		using value_type = char;
		using const_reference = const value_type&;
		using reference = value_type&;
		using const_pointer = const value_type*;
		using size_type = size_t;

		static constexpr size_type length{ sizeVal > 0 ? sizeVal - 1 : 0 };

		constexpr string_literal() noexcept = default;

		constexpr string_literal(const value_type (&str)[sizeVal]) {
			std::copy(str, str + length, values);
		}

		constexpr const_pointer data() const {
			return values;
		}

		constexpr reference operator[](size_type index) {
			return values[index];
		}

		constexpr const_reference operator[](size_type index) const {
			return values[index];
		}

		constexpr size_type size() const {
			return length;
		}

		BNCH_SWT_INLINE operator jsonifier::string() const {
			return { values, length };
		}

		BNCH_SWT_INLINE operator std::string() const {
			return { values, length };
		}

		constexpr operator jsonifier::string_view() const {
			return { values, length };
		}

		value_type values[sizeVal]{};
	};

	template<size_t N> constexpr auto stringLiteralFromView(jsonifier::string_view str) {
		string_literal<N + 1> sl{};
		std::copy_n(str.data(), str.size(), sl.values);
		sl[N] = '\0';
		return sl;
	}

	constexpr std::size_t countDigits(uint32_t number) {
		std::size_t count = 0;
		do {
			++count;
			number /= 10;
		} while (number != 0);
		return count;
	}

	template<uint32_t number> constexpr string_literal<countDigits(number) + 1> toStringLiteral() {
		constexpr std::size_t num_digits = countDigits(number);
		char buffer[num_digits + 1]{};
		char* ptr	  = buffer + num_digits;
		*ptr		  = '\0';
		uint32_t temp = number;
		do {
			*--ptr = '0' + (temp % 10);
			temp /= 10;
		} while (temp != 0);
		return string_literal<countDigits(number) + 1>{ buffer };
	}

	template<auto valueNew> struct make_static {
		static constexpr auto value{ valueNew };
	};

	template<uint32_t number> constexpr jsonifier::string_view toStringView() {
		constexpr auto& lit = jsonifier_internal::make_static<toStringLiteral<number>()>::value;
		return jsonifier::string_view(lit.value.data(), lit.value.size() - 1);
	}

	template<bnch_swt::string_literal... strings> constexpr auto joinLiterals() {
		constexpr size_t newSize = { (strings.size() + ...) };
		char returnValue[newSize + 1]{};
		returnValue[newSize] = '\0';
		auto copyLambda = [&](const char* ptr, size_t newSize, size_t& currentOffset) {
			std::copy(ptr, ptr + newSize, returnValue + currentOffset);
			currentOffset += newSize;
		};
		size_t currentOffset{};
		(copyLambda(strings.data(), strings.size(), currentOffset), ...);
		return bnch_swt::string_literal{ returnValue };
	}

}
