/*
	MIT License

	Copyright (c) 2024 RealTimeChris

	Permission is hereby granted, free of charge, to any person obtaining a copy of this
	software and associated documentation files (the "Software"), to deal in the Software
	without restriction, including without limitation the rights to use, copy, modify, merge,
	publish, distribute, sublicense, and/or sell copies of the Software, and to permit
	persons to whom the Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all copies or
	substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	DEALINGS IN THE SOFTWARE.
*/
/// https://github.com/RealTimeChris/benchmarksuite
/// Sep 1, 2024
#pragma once

#include <BnchSwt/Config.hpp>
#include <string_view>
#include <algorithm>
#include <string>
#include <array>

namespace bnch_swt {

	template<size_t sizeVal> struct string_literal {
		using value_type	  = char;
		using const_reference = const value_type&;
		using reference		  = value_type&;
		using const_pointer	  = const value_type*;
		using pointer		  = value_type*;
		using size_type		  = size_t;

		static constexpr size_type length{ sizeVal > 0 ? sizeVal - 1 : 0 };

		constexpr string_literal() noexcept = default;

		constexpr string_literal(const char (&str)[sizeVal]) noexcept {
			for (size_t x = 0; x < length; ++x) {
				values[x] = str[x];
			}
			values[length] = '\0';
		}

		constexpr const_pointer data() const noexcept {
			return values;
		}

		constexpr pointer data() noexcept {
			return values;
		}

		template<size_type sizeNew> constexpr auto operator+=(const string_literal<sizeNew>& str) const noexcept {
			string_literal<sizeNew + sizeVal - 1> newLiteral{};
			std::copy(values, values + size(), newLiteral.data());
			std::copy(str.data(), str.data() + sizeNew, newLiteral.data() + size());
			return newLiteral;
		}

		template<size_type sizeNew> constexpr auto operator+=(const value_type (&str)[sizeNew]) const noexcept {
			string_literal<sizeNew + sizeVal - 1> newLiteral{};
			std::copy(values, values + size(), newLiteral.data());
			std::copy(str, str + sizeNew, newLiteral.data() + size());
			return newLiteral;
		}

		template<size_type sizeNew> constexpr auto operator+(const string_literal<sizeNew>& str) const noexcept {
			string_literal<sizeNew + sizeVal - 1> newLiteral{};
			std::copy(values, values + size(), newLiteral.data());
			std::copy(str.data(), str.data() + sizeNew, newLiteral.data() + size());
			return newLiteral;
		}

		template<size_type sizeNew> constexpr auto operator+(const value_type (&str)[sizeNew]) const noexcept {
			string_literal<sizeNew + sizeVal - 1> newLiteral{};
			std::copy(values, values + size(), newLiteral.data());
			std::copy(str, str + sizeNew, newLiteral.data() + size());
			return newLiteral;
		}

		template<size_type sizeNew> constexpr friend auto operator+(const value_type (&lhs)[sizeNew], const string_literal<sizeVal>& str) noexcept {
			return string_literal<sizeNew>{ lhs } + str;
		}

		constexpr reference operator[](size_type index) noexcept {
			return values[index];
		}

		constexpr const_reference operator[](size_type index) const noexcept {
			return values[index];
		}

		constexpr size_type size() const noexcept {
			return length;
		}

		BNCH_SWT_INLINE operator std::string() const noexcept {
			std::string returnValues{ values, length };
			return returnValues;
		}

		BNCH_SWT_INLINE constexpr std::string_view view() const noexcept {
			return std::string_view{ values, length };
		}

		value_type values[sizeVal]{};
	};

	template<size_t N> constexpr auto stringLiteralFromView(std::string_view str) noexcept {
		bnch_swt::string_literal<N + 1> sl{};
		std::copy_n(str.data(), str.size(), sl.values);
		sl[N] = '\0';
		return sl;
	}

	template<size_t size> BNCH_SWT_INLINE std::ostream& operator<<(std::ostream& os, const string_literal<size>& input) noexcept {
		os << input.view();
		return os;
	}

	constexpr size_t countDigits(int64_t number) noexcept {
		size_t count = 0;
		if (number < 0) {
			number *= -1;
			++count;
		}
		do {
			++count;
			number /= 10;
		} while (number != 0);
		return count;
	}

	template<int64_t number, size_t numDigits = countDigits(number)> constexpr string_literal<numDigits + 1> toStringLiteral() noexcept {
		char buffer[numDigits + 1]{};
		char* ptr = buffer + numDigits;
		*ptr				  = '\0';
		int64_t temp{};
		if constexpr (number < 0) {
			temp			   = number * -1;
			*(ptr - numDigits) = '-';
		} else {
			temp = number;
		}
		do {
			*--ptr = '0' + (temp % 10);
			temp /= 10;
		} while (temp != 0);
		return string_literal<numDigits + 1>{ buffer };
	}

	template<auto valueNew> struct make_static {
		static constexpr auto value{ valueNew };
	};

	constexpr char toLower(char input) noexcept {
		return (input >= 'A' && input <= 'Z') ? (input + 32) : input;
	}

	template<size_t size> constexpr auto toLower(string_literal<size> input) noexcept {
		string_literal<size> output{};
		for (size_t x = 0; x < size; ++x) {
			output[x] = toLower(input[x]);
		}
		return output;
	}

	template<int64_t number> constexpr std::string_view toStringView() noexcept {
		constexpr auto& lit = bnch_swt::make_static<toStringLiteral<number>()>::value;
		return std::string_view{ lit.data(), lit.size() };
	}

}