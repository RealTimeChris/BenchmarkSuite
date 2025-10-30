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
#include <array>

namespace bnch_swt {

	template<uint64_t sizeVal> struct BNCH_SWT_ALIGN string_literal {
		using value_type	  = char;
		using const_reference = const value_type&;
		using reference		  = value_type&;
		using const_pointer	  = const value_type*;
		using pointer		  = value_type*;
		using uint64_type		  = uint64_t;

		static constexpr uint64_type length{ sizeVal > 0 ? sizeVal - 1 : 0 };
		static_assert(sizeVal > 0, "Sorry, but please instantiate string_literal with an actual string!");

		constexpr string_literal() noexcept = default;

		constexpr string_literal(const char (&str)[sizeVal]) noexcept {
			std::copy_n(str, sizeVal, values);
			values[length] = '\0';
		}

		constexpr const_pointer data() const noexcept {
			return values;
		}

		constexpr pointer data() noexcept {
			return values;
		}

		template<uint64_type sizeNew> constexpr auto operator+=(const string_literal<sizeNew>& str) const noexcept {
			string_literal<sizeNew + sizeVal - 1> newLiteral{};
			std::copy_n(values, size(), newLiteral.data());
			std::copy_n(str.data(), sizeNew, newLiteral.data() + size());
			return newLiteral;
		}

		template<uint64_type sizeNew> constexpr auto operator+=(const value_type (&str)[sizeNew]) const noexcept {
			string_literal<sizeNew + sizeVal - 1> newLiteral{};
			std::copy_n(values, size(), newLiteral.data());
			std::copy_n(str, sizeNew, newLiteral.data() + size());
			return newLiteral;
		}

		template<uint64_type sizeNew> constexpr auto operator+(const string_literal<sizeNew>& str) const noexcept {
			string_literal<sizeNew + sizeVal - 1> newLiteral{};
			std::copy_n(values, size(), newLiteral.data());
			std::copy_n(str.data(), sizeNew, newLiteral.data() + size());
			return newLiteral;
		}

		template<uint64_type sizeNew> constexpr auto operator+(const value_type (&str)[sizeNew]) const noexcept {
			string_literal<sizeNew + sizeVal - 1> newLiteral{};
			std::copy_n(values, size(), newLiteral.data());
			std::copy_n(str, sizeNew, newLiteral.data() + size());
			return newLiteral;
		}

		template<uint64_type sizeNew> constexpr friend auto operator+(const value_type (&lhs)[sizeNew], const string_literal<sizeVal>& str) noexcept {
			return string_literal<sizeNew>{ lhs } + str;
		}

		constexpr reference operator[](uint64_type index) noexcept {
			return values[index];
		}

		constexpr const_reference operator[](uint64_type index) const noexcept {
			return values[index];
		}

		constexpr uint64_type size() const noexcept {
			return length;
		}

		template<typename string_type> constexpr operator string_type() const {
			BNCH_SWT_ALIGN string_type returnValues{ values, length };
			return returnValues;
		}

		BNCH_SWT_ALIGN char values[sizeVal > 0 ? sizeVal : 1]{};
	};

	template<uint64_t size> string_literal(const char (&str)[size]) -> string_literal<size>;

	namespace internal {

		template<uint64_t N, typename string_type> constexpr auto stringLiteralFromView(string_type str) noexcept {
			string_literal<N + 1> sl{};
			std::copy_n(str.data(), str.size(), sl.values);
			sl[N] = '\0';
			return sl;
		}

		template<uint64_t size> BNCH_SWT_INLINE std::ostream& operator<<(std::ostream&, const string_literal<size>& input) noexcept {
			std::cout << input.operator std::string_view();
			return std::cout;
		}

		template<typename value_type> constexpr uint64_t countDigits(value_type number) noexcept {
			uint64_t count = 0;
			if (static_cast<int64_t>(number) < 0) {
				number *= -1;
				++count;
			}
			do {
				++count;
				number /= 10;
			} while (number != 0);
			return count;
		}

		template<auto number, uint64_t numDigits = countDigits(number)> constexpr string_literal<numDigits + 1> toStringLiteral() noexcept {
			char buffer[numDigits + 1]{};
			char* ptr = buffer + numDigits;
			*ptr	  = '\0';
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

		constexpr char toLower(char input) noexcept {
			return (input >= 'A' && input <= 'Z') ? (input + 32) : input;
		}

		template<uint64_t size> constexpr auto toLower(string_literal<size> input) noexcept {
			string_literal<size> output{};
			for (uint64_t x = 0; x < size; ++x) {
				output[x] = toLower(input[x]);
			}
			return output;
		}

	}

}
