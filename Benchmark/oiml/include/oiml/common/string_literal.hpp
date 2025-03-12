#pragma once

#include <oiml/common/config.hpp>
#include <string_view>
#include <algorithm>
#include <array>

namespace oiml {

	template<size_t sizeVal> struct oiml_string_literal {
		using value_type	  = char;
		using const_reference = const value_type&;
		using reference		  = value_type&;
		using const_pointer	  = const value_type*;
		using pointer		  = value_type*;
		using size_type		  = size_t;

		static constexpr size_type length{ sizeVal > 0 ? sizeVal - 1 : 0 };

		constexpr oiml_string_literal() noexcept = default;

		constexpr oiml_string_literal(const char (&str)[sizeVal]) noexcept {
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

		template<size_type size_new> constexpr auto operator+=(const oiml_string_literal<size_new>& str) const noexcept {
			oiml_string_literal<size_new + sizeVal - 1> newLiteral{};
			std::copy(values, values + size(), newLiteral.data());
			std::copy(str.data(), str.data() + size_new, newLiteral.data() + size());
			return newLiteral;
		}

		template<size_type size_new> constexpr auto operator+=(const value_type (&str)[size_new]) const noexcept {
			oiml_string_literal<size_new + sizeVal - 1> newLiteral{};
			std::copy(values, values + size(), newLiteral.data());
			std::copy(str, str + size_new, newLiteral.data() + size());
			return newLiteral;
		}

		template<size_type size_new> constexpr auto operator+(const oiml_string_literal<size_new>& str) const noexcept {
			oiml_string_literal<size_new + sizeVal - 1> newLiteral{};
			std::copy(values, values + size(), newLiteral.data());
			std::copy(str.data(), str.data() + size_new, newLiteral.data() + size());
			return newLiteral;
		}

		template<size_type size_new> constexpr auto operator+(const value_type (&str)[size_new]) const noexcept {
			oiml_string_literal<size_new + sizeVal - 1> newLiteral{};
			std::copy(values, values + size(), newLiteral.data());
			std::copy(str, str + size_new, newLiteral.data() + size());
			return newLiteral;
		}

		template<size_type size_new> constexpr friend auto operator+(const value_type (&lhs)[size_new], const oiml_string_literal<sizeVal>& str) noexcept {
			return oiml_string_literal<size_new>{ lhs } + str;
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

		template<typename string_type> constexpr operator string_type() const {
			string_type returnValues{ values, length };
			return returnValues;
		}

		alignas(64) char values[sizeVal]{};
	};

	template<size_t N, typename string_type> constexpr auto stringLiteralFromView(string_type str) noexcept {
		oiml_string_literal<N + 1> sl{};
		std::copy_n(str.data(), str.size(), sl.values);
		sl[N] = '\0';
		return sl;
	}

	template<size_t size> OIML_FORCE_INLINE std::ostream& operator<<(std::ostream& os, const oiml_string_literal<size>& input) noexcept {
		os << input.operator std::string_view();
		return os;
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

	template<auto number, size_t numDigits = countDigits(number)> constexpr oiml_string_literal<numDigits + 1> toStringLiteral() noexcept {
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
		return oiml_string_literal<numDigits + 1>{ buffer };
	}

	constexpr char toLower(char input) noexcept {
		return (input >= 'A' && input <= 'Z') ? (input + 32) : input;
	}

	template<size_t size> constexpr auto toLower(oiml_string_literal<size> input) noexcept {
		oiml_string_literal<size> output{};
		for (size_t x = 0; x < size; ++x) {
			output[x] = toLower(input[x]);
		}
		return output;
	}

}
