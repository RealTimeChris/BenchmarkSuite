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
/// https://github.com/RealTimeChris/jsonifier
/// Feb 3, 2023
#pragma once

#include "FastFloatNew.hpp"

namespace jsonifier_internal_new {

	JSONIFIER_ALWAYS_INLINE_VARIABLE bool expTable[]{ false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false };

	JSONIFIER_ALWAYS_INLINE_VARIABLE bool expFracTable[]{ false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
		false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false };

	JSONIFIER_ALWAYS_INLINE_VARIABLE char decimal{ '.' };
	JSONIFIER_ALWAYS_INLINE_VARIABLE char minus{ '-' };
	JSONIFIER_ALWAYS_INLINE_VARIABLE char plus{ '+' };
	JSONIFIER_ALWAYS_INLINE_VARIABLE char zero{ '0' };
	JSONIFIER_ALWAYS_INLINE_VARIABLE char nine{ '9' };

#define JSONIFIER_IS_DIGIT(x) ((static_cast<uint8_t>(x - zero)) <= 9)

	template<typename value_type, typename char_t> JSONIFIER_ALWAYS_INLINE bool finishParse(value_type& value, char_t const*& iter, char_t const*& startDigits,
		fast_float_new::span<const char_t>& integer, int16_t& digitCount, int64_t& expNumber, int64_t& exponent, uint64_t& mantissa, bool& negative, bool& tooManyDigits,
		char_t const* end = nullptr, fast_float_new::span<const char_t> fraction = {}) noexcept {
		using namespace fast_float_new;
		if (digitCount > 19) {
			char_t const* start = startDigits;
			while ((*start == zero || *start == decimal)) {
				if (*start == zero) {
					--digitCount;
				}
				++start;
			}

			if (digitCount > 19) {
				tooManyDigits		 = true;
				mantissa			 = 0;
				auto newIter		 = integer.ptr;
				char_t const* intEnd = newIter + integer.length;
				static constexpr uint64_t minNineteenDigitInteger{ 1000000000000000000 };
				while ((mantissa < minNineteenDigitInteger) && (newIter != intEnd)) {
					mantissa = mantissa * 10 + static_cast<uint8_t>(*newIter - zero);
					++newIter;
				}
				if (mantissa >= minNineteenDigitInteger) {
					exponent = intEnd - newIter + expNumber;
				} else {
					newIter				  = fraction.ptr;
					char_t const* fracEnd = newIter + fraction.length;
					while ((mantissa < minNineteenDigitInteger) && (newIter != fracEnd)) {
						mantissa = mantissa * 10 + static_cast<uint8_t>(*newIter - zero);
						++newIter;
					}
					exponent = fraction.ptr - newIter + expNumber;
				}
			}
		}

		if (binary_format<value_type>::min_exponent_fast_path <= exponent && exponent <= binary_format<value_type>::max_exponent_fast_path && !tooManyDigits) {
			if (rounds_to_nearest::roundsToNearest) {
				if (mantissa <= binary_format<value_type>::max_mantissa_fast_path_value) {
					value = value_type(mantissa);
					if (exponent < 0) {
						value = value / binary_format<value_type>::exact_power_of_ten(-exponent);
					} else {
						value = value * binary_format<value_type>::exact_power_of_ten(exponent);
					}
					if (negative) {
						value = -value;
					}
					return true;
				}
			} else {
				if (exponent >= 0 && mantissa <= binary_format<value_type>::max_mantissa_fast_path(exponent)) {
#if defined(__clang__) || defined(JSONIFIER_FASTFLOAT_32BIT)
					if (mantissa == 0) {
						value = negative ? value_type(-0.) : value_type(0.);
						return true;
					}
#endif
					value = value_type(mantissa) * binary_format<value_type>::exact_power_of_ten(exponent);
					if (negative) {
						value = -value;
					}
					return true;
				}
			}
		}
		adjusted_mantissa am = compute_float<binary_format<value_type>>(exponent, mantissa);
		if (tooManyDigits && am.power2 >= 0) {
			if (am != compute_float<binary_format<value_type>>(exponent, mantissa + 1)) {
				am = compute_error<binary_format<value_type>>(exponent, mantissa);
			}
		}
		if JSONIFIER_UNLIKELY (am.power2 < 0) {
			am = digit_comp<value_type>(integer, mantissa, exponent, am, fraction);
		}
		to_float(negative, am, value);
		if JSONIFIER_UNLIKELY ((mantissa != 0 && am.mantissa == 0 && am.power2 == 0) || am.power2 == binary_format<value_type>::infinite_power) {
			return false;
		}
		return true;
	}

	template<typename value_type, typename char_t> JSONIFIER_ALWAYS_INLINE bool parseFloatAfterFrac(value_type& value, char_t const*& iter, char_t const*& startDigits,
		fast_float_new::span<const char_t>& integer, int16_t& digitCount, int64_t& expNumber, int64_t& exponent, uint64_t& mantissa, bool& negative, bool& tooManyDigits,
		char_t const* end = nullptr) noexcept {
		using namespace fast_float_new;
		++iter;
		fast_float_new::span<const char_t> fraction;
		char_t const* before = iter;

		if (end - iter >= 8) {
			loop_parse_if_eight_digits(iter, end, mantissa);
		}

		while (JSONIFIER_IS_DIGIT(*iter)) {
			mantissa = mantissa * 10 + static_cast<uint8_t>(*iter - zero);
			++iter;
		}
		exponent		= before - iter;
		fraction.length = static_cast<size_t>(iter - before);
		fraction.ptr	= before;
		digitCount -= exponent;

		if JSONIFIER_UNLIKELY (exponent == 0) {
			return false;
		}

		if (expTable[*iter]) {
			char_t const* locationOfE = iter;
			++iter;
			bool neg_exp = false;
			if (minus == *iter) {
				neg_exp = true;
				++iter;
			} else if (plus == *iter) {
				++iter;
			}
			if (!JSONIFIER_IS_DIGIT(*iter)) {
				iter = locationOfE;
			} else {
				while (JSONIFIER_IS_DIGIT(*iter)) {
					if (expNumber < 0x10000000) {
						expNumber = 10 * expNumber + static_cast<uint8_t>(*iter - zero);
					}
					++iter;
				}
				if (neg_exp) {
					expNumber = -expNumber;
				}
				exponent += expNumber;
			}
		}

		return finishParse(value, iter, startDigits, integer, digitCount, expNumber, exponent, mantissa, negative, tooManyDigits, end, fraction);
	}

	template<typename value_type, typename char_t> JSONIFIER_ALWAYS_INLINE bool parseFloatAfterExp(value_type& value, char_t const*& iter, char_t const*& startDigits,
		fast_float_new::span<const char_t>& integer, int16_t& digitCount, int64_t& expNumber, int64_t& exponent, uint64_t& mantissa, bool& negative, bool& tooManyDigits,
		char_t const* end = nullptr) noexcept {
		using namespace fast_float_new;
		char_t const* locationOfE = iter;
		++iter;
		bool neg_exp = false;
		if (minus == *iter) {
			neg_exp = true;
			++iter;
		} else if (plus == *iter) {
			++iter;
		}
		if (!JSONIFIER_IS_DIGIT(*iter)) {
			iter = locationOfE;
		} else {
			while (JSONIFIER_IS_DIGIT(*iter)) {
				if (expNumber < 0x10000000) {
					expNumber = 10 * expNumber + static_cast<uint8_t>(*iter - zero);
				}
				++iter;
			}
			if (neg_exp) {
				expNumber = -expNumber;
			}
			exponent += expNumber;
		}

		return finishParse(value, iter, startDigits, integer, digitCount, expNumber, exponent, mantissa, negative, tooManyDigits, end);
	}

	template<typename value_type, typename char_t> JSONIFIER_ALWAYS_INLINE bool parseFloat(value_type& value, char_t const*& iter, char_t const* end = nullptr) noexcept {
		using namespace fast_float_new;

		span<const char_t> integer;
		int16_t digitCount;
		int64_t expNumber{};
		int64_t exponent{};
		uint64_t mantissa{};
		bool negative{ *iter == minus };
		bool tooManyDigits{ false };

		if (negative) {
			++iter;

			if JSONIFIER_UNLIKELY (!JSONIFIER_IS_DIGIT(*iter)) {
				return false;
			}
		}
		char_t const* startDigits = iter;

		while (JSONIFIER_IS_DIGIT(*iter)) {
			mantissa = 10 * mantissa + static_cast<uint8_t>(*iter - zero);
			++iter;
		}

		digitCount	   = static_cast<int64_t>(iter - startDigits);
		integer.length = static_cast<size_t>(digitCount);
		integer.ptr	   = startDigits;

		if JSONIFIER_UNLIKELY (digitCount == 0 || (startDigits[0] == zero && digitCount > 1)) {
			return false;
		}

		if (*iter == decimal) {
			return parseFloatAfterFrac(value, iter, startDigits, integer, digitCount, expNumber, exponent, mantissa, negative, tooManyDigits, end);
		}
		if (expTable[*iter]) {
			return parseFloatAfterExp(value, iter, startDigits, integer, digitCount, expNumber, exponent, mantissa, negative, tooManyDigits, end);
		}
		return finishParse(value, iter, startDigits, integer, digitCount, expNumber, exponent, mantissa, negative, tooManyDigits);
	}
}