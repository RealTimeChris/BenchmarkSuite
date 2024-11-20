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

	template<typename T, typename UC> JSONIFIER_ALWAYS_INLINE bool parseFloat(UC const*& iter, UC const* end, T& value) noexcept {
		using namespace fast_float_new;
		static_assert(is_supported_float_t<T>(), "only some floating-point types are supported");
		static_assert(is_supported_char_t<UC>(), "only char, wchar_t, char16_t and char32_t are supported");

		static constexpr UC decimalNew = '.';
		static constexpr UC smallE	   = 'e';
		static constexpr UC bigE	   = 'E';
		static constexpr UC minusNew   = '-';
		static constexpr UC plusNew	   = '+';
		static constexpr UC zeroNew	   = '0';
		span<const UC> fraction;
		bool too_many_digits = false;
		bool negative		 = (*iter == minusNew);
		if (negative) {
			++iter;

			if JSONIFIER_UNLIKELY (!JSONIFIER_IS_DIGIT(*iter)) {
				return false;
			}
		}
		UC const* const start_digits = iter;

		uint64_t mantissa = 0;

		while (JSONIFIER_IS_DIGIT(*iter)) {
			mantissa = 10 * mantissa + static_cast<uint64_t>(*iter - zeroNew);
			++iter;
		}

		UC const* const end_of_integer_part = iter;
		int64_t digit_count					= static_cast<int64_t>(end_of_integer_part - start_digits);
		span<const UC> integer{ start_digits, static_cast<size_t>(digit_count) };

		if (digit_count == 0 || (start_digits[0] == zeroNew && digit_count > 1)) {
			return false;
		}

		int64_t exponent			 = 0;
		if (*iter == decimalNew) {
			++iter;
			UC const* before = iter;

			if (end - iter >= 8) {
				loop_parse_if_eight_digits(iter, end, mantissa);
			}

			while (JSONIFIER_IS_DIGIT(*iter)) {
				uint8_t digit = static_cast<uint8_t>(*iter - zeroNew);
				++iter;
				mantissa = mantissa * 10 + digit;
			}
			exponent			   = before - iter;
			fraction.length = static_cast<size_t>(iter - before);
			fraction.ptr	   = before;
			digit_count -= exponent;

			if (exponent == 0) {
				return false;
			}
		}

		int64_t exp_number = 0;

		if ((smallE == *iter) || (bigE == *iter)) {
			UC const* location_of_e = iter;
			++iter;
			bool neg_exp = false;
			if (minusNew == *iter) {
				neg_exp = true;
				++iter;
			} else if (plusNew == *iter) {
				++iter;
			}
			if (!JSONIFIER_IS_DIGIT(*iter)) {
				iter = location_of_e;
			} else {
				while (JSONIFIER_IS_DIGIT(*iter)) {
					uint8_t digit = static_cast<uint8_t>(*iter - zeroNew);
					if (exp_number < 0x10000000) {
						exp_number = 10 * exp_number + digit;
					}
					++iter;
				}
				if (neg_exp) {
					exp_number = -exp_number;
				}
				exponent += exp_number;
			}
		}

		if (digit_count > 19) {
			UC const* start = start_digits;
			while ((*start == zeroNew || *start == decimalNew)) {
				if (*start == zeroNew) {
					--digit_count;
				}
				++start;
			}

			if (digit_count > 19) {
				too_many_digits = true;
				mantissa					   = 0;
				iter				   = integer.ptr;
				UC const* int_end	   = iter + integer.len();
				static constexpr uint64_t minimal_nineteen_digit_integer{ 1000000000000000000 };
				while ((mantissa < minimal_nineteen_digit_integer) && (iter != int_end)) {
					mantissa = mantissa * 10 + static_cast<uint64_t>(*iter - zeroNew);
					++iter;
				}
				if (mantissa >= minimal_nineteen_digit_integer) {
					exponent = end_of_integer_part - iter + exp_number;
				} else {
					iter			   = fraction.ptr;
					UC const* frac_end = iter + fraction.len();
					while ((mantissa < minimal_nineteen_digit_integer) && (iter != frac_end)) {
						mantissa = mantissa * 10 + static_cast<uint64_t>(*iter - zeroNew);
						++iter;
					}
					exponent = fraction.ptr - iter + exp_number;
				}
			}
		}

		if (binary_format<T>::min_exponent_fast_path() <= exponent && exponent <= binary_format<T>::max_exponent_fast_path() && !too_many_digits) {
			if (rounds_to_nearest::roundsToNearest) {
				if (mantissa <= binary_format<T>::max_mantissa_fast_path()) {
					value = T(mantissa);
					if (exponent < 0) {
						value = value / binary_format<T>::exact_power_of_ten(-exponent);
					} else {
						value = value * binary_format<T>::exact_power_of_ten(exponent);
					}
					if (negative) {
						value = -value;
					}
					return true;
				}
			} else {
				if (exponent >= 0 && mantissa <= binary_format<T>::max_mantissa_fast_path(exponent)) {
#if defined(__clang__) || defined(FASTFLOAT_NEWER_32BIT)
					if (mantissa == 0) {
						value = negative ? T(-0.) : T(0.);
						return true;
					}
#endif
					value = T(mantissa) * binary_format<T>::exact_power_of_ten(exponent);
					if (negative) {
						value = -value;
					}
					return true;
				}
			}
		}
		adjusted_mantissa am = compute_float<binary_format<T>>(exponent, mantissa);
		if (too_many_digits && am.power2 >= 0) {
			if (am != compute_float<binary_format<T>>(exponent, mantissa + 1)) {
				am = compute_error<binary_format<T>>(exponent, mantissa);
			}
		}
		if (am.power2 < 0) {
			am = digit_comp<T>(integer, fraction, am, mantissa, exponent);
		}
		to_float(negative, am, value);
		if ((mantissa != 0 && am.mantissa == 0 && am.power2 == 0) || am.power2 == binary_format<T>::infinite_power()) {
			return false;
		}
		return true;
	}
}