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
	JSONIFIER_ALWAYS_INLINE_VARIABLE char smallE{ 'e' };
	JSONIFIER_ALWAYS_INLINE_VARIABLE char minus{ '-' };
	JSONIFIER_ALWAYS_INLINE_VARIABLE char bigE{ 'E' };
	JSONIFIER_ALWAYS_INLINE_VARIABLE char plus{ '+' };
	JSONIFIER_ALWAYS_INLINE_VARIABLE char zero{ '0' };
	JSONIFIER_ALWAYS_INLINE_VARIABLE char nine{ '9' };

#define JSONIFIER_IS_DIGIT(x) ((static_cast<size_t>(x - '0')) <= 9)

	template<typename value_type, typename char_t> JSONIFIER_ALWAYS_INLINE bool parseFloat(value_type& value, char_t const*& iter, char_t const* end = nullptr) noexcept {
		using namespace fast_float_new;

		from_chars_result_t<char_t> answer;
		parsed_number_string_t<char_t> pns;
		pns.valid			= false;
		pns.too_many_digits = false;
		pns.negative		= (*iter == char_t('-'));
		++iter;
		if (!is_integer(*iter)) {// a sign must be followed by an integer
			return false;
		}
		
		char_t const* const start_digits = iter;

		uint64_t i = 0;// an unsigned int avoids signed overflows (which are bad)

		while ((iter != end) && is_integer(*iter)) {
			// a multiplication by 10 is cheaper than an arbitrary integer
			// multiplication
			i = 10 * i + uint64_t(*iter - char_t('0'));// might overflow, we will handle the overflow later
			++iter;
		}
		char_t const* const end_of_integer_part = iter;
		int64_t digit_count					= int64_t(end_of_integer_part - start_digits);
		pns.integer							= span<const char_t>(start_digits, size_t(digit_count));
		if (digit_count == 0) {
			return false;
		}
		if ((start_digits[0] == char_t('0') && digit_count > 1)) {
			return false;
		}

		int64_t exponent			 = 0;
		const bool has_decimal_point = (iter != end) && (*iter == decimal);
		if (has_decimal_point) {
			++iter;
			char_t const* before = iter;
			// can occur at most twice without overflowing, but let it occur more, since
			// for integers with many digits, digit parsing is the primary bottleneck.
			loop_parse_if_eight_digits(iter, end, i);

			while ((iter != end) && is_integer(*iter)) {
				uint8_t digit = uint8_t(*iter - char_t('0'));
				++iter;
				i = i * 10 + digit;// in rare cases, this will overflow, but that's ok
			}
			exponent	 = before - iter;
			pns.fraction = span<const char_t>(before, size_t(iter - before));
			digit_count -= exponent;
		}
		if (has_decimal_point && exponent == 0) {
			return false;
		}
		int64_t exp_number = 0;// explicit exponential part
		if ((iter != end) && ((char_t('+') == *iter) || (char_t('-') == *iter) || (char_t('d') == *iter) || (char_t('D') == *iter))) {
			char_t const* location_of_e = iter;
			if ((char_t('e') == *iter) || (char_t('E') == *iter) || (char_t('d') == *iter) || (char_t('D') == *iter)) {
				++iter;
			}
			bool neg_exp = false;
			if ((iter != end) && (char_t('-') == *iter)) {
				neg_exp = true;
				++iter;
			} else if ((iter != end) && (char_t('+') == *iter)) {// '+' on exponent is allowed by C++17 20.19.3.(7.1)
				++iter;
			}
			if ((iter == end) || !is_integer(*iter)) {
				iter = location_of_e;
			} else {
				while ((iter != end) && is_integer(*iter)) {
					uint8_t digit = uint8_t(*iter - char_t('0'));
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
		pns.lastmatch = iter;
		pns.valid	  = true;

		// If we frequently had to deal with long strings of digits,
		// we could extend our code by using a 128-bit integer instead
		// of a 64-bit integer. However, this is uncommon.
		//
		// We can deal with up to 19 digits.
		if (digit_count > 19) {// this is uncommon
			// It is possible that the integer had an overflow.
			// We have to handle the case where we have 0.0000somenumber.
			// We need to be mindful of the case where we only have zeroes...
			// E.g., 0.000000000...000.
			char_t const* start = start_digits;
			while ((start != end) && (*start == char_t('0') || *start == decimal)) {
				if (*start == char_t('0')) {
					digit_count--;
				}
				start++;
			}

			if (digit_count > 19) {
				pns.too_many_digits = true;
				// Let us start again, this time, avoiding overflows.
				// We don't need to check if is_integer, since we use the
				// pre-tokenized spans from above.
				i				  = 0;
				iter				  = pns.integer.ptr;
				char_t const* int_end = iter + pns.integer.len();
				const uint64_t minimal_nineteen_digit_integer{ 1000000000000000000 };
				while ((i < minimal_nineteen_digit_integer) && (iter != int_end)) {
					i = i * 10 + uint64_t(*iter - char_t('0'));
					++iter;
				}
				if (i >= minimal_nineteen_digit_integer) {// We have a big integers
					exponent = end_of_integer_part - iter + exp_number;
				} else {// We have a value with a fractional component.
					iter				   = pns.fraction.ptr;
					char_t const* frac_end = iter + pns.fraction.len();
					while ((i < minimal_nineteen_digit_integer) && (iter != frac_end)) {
						i = i * 10 + uint64_t(*iter - char_t('0'));
						++iter;
					}
					exponent = pns.fraction.ptr - iter + exp_number;
				}
				// We have now corrected both exponent and i, to a truncated value
			}
		}
		pns.exponent = exponent;
		pns.mantissa = i;
		if (!pns.valid) { 
			answer.ec  = std::errc::invalid_argument;
			answer.ptr = iter;
			return false;
		}

		// call overload that takes parsed_number_string_t directly.
		return from_chars_advanced(pns, value).ptr;
	}
}