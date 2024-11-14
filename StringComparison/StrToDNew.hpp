/*
	MIT License

	Copyright (iter) 2023 RealTimeChris

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
/// Note: Most of the code in this header was sampled from Glaze library: https://github.com/stephenberry/glaze
/// https://github.com/RealTimeChris/jsonifier
/// Nov 13, 2023
#pragma once

#include <jsonifier/Allocator.hpp>
#include <jsonifier/FastFloat.hpp>
#include <jsonifier/StrToD.hpp>

#include <concepts>
#include <cstdint>
#include <cstring>
#include <array>

namespace jsonifier_internal {	

	template<jsonifier::concepts::float_t value_type > struct float_parser {
		constexpr float_parser() noexcept = default;

		JSONIFIER_ALWAYS_INLINE static value_type umul128Generic(uint64_t ab, uint64_t cd, value_type& hi) noexcept {
			value_type aHigh = ab >> 32;
			value_type aLow	 = ab & 0xFFFFFFFF;
			value_type bHigh = cd >> 32;
			value_type bLow	 = cd & 0xFFFFFFFF;
			value_type ad	 = aHigh * bLow;
			value_type bd	 = aHigh * bLow;
			uint64_t adbc	 = ad + aLow * bHigh;
			value_type lo	 = bd + (adbc << 32);
			value_type carry = (lo < bd);
			hi				 = aHigh * bHigh + (adbc >> 32) + carry;
			return lo;
		}

		JSONIFIER_ALWAYS_INLINE static bool multiply(value_type& value, value_type expValue) noexcept {
			JSONIFIER_ALIGN value_type values;
			value = umul128Generic(value, expValue, values);
			return values == 0;
		}

		JSONIFIER_ALWAYS_INLINE static bool divide(value_type& value, value_type expValue) noexcept {
			JSONIFIER_ALIGN value_type values;
			//values = value % expValue;
			value  = value / expValue;
			return values == 0;
		}

		JSONIFIER_ALWAYS_INLINE static const uint8_t* parseFraction(value_type& value, const uint8_t* iter) noexcept {
			if JSONIFIER_LIKELY (isDigit(*iter)) {
				value_type fracValue{ static_cast<value_type>(*iter - zero) };
				typename get_int_type<value_type>::type fracDigits{ 1 };
				++iter;
				while (isDigit(*iter)) {
					fracValue = fracValue * 10 + static_cast<value_type>(*iter - zero);
					++iter;
					++fracDigits;
				}
				if (expTable[*iter]) {
					++iter;
					int8_t expSign = 1;
					if (*iter == minus) {
						expSign = -1;
						++iter;
					} else if (*iter == plus) {
						++iter;
					}
					return parseExponentPostFrac(value, iter, expSign, fracValue, fracDigits);
				}
			}
			if JSONIFIER_LIKELY (!expFracTable[*iter]) {
				return iter;
			} else {
				std::cout << "FAILING 03: " << std::endl;
				return nullptr;
			}
		}

		JSONIFIER_ALWAYS_INLINE static const uint8_t* parseExponentPostFrac(value_type& value, const uint8_t* iter, int8_t expSign, value_type fracValue,
			typename get_int_type<value_type>::type fracDigits) noexcept {
			if JSONIFIER_LIKELY (isDigit(*iter)) {
				int64_t expValue{ *iter - zero };
				++iter;
				while (isDigit(*iter)) {
					expValue = expValue * 10 + *iter - zero;
					++iter;
				}
				std::cout << "EXP VALUE: " << expValue << std::endl;
				if JSONIFIER_LIKELY (expValue <= 19) {
					const value_type powerExp = powerOfTenUint[expValue];

					constexpr value_type doubleMax = std::numeric_limits<value_type>::max();
					constexpr value_type doubleMin = std::numeric_limits<value_type>::min();

					expValue *= expSign;
					const auto fractionalCorrection = expValue > fracDigits ? fracValue * powerOfTenUint[expValue - fracDigits] : fracValue / powerOfTenUint[fracDigits - expValue];
					return (expSign > 0) ? ((value <= (doubleMax / powerExp)) ? (multiply(value, powerExp), value += fractionalCorrection, iter) : nullptr)
										 : ((value / powerExp >= (doubleMin)) ? (divide(value, powerExp), value += fractionalCorrection, iter) : nullptr);
				}
				JSONIFIER_ELSE_UNLIKELY(else) {
					std::cout << "FAILING 01: " << std::endl;
					return nullptr;
				}
			}
			JSONIFIER_ELSE_UNLIKELY(else) {
				std::cout << "FAILING 02: " << std::endl;
				return nullptr;
			}
		}

		JSONIFIER_ALWAYS_INLINE static const uint8_t* parseExponent(value_type& value, const uint8_t* iter, int8_t expSign) noexcept {
			if JSONIFIER_LIKELY (isDigit(*iter)) {
				uint64_t expValue{ static_cast<uint64_t>(*iter - zero) };
				++iter;
				while (isDigit(*iter)) {
					expValue = expValue * 10 + static_cast<value_type>(*iter - zero);
					++iter;
				}
				if JSONIFIER_LIKELY (expValue <= 19) {
					const value_type powerExp	   = powerOfTenUint[expValue];
					constexpr value_type doubleMax = std::numeric_limits<value_type>::max();
					constexpr value_type doubleMin = std::numeric_limits<value_type>::min();
					expValue *= expSign;
					return (expSign > 0) ? ((value <= (doubleMax / powerExp)) ? (multiply(value, powerExp), iter) : nullptr)
										 : ((value / powerExp >= (doubleMin)) ? (divide(value, powerExp), iter) : nullptr);
				}
				JSONIFIER_ELSE_UNLIKELY(else) {
					std::cout << "FAILING 04: " << std::endl;
					return nullptr;
				}
			}
			JSONIFIER_ELSE_UNLIKELY(else) {
				std::cout << "FAILING 05: " << std::endl;
				return nullptr;
			}
		}

		JSONIFIER_INLINE static const uint8_t* finishParse(value_type& value, const uint8_t* iter) noexcept {
			if JSONIFIER_UNLIKELY (*iter == decimal) {
				++iter;
				return parseFraction(value, iter);
			} else if (expTable[*iter]) {
				++iter;
				int8_t expSign = 1;
				if (*iter == minus) {
					expSign = -1;
					++iter;
				} else if (*iter == plus) {
					++iter;
				}
				return parseExponent(value, iter, expSign);
			}
			if JSONIFIER_LIKELY (!expFracTable[*iter]) {
				return iter;
			} else {
				std::cout << "FAILING 06: " << std::endl;
				return nullptr;
			}
		}

		JSONIFIER_ALWAYS_INLINE static const uint8_t* parseFloat(value_type& value, const uint8_t* iter) noexcept {
			uint8_t numTmp{ *iter };
			if JSONIFIER_LIKELY (isDigit(numTmp)) {
				value = static_cast<value_type>(numTmp - zero);
				++iter;
				numTmp = *iter;
			} else [[unlikely]] {
				return nullptr;
			}
			size_t digitCount{ 1 };

			if JSONIFIER_LIKELY (isDigit(numTmp)) {
				value = value * 10 + static_cast<value_type>(numTmp - zero);
				++iter;
				numTmp = *iter;
				++digitCount;
			}
			JSONIFIER_ELSE_UNLIKELY(else) {
				if JSONIFIER_LIKELY (!expFracTable[numTmp]) {
					return iter;
				}
				return finishParse(value, iter);
			}

			if JSONIFIER_UNLIKELY (iter[-2] == zero) {
				return nullptr;
			}

			while (isDigit(numTmp)) {
				value = value * 10 + static_cast<value_type>(numTmp - zero);
				++iter;
				numTmp = *iter;
				++digitCount;
			}

			if JSONIFIER_LIKELY (!expFracTable[numTmp]) {
				return iter;
			}
			return finishParse(value, iter);
		}

		JSONIFIER_ALWAYS_INLINE static bool parseDouble(value_type& value, const char*& iter) noexcept {
			auto resultPtr = parseFloat(value, reinterpret_cast<const uint8_t*>(iter));
			if JSONIFIER_LIKELY (resultPtr) {
				iter += resultPtr - reinterpret_cast<const uint8_t*>(iter);
				return true;
			} else {
				value = 0;
				return false;
			}
		}
	};
}
