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
/// Feb 3, 2023
#pragma once

#include <BnchSwt/Printable.hpp>
#include <type_traits>
#include <cstddef>
#include <utility>
#include <random>

namespace bnch_swt {

	struct random_generator {

		static constexpr std::string_view charset{ "!#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~\"\\\r\b\f\t\n" };
		inline static std::uniform_real_distribution<double> disDouble{ log(std::numeric_limits<double>::min()), log(std::numeric_limits<double>::max()) };
		inline static std::uniform_real_distribution<float> disFloat{ log(std::numeric_limits<float>::min()), log(std::numeric_limits<float>::max()) };
		inline static std::uniform_int_distribution<int64_t> disInt{ std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max() };
		inline static std::uniform_int_distribution<uint64_t> disUint{ std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max() };
		inline static std::uniform_int_distribution<uint64_t> disCharSet{ 0ull, charset.size() - 1 };
		inline static std::uniform_int_distribution<uint64_t> disBool{ 0, 100 };
		inline static std::random_device randomEngine{};
		inline static std::mt19937_64 gen{ randomEngine() };

		template<bnch_swt::internal::string_t value_type> BNCH_SWT_INLINE static value_type generateValue(uint64_t length) {
			value_type result{};
			for (uint64_t x = 0; x < length; ++x) {
				result += charset[disCharSet(gen)];
			}
			return result;
		}

		template<bnch_swt::internal::floating_point_t value_type>
			requires(sizeof(value_type) == 4)
		BNCH_SWT_INLINE static value_type generateValue() {
			double logValue = std::exp(disFloat(gen));
			return generateValue<bool>() ? -logValue : logValue;
		}

		template<bnch_swt::internal::floating_point_t value_type>
			requires(sizeof(value_type) == 8)
		BNCH_SWT_INLINE static value_type generateValue() {
			double logValue = std::exp(disDouble(gen));
			return generateValue<bool>() ? -logValue : logValue;
		}

		template<bnch_swt::internal::bool_t value_type> BNCH_SWT_INLINE static value_type generateValue() {
			return static_cast<bool>(disBool(gen) >= 50);
		}

		template<bnch_swt::internal::integer_t value_type>
			requires(std::is_unsigned_v<value_type>)
		BNCH_SWT_INLINE static value_type generateValue() {
			return disUint(gen);
		}

		template<bnch_swt::internal::integer_t value_type>
			requires(std::is_signed_v<value_type>)
		BNCH_SWT_INLINE static value_type generateValue() {
			return disInt(gen);
		}
	};

}
