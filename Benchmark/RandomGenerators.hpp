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

#include <cstddef>
#include <type_traits>
#include <utility>

namespace bnch_swt {

	inline static std::random_device randomEngine{};
	inline static std::mt19937_64 gen{ randomEngine() };
	static constexpr std::string_view charset{ "!#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~\"\\\r\b\f\t\n" };

	struct test_generator {

		template<typename value_type01, typename value_type02> static value_type01 randomizeNumberUniform(value_type01 start, value_type02 end) {
			std::uniform_real_distribution<value_type01> dis{ start, static_cast<value_type01>(end) };
			return dis(gen);
		}

		template<jsonifier::concepts::integer_t value_type01, jsonifier::concepts::integer_t value_type02>
		static value_type01 randomizeNumberUniform(value_type01 start, value_type02 end) {
			std::uniform_int_distribution<value_type01> dis{ start, static_cast<value_type01>(end) };
			return dis(gen);
		}

		static std::string generateString(size_t length) {
			constexpr size_t charsetSize = charset.size();
			std::string result{};
			for (size_t x = 0; x < length; ++x) {
				result += charset[randomizeNumberUniform(0ull, charsetSize - 1)];
			}
			return result;
		}

		static double generateDouble() {
			static constexpr double min = std::numeric_limits<double>::min();
			static constexpr double max = std::numeric_limits<double>::max();
			std::uniform_real_distribution<double> dis(log(min), log(max));
			double logValue = dis(gen);
			bool negative{ generateBool() };
			return negative ? -std::exp(logValue) : std::exp(logValue);
		}

		static bool generateBool() {
			return static_cast<bool>(randomizeNumberUniform(0, 100) >= 50);
		};

		static size_t generateUint() {
			return randomizeNumberUniform(std::numeric_limits<size_t>::min(), std::numeric_limits<size_t>::max());
		};

		static int64_t generateInt() {
			return randomizeNumberUniform(std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max());
		};

		test_generator() {
			auto fill = [&](auto& v) {
				auto arraySize01 = randomizeNumberUniform(15ull, 25ull);
				v.resize(arraySize01);
				for (size_t x = 0; x < arraySize01; ++x) {
					auto arraySize02 = randomizeNumberUniform(15ull, 35ull);
					auto arraySize03 = randomizeNumberUniform(0ull, arraySize02);
					for (size_t y = 0; y < arraySize03; ++y) {
						auto newString = generateString(y);
						v[x].testVals01.emplace_back(newString);
					}
					arraySize03 = randomizeNumberUniform(0ull, arraySize02);
					for (size_t y = 0; y < arraySize03; ++y) {
						v[x].testVals02.emplace_back(generateUint());
					}
					arraySize03 = randomizeNumberUniform(0ull, arraySize02);
					for (size_t y = 0; y < arraySize03; ++y) {
						v[x].testVals03.emplace_back(generateInt());
					}
					arraySize03 = randomizeNumberUniform(0ull, arraySize02);
					for (size_t y = 0; y < arraySize03; ++y) {
						auto newBool = generateBool();
						v[x].testVals05.emplace_back(newBool);
					}
					arraySize03 = randomizeNumberUniform(0ull, arraySize02);
					for (size_t y = 0; y < arraySize03; ++y) {
						v[x].testVals04.emplace_back(generateDouble());
					}
				}
			};
		}
	};
}
