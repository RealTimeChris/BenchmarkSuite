/*
	MIT License

	DiscordCoreAPI, A bot library for Discord, written in C++, and featuring explicit multithreading through the usage of custom, asynchronous C++ CoRoutines.

	Copyright 2022, 2023 Chris M. (RealTimeChris)

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
*/
/// Hash.hpp - Header file for the hash related stuff.
/// May 12, 2021
/// https://github.com/realtimechris/jsonifier
/// \file Hash.hpp
#pragma once

#include <memory_resource>
#include <exception>
#include <utility>
#include <vector>

namespace jsonifier {

	struct object_compare {
		template<typename value_type01, typename value_type02> BNCH_SWT_INLINE bool operator()(const value_type01& lhs, const value_type02& rhs) {
			return lhs == static_cast<value_type01>(rhs);
		}

		template<typename value_type01> BNCH_SWT_INLINE bool operator()(const value_type01& lhs, const value_type01& rhs) {
			return lhs == rhs;
		}
	};

	BNCH_SWT_INLINE uint64_t internalHashFunction(const void* value, uint64_t count) {
		static constexpr uint64_t fnvOffsetBasis{ 0xcbf29ce484222325 };
		static constexpr uint64_t fnvPrime{ 0x00000100000001B3 };
		uint64_t hash{ fnvOffsetBasis };
		for (uint64_t x = 0; x < count; ++x) {
			hash ^= static_cast<const uint8_t*>(value)[x];
			hash *= fnvPrime;
		}
		return hash;
	}

	template<typename value_type> struct key_hasher;

	template<> struct key_hasher<const char*> {
		BNCH_SWT_INLINE static uint64_t getHashKey(const char* other) {
			return internalHashFunction(other, std::char_traits<char>::length(other));
		}
	};

	template<uint64_t size> struct key_hasher<char[size]> {
		BNCH_SWT_INLINE static uint64_t getHashKey(const char (&other)[size]) {
			return internalHashFunction(other, std::char_traits<char>::length(other));
		}
	};

	template<jsonifier::concepts::integer_t value_type> struct key_hasher<value_type> {
		BNCH_SWT_INLINE static uint64_t getHashKey(const value_type& other) {
			return internalHashFunction(&other, sizeof(other));
		}
	};

	template<jsonifier::concepts::enum_t value_type> struct key_hasher<value_type> {
		BNCH_SWT_INLINE static uint64_t getHashKey(const value_type& other) {
			return internalHashFunction(&other, sizeof(other));
		}
	};

	template<jsonifier::concepts::string_t value_type> struct key_hasher<value_type> {
		BNCH_SWT_INLINE static uint64_t getHashKey(const value_type& other) {
			return internalHashFunction(other.data(), other.size());
		}
	};

	template<> struct key_hasher<jsonifier::vector<jsonifier::string>> {
		BNCH_SWT_INLINE static uint64_t getHashKey(const jsonifier::vector<jsonifier::string>& data) {
			jsonifier::string newString{};
			for (auto& value: data) {
				newString.append(value);
			}
			return internalHashFunction(newString.data(), newString.size());
		}
	};

	template<typename value_Type> struct key_accessor;

	template<> struct key_accessor<const char*> {
		BNCH_SWT_INLINE static uint64_t getHashKey(const char* other) {
			return key_hasher<const char*>::getHashKey(other);
		}
	};

	template<uint64_t size> struct key_accessor<char[size]> {
		BNCH_SWT_INLINE static uint64_t getHashKey(const char (&other)[size]) {
			return key_hasher<char[size]>::getHashKey(other);
		}
	};

	template<jsonifier::concepts::string_t value_type> struct key_accessor<value_type> {
		BNCH_SWT_INLINE static uint64_t getHashKey(const value_type& other) {
			return key_hasher<value_type>::getHashKey(other);
		}
	};

	template<> struct key_accessor<jsonifier::vector<jsonifier::string>> {
		BNCH_SWT_INLINE static uint64_t getHashKey(const jsonifier::vector<jsonifier::string>& other) {
			return key_hasher<jsonifier::vector<jsonifier::string>>::getHashKey(other);
		}
	};

	template<typename value_type> struct hash_policy {
		template<typename key_type> BNCH_SWT_INLINE uint64_t indexForHash(key_type&& key) const {
			return key_hasher<std::remove_cvref_t<key_type>>::getHashKey(key) & (static_cast<const value_type*>(this)->capacityVal - 1);
		}

		BNCH_SWT_INLINE static int8_t log2(uint64_t value) {
			static constexpr int8_t table[64] = { 63, 0, 58, 1, 59, 47, 53, 2, 60, 39, 48, 27, 54, 33, 42, 3, 61, 51, 37, 40, 49, 18, 28, 20, 55, 30, 34, 11, 43, 14, 22, 4, 62, 57,
				46, 52, 38, 26, 32, 41, 50, 36, 17, 19, 29, 10, 13, 21, 56, 45, 25, 31, 35, 16, 9, 12, 44, 24, 15, 8, 23, 7, 6, 5 };
			value |= value >> 1;
			value |= value >> 2;
			value |= value >> 4;
			value |= value >> 8;
			value |= value >> 16;
			value |= value >> 32;
			return table[((value - (value >> 1)) * 0x07EDD5E59A4E28C2) >> 58];
		}

		BNCH_SWT_INLINE static uint64_t nextPowerOfTwo(uint64_t size) {
			--size;
			size |= size >> 1;
			size |= size >> 2;
			size |= size >> 4;
			size |= size >> 8;
			size |= size >> 16;
			size |= size >> 32;
			++size;
			return size;
		}

		static int8_t computeMaxLookAheadDistance(uint64_t num_buckets) {
			return log2(num_buckets);
		}
	};

	template<typename value_type_internal_new> class hash_iterator {
	  public:
		using iterator_category	  = std::forward_iterator_tag;
		using value_type_internal = value_type_internal_new;
		using value_type		  = typename value_type_internal::value_type;
		using reference			  = value_type&;
		using pointer			  = value_type*;
		using pointer_internal	  = value_type_internal*;
		using size_type			  = uint64_t;

		BNCH_SWT_INLINE hash_iterator() = default;

		BNCH_SWT_INLINE hash_iterator(pointer_internal valueNew, size_type currentIndexNew) : value{ valueNew }, currentIndex{ currentIndexNew } {};

		BNCH_SWT_INLINE hash_iterator& operator++() {
			skipEmptySlots();
			return *this;
		}

		BNCH_SWT_INLINE hash_iterator& operator--() {
			skipEmptySlotsRev();
			return *this;
		}

		BNCH_SWT_INLINE hash_iterator& operator-(size_type amountToReverse) {
			for (size_type x = 0; x < amountToReverse; ++x) {
				skipEmptySlotsRev();
			}
			return *this;
		}

		BNCH_SWT_INLINE hash_iterator& operator+(size_type amountToAdd) {
			for (size_type x = 0; x < amountToAdd; ++x) {
				skipEmptySlots();
			}
			return *this;
		}

		BNCH_SWT_INLINE pointer getRawPtr() {
			return &value->data[currentIndex];
		}

		BNCH_SWT_INLINE bool operator==(const hash_iterator&) const {
			return !value || value->sentinelVector[currentIndex] == -1;
		}

		BNCH_SWT_INLINE pointer operator->() {
			return &value->data[currentIndex];
		}

		BNCH_SWT_INLINE reference operator*() {
			return value->data[currentIndex];
		}

	  protected:
		pointer_internal value{};
		size_type currentIndex{};

		void skipEmptySlots() {
			if (currentIndex < value->sentinelVector.size()) {
				++currentIndex;
				while (value && value->sentinelVector[currentIndex] == 0 && currentIndex < value->sentinelVector.size()) {
					++currentIndex;
				};
			}
		}

		void skipEmptySlotsRev() {
			if (static_cast<int64_t>(currentIndex) > 0) {
				--currentIndex;
				while (value && value->sentinelVector[currentIndex] == 0 && static_cast<int64_t>(currentIndex) > 0) {
					--currentIndex;
				};
			}
		}
	};	
}
