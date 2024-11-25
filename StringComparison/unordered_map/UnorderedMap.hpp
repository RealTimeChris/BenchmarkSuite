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
/// unordered_map.hpp - Header file for the unordered_map class.
/// May 12, 2021
/// https://discordcoreapi.com
/// \file unordered_map.hpp
#pragma once

#include "HashTable.hpp"

#include <memory_resource>
#include <shared_mutex>
#include <exception>
#include <optional>
#include <utility>
#include <vector>
#include <mutex>

namespace jsonifier {

	template<typename key_type_new, typename mapped_type_new, typename hasher, typename key_equals, typename allocator> class unordered_map
		: protected hash_table<key_type_new, mapped_type_new, hasher, key_equals, allocator> {
	  public:
		using key_type		   = key_type_new;
		using value_type	   = std::pair<key_type_new, mapped_type_new>;
		using allocator_type   = jsonifier_internal::alloc_wrapper<value_type>;
		using allocator_traits = std::allocator_traits<allocator_type>;
		using size_type		   = uint64_t;
		using difference_type  = int64_t;
		using pointer		   = typename allocator_traits::pointer;
		using const_pointer	   = typename allocator_traits::const_pointer;
		using mapped_type	   = mapped_type_new;
		using reference		   = value_type&;
		using const_reference  = const value_type&;
		using iterator		   = hash_map_iterator<hash_table<key_type_new, mapped_type_new, hasher, key_equals, allocator>> ;
		using const_iterator   = hash_map_iterator<const hash_table<key_type_new, mapped_type_new, hasher, key_equals, allocator>>;
		using key_compare	   = key_equals;
		using hash_table_new   = hash_table<key_type_new, mapped_type_new, hasher, key_equals, allocator>;

		friend hash_table<key_type_new, mapped_type_new, hasher, key_equals, allocator>;
		friend iterator;
		friend const_iterator;

		JSONIFIER_ALWAYS_INLINE unordered_map() {};

		JSONIFIER_ALWAYS_INLINE unordered_map& operator=(unordered_map&& other) noexcept {
			if (this != &other) {
				hash_table_new::reset();
				swap(other);
			}
			return *this;
		}

		JSONIFIER_ALWAYS_INLINE unordered_map(unordered_map&& other) noexcept {
			*this = std::move(other);
		}

		JSONIFIER_ALWAYS_INLINE unordered_map& operator=(const unordered_map& other) {
			if (this != &other) {
				std::cout << "WERTE BEING COPIED!" << std::endl;
				hash_table_new::reset();
				hash_table_new::reserve(other.capacity());
				for (const auto& [key, value]: other) {
					emplace(key, value);
				}
			}
			return *this;
		}

		JSONIFIER_ALWAYS_INLINE unordered_map(const unordered_map& other) {
			*this = other;
		}

		JSONIFIER_ALWAYS_INLINE unordered_map(std::initializer_list<value_type> list) {
			hash_table_new::reserve(list.size());
			for (auto& value: list) {
				emplace(std::move(value.first), std::move(value.second));
			}
		};

		template<typename... Args> JSONIFIER_ALWAYS_INLINE iterator emplace(Args&&... value) {
			return hash_table_new::emplaceInternal(std::forward<Args>(value)...);
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE const_iterator find(key_type_newer&& key) const {
			return hash_table_new::find(std::forward<key_type_newer>(key));
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE iterator find(key_type_newer&& key) {
			return hash_table_new::find(std::forward<key_type_newer>(key));
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE const mapped_type& operator[](key_type_newer&& key) const {
			return hash_table_new::emplaceInternal(std::forward<key_type_newer>(key))->second;
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE mapped_type& operator[](key_type_newer&& key) {
			return hash_table_new::emplaceInternal(std::forward<key_type_newer>(key))->second;
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE const mapped_type& at(key_type_newer&& key) const {
			return hash_table_new::at(std::forward<key_type_newer>(key));
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE mapped_type& at(key_type_newer&& key) {
			return hash_table_new::at(std::forward<key_type_newer>(key));
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE bool contains(key_type_newer&& key) const {
			return hash_table_new::contains(std::forward<key_type_newer>(key));
		}

		template<map_container_iterator_t<key_type, mapped_type> map_iterator> JSONIFIER_ALWAYS_INLINE iterator erase(map_iterator&& iter) {
			return hash_table_new::erase(std::forward<map_iterator>(iter));
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE iterator erase(key_type_newer&& key) {
			return hash_table_new::erase(std::forward<key_type_newer>(key));
		}

		JSONIFIER_ALWAYS_INLINE const_iterator begin() const {
			return hash_table_new::begin();
		}

		JSONIFIER_ALWAYS_INLINE const_iterator end() const {
			return hash_table_new::end();
		}

		JSONIFIER_ALWAYS_INLINE iterator begin() {
			return hash_table_new::begin();
		}

		JSONIFIER_ALWAYS_INLINE iterator end() {
			return hash_table_new::end();
		}

		JSONIFIER_ALWAYS_INLINE bool full() const {
			return static_cast<float>(hash_table_new::sizeVal) >= static_cast<float>(hash_table_new::capacityVal) * 0.90f;
		}

		JSONIFIER_ALWAYS_INLINE size_type size() const {
			return hash_table_new::sizeVal;
		}

		JSONIFIER_ALWAYS_INLINE bool empty() const {
			return hash_table_new::sizeVal == 0;
		}

		JSONIFIER_ALWAYS_INLINE void swap(unordered_map& other) {
			std::swap(hash_table_new::maxLookAheadDistance, other.hash_table_new::maxLookAheadDistance);
			std::swap(hash_table_new::sentinelVector, other.hash_table_new::sentinelVector);
			std::swap(hash_table_new::capacityVal, other.hash_table_new::capacityVal);
			std::swap(hash_table_new::sizeVal, other.hash_table_new::sizeVal);
			std::swap(hash_table_new::data, other.hash_table_new::data);
		}

		JSONIFIER_ALWAYS_INLINE size_type capacity() const {
			return hash_table_new::capacityVal;
		}

		JSONIFIER_ALWAYS_INLINE bool operator==(const unordered_map& other) const {
			if (hash_table_new::capacityVal != other.hash_table_new::capacityVal || hash_table_new::sizeVal != other.hash_table_new::sizeVal ||
				hash_table_new::data != other.hash_table_new::data) {
				return false;
			}
			for (auto iter01{ begin() }, iter02{ other.begin() }; iter01 != end(); ++iter01, ++iter02) {
				if (!object_compare()(iter01.operator*().second, iter02.operator*().second) || !key_compare()(iter01.operator*().first, iter02.operator*().first)) {
					return false;
				}
			}
			return true;
		}

		JSONIFIER_ALWAYS_INLINE void clear() {
			hash_table_new::clear();
		}

		JSONIFIER_ALWAYS_INLINE ~unordered_map() {
			hash_table_new::reset();
		};
	};

}