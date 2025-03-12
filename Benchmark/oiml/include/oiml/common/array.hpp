#pragma once

#include <oiml/common/iterator.hpp>
#include <algorithm>
#include <stdexcept>

namespace oiml {

	template<typename value_type_new, size_t size_new> struct oiml_array {
	  public:
		static_assert(std::is_object_v<value_type_new>, "The C++ Standard forbids containers of non-object types because of [container.requirements].");
		static constexpr size_t size_val{ size_new };
		using value_type			 = value_type_new;
		using size_type				 = size_t;
		using difference_type		 = ptrdiff_t;
		using pointer				 = value_type*;
		using const_pointer			 = const value_type*;
		using reference				 = value_type&;
		using const_reference		 = const value_type&;
		using iterator				 = oiml_array_iterator<value_type, size_new>;
		using const_iterator		 = const oiml_array_iterator<value_type, size_new>;
		using reverse_iterator		 = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		OIML_FORCE_INLINE constexpr oiml_array(const std::initializer_list<value_type>& values) {
			std::copy_n(values.begin(), values.size(), data_val);
		}

		OIML_FORCE_INLINE constexpr oiml_array(const value_type (&values)[size_new]) {
			std::copy_n(&values, size_new, data_val);
		}

		OIML_FORCE_INLINE constexpr void fill(const value_type& _Value) {
			std::fill_n(data_val, size_new, _Value);
		}

		OIML_FORCE_INLINE constexpr iterator begin() noexcept {
			return iterator(data_val);
		}

		OIML_FORCE_INLINE constexpr const_iterator begin() const noexcept {
			return const_iterator(data_val);
		}

		OIML_FORCE_INLINE constexpr iterator end() noexcept {
			return iterator(data_val);
		}

		OIML_FORCE_INLINE constexpr const_iterator end() const noexcept {
			return const_iterator(data_val);
		}

		OIML_FORCE_INLINE constexpr reverse_iterator rbegin() noexcept {
			return reverse_iterator(end());
		}

		OIML_FORCE_INLINE constexpr const_reverse_iterator rbegin() const noexcept {
			return const_reverse_iterator(end());
		}

		OIML_FORCE_INLINE constexpr reverse_iterator rend() noexcept {
			return reverse_iterator(begin());
		}

		OIML_FORCE_INLINE constexpr const_reverse_iterator rend() const noexcept {
			return const_reverse_iterator(begin());
		}

		OIML_FORCE_INLINE constexpr const_iterator cbegin() const noexcept {
			return begin();
		}

		OIML_FORCE_INLINE constexpr const_iterator cend() const noexcept {
			return end();
		}

		OIML_FORCE_INLINE constexpr const_reverse_iterator crbegin() const noexcept {
			return rbegin();
		}

		OIML_FORCE_INLINE constexpr const_reverse_iterator crend() const noexcept {
			return rend();
		}

		OIML_FORCE_INLINE constexpr size_type size() const noexcept {
			return size_new;
		}

		OIML_FORCE_INLINE constexpr size_type max_size() const noexcept {
			return size_new;
		}

		OIML_FORCE_INLINE constexpr bool empty() const noexcept {
			return false;
		}

		OIML_FORCE_INLINE constexpr reference at(size_type position) {
			if (size_new <= position) {
				std::runtime_error{ "invalid oiml_array<T, N> subscript" };
			}

			return data_val[position];
		}

		OIML_FORCE_INLINE constexpr const_reference at(size_type position) const {
			if (size_new <= position) {
				std::runtime_error{ "invalid oiml_array<T, N> subscript" };
			}

			return data_val[position];
		}

		OIML_FORCE_INLINE constexpr reference operator[](size_type position) noexcept {
			return data_val[position];
		}

		OIML_FORCE_INLINE constexpr const_reference operator[](size_type position) const noexcept {
			return data_val[position];
		}

		OIML_FORCE_INLINE constexpr reference front() noexcept {
			return data_val[0];
		}

		OIML_FORCE_INLINE constexpr const_reference front() const noexcept {
			return data_val[0];
		}

		OIML_FORCE_INLINE constexpr reference back() noexcept {
			return data_val[size_new - 1];
		}

		OIML_FORCE_INLINE constexpr const_reference back() const noexcept {
			return data_val[size_new - 1];
		}

		OIML_FORCE_INLINE constexpr value_type* data() noexcept {
			return data_val;
		}

		OIML_FORCE_INLINE constexpr const value_type* data() const noexcept {
			return data_val;
		}

		OIML_FORCE_INLINE constexpr friend bool operator==(const oiml_array& lhs, const oiml_array& rhs) {
			if (lhs.size_val == rhs.size_val) {
				for (size_t x = 0; x < size_val; ++x) {
					if (lhs[x] != rhs[x]) {
						return false;
					}
				}
				return true;
			} else {
				return false;
			}
		}

		value_type data_val[size_val]{};
	};

	struct empty_array_element {};

	template<class value_type_new> class oiml_array<value_type_new, 0> {
	  public:
		static_assert(std::is_object_v<value_type_new>, "The C++ Standard forbids containers of non-object types because of [container.requirements].");

		using value_type			 = value_type_new;
		using size_type				 = size_t;
		using difference_type		 = ptrdiff_t;
		using pointer				 = value_type*;
		using const_pointer			 = const value_type*;
		using reference				 = value_type&;
		using const_reference		 = const value_type&;
		using iterator				 = oiml_array_iterator<value_type, 0>;
		using const_iterator		 = const oiml_array_iterator<value_type, 0>;
		using reverse_iterator		 = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		OIML_FORCE_INLINE constexpr void fill(const value_type&) {
		}

		OIML_FORCE_INLINE constexpr void swap(oiml_array&) noexcept {
		}

		OIML_FORCE_INLINE constexpr iterator begin() noexcept {
			return iterator{};
		}

		OIML_FORCE_INLINE constexpr const_iterator begin() const noexcept {
			return const_iterator{};
		}

		OIML_FORCE_INLINE constexpr iterator end() noexcept {
			return iterator{};
		}

		OIML_FORCE_INLINE constexpr const_iterator end() const noexcept {
			return const_iterator{};
		}

		OIML_FORCE_INLINE constexpr reverse_iterator rbegin() noexcept {
			return reverse_iterator(end());
		}

		OIML_FORCE_INLINE constexpr const_reverse_iterator rbegin() const noexcept {
			return const_reverse_iterator(end());
		}

		OIML_FORCE_INLINE constexpr reverse_iterator rend() noexcept {
			return reverse_iterator(begin());
		}

		OIML_FORCE_INLINE constexpr const_reverse_iterator rend() const noexcept {
			return const_reverse_iterator(begin());
		}

		OIML_FORCE_INLINE constexpr const_iterator cbegin() const noexcept {
			return begin();
		}

		OIML_FORCE_INLINE constexpr const_iterator cend() const noexcept {
			return end();
		}

		OIML_FORCE_INLINE constexpr const_reverse_iterator crbegin() const noexcept {
			return rbegin();
		}

		OIML_FORCE_INLINE constexpr const_reverse_iterator crend() const noexcept {
			return rend();
		}

		OIML_FORCE_INLINE constexpr size_type size() const noexcept {
			return 0;
		}

		OIML_FORCE_INLINE constexpr size_type max_size() const noexcept {
			return 0;
		}

		OIML_FORCE_INLINE constexpr bool empty() const noexcept {
			return true;
		}

		OIML_FORCE_INLINE constexpr reference at(size_type) {
			std::runtime_error{ "invalid oiml_array<T, N> subscript" };
		}

		OIML_FORCE_INLINE constexpr const_reference at(size_type) const {
			std::runtime_error{ "invalid oiml_array<T, N> subscript" };
		}

		OIML_FORCE_INLINE constexpr reference operator[](size_type) noexcept {
			return *data();
		}

		OIML_FORCE_INLINE constexpr const_reference operator[](size_type) const noexcept {
			return *data();
		}

		OIML_FORCE_INLINE constexpr reference front() noexcept {
			return *data();
		}

		OIML_FORCE_INLINE constexpr const_reference front() const noexcept {
			return *data();
		}

		OIML_FORCE_INLINE constexpr reference back() noexcept {
			return *data();
		}

		OIML_FORCE_INLINE constexpr const_reference back() const noexcept {
			return *data();
		}

		OIML_FORCE_INLINE constexpr value_type* data() noexcept {
			return nullptr;
		}

		OIML_FORCE_INLINE constexpr const value_type* data() const noexcept {
			return nullptr;
		}

		OIML_FORCE_INLINE constexpr friend bool operator==(const oiml_array& lhs, const oiml_array& rhs) {
			return true;
		}

	  private:
		std::conditional_t<std::disjunction_v<std::is_default_constructible<value_type>, std::is_default_constructible<value_type>>, value_type, empty_array_element> data_val[1]{};
	};

}