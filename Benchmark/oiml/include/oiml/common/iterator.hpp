#pragma once

#include <oiml/common/config.hpp>
#include <iterator>

namespace oiml {

	template<typename value_type_new, size_t size> class oiml_array_iterator {
	  public:
		using iterator_concept	= std::contiguous_iterator_tag;
		using iterator_category = std::random_access_iterator_tag;
		using element_type		= value_type_new;
		using value_type		= value_type_new;
		using difference_type	= std::ptrdiff_t;
		using pointer			= value_type*;
		using reference			= value_type&;

		OIML_FORCE_INLINE constexpr oiml_array_iterator() noexcept : ptr() {
		}

		OIML_FORCE_INLINE constexpr oiml_array_iterator(pointer ptrNew) noexcept : ptr(ptrNew) {
		}

		OIML_FORCE_INLINE constexpr reference operator*() const noexcept {
			return *ptr;
		}

		OIML_FORCE_INLINE constexpr pointer operator->() const noexcept {
			return std::pointer_traits<pointer>::pointer_to(**this);
		}

		OIML_FORCE_INLINE constexpr oiml_array_iterator& operator++() noexcept {
			++ptr;
			return *this;
		}

		OIML_FORCE_INLINE constexpr oiml_array_iterator operator++(int32_t) noexcept {
			oiml_array_iterator temp = *this;
			++*this;
			return temp;
		}

		OIML_FORCE_INLINE constexpr oiml_array_iterator& operator--() noexcept {
			--ptr;
			return *this;
		}

		OIML_FORCE_INLINE constexpr oiml_array_iterator operator--(int32_t) noexcept {
			oiml_array_iterator temp = *this;
			--*this;
			return temp;
		}

		OIML_FORCE_INLINE constexpr oiml_array_iterator& operator+=(const difference_type offSet) noexcept {
			ptr += offSet;
			return *this;
		}

		OIML_FORCE_INLINE constexpr oiml_array_iterator operator+(const difference_type offSet) const noexcept {
			oiml_array_iterator temp = *this;
			temp += offSet;
			return temp;
		}

		OIML_FORCE_INLINE friend constexpr oiml_array_iterator operator+(const difference_type offSet, oiml_array_iterator _Next) noexcept {
			_Next += offSet;
			return _Next;
		}

		OIML_FORCE_INLINE constexpr oiml_array_iterator& operator-=(const difference_type offSet) noexcept {
			return *this += -offSet;
		}

		OIML_FORCE_INLINE constexpr oiml_array_iterator operator-(const difference_type offSet) const noexcept {
			oiml_array_iterator temp = *this;
			temp -= offSet;
			return temp;
		}

		OIML_FORCE_INLINE constexpr difference_type operator-(const oiml_array_iterator& other) const noexcept {
			return static_cast<difference_type>(ptr - other.ptr);
		}

		OIML_FORCE_INLINE constexpr reference operator[](const difference_type offSet) const noexcept {
			return *(*this + offSet);
		}

		OIML_FORCE_INLINE constexpr bool operator==(const oiml_array_iterator& other) const noexcept {
			return ptr == other.ptr;
		}

		OIML_FORCE_INLINE constexpr std::strong_ordering operator<=>(const oiml_array_iterator& other) const noexcept {
			return ptr <=> other.ptr;
		}

		OIML_FORCE_INLINE constexpr bool operator!=(const oiml_array_iterator& other) const noexcept {
			return !(*this == other);
		}

		OIML_FORCE_INLINE constexpr bool operator<(const oiml_array_iterator& other) const noexcept {
			return ptr < other.ptr;
		}

		OIML_FORCE_INLINE constexpr bool operator>(const oiml_array_iterator& other) const noexcept {
			return other < *this;
		}

		OIML_FORCE_INLINE constexpr bool operator<=(const oiml_array_iterator& other) const noexcept {
			return !(other < *this);
		}

		OIML_FORCE_INLINE constexpr bool operator>=(const oiml_array_iterator& other) const noexcept {
			return !(*this < other);
		}

		pointer ptr;
	};

	template<typename value_type_new> class oiml_array_iterator<value_type_new, 0> {
	  public:
		using iterator_concept	= std::contiguous_iterator_tag;
		using iterator_category = std::random_access_iterator_tag;
		using element_type		= value_type_new;
		using value_type		= value_type_new;
		using difference_type	= std::ptrdiff_t;
		using pointer			= value_type*;
		using reference			= value_type&;

		OIML_FORCE_INLINE constexpr oiml_array_iterator() noexcept {
		}

		OIML_FORCE_INLINE constexpr oiml_array_iterator(std::nullptr_t ptrNew) noexcept {
		}

		OIML_FORCE_INLINE constexpr bool operator==(const oiml_array_iterator& other) const noexcept {
			return true;
		}

		OIML_FORCE_INLINE constexpr bool operator!=(const oiml_array_iterator& other) const noexcept {
			return !(*this == other);
		}

		OIML_FORCE_INLINE constexpr bool operator>=(const oiml_array_iterator& other) const noexcept {
			return !(*this < other);
		}
	};

}