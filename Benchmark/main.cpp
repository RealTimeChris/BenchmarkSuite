#include <BnchSwt/BenchmarkSuite.hpp>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>
#include <source_location>
#include <array>
#include <queue>
#include <latch>
#include <bit>

namespace detail {

	template<typename value_01_type, typename value_02_type>
	concept convertible_to = std::is_convertible_v<value_02_type, value_01_type>;

	template<typename value_01_type, convertible_to<value_01_type> value_02_type> BNCH_SWT_INLINE constexpr value_01_type max(value_01_type val01, value_02_type val02) noexcept {
		return val01 > static_cast<std::remove_cvref_t<value_01_type>>(val02) ? val01 : static_cast<std::remove_cvref_t<value_01_type>>(val02);
	}

	template<typename value_01_type, convertible_to<value_01_type> value_02_type> BNCH_SWT_INLINE constexpr value_01_type min(value_01_type val01, value_02_type val02) noexcept {
		return val01 < static_cast<std::remove_cvref_t<value_01_type>>(val02) ? val01 : static_cast<std::remove_cvref_t<value_01_type>>(val02);
	}

	template<class value_type> BNCH_SWT_INLINE constexpr value_type&& forward(std::remove_reference_t<value_type>& arg) noexcept {
		return static_cast<value_type&&>(arg);
	}

	template<class value_type> BNCH_SWT_INLINE constexpr value_type&& forward(std::remove_reference_t<value_type>&& arg) noexcept {
		static_assert(!std::is_lvalue_reference_v<value_type>, "bad detail::forward call");
		return static_cast<std::remove_reference_t<value_type>&&>(arg);
	}

	template<class value_type> BNCH_SWT_INLINE constexpr std::remove_reference_t<value_type>&& move(value_type&& arg) noexcept {
		return static_cast<std::remove_reference_t<value_type>&&>(arg);
	}

}


template<typename value_type>
concept integral_or_enum_types = std::integral<value_type> || std::is_enum_v<std::remove_cvref_t<value_type>>;

template<typename value_type_new, uint64_t size> class array_iterator {
  public:
	using iterator_concept	= std::contiguous_iterator_tag;
	using iterator_category = std::random_access_iterator_tag;
	using element_type		= value_type_new;
	using value_type		= value_type_new;
	using difference_type	= std::ptrdiff_t;
	using pointer			= value_type*;
	using reference			= value_type&;

	BNCH_SWT_INLINE constexpr array_iterator() noexcept : ptr() {
	}

	BNCH_SWT_INLINE constexpr array_iterator(pointer ptrNew) noexcept : ptr(ptrNew) {
	}

	BNCH_SWT_INLINE constexpr reference operator*() const noexcept {
		return *ptr;
	}

	BNCH_SWT_INLINE constexpr pointer operator->() const noexcept {
		return std::pointer_traits<pointer>::pointer_to(**this);
	}

	BNCH_SWT_INLINE constexpr array_iterator& operator++() noexcept {
		++ptr;
		return *this;
	}

	BNCH_SWT_INLINE constexpr array_iterator operator++(int32_t) noexcept {
		array_iterator temp = *this;
		++*this;
		return temp;
	}

	BNCH_SWT_INLINE constexpr array_iterator& operator--() noexcept {
		--ptr;
		return *this;
	}

	BNCH_SWT_INLINE constexpr array_iterator operator--(int32_t) noexcept {
		array_iterator temp = *this;
		--*this;
		return temp;
	}

	BNCH_SWT_INLINE constexpr array_iterator& operator+=(const difference_type offSet) noexcept {
		ptr += offSet;
		return *this;
	}

	BNCH_SWT_INLINE constexpr array_iterator operator+(const difference_type offSet) const noexcept {
		array_iterator temp = *this;
		temp += offSet;
		return temp;
	}

	BNCH_SWT_INLINE friend constexpr array_iterator operator+(const difference_type offSet, array_iterator next) noexcept {
		next += offSet;
		return next;
	}

	BNCH_SWT_INLINE constexpr array_iterator& operator-=(const difference_type offSet) noexcept {
		return *this += -offSet;
	}

	BNCH_SWT_INLINE constexpr array_iterator operator-(const difference_type offSet) const noexcept {
		array_iterator temp = *this;
		temp -= offSet;
		return temp;
	}

	BNCH_SWT_INLINE constexpr difference_type operator-(const array_iterator& other) const noexcept {
		return static_cast<difference_type>(ptr - other.ptr);
	}

	BNCH_SWT_INLINE constexpr reference operator[](const difference_type offSet) const noexcept {
		return *(*this + offSet);
	}

	BNCH_SWT_INLINE constexpr bool operator==(const array_iterator& other) const noexcept {
		return ptr == other.ptr;
	}

	BNCH_SWT_INLINE constexpr std::strong_ordering operator<=>(const array_iterator& other) const noexcept {
		return ptr <=> other.ptr;
	}

	BNCH_SWT_INLINE constexpr bool operator!=(const array_iterator& other) const noexcept {
		return !(*this == other);
	}

	BNCH_SWT_INLINE constexpr bool operator<(const array_iterator& other) const noexcept {
		return ptr < other.ptr;
	}

	BNCH_SWT_INLINE constexpr bool operator>(const array_iterator& other) const noexcept {
		return other < *this;
	}

	BNCH_SWT_INLINE constexpr bool operator<=(const array_iterator& other) const noexcept {
		return !(other < *this);
	}

	BNCH_SWT_INLINE constexpr bool operator>=(const array_iterator& other) const noexcept {
		return !(*this < other);
	}

	pointer ptr;
};

template<typename value_type_new> class array_iterator<value_type_new, 0> {
  public:
	using iterator_concept	= std::contiguous_iterator_tag;
	using iterator_category = std::random_access_iterator_tag;
	using element_type		= value_type_new;
	using value_type		= value_type_new;
	using difference_type	= std::ptrdiff_t;
	using pointer			= value_type*;
	using reference			= value_type&;

	BNCH_SWT_INLINE constexpr array_iterator() noexcept {
	}

	BNCH_SWT_INLINE constexpr array_iterator(std::nullptr_t ptrNew) noexcept {
		( void )ptrNew;
	}

	BNCH_SWT_INLINE constexpr bool operator==(const array_iterator& other) const noexcept {
		( void )other;
		return true;
	}

	BNCH_SWT_INLINE constexpr bool operator!=(const array_iterator& other) const noexcept {
		return !(*this == other);
	}

	BNCH_SWT_INLINE constexpr bool operator>=(const array_iterator& other) const noexcept {
		return !(*this < other);
	}
};

enum class array_static_assert_errors {
	invalid_index_type,
};

template<auto index> using tag = std::integral_constant<uint64_t, static_cast<uint64_t>(index)>;

template<typename value_type_new, auto size_new> struct array {
  public:
	static_assert(integral_or_enum_types<decltype(size_new)>, "Sorry, but the size val passed to array must be integral or enum!");
	static constexpr uint64_t size_val{ static_cast<uint64_t>(size_new) };
	using value_type			 = value_type_new;
	using uint64_types			 = decltype(size_new);
	using size_type				 = uint64_t;
	using difference_type		 = ptrdiff_t;
	using pointer				 = value_type*;
	using const_pointer			 = const value_type*;
	using reference				 = value_type&;
	using const_reference		 = const value_type&;
	using iterator				 = array_iterator<value_type, static_cast<uint64_t>(size_new)>;
	using const_iterator		 = array_iterator<const value_type, static_cast<uint64_t>(size_new)>;
	using reverse_iterator		 = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	BNCH_SWT_INLINE constexpr array() = default;

	BNCH_SWT_INLINE constexpr array(std::initializer_list<value_type> values) {
		for (uint64_t x = 0; x < values.size(); ++x) {
			data_val[x] = values.begin()[x];
		}
	}

	BNCH_SWT_INLINE constexpr array(const array& other)
		requires(std::is_copy_constructible_v<value_type>)
	{
		for (uint64_t i = 0; i < size_val; ++i) {
			data_val[i] = other.data_val[i];
		}
	}

	BNCH_SWT_INLINE array(const array& other)
		requires(!std::is_copy_constructible_v<value_type>)
	= delete;

	BNCH_SWT_INLINE constexpr array& operator=(const array& other)
		requires(std::is_copy_assignable_v<value_type>)
	{
		if (this != &other) {
			for (uint64_t i = 0; i < size_val; ++i) {
				data_val[i] = other.data_val[i];
			}
		}
		return *this;
	}

	BNCH_SWT_INLINE array& operator=(const array& other)
		requires(!std::is_copy_assignable_v<value_type>)
	= delete;

	BNCH_SWT_INLINE constexpr array(array&& other) noexcept
		requires(std::is_move_constructible_v<value_type>)
	{
		for (uint64_t i = 0; i < size_val; ++i) {
			data_val[i] = detail::move(other.data_val[i]);
		}
	}

	BNCH_SWT_INLINE constexpr array& operator=(array&& other) noexcept
		requires(std::is_move_assignable_v<value_type>)
	{
		if (this != &other) {
			for (uint64_t i = 0; i < size_val; ++i) {
				data_val[i] = detail::move(other.data_val[i]);
			}
		}
		return *this;
	}

	template<typename... Args> BNCH_SWT_INLINE constexpr array(Args&&... args)
		requires(sizeof...(Args) == size_val && (std::is_constructible_v<value_type, Args> && ...) && std::is_copy_constructible_v<value_type>)
		: data_val{ static_cast<value_type>(forward<Args>(args))... } {
	}

	BNCH_SWT_INLINE constexpr void fill(const value_type& value) {
		std::fill_n(data_val, static_cast<int64_t>(size_new), value);
	}

	BNCH_SWT_INLINE constexpr iterator begin() noexcept {
		return iterator(data_val);
	}

	BNCH_SWT_INLINE constexpr const_iterator begin() const noexcept {
		return const_iterator(data_val);
	}

	BNCH_SWT_INLINE constexpr iterator end() noexcept {
		return iterator(data_val + size_val);
	}

	BNCH_SWT_INLINE constexpr const_iterator end() const noexcept {
		return const_iterator(data_val + size_val);
	}

	BNCH_SWT_INLINE constexpr reverse_iterator rbegin() noexcept {
		return reverse_iterator(end());
	}

	BNCH_SWT_INLINE constexpr const_reverse_iterator rbegin() const noexcept {
		return const_reverse_iterator(end());
	}

	BNCH_SWT_INLINE constexpr reverse_iterator rend() noexcept {
		return reverse_iterator(begin());
	}

	BNCH_SWT_INLINE constexpr const_reverse_iterator rend() const noexcept {
		return const_reverse_iterator(begin());
	}

	BNCH_SWT_INLINE constexpr const_iterator cbegin() const noexcept {
		return begin();
	}

	BNCH_SWT_INLINE constexpr const_iterator cend() const noexcept {
		return end();
	}

	BNCH_SWT_INLINE constexpr const_reverse_iterator crbegin() const noexcept {
		return rbegin();
	}

	BNCH_SWT_INLINE constexpr const_reverse_iterator crend() const noexcept {
		return rend();
	}

	BNCH_SWT_INLINE constexpr uint64_t size() const noexcept {
		return size_val;
	}

	BNCH_SWT_INLINE constexpr uint64_t max_size() const noexcept {
		return size_val;
	}

	BNCH_SWT_INLINE constexpr bool empty() const noexcept {
		return false;
	}

	template<integral_or_enum_types index_type> BNCH_SWT_INLINE constexpr reference at(index_type position) {
		//static_assert(is_indexable<index_type, decltype(size_new)>, array_static_assert_errors::invalid_index_type, index_type>::impl,
		//"Sorry, but please index into this array using the correct enum type!");
		if (size_new <= position) {
			throw std::runtime_error{ "invalid array<value_type, N> subscript" };
		}

		return data_val[static_cast<uint64_t>(position)];
	}

	template<integral_or_enum_types index_type> BNCH_SWT_INLINE constexpr const_reference at(index_type position) const {
		//static_assert(static_assert_printer<is_indexable<index_type, decltype(size_new)>, array_static_assert_errors::invalid_index_type, index_type>::impl,
		//			"Sorry, but please index into this array using the correct enum type!");
		if (size_new <= position) {
			throw std::runtime_error{ "invalid array<value_type, N> subscript" };
		}

		return data_val[static_cast<uint64_t>(position)];
	}

	template<integral_or_enum_types index_type> BNCH_SWT_INLINE constexpr reference operator[](index_type position) noexcept {
		//static_assert(static_assert_printer<is_indexable<index_type, decltype(size_new)>, array_static_assert_errors::invalid_index_type, index_type>::impl,
		//			"Sorry, but please index into this array using the correct enum type!");
		return data_val[static_cast<uint64_t>(position)];
	}

	template<integral_or_enum_types index_type> BNCH_SWT_INLINE constexpr const_reference operator[](index_type position) const noexcept {
		//static_assert(static_assert_printer<is_indexable<index_type, decltype(size_new)>, array_static_assert_errors::invalid_index_type, index_type>::impl,
		//			"Sorry, but please index into this array using the correct enum type!");
		return data_val[static_cast<uint64_t>(position)];
	}

	template<uint64_t index> BNCH_SWT_INLINE constexpr uint64_t& operator[](tag<index> index_new) {
		return data_val[index_new];
	}

	template<uint64_t index> BNCH_SWT_INLINE constexpr uint64_t operator[](tag<index> index_new) const {
		return data_val[index_new];
	}

	BNCH_SWT_INLINE constexpr reference front() noexcept {
		return data_val[0];
	}

	BNCH_SWT_INLINE constexpr const_reference front() const noexcept {
		return data_val[0];
	}

	BNCH_SWT_INLINE constexpr reference back() noexcept {
		return data_val[size_new - 1];
	}

	BNCH_SWT_INLINE constexpr const_reference back() const noexcept {
		return data_val[size_new - 1];
	}

	BNCH_SWT_INLINE constexpr value_type* data() noexcept {
		return data_val;
	}

	BNCH_SWT_INLINE constexpr const value_type* data() const noexcept {
		return data_val;
	}

	BNCH_SWT_INLINE constexpr friend bool operator==(const array& lhs, const array& rhs) {
		for (uint64_t x = 0; x < size_val; ++x) {
			if (lhs[x] != rhs[x]) {
				return false;
			}
		}
		return true;
	}

	value_type data_val[size_val]{};
};

template<typename value_type, typename... U> array(value_type, U...) -> array<value_type, 1 + sizeof...(U)>;

struct empty_array_element {};

template<class value_type_new> class array<value_type_new, 0> {
  public:
	using value_type			 = value_type_new;
	using uint64_types			 = uint64_t;
	using difference_type		 = ptrdiff_t;
	using pointer				 = value_type*;
	using const_pointer			 = const value_type*;
	using reference				 = value_type&;
	using const_reference		 = const value_type&;
	using iterator				 = array_iterator<value_type, 0>;
	using const_iterator		 = const array_iterator<value_type, 0>;
	using reverse_iterator		 = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	BNCH_SWT_INLINE constexpr void fill(const value_type&) {
	}

	BNCH_SWT_INLINE constexpr void swap(array&) noexcept {
	}

	BNCH_SWT_INLINE constexpr iterator begin() noexcept {
		return iterator{};
	}

	BNCH_SWT_INLINE constexpr const_iterator begin() const noexcept {
		return const_iterator{};
	}

	BNCH_SWT_INLINE constexpr iterator end() noexcept {
		return iterator{};
	}

	BNCH_SWT_INLINE constexpr const_iterator end() const noexcept {
		return const_iterator{};
	}

	BNCH_SWT_INLINE constexpr reverse_iterator rbegin() noexcept {
		return reverse_iterator(end());
	}

	BNCH_SWT_INLINE constexpr const_reverse_iterator rbegin() const noexcept {
		return const_reverse_iterator(end());
	}

	BNCH_SWT_INLINE constexpr reverse_iterator rend() noexcept {
		return reverse_iterator(begin());
	}

	BNCH_SWT_INLINE constexpr const_reverse_iterator rend() const noexcept {
		return const_reverse_iterator(begin());
	}

	BNCH_SWT_INLINE constexpr const_iterator cbegin() const noexcept {
		return begin();
	}

	BNCH_SWT_INLINE constexpr const_iterator cend() const noexcept {
		return end();
	}

	BNCH_SWT_INLINE constexpr const_reverse_iterator crbegin() const noexcept {
		return rbegin();
	}

	BNCH_SWT_INLINE constexpr const_reverse_iterator crend() const noexcept {
		return rend();
	}

	BNCH_SWT_INLINE constexpr uint64_types size() const noexcept {
		return 0;
	}

	BNCH_SWT_INLINE constexpr uint64_types max_size() const noexcept {
		return 0;
	}

	BNCH_SWT_INLINE constexpr bool empty() const noexcept {
		return true;
	}

	BNCH_SWT_INLINE constexpr reference at(uint64_types) {
		throw std::runtime_error{ "invalid array<value_type, N> subscript" };
	}

	BNCH_SWT_INLINE constexpr const_reference at(uint64_types) const {
		throw std::runtime_error{ "invalid array<value_type, N> subscript" };
	}

	BNCH_SWT_INLINE constexpr reference operator[](uint64_types) noexcept {
		return *data();
	}

	BNCH_SWT_INLINE constexpr const_reference operator[](uint64_types) const noexcept {
		return *data();
	}

	BNCH_SWT_INLINE constexpr reference front() noexcept {
		return *data();
	}

	BNCH_SWT_INLINE constexpr const_reference front() const noexcept {
		return *data();
	}

	BNCH_SWT_INLINE constexpr reference back() noexcept {
		return *data();
	}

	BNCH_SWT_INLINE constexpr const_reference back() const noexcept {
		return *data();
	}

	BNCH_SWT_INLINE constexpr value_type* data() noexcept {
		return nullptr;
	}

	BNCH_SWT_INLINE constexpr const value_type* data() const noexcept {
		return nullptr;
	}

	BNCH_SWT_INLINE constexpr friend bool operator==(const array& lhs, const array& rhs) {
		( void )lhs;
		( void )rhs;
		return true;
	}

  protected:
	std::conditional_t<std::disjunction_v<std::is_default_constructible<value_type>, std::is_default_constructible<value_type>>, value_type, empty_array_element> data_val[1]{};
};

template<size_t alignment, typename value_type> struct alignas(alignment) static_aligned_const {
	alignas(alignment) value_type value{};

	BNCH_SWT_INLINE constexpr static_aligned_const() noexcept : value{} {
	}

	BNCH_SWT_INLINE constexpr static_aligned_const(value_type new_value) noexcept : value{ new_value } {
	}

	BNCH_SWT_INLINE constexpr operator const value_type&() const {
		return value;
	}

	BNCH_SWT_INLINE operator value_type&() {
		return value;
	}
};

template<auto value_new> struct make_static {
	static constexpr auto value{ value_new };
};

template<typename value_type> BNCH_SWT_INLINE static constexpr bool is_alpha_lookup(value_type c) noexcept {
	alignas(64) static constexpr const static_aligned_const<64, bool>* __restrict alpha_table{ [] constexpr {
		alignas(64) constexpr array<static_aligned_const<64, bool>, 256> return_values{ [] {
			array<static_aligned_const<64, bool>, 256> return_values{};
			for (int32_t i = 'A'; i <= 'Z'; ++i) {
				return_values[static_cast<uint64_t>(i)] = true;
			}

			for (int32_t i = 'a'; i <= 'z'; ++i) {
				return_values[static_cast<uint64_t>(i)] = true;
			}
			return return_values;
		}() };
		return make_static<return_values>::value.data();
	}() };
	return alpha_table[static_cast<uint8_t>(c)];
}

template<typename value_type> BNCH_SWT_INLINE static constexpr bool is_alpha_lookup_raw(value_type c) noexcept {
	alignas(64) static constexpr const bool* __restrict alpha_table_raw{ [] constexpr {
		alignas(64) constexpr std::array<bool, 256> return_values{ [] {
			std::array<bool, 256> return_values{};
			for (int32_t i = 'A'; i <= 'Z'; ++i) {
				return_values[static_cast<uint64_t>(i)] = true;
			}

			for (int32_t i = 'a'; i <= 'z'; ++i) {
				return_values[static_cast<uint64_t>(i)] = true;
			}
			return return_values;
		}() };
		return make_static<return_values>::value.data();
	}() };
	return alpha_table_raw[static_cast<uint8_t>(c)];
}

template<typename value_type> BNCH_SWT_INLINE static constexpr bool is_alpha_comparison(value_type c) noexcept {
	const uint8_t ch = static_cast<uint8_t>(c);
	return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
}

template<typename value_type> BNCH_SWT_INLINE static constexpr bool is_digit_comparison(value_type c) noexcept {
	return static_cast<uint8_t>(c - '0') < 10;
}

template<typename value_type> BNCH_SWT_INLINE static constexpr bool is_digit_table(value_type c) noexcept {
	alignas(64) static constexpr const static_aligned_const<64, bool>* __restrict digit_table{ [] {
		alignas(64) constexpr array<static_aligned_const<64, bool>, 256> return_values{ [] {
			array<static_aligned_const<64, bool>, 256> return_values{};
			for (uint64_t x = 0; x < 256; ++x) {
				return_values[x] = (static_cast<uint8_t>(x) - '0') < 10;
			}
			return return_values;
		}() };
		return make_static<return_values>::value.data();
	}() };
	return digit_table[static_cast<uint8_t>(c)];
}

template<typename value_type> BNCH_SWT_INLINE static constexpr bool is_digit_table_raw(value_type c) noexcept {
	alignas(64) static constexpr const bool* __restrict digit_table_raw{ [] {
		alignas(64) constexpr std::array<bool, 256> return_values{ [] {
			std::array<bool, 256> return_values{};
			for (uint64_t x = 0; x < 256; ++x) {
				return_values[x] = (static_cast<uint8_t>(x) - '0') < 10;
			}
			return return_values;
		}() };
		return make_static<return_values>::value.data();
	}() };
	return digit_table_raw[static_cast<uint8_t>(c)];
}

template<typename value_type> BNCH_SWT_INLINE static constexpr bool is_space_lookup(value_type c) noexcept {
	alignas(64) static constexpr const static_aligned_const<64, bool>* __restrict space_table{ [] {
		alignas(64) constexpr array<static_aligned_const<64, bool>, 256> return_values{ [] {
			array<static_aligned_const<64, bool>, 256> return_values{};
			return_values[static_cast<uint64_t>('\r')] = true;
			return_values[static_cast<uint64_t>('\n')] = true;
			return_values[static_cast<uint64_t>(' ')]  = true;
			return_values[static_cast<uint64_t>('\t')] = true;
			return_values[static_cast<uint64_t>('\v')] = true;
			return_values[static_cast<uint64_t>('\f')] = true;
			return return_values;
		}() };
		return make_static<return_values>::value.data();
	}() };
	return space_table[static_cast<uint8_t>(c)];
}

template<typename value_type> BNCH_SWT_INLINE static constexpr bool is_space_lookup_raw(value_type c) noexcept {
	alignas(64) static constexpr const bool* __restrict space_table_raw{ [] {
		alignas(64) constexpr std::array<bool, 256> return_values{ [] {
			std::array<bool, 256> return_values{};
			return_values[static_cast<uint64_t>('\r')] = true;
			return_values[static_cast<uint64_t>('\n')] = true;
			return_values[static_cast<uint64_t>(' ')]  = true;
			return_values[static_cast<uint64_t>('\t')] = true;
			return_values[static_cast<uint64_t>('\v')] = true;
			return_values[static_cast<uint64_t>('\f')] = true;
			return return_values;
		}() };
		return make_static<return_values>::value.data();
	}() };
	return space_table_raw[static_cast<uint8_t>(c)];
}

template<typename value_type> BNCH_SWT_INLINE static constexpr bool is_space_comparison(value_type c) noexcept {
	const uint8_t ch = static_cast<uint8_t>(c);
	if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
		return true;
	}
	return ch == '\v' || ch == '\f';
}

int main() {
	static constexpr uint64_t total_iterations{ 20 };
	static constexpr uint64_t measured_iterations{ 5 };
	static constexpr uint64_t numbers_to_check{ 1 };
	static constexpr uint64_t total_bytes{ numbers_to_check * sizeof(uint8_t) };
	std::vector<std::vector<uint8_t>> values_to_test{};
	std::vector<std::vector<bool>> results01{};
	std::vector<std::vector<bool>> results02{};
	values_to_test.resize(total_iterations);
	results01.resize(total_iterations);
	results02.resize(total_iterations);
	for (uint64_t x = 0; x < total_iterations; ++x) {
		for (uint64_t y = 0; y < numbers_to_check; ++y) {
			values_to_test[x].emplace_back(bnch_swt::random_generator::generateValue<uint8_t>());
			results01[x].emplace_back(bnch_swt::random_generator::generateValue<uint8_t>());
			results02[x].emplace_back(bnch_swt::random_generator::generateValue<uint8_t>());
		}
	}

	uint64_t current_iteration{};
	struct comparison {
		BNCH_SWT_INLINE static uint64_t impl(std::vector<std::vector<uint8_t>>& values, std::vector<std::vector<bool>>& results, uint64_t& current_iteration) {
			for (uint64_t x = 0; x < numbers_to_check; ++x) {
				results[current_iteration][x] = is_digit_comparison(values[current_iteration][x]);
			}
			++current_iteration;
			return total_bytes;
		};
	};

	struct lookup_table {
		BNCH_SWT_INLINE static uint64_t impl(std::vector<std::vector<uint8_t>>& values, std::vector<std::vector<bool>>& results, uint64_t& current_iteration) {
			for (uint64_t x = 0; x < numbers_to_check; ++x) {
				results[current_iteration][x] = is_digit_table(values[current_iteration][x]);
			}
			++current_iteration;
			return total_bytes;
		};
	};

	struct lookup_table_raw {
		BNCH_SWT_INLINE static uint64_t impl(std::vector<std::vector<uint8_t>>& values, std::vector<std::vector<bool>>& results, uint64_t& current_iteration) {
			for (uint64_t x = 0; x < numbers_to_check; ++x) {
				results[current_iteration][x] = is_digit_table_raw(values[current_iteration][x]);
			}
			++current_iteration;
			return total_bytes;
		};
	};

	struct alpha_comparison {
		BNCH_SWT_INLINE static uint64_t impl(std::vector<std::vector<uint8_t>>& values, std::vector<std::vector<bool>>& results, uint64_t& current_iteration) {
			for (uint64_t x = 0; x < numbers_to_check; ++x) {
				results[current_iteration][x] = is_alpha_comparison(values[current_iteration][x]);
			}
			++current_iteration;
			return total_bytes;
		};
	};

	struct alpha_lookup_table {
		BNCH_SWT_INLINE static uint64_t impl(std::vector<std::vector<uint8_t>>& values, std::vector<std::vector<bool>>& results, uint64_t& current_iteration) {
			for (uint64_t x = 0; x < numbers_to_check; ++x) {
				results[current_iteration][x] = is_alpha_lookup(values[current_iteration][x]);
			}
			++current_iteration;
			return total_bytes;
		};
	};

	struct alpha_lookup_table_raw {
		BNCH_SWT_INLINE static uint64_t impl(std::vector<std::vector<uint8_t>>& values, std::vector<std::vector<bool>>& results, uint64_t& current_iteration) {
			for (uint64_t x = 0; x < numbers_to_check; ++x) {
				results[current_iteration][x] = is_alpha_lookup_raw(values[current_iteration][x]);
			}
			++current_iteration;
			return total_bytes;
		};
	};

	struct space_comparison {
		BNCH_SWT_INLINE static uint64_t impl(std::vector<std::vector<uint8_t>>& values, std::vector<std::vector<bool>>& results, uint64_t& current_iteration) {
			for (uint64_t x = 0; x < numbers_to_check; ++x) {
				results[current_iteration][x] = is_space_comparison(values[current_iteration][x]);
			}
			++current_iteration;
			return total_bytes;
		};
	};

	struct space_lookup_table {
		BNCH_SWT_INLINE static uint64_t impl(std::vector<std::vector<uint8_t>>& values, std::vector<std::vector<bool>>& results, uint64_t& current_iteration) {
			for (uint64_t x = 0; x < numbers_to_check; ++x) {
				results[current_iteration][x] = is_space_lookup(values[current_iteration][x]);
			}
			++current_iteration;
			return total_bytes;
		};
	};

	struct space_lookup_table_raw {
		BNCH_SWT_INLINE static uint64_t impl(std::vector<std::vector<uint8_t>>& values, std::vector<std::vector<bool>>& results, uint64_t& current_iteration) {
			for (uint64_t x = 0; x < numbers_to_check; ++x) {
				results[current_iteration][x] = is_space_lookup_raw(values[current_iteration][x]);
			}
			++current_iteration;
			return total_bytes;
		};
	};

	bnch_swt::benchmark_stage<"test_stage", total_iterations, measured_iterations, true>::runBenchmark<"comparison", comparison>(values_to_test, results01, current_iteration);
	current_iteration = 0;
	bnch_swt::benchmark_stage<"test_stage", total_iterations, measured_iterations, true>::runBenchmark<"lookup_table", lookup_table>(values_to_test, results01, current_iteration);
	current_iteration = 0;
	bnch_swt::benchmark_stage<"test_stage", total_iterations, measured_iterations, true>::runBenchmark<"lookup_table_raw", lookup_table_raw>(values_to_test, results01,
		current_iteration);
	current_iteration = 0;

	bnch_swt::benchmark_stage<"alpha_test_stage", total_iterations, measured_iterations, true>::runBenchmark<"alpha_comparison", alpha_comparison>(values_to_test, results01,
		current_iteration);
	current_iteration = 0;
	bnch_swt::benchmark_stage<"alpha_test_stage", total_iterations, measured_iterations, true>::runBenchmark<"alpha_lookup_table", alpha_lookup_table>(values_to_test, results01,
		current_iteration);
	current_iteration = 0;
	bnch_swt::benchmark_stage<"alpha_test_stage", total_iterations, measured_iterations, true>::runBenchmark<"alpha_lookup_table_raw", alpha_lookup_table_raw>(values_to_test,
		results01, current_iteration);
	current_iteration = 0;

	bnch_swt::benchmark_stage<"space_test_stage", total_iterations, measured_iterations, true>::runBenchmark<"space_comparison", space_comparison>(values_to_test, results01,
		current_iteration);
	current_iteration = 0;
	bnch_swt::benchmark_stage<"space_test_stage", total_iterations, measured_iterations, true>::runBenchmark<"space_lookup_table", space_lookup_table>(values_to_test, results01,
		current_iteration);
	current_iteration = 0;
	bnch_swt::benchmark_stage<"space_test_stage", total_iterations, measured_iterations, true>::runBenchmark<"space_lookup_table_raw", space_lookup_table_raw>(values_to_test,
		results01, current_iteration);

	bnch_swt::benchmark_stage<"test_stage", total_iterations, measured_iterations, true>::printResults();
	bnch_swt::benchmark_stage<"alpha_test_stage", total_iterations, measured_iterations, true>::printResults();
	bnch_swt::benchmark_stage<"space_test_stage", total_iterations, measured_iterations, true>::printResults();
	return 0;
}
