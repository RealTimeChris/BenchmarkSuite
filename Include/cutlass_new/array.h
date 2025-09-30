/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Statically sized array of elements that accommodates all CUTLASS-supported numeric types
           and is safe to use in a union.
*/

#pragma once
#include "cutlass_new/cutlass.h"
#include "cutlass_new/functional.h"
#include "cutlass_new/numeric_types.h"
#include "cutlass_new/platform/platform.h"
namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Statically sized array for any data type
template <
  typename T,
  int N,
  bool RegisterSized = sizeof_bits<T>::value >= 32
>
struct Array;

namespace detail {

template<class T>
struct is_Array : platform::false_type {};

template <
  typename T,
  int N,
  bool RegisterSized
>
struct is_Array<Array<T, N, RegisterSized> > : platform::true_type {};

template<typename T>
constexpr bool is_Array_v = is_Array<T>::value;

} // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the size of an Array<> in bits
template <typename T, int N, bool RegisterSized>
struct sizeof_bits<Array<T, N, RegisterSized> > {
  static constexpr int value = sizeof(Array<T, N, RegisterSized>) * 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if the argument is a power of 2
CUTLASS_HOST_DEVICE
constexpr bool ispow2(unsigned x) {
  return x && (!(x & (x - 1)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns the largest power of two not greater than the argument.
CUTLASS_HOST_DEVICE
constexpr unsigned floor_pow_2(unsigned x) {
  return (x == 0 || ispow2(x)) ? x : ((floor_pow_2(x >> 1)) << 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////


template<uint64_t index> struct tag : public std::integral_constant<uint64_t, index> {};
/// Statically sized array for any data type
template <
  typename T,
  int N
>
struct Array<T, N, true> {

  /// Storage type
  using Storage = T;

  /// Element type
  using Element = T;

  /// Number of storage elements
  //static std::size_t const kStorageElements = N;
  static constexpr size_t kStorageElements = N;

  /// Number of logical elements
  static constexpr size_t kElements = N;

  //
  // C++ standard members
  //

  typedef T value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef value_type &reference;
  typedef value_type const & const_reference;
  typedef value_type *pointer;
  typedef value_type const * const_pointer;

  //
  // Iterators
  //

  /// Bidirectional iterator over elements
  class iterator {

    /// Pointer to object
    T *ptr_;

  public:

    CUTLASS_HOST_DEVICE
    iterator(): ptr_(nullptr) { }

    CUTLASS_HOST_DEVICE
    iterator(T *_ptr): ptr_(_ptr) { }

    CUTLASS_HOST_DEVICE
    iterator &operator++() {
      ++ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator &operator--() {
      --ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator operator++(int) {
      iterator ret(*this);
      ++ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    iterator operator--(int) {
      iterator ret(*this);
      --ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    T &operator*() const {
      return *ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator==(iterator const &other) const {
      return ptr_ == other.ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(iterator const &other) const {
      return ptr_ != other.ptr_;
    }
  };

  /// Bidirectional constant iterator over elements
  class const_iterator {

    /// Pointer to object
    const T *ptr_;

  public:

    CUTLASS_HOST_DEVICE
    const_iterator(): ptr_(nullptr) { }

    CUTLASS_HOST_DEVICE
    const_iterator(T const *_ptr): ptr_(_ptr) { }

    CUTLASS_HOST_DEVICE
    const_iterator &operator++() {
      ++ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    const_iterator &operator--() {
      --ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    const_iterator operator++(int) {
      const_iterator ret(*this);
      ++ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    const_iterator operator--(int) {
      const_iterator ret(*this);
      --ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    T const &operator*() const {
      return *ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator==(const_iterator const &other) const {
      return ptr_ == other.ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(const_iterator const &other) const {
      return ptr_ != other.ptr_;
    }
  };

  /// Bidirectional iterator over elements
  class reverse_iterator {

    /// Pointer to object
    T *ptr_;

  public:

    CUTLASS_HOST_DEVICE
    reverse_iterator(): ptr_(nullptr) { }

    CUTLASS_HOST_DEVICE
    reverse_iterator(T *_ptr): ptr_(_ptr) { }

    CUTLASS_HOST_DEVICE
    reverse_iterator &operator++() {
      --ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    reverse_iterator &operator--() {
      ++ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    reverse_iterator operator++(int) {
      iterator ret(*this);
      --ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    reverse_iterator operator--(int) {
      iterator ret(*this);
      ++ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    T &operator*() const {
      return *(ptr_ - 1);
    }

    CUTLASS_HOST_DEVICE
    bool operator==(reverse_iterator const &other) const {
      return ptr_ == other.ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(reverse_iterator const &other) const {
      return ptr_ != other.ptr_;
    }
  };

  /// Bidirectional constant iterator over elements
  class const_reverse_iterator {

    /// Pointer to object
    T const *ptr_;

  public:

    CUTLASS_HOST_DEVICE
    const_reverse_iterator(): ptr_(nullptr) { }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator(T const *_ptr): ptr_(_ptr) { }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator &operator++() {
      --ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator &operator--() {
      ++ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator operator++(int) {
      const_reverse_iterator ret(*this);
      --ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator operator--(int) {
      const_reverse_iterator ret(*this);
      ++ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    T const &operator*() const {
      return *(ptr_ - 1);
    }

    CUTLASS_HOST_DEVICE
    bool operator==(const_iterator const &other) const {
      return ptr_ == other.ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(const_iterator const &other) const {
      return ptr_ != other.ptr_;
    }
  };

  /// Internal storage
  Storage storage[kElements];

  /// Efficient clear method
  CUTLASS_HOST_DEVICE
  void clear() {
    fill(T(0));
  }

  CUTLASS_HOST_DEVICE
  reference at(size_type pos) {
    return reinterpret_cast<reference>(storage[pos]);
  }

  CUTLASS_HOST_DEVICE
  const_reference at(size_type pos) const {
    return reinterpret_cast<const_reference>(storage[pos]);
  }

  CUTLASS_HOST_DEVICE reference operator[](size_type pos) {
	  return reinterpret_cast<reference>(storage[pos]);
  }

  CUTLASS_HOST_DEVICE const_reference operator[](size_type pos) const {
	  return reinterpret_cast<const_reference>(storage[pos]);
  }
  
  template<uint64_t index> CUTLASS_HOST_DEVICE reference operator[](tag<index> pos) {
	  return reinterpret_cast<reference>(storage[pos]);
  }

  template<uint64_t index> CUTLASS_HOST_DEVICE const_reference operator[](tag<index> pos) const {
    return reinterpret_cast<const_reference>(storage[pos]);
  }

  CUTLASS_HOST_DEVICE
  reference front() {
    return reinterpret_cast<reference>(storage[0]);
  }

  CUTLASS_HOST_DEVICE
  const_reference front() const {
    return reinterpret_cast<const_reference>(storage[0]);
  }

  CUTLASS_HOST_DEVICE
  reference back() {
    return reinterpret_cast<reference>(storage[kStorageElements - 1]);
  }

  CUTLASS_HOST_DEVICE
  const_reference back() const {
    return reinterpret_cast<const_reference>(storage[kStorageElements - 1]);
  }

  CUTLASS_HOST_DEVICE
  pointer data() {
    return reinterpret_cast<pointer>(storage);
  }

  CUTLASS_HOST_DEVICE
  const_pointer data() const {
    return reinterpret_cast<const_pointer>(storage);
  }
  
  CUTLASS_HOST_DEVICE
  pointer raw_data() {
    return reinterpret_cast<pointer>(storage);
  }

  CUTLASS_HOST_DEVICE
  const_pointer raw_data() const {
    return reinterpret_cast<const_pointer>(storage);
  }


  CUTLASS_HOST_DEVICE
  constexpr bool empty() const {
    return !kElements;
  }

  CUTLASS_HOST_DEVICE
  constexpr size_type size() const {
    return kElements;
  }

  CUTLASS_HOST_DEVICE
  constexpr size_type max_size() const {
    return kElements;
  }

  CUTLASS_HOST_DEVICE
  void fill(T const &value) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(kElements); ++i) {
      storage[i] = static_cast<Storage>(value);
    }
  }

  CUTLASS_HOST_DEVICE
  iterator begin() {
    return iterator(storage);
  }

  CUTLASS_HOST_DEVICE
  const_iterator begin() const {
    return cbegin();
  }

  CUTLASS_HOST_DEVICE
  const_iterator cbegin() const {
    return const_iterator(storage);
  }

  CUTLASS_HOST_DEVICE
  iterator end() {
    return iterator(reinterpret_cast<pointer>(storage + kStorageElements));
  }

  CUTLASS_HOST_DEVICE
  const_iterator end() const {
    return cend();
  }

  CUTLASS_HOST_DEVICE
  const_iterator cend() const {
    return const_iterator(reinterpret_cast<const_pointer>(storage + kStorageElements));
  }

  CUTLASS_HOST_DEVICE
  reverse_iterator rbegin() {
    return reverse_iterator(reinterpret_cast<pointer>(storage + kStorageElements));
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator rbegin() const {
    return crbegin();
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator(reinterpret_cast<const_pointer>(storage + kStorageElements));
  }

  CUTLASS_HOST_DEVICE
  reverse_iterator rend() {
    return reverse_iterator(reinterpret_cast<pointer>(storage));
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator rend() const {
    return crend();
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator crend() const {
    return const_reverse_iterator(reinterpret_cast<const_pointer>(storage));
  }

  //
  // Comparison operators
  //

};

/////////////////////////////////////////////////////////////////////////////////////////////////
// functional.h numeric specializations
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int N>
struct plus<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {

    Array<T, N> result;
    plus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {

    Array<T, N> result;
    plus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {

    Array<T, N> result;
    plus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct multiplies<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {

    Array<T, N> result;
    multiplies<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {

    Array<T, N> result;
    multiplies<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {

    Array<T, N> result;
    multiplies<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N, bool PropogateNaN>
struct maximum_absolute_value_reduction<Array<T, N>, PropogateNaN> {

  CUTLASS_HOST_DEVICE
  T operator() (T const& scalar, Array<T, N> const& rhs) const {

    T result = scalar;
    maximum_absolute_value_reduction<T, PropogateNaN> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result = scalar_op(result, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct scale<Array<T, N>> {
  T const scaling_factor_;

  CUTLASS_HOST_DEVICE
  scale(T scaling_factor) : scaling_factor_(scaling_factor) {
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const & rhs) const {
    Array<T, N> result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = rhs[i] * scaling_factor_;
    }

    return result;
  }
};

/// Fused multiply-add
template <typename T, int N>
struct multiply_add<Array<T, N>, Array<T, N>, Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, Array<T, N> const &b, Array<T, N> const &c) const {

    Array<T, N> result;
    multiply_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], b[i], c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, T const &scalar, Array<T, N> const &c) const {

    Array<T, N> result;
    multiply_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], scalar, c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &b, Array<T, N> const &c) const {

    Array<T, N> result;
    multiply_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, b[i], c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, Array<T, N> const &b, T const &scalar) const {

    Array<T, N> result;
    multiply_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], b[i], scalar);
    }

    return result;
  }


  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, T const &scalar_b, T const &scalar_c) const {

    Array<T, N> result;
    multiply_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], scalar_b, scalar_c);
    }

    return result;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
  

////////////////////////////////////////////////////////////////////////////////////////////////////
// AlignedArray
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Aligned array type
template <
  /// Element type
  typename T,
  /// Number of elements in the array
  int N,
  /// Alignment requirement in bytes
  int Alignment = ( sizeof_bits<T>::value * N + 7 ) / 8
>
class alignas(Alignment) AlignedArray: public Array<T, N> {
public:

};

} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass_new/array_subbyte.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
