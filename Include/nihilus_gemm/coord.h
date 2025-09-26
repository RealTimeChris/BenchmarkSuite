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
    \brief A Coord is a coordinate of arbitrary rank into a tensor or matrix
*/

#pragma once
#include "nihilus_gemm/nihilus_gemm.h"
#if defined(__CUDACC_RTC__)
	#include CUDA_STD_HEADER(cstdint)
#else
	#include <cstdint>
#endif

namespace nihilus_gemm {

	////////////////////////////////////////////////////////////////////////////////////////////////////

	/// Statically-sized array specifying Coords within a tensor
	template<int Rank_,///< Logical rank of coordinate
		typename Index_		= int,///< Index type used for each dimension
		typename LongIndex_ = int64_t///< Long index type used for linear offsets
		>
	struct Coord {
	  public:
		//
		// Type and constant definitions
		//

		/// Number of elements in Coord
		static constexpr int kRank = Rank_;

		/// Index type used to store elements
		using Index = Index_;

		/// Type used to represent linear offsets
		using LongIndex = LongIndex_;

	  public:
		//
		// Data members
		//

		/// Indices
		Index idx[kRank];

	  public:
		//
		// Methods
		//

		/// Default ctor initializes uniformly
		NIHILUS_HOST_DEVICE
		constexpr explicit Coord(Index value = Index(0)) {
			for (int i = 0; i < kRank; ++i) {
				idx[i] = value;
			}
		}

		/// Constructs from an array of integers
		NIHILUS_HOST_DEVICE
		constexpr Coord(Index const (&_idx)[kRank]) {
			for (int i = 0; i < kRank; ++i) {
				idx[i] = _idx[i];
			}
		}

		/// Constructs from some other Coord
		template<int R, typename I, typename L> NIHILUS_HOST_DEVICE Coord(Coord<R, I, L> other) {
			for (int i = 0; i < kRank; ++i) {
				idx[i] = other[i];
			}
		}

		/// Returns a slice of the Coord which may be larger or smaller in rank
		/// than this.
		template<int Slice> NIHILUS_HOST_DEVICE Coord<Slice, Index, LongIndex> slice(int start = 0, Index identity = 0) const {
			Coord<Slice, Index, LongIndex> result;
			for (int i = 0; i < Slice; ++i) {
				if (i + start < kRank) {
					result[i] = idx[i + start];
				} else {
					result[i] = identity;
				}
			}
			return result;
		}

		/// Returns the index of the dimension with least value
		NIHILUS_HOST_DEVICE
		int min_dim_index() const {
			int i = 0;
			for (int j = 1; j < kRank; ++j) {
				if (idx[j] < idx[i]) {
					i = j;
				}
			}
			return i;
		}

		/// Returns the index of the dimension with greatest value
		NIHILUS_HOST_DEVICE
		int max_dim_index() const {
			int i = 0;
			for (int j = 1; j < kRank; ++j) {
				if (idx[j] > idx[i]) {
					i = j;
				}
			}
			return i;
		}

		/// Returns true if Coord is non-zero.
		NIHILUS_HOST_DEVICE
		explicit operator bool() const {
			for (int i = 0; i < kRank; ++i) {
				if (idx[i]) {
					return true;
				}
			}
			return false;
		}

		/// Returns true if Coord is uniformly zero.
		NIHILUS_HOST_DEVICE
		bool operator!() const {
			for (int i = 0; i < kRank; ++i) {
				if (idx[i]) {
					return false;
				}
			}
			return true;
		}

		/// Element-wise addition
		NIHILUS_HOST_DEVICE
		Coord operator+(Coord const& b) const {
			Coord c;
			for (int i = 0; i < kRank; ++i) {
				c.idx[i] = idx[i] + b.idx[i];
			}
			return c;
		}

		/// Element-wise subtraction
		NIHILUS_HOST_DEVICE
		Coord operator-(Coord const& b) const {
			Coord c;
			for (int i = 0; i < kRank; ++i) {
				c.idx[i] = idx[i] - b.idx[i];
			}
			return c;
		}

		/// Element-wise multiplication
		NIHILUS_HOST_DEVICE
		Coord operator*(Coord const& b) const {
			Coord c;
			for (int i = 0; i < kRank; ++i) {
				c.idx[i] = idx[i] * b.idx[i];
			}
			return c;
		}

		/// Element-wise division
		NIHILUS_HOST_DEVICE
		Coord operator/(Coord const& b) const {
			Coord c;
			for (int i = 0; i < kRank; ++i) {
				c.idx[i] = idx[i] / b.idx[i];
			}
			return c;
		}

		/// In-place addition
		NIHILUS_HOST_DEVICE
		Coord& operator+=(Coord const& b) {
			for (int i = 0; i < kRank; ++i) {
				idx[i] += b.idx[i];
			}
			return *this;
		}

		/// In-place subtraction
		NIHILUS_HOST_DEVICE
		Coord& operator-=(Coord const& b) {
			for (int i = 0; i < kRank; ++i) {
				idx[i] -= b.idx[i];
			}
			return *this;
		}

		/// In-place multiplication
		NIHILUS_HOST_DEVICE
		Coord& operator*=(Coord const& b) {
			for (int i = 0; i < kRank; ++i) {
				idx[i] *= b.idx[i];
			}
			return *this;
		}

		/// In-place division
		NIHILUS_HOST_DEVICE
		Coord& operator/=(Coord const& b) {
			for (int i = 0; i < kRank; ++i) {
				idx[i] /= b.idx[i];
			}
			return *this;
		}

		/// Member access operator
		NIHILUS_HOST_DEVICE Index& operator[](int dim) {
			return idx[dim];
		}

		/// Member access operator
		NIHILUS_HOST_DEVICE Index const& operator[](int dim) const {
			return idx[dim];
		}

		/// Computes the dot product with anotherCoord object
		NIHILUS_HOST_DEVICE
		LongIndex dot(Coord const& b, LongIndex sum = LongIndex(0)) const {
			for (int i = 0; i < kRank; ++i) {
				sum += idx[i] * b.idx[i];
			}
			return sum;
		}

		/// Gets the index of a given Coord element
		template<int Dim> NIHILUS_HOST_DEVICE Index& at() {
			return idx[Dim];
		}

		/// Access via index; may limit unrolling potential
		NIHILUS_HOST_DEVICE
		Index& at(int dim) {
			return idx[dim];
		}

		/// Gets the index of a given Coord element
		template<int Dim> NIHILUS_HOST_DEVICE Index const& at() const {
			return idx[Dim];
		}

		/// Access via index; may limit unrolling potential
		NIHILUS_HOST_DEVICE
		Index const& at(int dim) const {
			return idx[dim];
		}

		/// Determines if two Coord<> objects are equal
		NIHILUS_HOST_DEVICE
		bool operator==(Coord const& b) const {
			bool equal = true;
			for (int i = 0; equal && i < kRank; ++i) {
				equal = (idx[i] == b.idx[i]);
			}
			return equal;
		}

		/// Not equal
		NIHILUS_HOST_DEVICE
		bool operator!=(Coord const& b) const {
			return !(*this == b);
		}

		/// Clamps a coordinate to a range specified by maximum and minimum values
		NIHILUS_HOST_DEVICE
		Coord& clamp(Coord const& max, Coord const& min = Coord()) {
			for (int i = 0; i < kRank; ++i) {
				idx[i] = __NV_STD_MAX(__NV_STD_MIN(idx[i], max.idx[i]), min.idx[i]);
			}
			return *this;
		}

		/// Returns the sum of all elements
		NIHILUS_HOST_DEVICE
		Index sum() const {
			Index sum_(idx[0]);
			for (int i = 1; i < kRank; ++i) {
				sum_ += idx[i];
			}
			return sum_;
		}

		/// Returns the product of all elements
		NIHILUS_HOST_DEVICE
		LongIndex product() const {
			LongIndex product_(idx[0]);
			for (int i = 1; i < kRank; ++i) {
				product_ *= idx[i];
			}
			return product_;
		}

		/// Less than operator
		NIHILUS_HOST_DEVICE
		bool operator<(Coord const& b) const {
			for (int i = 0; i < kRank; ++i) {
				if (!(idx[i] < b[i])) {
					return false;
				}
			}
			return true;
		}

		/// Less than or equals operator
		NIHILUS_HOST_DEVICE
		bool operator<=(Coord const& b) const {
			for (int i = 0; i < kRank; ++i) {
				if (!(idx[i] <= b[i])) {
					return false;
				}
			}
			return true;
		}

		/// Greater than operator
		NIHILUS_HOST_DEVICE
		bool operator>(Coord const& b) const {
			return !(*this <= b);
		}

		/// Greater than or equals operator
		NIHILUS_HOST_DEVICE
		bool operator>=(Coord const& b) const {
			return !(*this < b);
		}
	};

	template<uint64_t index> using tag = std::integral_constant<uint64_t, index>;

	template<uint64_t Rank_, uint64_t... dimensions> struct constexpresh_coord;

	template<uint64_t M_new> struct constexpresh_coord<1, M_new> {
		static constexpr uint64_t kRank = 1ull;
		using Index = uint64_t;
		using LongIndex = int64_t;
		static constexpr uint64_t M{ M_new };

		NIHILUS_HOST_DEVICE
		explicit constexpresh_coord() {
		}

		NIHILUS_HOST_DEVICE constexpresh_coord(constexpresh_coord<1, M> other) {
		}

		NIHILUS_HOST_DEVICE
		consteval uint64_t min_dim_index() const {
			return 0;
		}

		NIHILUS_HOST_DEVICE
		consteval uint64_t max_dim_index() const {
			return 0;
		}

		NIHILUS_HOST_DEVICE
		consteval explicit operator bool() const {
			if (M) {
				return true;
			}
			return false;
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator!() const {
			if (M) {
				return false;
			}
			return true;
		}

		template<uint64_t M_other> NIHILUS_HOST_DEVICE consteval constexpresh_coord<1, M / M_other> operator/(constexpresh_coord<1, M_other> const& b) const {
			static_assert(M % M_other == 0, "M must be divisible by M_other");
			return constexpresh_coord<1, M / M_other>{};
		}

		template<uint64_t M_other> NIHILUS_HOST_DEVICE consteval constexpresh_coord<1, M + M_other> operator+(constexpresh_coord<1, M_other> const& b) const {
			return constexpresh_coord<1, M + M_other>{};
		}

		template<uint64_t M_other> NIHILUS_HOST_DEVICE consteval constexpresh_coord<1, M - M_other> operator-(constexpresh_coord<1, M_other> const& b) const {
			return constexpresh_coord<1, M - M_other>{};
		}

		template<uint64_t M_other> NIHILUS_HOST_DEVICE consteval constexpresh_coord<1, M * M_other> operator*(constexpresh_coord<1, M_other> const& b) const {
			return constexpresh_coord<1, M * M_other>{};
		}

		template<uint64_t M_other> NIHILUS_HOST_DEVICE consteval decltype(auto) operator+=(constexpresh_coord<1, M_other> const& b) {
			return *this + b;
		}

		template<uint64_t M_other> NIHILUS_HOST_DEVICE consteval decltype(auto) operator-=(constexpresh_coord<1, M_other> const& b) {
			return *this - b;
		}

		template<uint64_t M_other> NIHILUS_HOST_DEVICE consteval decltype(auto) operator*=(constexpresh_coord<1, M_other> const& b) {
			return *this * b;
		}

		template<uint64_t M_other> NIHILUS_HOST_DEVICE consteval decltype(auto) operator/=(constexpresh_coord<1, M_other> const& b) {
			return *this / b;
		}

		template<uint64_t index> NIHILUS_HOST_DEVICE consteval decltype(auto) operator[](tag<index> index_new) {
			if constexpr (index == 0) {
				return M;
			}
		}

		NIHILUS_HOST_DEVICE
		consteval LongIndex dot(constexpresh_coord const& b, LongIndex sum = LongIndex(0)) const {
			sum += M * b.M;
			return sum;
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator==(constexpresh_coord const& b) const {
			return (M == b.M);
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator!=(constexpresh_coord const& b) const {
			return !(*this == b);
		}

		template<uint64_t M1, uint64_t M2>
		NIHILUS_HOST_DEVICE consteval decltype(auto) clamp(constexpresh_coord<1, M1> const& max, constexpresh_coord<1, M2> const& min = constexpresh_coord<1, M2>()) {
			constexpr uint64_t M_clamped = __NV_STD_MAX(__NV_STD_MIN(M, max.M), min.M);
			return constexpresh_coord<1, M_clamped>{};
		}

		NIHILUS_HOST_DEVICE
		consteval Index sum() const {
			return M;
		}

		NIHILUS_HOST_DEVICE
		consteval LongIndex product() const {
			return M;
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator<(constexpresh_coord const& b) const {
			return (M < b.M);
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator<=(constexpresh_coord const& b) const {
			return (M <= b.M);
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator>(constexpresh_coord const& b) const {
			return !(*this <= b);
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator>=(constexpresh_coord const& b) const {
			return !(*this < b);
		}
	};

	template<uint64_t M_new, uint64_t K_new> struct constexpresh_coord<2, M_new, K_new> {
		static constexpr uint64_t kRank = 2ull;
		using Index = uint64_t;
		using LongIndex = int64_t;
		static constexpr uint64_t M{ M_new };
		static constexpr uint64_t K{ K_new };

		NIHILUS_HOST_DEVICE
		explicit constexpresh_coord() {
		}

		NIHILUS_HOST_DEVICE constexpresh_coord(constexpresh_coord<2, M, K> other) {
		}

		NIHILUS_HOST_DEVICE
		consteval uint64_t min_dim_index() const {
			uint64_t i			  = 0;
			uint64_t lowest_value = M;
			if (K < lowest_value) {
				i			 = 1;
				lowest_value = K;
			}
			return i;
		}

		NIHILUS_HOST_DEVICE
		consteval uint64_t max_dim_index() const {
			uint64_t i			  = 0;
			uint64_t lowest_value = M;
			if (K > lowest_value) {
				i			 = 1;
				lowest_value = K;
			}
			return i;
		}

		NIHILUS_HOST_DEVICE
		consteval explicit operator bool() const {
			if (M) {
				return true;
			}
			if (K) {
				return true;
			}
			return false;
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator!() const {
			if (M) {
				return false;
			}
			if (K) {
				return false;
			}
			return true;
		}

		template<uint64_t M_other, uint64_t K_other>
		NIHILUS_HOST_DEVICE consteval constexpresh_coord<2, M / M_other, K / K_other> operator/(constexpresh_coord<2, M_other, K_other> const& b) const {
			static_assert(M % M_other == 0, "M must be divisible by M_other");
			static_assert(K % K_other == 0, "K must be divisible by K_other");
			return constexpresh_coord<2, M / M_other, K / K_other>{};
		}

		template<uint64_t M_other, uint64_t K_other>
		NIHILUS_HOST_DEVICE consteval constexpresh_coord<2, M + M_other, K + K_other> operator+(constexpresh_coord<2, M_other, K_other> const& b) const {
			return constexpresh_coord<2, M + M_other, K + K_other>{};
		}

		template<uint64_t M_other, uint64_t K_other>
		NIHILUS_HOST_DEVICE consteval constexpresh_coord<2, M - M_other, K - K_other> operator-(constexpresh_coord<2, M_other, K_other> const& b) const {
			return constexpresh_coord<2, M - M_other, K - K_other>{};
		}

		template<uint64_t M_other, uint64_t K_other>
		NIHILUS_HOST_DEVICE consteval constexpresh_coord<2, M * M_other, K * K_other> operator*(constexpresh_coord<2, M_other, K_other> const& b) const {
			return constexpresh_coord<2, M * M_other, K * K_other>{};
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE consteval decltype(auto) operator+=(constexpresh_coord<2, M_other, K_other> const& b) {
			return *this + b;
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE consteval decltype(auto) operator-=(constexpresh_coord<2, M_other, K_other> const& b) {
			return *this - b;
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE consteval decltype(auto) operator*=(constexpresh_coord<2, M_other, K_other> const& b) {
			return *this * b;
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE consteval decltype(auto) operator/=(constexpresh_coord<2, M_other, K_other> const& b) {
			return *this / b;
		}

		template<uint64_t index> NIHILUS_HOST_DEVICE consteval decltype(auto) operator[](tag<index> index_new) {
			if constexpr (index == 0) {
				return M;
			} else if constexpr (index == 1) {
				return K;
			}
		}

		NIHILUS_HOST_DEVICE
		consteval LongIndex dot(constexpresh_coord const& b, LongIndex sum = LongIndex(0)) const {
			sum += M * b.M;
			sum += K * b.K;
			return sum;
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator==(constexpresh_coord const& b) const {
			return (M == b.M) && (K == b.K);
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator!=(constexpresh_coord const& b) const {
			return !(*this == b);
		}

		template<uint64_t M1, uint64_t K1, uint64_t M2, uint64_t K2>
		NIHILUS_HOST_DEVICE consteval decltype(auto) clamp(constexpresh_coord<2, M1, K1> const& max, constexpresh_coord<2, M2, K2> const& min = constexpresh_coord<2, M2, K2>()) {
			constexpr uint64_t M_clamped = __NV_STD_MAX(__NV_STD_MIN(M, max.M), min.M);
			constexpr uint64_t K_clamped = __NV_STD_MAX(__NV_STD_MIN(K, max.K), min.K);
			return constexpresh_coord<2, M_clamped, K_clamped>{};
		}

		NIHILUS_HOST_DEVICE
		consteval Index sum() const {
			Index sum_(M);
			sum_ += K;
			return sum_;
		}

		NIHILUS_HOST_DEVICE
		consteval LongIndex product() const {
			LongIndex product_(M);
			product_ *= K;
			return product_;
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator<(constexpresh_coord const& b) const {
			return (M < b.M) && (K < b.K);
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator<=(constexpresh_coord const& b) const {
			return (M <= b.M) && (K <= b.K);
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator>(constexpresh_coord const& b) const {
			return !(*this <= b);
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator>=(constexpresh_coord const& b) const {
			return !(*this < b);
		}
	};

	template<uint64_t M_new> struct constexpresh_coord<2, M_new> {
		static constexpr uint64_t kRank = 2ull;
		using Index = uint64_t;
		using LongIndex = int64_t;
		static constexpr uint64_t M{ M_new };
		mutable uint64_t N{};

		NIHILUS_HOST_DEVICE
		explicit constexpresh_coord() {
		}

		NIHILUS_HOST_DEVICE constexpresh_coord(const constexpresh_coord<2, M>& other) : N{ other.N } {}

		NIHILUS_HOST_DEVICE constexpresh_coord(Index other) : N{ other } {}

		NIHILUS_HOST_DEVICE operator Coord<2>() const {
			int32_t values[2]{ static_cast<int32_t>(M), static_cast<int32_t>(N) };
			return Coord<2>{ values };
		}

		NIHILUS_HOST_DEVICE
		consteval uint64_t min_dim_index() const {
			uint64_t i			  = 0;
			uint64_t lowest_value = M;
			if (N < lowest_value) {
				i			 = 1;
				lowest_value = N;
			}
			return i;
		}

		NIHILUS_HOST_DEVICE
		consteval uint64_t max_dim_index() const {
			uint64_t i			  = 0;
			uint64_t lowest_value = M;
			if (N > lowest_value) {
				i			 = 1;
				lowest_value = N;
			}
			return i;
		}

		NIHILUS_HOST_DEVICE
		consteval explicit operator bool() const {
			if (M) {
				return true;
			}
			if (N) {
				return true;
			}
			return false;
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator!() const {
			if (M) {
				return false;
			}
			if (N) {
				return false;
			}
			return true;
		}

		template<uint64_t M_other>
		NIHILUS_HOST_DEVICE consteval constexpresh_coord<2, M / M_other> operator/(constexpresh_coord<2, M_other> const& b) const {
			static_assert(M % M_other == 0, "M must be divisible by M_other");
			return constexpresh_coord<2, M / M_other>{ N / b.N };
		}

		template<uint64_t M_other>
		NIHILUS_HOST_DEVICE consteval constexpresh_coord<2, M + M_other> operator+(constexpresh_coord<2, M_other> const& b) const {
			return constexpresh_coord<2, M + M_other>{ N + b.N };
		}

		template<uint64_t M_other>
		NIHILUS_HOST_DEVICE consteval constexpresh_coord<2, M - M_other> operator-(constexpresh_coord<2, M_other> const& b) const {
			return constexpresh_coord<2, M - M_other>{ N - b.N };
		}

		template<uint64_t M_other>
		NIHILUS_HOST_DEVICE consteval constexpresh_coord<2, M * M_other> operator*(constexpresh_coord<2, M_other> const& b) const {
			return constexpresh_coord<2, M * M_other>{ N * b.N };
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE consteval decltype(auto) operator+=(constexpresh_coord<2, M_other> const& b) {
			return *this + b;
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE consteval decltype(auto) operator-=(constexpresh_coord<2, M_other> const& b) {
			return *this - b;
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE consteval decltype(auto) operator*=(constexpresh_coord<2, M_other> const& b) {
			return *this * b;
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE consteval decltype(auto) operator/=(constexpresh_coord<2, M_other> const& b) {
			return *this / b;
		}

		template<uint64_t index> NIHILUS_HOST_DEVICE consteval decltype(auto) operator[](tag<index> index_new) {
			if constexpr (index == 0) {
				return M;
			} else if constexpr (index == 1) {
				return N;
			}
		}

		NIHILUS_HOST_DEVICE
		consteval LongIndex dot(constexpresh_coord const& b, LongIndex sum = LongIndex(0)) const {
			sum += M * b.M;
			sum += N * b.N;
			return sum;
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator==(constexpresh_coord const& b) const {
			return (M == b.M) && (N == b.N);
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator!=(constexpresh_coord const& b) const {
			return !(*this == b);
		}

		template<uint64_t M1, uint64_t K1, uint64_t M2, uint64_t K2>
		NIHILUS_HOST_DEVICE consteval decltype(auto) clamp(constexpresh_coord<2, M1, K1> const& max, constexpresh_coord<2, M2, K2> const& min = constexpresh_coord<2, M2, K2>()) {
			constexpr uint64_t M_clamped = __NV_STD_MAX(__NV_STD_MIN(M, max.M), min.M);
			constexpr uint64_t K_clamped = __NV_STD_MAX(__NV_STD_MIN(N, max.N), min.N);
			return constexpresh_coord<2, M_clamped, K_clamped>{};
		}

		NIHILUS_HOST_DEVICE
		consteval Index sum() const {
			Index sum_(M);
			sum_ += N;
			return sum_;
		}

		NIHILUS_HOST_DEVICE
		consteval LongIndex product() const {
			LongIndex product_(M);
			product_ *= N;
			return product_;
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator<(constexpresh_coord const& b) const {
			return (M < b.M) && (N < b.N);
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator<=(constexpresh_coord const& b) const {
			return (M <= b.M) && (N <= b.N);
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator>(constexpresh_coord const& b) const {
			return !(*this <= b);
		}

		NIHILUS_HOST_DEVICE
		consteval bool operator>=(constexpresh_coord const& b) const {
			return !(*this < b);
		}
	};

	template<uint64_t M_new, uint64_t K_new> struct constexpresh_coord<3, M_new, K_new> {
		static constexpr uint64_t kRank = 3ull;
		using Index = uint64_t;
		using LongIndex = int64_t;
		static constexpr uint64_t M{ M_new };
		static constexpr uint64_t K{ K_new };
		mutable uint64_t N{};

		NIHILUS_HOST_DEVICE
		constexpr explicit constexpresh_coord(Index index = 0) {
			N = index;
		}

		NIHILUS_HOST_DEVICE constexpr constexpresh_coord(const constexpresh_coord& other){};

		NIHILUS_HOST_DEVICE
		constexpr uint64_t min_dim_index() const {
			uint64_t i			  = 0;
			uint64_t lowest_value = M;
			if (K < lowest_value) {
				i			 = 1;
				lowest_value = K;
			}
			if (N < lowest_value) {
				i			 = 2;
				lowest_value = N;
			}
			return i;
		}

		NIHILUS_HOST_DEVICE
		constexpr uint64_t max_dim_index() const {
			uint64_t i			  = 0;
			uint64_t lowest_value = M;
			if (K > lowest_value) {
				i			 = 1;
				lowest_value = K;
			}
			if (N > lowest_value) {
				i			 = 2;
				lowest_value = N;
			}
			return i;
		}

		NIHILUS_HOST_DEVICE
		const Index m() const {
			return M;
		}

		NIHILUS_HOST_DEVICE
		explicit operator bool() const {
			if (M) {
				return true;
			}
			if (K) {
				return true;
			}
			if (N) {
				return true;
			}
			return false;
		}

		NIHILUS_HOST_DEVICE
		bool operator!() const {
			if (M) {
				return false;
			}
			if (K) {
				return false;
			}
			if (N) {
				return false;
			}
			return true;
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE decltype(auto) operator/(constexpresh_coord<3, M_other, K_other> const& b) const {
			static_assert(M % M_other == 0, "M must be divisible by M_other");
			static_assert(K % K_other == 0, "K must be divisible by K_other");
			return constexpresh_coord<3, M / M_other, K / K_other>{ N / b.N };
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE decltype(auto) operator+(constexpresh_coord<3, M_other, K_other> const& b) const {
			return constexpresh_coord<3, M + M_other, K + K_other>{ N + b.N };
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE decltype(auto) operator-(constexpresh_coord<3, M_other, K_other> const& b) const {
			return constexpresh_coord<3, M - M_other, K - K_other>{ N - b.N };
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE decltype(auto) operator*(constexpresh_coord<3, M_other, K_other> const& b) const {
			return constexpresh_coord<3, M * M_other, K * K_other>{ N * b.N };
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE decltype(auto) operator+=(constexpresh_coord<3, M_other, K_other> const& b) {
			return *this + b;
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE decltype(auto) operator-=(constexpresh_coord<3, M_other, K_other> const& b) {
			return *this - b;
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE decltype(auto) operator*=(constexpresh_coord<3, M_other, K_other> const& b) {
			return *this * b;
		}

		template<uint64_t M_other, uint64_t K_other> NIHILUS_HOST_DEVICE decltype(auto) operator/=(constexpresh_coord<3, M_other, K_other> const& b) {
			return *this / b;
		}

		template<uint64_t index> NIHILUS_HOST_DEVICE constexpr auto& operator[](tag<index> index_new) {
			if constexpr (index == 0) {
				return M;
			} else if constexpr (index == 1) {
				return K;
			} else {
				return N;
			}
		}

		NIHILUS_HOST_DEVICE
		LongIndex dot(constexpresh_coord const& b, LongIndex sum = LongIndex(0)) const {
			sum += M * b.M;
			sum += K * b.K;
			sum += N * b.N;
			return sum;
		}

		NIHILUS_HOST_DEVICE
		bool operator==(constexpresh_coord const& b) const {
			return (M == b.M) && (K == b.K) && (N == b.N);
		}

		NIHILUS_HOST_DEVICE
		bool operator!=(constexpresh_coord const& b) const {
			return !(*this == b);
		}

		template<uint64_t M1, uint64_t K1, uint64_t M2, uint64_t K2>
		NIHILUS_HOST_DEVICE decltype(auto) clamp(constexpresh_coord<3, M1, K1> const& max, constexpresh_coord<3, M2, K2> const& min = constexpresh_coord<3, M2, K2>()) {
			static constexpr uint64_t M_clamped = __NV_STD_MAX(__NV_STD_MIN(M, max.M), min.M);
			static constexpr uint64_t K_clamped = __NV_STD_MAX(__NV_STD_MIN(K, max.K), min.K);
			uint64_t N_clamped					= __NV_STD_MAX(__NV_STD_MIN(N, max.N), min.N);
			return constexpresh_coord<3, M_clamped, K_clamped>{ N_clamped };
		}

		NIHILUS_HOST_DEVICE
		Index sum() const {
			Index sum_(M);
			sum_ += K;
			sum_ += N;
			return sum_;
		}

		NIHILUS_HOST_DEVICE
		LongIndex product() const {
			LongIndex product_(M);
			product_ *= K;
			product_ *= N;
			return product_;
		}

		NIHILUS_HOST_DEVICE
		bool operator<(constexpresh_coord const& b) const {
			return (M < b.M) && (K < b.K) && (N < b.N);
		}

		NIHILUS_HOST_DEVICE
		bool operator<=(constexpresh_coord const& b) const {
			return (M <= b.M) && (K <= b.K) && (N <= b.N);
		}

		NIHILUS_HOST_DEVICE
		bool operator>(constexpresh_coord const& b) const {
			return !(*this <= b);
		}

		NIHILUS_HOST_DEVICE
		bool operator>=(constexpresh_coord const& b) const {
			return !(*this < b);
		}
	};

}// namespace nihilus_gemm

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace nihilus_gemm {

	template<uint64_t M, uint64_t K> NIHILUS_HOST_DEVICE constexpresh_coord<3, M, K> make_Coord(uint64_t _2) {
		return constexpresh_coord<3, M, K>{ _2 };
	}

	/// Scalar multiplication
	template<int Rank, typename Index> NIHILUS_HOST_DEVICE Coord<Rank, Index> operator*(Index s, Coord<Rank, Index> coord) {
		NIHILUS_PRAGMA_UNROLL
		for (int i = 0; i < Rank; ++i) {
			coord[i] *= s;
		}
		return coord;
	}

	/// Scalar multiplication
	template<int Rank, typename Index> NIHILUS_HOST_DEVICE Coord<Rank, Index> operator*(Coord<Rank, Index> coord, Index s) {
		NIHILUS_PRAGMA_UNROLL
		for (int i = 0; i < Rank; ++i) {
			coord[i] *= s;
		}
		return coord;
	}

	/// Scalar division
	template<int Rank, typename Index> NIHILUS_HOST_DEVICE Coord<Rank, Index> operator/(Index s, Coord<Rank, Index> coord) {
		NIHILUS_PRAGMA_UNROLL
		for (int i = 0; i < Rank; ++i) {
			coord[i] = s / coord[i];
		}
		return coord;
	}

	/// Scalar division
	template<int Rank, typename Index> NIHILUS_HOST_DEVICE Coord<Rank, Index> operator/(Coord<Rank, Index> coord, Index s) {
		NIHILUS_PRAGMA_UNROLL
		for (int i = 0; i < Rank; ++i) {
			coord[i] /= s;
		}
		return coord;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	// Integer-valued make_Coord
	//
	////////////////////////////////////////////////////////////////////////////////////////////////////

	/// Helper to make a 1-element coordinate
	template<typename T> NIHILUS_HOST_DEVICE constexpr Coord<1, T> make_Coord(T _0) {
		T values[1] = { _0 };
		return Coord<1, T>(values);
	}

	/// Helper to make a 2-element coordinate
	template<typename T> NIHILUS_HOST_DEVICE constexpr Coord<2, T> make_Coord(T _0, T _1) {
		T values[2] = { _0, _1 };
		return Coord<2, T>(values);
	}

	/// Helper to make a 3-element coordinate
	template<typename T> NIHILUS_HOST_DEVICE constexpr Coord<3, T> make_Coord(T _0, T _1, T _2) {
		T values[3] = { _0, _1, _2 };
		return Coord<3, T>(values);
	}

	/// Helper to make a 4-element coordinate
	template<typename T> NIHILUS_HOST_DEVICE constexpr Coord<4, T> make_Coord(T _0, T _1, T _2, T _3) {
		T values[4] = { _0, _1, _2, _3 };
		return Coord<4, T>(values);
	}

	/// Helper to make a 5-element coordinate
	template<typename T> NIHILUS_HOST_DEVICE constexpr Coord<5, T> make_Coord(T _0, T _1, T _2, T _3, T _4) {
		T values[5] = { _0, _1, _2, _3, _4 };
		return Coord<5, T>(values);
	}

	/// Helper to make a 1-element coordinate
	template<int N, typename T> NIHILUS_HOST_DEVICE Coord<N, T> make_Coord_with_padding(T _0) {
		Coord<N, T> coord;

		NIHILUS_PRAGMA_UNROLL
		for (int i = N - 1; i > 0; --i) {
			coord[i] = 0;
		}

		coord[0] = _0;

		return coord;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////

}// namespace nihilus_gemm
