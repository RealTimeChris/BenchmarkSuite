#include <BnchSwt/BenchmarkSuite.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass_new/detail/helper_macros.hpp>

static constexpr uint64_t total_iterations{ 6 };
static constexpr uint64_t measured_iterations{ 3 };

struct OpMultiplyAdd {};

struct OpMultiplyAddSaturate {};

struct OpMultiplyAddFastBF16 {};

struct OpMultiplyAddFastF16 {};

struct OpMultiplyAddMixedInputUpcast {};

struct OpMultiplyAddFastF32 {};

struct OpMultiplyAddComplexFastF32 {};

struct OpMultiplyAddFastAccum;

struct OpMultiplyAddComplex {};

struct OpMultiplyAddGaussianComplex {};

struct OpXorPopc {};

struct OpAndPopc {};

struct OpClassSimt {};

struct OpClassTensorOp {};

struct OpClassWmmaTensorOp {};

struct OpClassSparseTensorOp {};

struct OpClassBlockScaledTensorOp {};

struct OpClassBlockScaledSparseTensorOp {};

template<int Rank_, typename Index_ = int, typename LongIndex_ = int64_t> struct Coord {
  public:
	static constexpr int kRank = Rank_;

	using Index = Index_;

	using LongIndex = LongIndex_;

  private:
	Index idx[kRank];

  public:
	CUTLASS_HOST_DEVICE explicit Coord(Index value = Index(0)) {
		for (int i = 0; i < kRank; ++i) {
			idx[i] = value;
		}
	}

	CUTLASS_HOST_DEVICE Coord(Index const (&_idx)[kRank]) {
		for (int i = 0; i < kRank; ++i) {
			idx[i] = _idx[i];
		}
	}

	template<int R, typename I, typename L> CUTLASS_HOST_DEVICE Coord(Coord<R, I, L> other) {
		for (int i = 0; i < kRank; ++i) {
			idx[i] = other[i];
		}
	}

	template<int Slice> CUTLASS_HOST_DEVICE Coord<Slice, Index, LongIndex> slice(int start = 0, Index identity = 0) const {
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

	CUTLASS_HOST_DEVICE int min_dim_index() const {
		int i = 0;
		for (int j = 1; j < kRank; ++j) {
			if (idx[j] < idx[i]) {
				i = j;
			}
		}
		return i;
	}

	CUTLASS_HOST_DEVICE int max_dim_index() const {
		int i = 0;
		for (int j = 1; j < kRank; ++j) {
			if (idx[j] > idx[i]) {
				i = j;
			}
		}
		return i;
	}

	CUTLASS_HOST_DEVICE explicit operator bool() const {
		for (int i = 0; i < kRank; ++i) {
			if (idx[i]) {
				return true;
			}
		}
		return false;
	}

	CUTLASS_HOST_DEVICE bool operator!() const {
		for (int i = 0; i < kRank; ++i) {
			if (idx[i]) {
				return false;
			}
		}
		return true;
	}

	CUTLASS_HOST_DEVICE Coord operator+(Coord const& b) const {
		Coord c;
		for (int i = 0; i < kRank; ++i) {
			c.idx[i] = idx[i] + b.idx[i];
		}
		return c;
	}

	CUTLASS_HOST_DEVICE Coord operator-(Coord const& b) const {
		Coord c;
		for (int i = 0; i < kRank; ++i) {
			c.idx[i] = idx[i] - b.idx[i];
		}
		return c;
	}

	CUTLASS_HOST_DEVICE Coord operator*(Coord const& b) const {
		Coord c;
		for (int i = 0; i < kRank; ++i) {
			c.idx[i] = idx[i] * b.idx[i];
		}
		return c;
	}

	CUTLASS_HOST_DEVICE Coord operator/(Coord const& b) const {
		Coord c;
		for (int i = 0; i < kRank; ++i) {
			c.idx[i] = idx[i] / b.idx[i];
		}
		return c;
	}

	CUTLASS_HOST_DEVICE Coord& operator+=(Coord const& b) {
		for (int i = 0; i < kRank; ++i) {
			idx[i] += b.idx[i];
		}
		return *this;
	}

	CUTLASS_HOST_DEVICE Coord& operator-=(Coord const& b) {
		for (int i = 0; i < kRank; ++i) {
			idx[i] -= b.idx[i];
		}
		return *this;
	}

	CUTLASS_HOST_DEVICE Coord& operator*=(Coord const& b) {
		for (int i = 0; i < kRank; ++i) {
			idx[i] *= b.idx[i];
		}
		return *this;
	}

	CUTLASS_HOST_DEVICE Coord& operator/=(Coord const& b) {
		for (int i = 0; i < kRank; ++i) {
			idx[i] /= b.idx[i];
		}
		return *this;
	}

	CUTLASS_HOST_DEVICE Index& operator[](int dim) {
		return idx[dim];
	}

	CUTLASS_HOST_DEVICE Index const& operator[](int dim) const {
		return idx[dim];
	}

	CUTLASS_HOST_DEVICE LongIndex dot(Coord const& b, LongIndex sum = LongIndex(0)) const {
		for (int i = 0; i < kRank; ++i) {
			sum += idx[i] * b.idx[i];
		}
		return sum;
	}

	template<int Dim> CUTLASS_HOST_DEVICE Index& at() {
		return idx[Dim];
	}

	CUTLASS_HOST_DEVICE Index& at(int dim) {
		return idx[dim];
	}

	template<int Dim> CUTLASS_HOST_DEVICE Index const& at() const {
		return idx[Dim];
	}

	CUTLASS_HOST_DEVICE Index const& at(int dim) const {
		return idx[dim];
	}

	CUTLASS_HOST_DEVICE bool operator==(Coord const& b) const {
		bool equal = true;
		for (int i = 0; equal && i < kRank; ++i) {
			equal = (idx[i] == b.idx[i]);
		}
		return equal;
	}

	CUTLASS_HOST_DEVICE bool operator!=(Coord const& b) const {
		return !(*this == b);
	}

	CUTLASS_HOST_DEVICE Coord& clamp(Coord const& max, Coord const& min = Coord()) {
		for (int i = 0; i < kRank; ++i) {
			idx[i] = __NV_STD_MAX(__NV_STD_MIN(idx[i], max.idx[i]), min.idx[i]);
		}
		return *this;
	}

	CUTLASS_HOST_DEVICE Index sum() const {
		Index sum_(idx[0]);
		for (int i = 1; i < kRank; ++i) {
			sum_ += idx[i];
		}
		return sum_;
	}

	CUTLASS_HOST_DEVICE LongIndex product() const {
		LongIndex product_(idx[0]);
		for (int i = 1; i < kRank; ++i) {
			product_ *= idx[i];
		}
		return product_;
	}

	CUTLASS_HOST_DEVICE bool operator<(Coord const& b) const {
		for (int i = 0; i < kRank; ++i) {
			if (!(idx[i] < b[i])) {
				return false;
			}
		}
		return true;
	}

	CUTLASS_HOST_DEVICE bool operator<=(Coord const& b) const {
		for (int i = 0; i < kRank; ++i) {
			if (!(idx[i] <= b[i])) {
				return false;
			}
		}
		return true;
	}

	CUTLASS_HOST_DEVICE bool operator>(Coord const& b) const {
		return !(*this <= b);
	}

	CUTLASS_HOST_DEVICE bool operator>=(Coord const& b) const {
		return !(*this < b);
	}
};


template<int M = 1, int N = 1, int K = 1> struct GemmShape {
	static constexpr int kM = M;
	static constexpr int kN = N;
	static constexpr int kK = K;

	static constexpr int kMN  = M * N;
	static constexpr int kMK  = M * K;
	static constexpr int kKN  = N * K;
	static constexpr int kMNK = M * N * K;

	static constexpr int kCount = kMNK;


	CUTLASS_HOST_DEVICE static Coord<3> toCoord() {
		return make_Coord(kM, kN, kK);
	}
};

template<typename Shape> using GemmShapeTranspose = GemmShape<Shape::kN, Shape::kM, Shape::kK>;

template<typename T> CUTLASS_HOST_DEVICE Coord<3, T> make_Coord(T _0, T _1, T _2) {
	T values[3] = { _0, _1, _2 };
	return Coord<3, T>(values);
}

template<typename T> CUTLASS_HOST_DEVICE Coord<2, T> make_Coord(T _0, T _1) {
	T values[2] = { _0, _1 };
	return Coord<2, T>(values);
}


struct GemmCoord : public Coord<3, int> {
	typedef int Index;

	typedef Coord<3, Index> Base;

	static constexpr int kM = 0;

	static constexpr int kN = 1;

	static constexpr int kK = 2;


	CUTLASS_HOST_DEVICE GemmCoord() {
	}

	CUTLASS_HOST_DEVICE GemmCoord(Coord<3, Index> const& coord) : Base(make_Coord(coord[0], coord[1], coord[2])) {
	}

	CUTLASS_HOST_DEVICE GemmCoord(Index m, Index n, Index k) : Base(make_Coord(m, n, k)) {
	}

	CUTLASS_HOST_DEVICE Index const& m() const {
		return this->at(kM);
	}

	CUTLASS_HOST_DEVICE Index& m() {
		return this->at(kM);
	}

	CUTLASS_HOST_DEVICE Index const& n() const {
		return this->at(kN);
	}

	CUTLASS_HOST_DEVICE Index& n() {
		return this->at(kN);
	}

	CUTLASS_HOST_DEVICE Index const& k() const {
		return this->at(kK);
	}

	CUTLASS_HOST_DEVICE Index& k() {
		return this->at(kK);
	}

	CUTLASS_HOST_DEVICE Coord<3> mnk() const {
		return make_Coord(m(), n(), k());
	}

	CUTLASS_HOST_DEVICE Coord<3> knm() const {
		return make_Coord(k(), n(), m());
	}

	CUTLASS_HOST_DEVICE Coord<2> nm() const {
		return make_Coord(n(), m());
	}

	CUTLASS_HOST_DEVICE Coord<2> mn() const {
		return make_Coord(m(), n());
	}

	CUTLASS_HOST_DEVICE Coord<2> mk() const {
		return make_Coord(m(), k());
	}

	CUTLASS_HOST_DEVICE Coord<2> km() const {
		return make_Coord(k(), m());
	}

	CUTLASS_HOST_DEVICE Coord<2> nk() const {
		return make_Coord(n(), k());
	}

	CUTLASS_HOST_DEVICE Coord<2> kn() const {
		return make_Coord(k(), n());
	}


	CUTLASS_HOST_DEVICE GemmCoord operator+(Base const& b) const {
		return GemmCoord(Base::operator+(b));
	}

	CUTLASS_HOST_DEVICE GemmCoord operator-(Base const& b) const {
		return GemmCoord(Base::operator-(b));
	}

	CUTLASS_HOST_DEVICE GemmCoord operator*(Base const& b) const {
		return GemmCoord(Base::operator*(b));
	}

	CUTLASS_HOST_DEVICE GemmCoord operator/(Base const& b) const {
		return GemmCoord(Base::operator/(b));
	}

	CUTLASS_HOST_DEVICE GemmCoord& operator+=(Base const& b) {
		Base::operator+=(b);
		return *this;
	}

	CUTLASS_HOST_DEVICE GemmCoord& operator-=(Base const& b) {
		Base::operator-=(b);
		return *this;
	}

	CUTLASS_HOST_DEVICE GemmCoord& operator*=(Base const& b) {
		Base::operator*=(b);
		return *this;
	}

	CUTLASS_HOST_DEVICE GemmCoord& operator/=(Base const& b) {
		Base::operator/=(b);
		return *this;
	}
};

enum class FloatRoundStyle {
	round_indeterminate,
	round_toward_zero,
	round_to_nearest,
	round_to_nearest_satfinite,
	round_toward_infinity,
	round_toward_neg_infinity,
	round_half_ulp_truncate,
	round_half_ulp_trunc_dntz
};

struct ScaleType {
	enum Kind { Default, NoBetaScaling, OnlyAlphaScaling, PerChannelScaling, OnlyAlphaPerChannelScaling, Nothing };
};

template<typename T> struct sizeof_bits {
	static constexpr int value = int(sizeof(T) * 8);
};

template<typename T> struct sizeof_bits<T const> : sizeof_bits<T> {};

template<typename T> struct sizeof_bits<T volatile> : sizeof_bits<T> {};

template<typename T> struct sizeof_bits<T const volatile> : sizeof_bits<T> {};

template<> struct sizeof_bits<void> {
	static constexpr int value = 0;
};

template<typename T, int N, bool RegisterSized = sizeof_bits<T>::value >= 32> struct Array;

using true_type	 = std::bool_constant<true>;
using false_type = std::bool_constant<false>;

template<class T> struct is_Array : false_type {};

template<typename T, int N, bool RegisterSized> struct is_Array<Array<T, N, RegisterSized>> : true_type {};

template<typename T> constexpr bool is_Array_v = is_Array<T>::value;
template<typename T, int N, bool RegisterSized> struct sizeof_bits<Array<T, N, RegisterSized>> {
	static constexpr int value = sizeof(Array<T, N, RegisterSized>) * 8;
};


CUTLASS_HOST_DEVICE constexpr bool ispow2(unsigned x) {
	return x && (!(x & (x - 1)));
}

CUTLASS_HOST_DEVICE constexpr unsigned floor_pow_2(unsigned x) {
	return (x == 0 || ispow2(x)) ? x : ((floor_pow_2(x >> 1)) << 1);
}

template<uint64_t index> struct tag : public std::integral_constant<uint64_t, index> {};
template<typename T, int N> struct Array<T, N, true> {
	using Storage = T;

	using Element = T;

	static constexpr size_t kStorageElements = N;

	static constexpr size_t kElements = N;

	typedef T value_type;
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;
	typedef value_type& reference;
	typedef value_type const& const_reference;
	typedef value_type* pointer;
	typedef value_type const* const_pointer;

	class iterator {
		T* ptr_;

	  public:
		CUTLASS_HOST_DEVICE iterator() : ptr_(nullptr) {
		}

		CUTLASS_HOST_DEVICE iterator(T* _ptr) : ptr_(_ptr) {
		}

		CUTLASS_HOST_DEVICE iterator& operator++() {
			++ptr_;
			return *this;
		}

		CUTLASS_HOST_DEVICE iterator& operator--() {
			--ptr_;
			return *this;
		}

		CUTLASS_HOST_DEVICE iterator operator++(int) {
			iterator ret(*this);
			++ptr_;
			return ret;
		}

		CUTLASS_HOST_DEVICE iterator operator--(int) {
			iterator ret(*this);
			--ptr_;
			return ret;
		}

		CUTLASS_HOST_DEVICE T& operator*() const {
			return *ptr_;
		}

		CUTLASS_HOST_DEVICE bool operator==(iterator const& other) const {
			return ptr_ == other.ptr_;
		}

		CUTLASS_HOST_DEVICE bool operator!=(iterator const& other) const {
			return ptr_ != other.ptr_;
		}
	};

	class const_iterator {
		const T* ptr_;

	  public:
		CUTLASS_HOST_DEVICE const_iterator() : ptr_(nullptr) {
		}

		CUTLASS_HOST_DEVICE const_iterator(T const* _ptr) : ptr_(_ptr) {
		}

		CUTLASS_HOST_DEVICE const_iterator& operator++() {
			++ptr_;
			return *this;
		}

		CUTLASS_HOST_DEVICE const_iterator& operator--() {
			--ptr_;
			return *this;
		}

		CUTLASS_HOST_DEVICE const_iterator operator++(int) {
			const_iterator ret(*this);
			++ptr_;
			return ret;
		}

		CUTLASS_HOST_DEVICE const_iterator operator--(int) {
			const_iterator ret(*this);
			--ptr_;
			return ret;
		}

		CUTLASS_HOST_DEVICE T const& operator*() const {
			return *ptr_;
		}

		CUTLASS_HOST_DEVICE bool operator==(const_iterator const& other) const {
			return ptr_ == other.ptr_;
		}

		CUTLASS_HOST_DEVICE bool operator!=(const_iterator const& other) const {
			return ptr_ != other.ptr_;
		}
	};

	class reverse_iterator {
		T* ptr_;

	  public:
		CUTLASS_HOST_DEVICE reverse_iterator() : ptr_(nullptr) {
		}

		CUTLASS_HOST_DEVICE reverse_iterator(T* _ptr) : ptr_(_ptr) {
		}

		CUTLASS_HOST_DEVICE reverse_iterator& operator++() {
			--ptr_;
			return *this;
		}

		CUTLASS_HOST_DEVICE reverse_iterator& operator--() {
			++ptr_;
			return *this;
		}

		CUTLASS_HOST_DEVICE reverse_iterator operator++(int) {
			iterator ret(*this);
			--ptr_;
			return ret;
		}

		CUTLASS_HOST_DEVICE reverse_iterator operator--(int) {
			iterator ret(*this);
			++ptr_;
			return ret;
		}

		CUTLASS_HOST_DEVICE T& operator*() const {
			return *(ptr_ - 1);
		}

		CUTLASS_HOST_DEVICE bool operator==(reverse_iterator const& other) const {
			return ptr_ == other.ptr_;
		}

		CUTLASS_HOST_DEVICE bool operator!=(reverse_iterator const& other) const {
			return ptr_ != other.ptr_;
		}
	};

	class const_reverse_iterator {
		T const* ptr_;

	  public:
		CUTLASS_HOST_DEVICE const_reverse_iterator() : ptr_(nullptr) {
		}

		CUTLASS_HOST_DEVICE const_reverse_iterator(T const* _ptr) : ptr_(_ptr) {
		}

		CUTLASS_HOST_DEVICE const_reverse_iterator& operator++() {
			--ptr_;
			return *this;
		}

		CUTLASS_HOST_DEVICE const_reverse_iterator& operator--() {
			++ptr_;
			return *this;
		}

		CUTLASS_HOST_DEVICE const_reverse_iterator operator++(int) {
			const_reverse_iterator ret(*this);
			--ptr_;
			return ret;
		}

		CUTLASS_HOST_DEVICE const_reverse_iterator operator--(int) {
			const_reverse_iterator ret(*this);
			++ptr_;
			return ret;
		}

		CUTLASS_HOST_DEVICE T const& operator*() const {
			return *(ptr_ - 1);
		}

		CUTLASS_HOST_DEVICE bool operator==(const_iterator const& other) const {
			return ptr_ == other.ptr_;
		}

		CUTLASS_HOST_DEVICE bool operator!=(const_iterator const& other) const {
			return ptr_ != other.ptr_;
		}
	};

	Storage storage[kElements];

	CUTLASS_HOST_DEVICE void clear() {
		fill(T(0));
	}

	CUTLASS_HOST_DEVICE reference at(size_type pos) {
		return reinterpret_cast<reference>(storage[pos]);
	}

	CUTLASS_HOST_DEVICE const_reference at(size_type pos) const {
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

	CUTLASS_HOST_DEVICE reference front() {
		return reinterpret_cast<reference>(storage[0]);
	}

	CUTLASS_HOST_DEVICE const_reference front() const {
		return reinterpret_cast<const_reference>(storage[0]);
	}

	CUTLASS_HOST_DEVICE reference back() {
		return reinterpret_cast<reference>(storage[kStorageElements - 1]);
	}

	CUTLASS_HOST_DEVICE const_reference back() const {
		return reinterpret_cast<const_reference>(storage[kStorageElements - 1]);
	}

	CUTLASS_HOST_DEVICE pointer data() {
		return reinterpret_cast<pointer>(storage);
	}

	CUTLASS_HOST_DEVICE const_pointer data() const {
		return reinterpret_cast<const_pointer>(storage);
	}

	CUTLASS_HOST_DEVICE pointer raw_data() {
		return reinterpret_cast<pointer>(storage);
	}

	CUTLASS_HOST_DEVICE const_pointer raw_data() const {
		return reinterpret_cast<const_pointer>(storage);
	}


	CUTLASS_HOST_DEVICE constexpr bool empty() const {
		return !kElements;
	}

	CUTLASS_HOST_DEVICE constexpr size_type size() const {
		return kElements;
	}

	CUTLASS_HOST_DEVICE constexpr size_type max_size() const {
		return kElements;
	}

	CUTLASS_HOST_DEVICE void fill(T const& value) {
		CUTLASS_PRAGMA_UNROLL
		for (int i = 0; i < int(kElements); ++i) {
			storage[i] = static_cast<Storage>(value);
		}
	}

	CUTLASS_HOST_DEVICE iterator begin() {
		return iterator(storage);
	}

	CUTLASS_HOST_DEVICE const_iterator begin() const {
		return cbegin();
	}

	CUTLASS_HOST_DEVICE const_iterator cbegin() const {
		return const_iterator(storage);
	}

	CUTLASS_HOST_DEVICE iterator end() {
		return iterator(reinterpret_cast<pointer>(storage + kStorageElements));
	}

	CUTLASS_HOST_DEVICE const_iterator end() const {
		return cend();
	}

	CUTLASS_HOST_DEVICE const_iterator cend() const {
		return const_iterator(reinterpret_cast<const_pointer>(storage + kStorageElements));
	}

	CUTLASS_HOST_DEVICE reverse_iterator rbegin() {
		return reverse_iterator(reinterpret_cast<pointer>(storage + kStorageElements));
	}

	CUTLASS_HOST_DEVICE const_reverse_iterator rbegin() const {
		return crbegin();
	}

	CUTLASS_HOST_DEVICE const_reverse_iterator crbegin() const {
		return const_reverse_iterator(reinterpret_cast<const_pointer>(storage + kStorageElements));
	}

	CUTLASS_HOST_DEVICE reverse_iterator rend() {
		return reverse_iterator(reinterpret_cast<pointer>(storage));
	}

	CUTLASS_HOST_DEVICE const_reverse_iterator rend() const {
		return crend();
	}

	CUTLASS_HOST_DEVICE const_reverse_iterator crend() const {
		return const_reverse_iterator(reinterpret_cast<const_pointer>(storage));
	}
};

template<typename T, typename S, FloatRoundStyle Round = FloatRoundStyle::round_to_nearest> struct NumericConverter {
	using result_type							 = T;
	using source_type							 = S;
	static constexpr FloatRoundStyle round_style = Round;

	CUTLASS_HOST_DEVICE static result_type convert(source_type const& s) {
		return static_cast<result_type>(s);
	}

	CUTLASS_HOST_DEVICE result_type operator()(source_type const& s) const {
		return convert(s);
	}
};

struct Identity;
struct Conjugate;
template<typename T, typename S, int N, FloatRoundStyle Round = FloatRoundStyle::round_to_nearest, typename Transform = Identity> struct NumericArrayConverter {
	using result_type							 = Array<T, N>;
	using source_type							 = Array<S, N>;
	static constexpr FloatRoundStyle round_style = Round;

	CUTLASS_HOST_DEVICE static result_type convert(source_type const& s) {
		result_type result;
		NumericConverter<T, S, Round> convert_;

		CUTLASS_PRAGMA_UNROLL
		for (int i = 0; i < N; ++i) {
			if (std::is_same<Transform, Identity>::value) {
				result[i] = convert_(s[i]);
			} else {
				result[i] = conj(convert_(s[i]));
			}
		}

		return result;
	}

	CUTLASS_HOST_DEVICE result_type operator()(source_type const& s) const {
		return convert(s);
	}
};

template<typename T> struct multiplies {
	CUTLASS_HOST_DEVICE T operator()(T lhs, T const& rhs) const {
		lhs *= rhs;
		return lhs;
	}
};

template<typename A, typename B = A, typename C = A> struct multiply_add {
	CUTLASS_HOST_DEVICE C operator()(A const& a, B const& b, C const& c) const {
		return C(a) * C(b) + c;
	}
};

template<typename ElementOutput_, int Count, typename ElementAccumulator_ = ElementOutput_, typename ElementCompute_ = ElementOutput_, ScaleType::Kind Scale = ScaleType::Default,
	FloatRoundStyle Round = FloatRoundStyle::round_to_nearest, typename ElementSource_ = ElementOutput_>
class LinearCombination {
  public:
	using ElementOutput		 = ElementOutput_;
	using ElementSource		 = ElementSource_;
	using ElementAccumulator = ElementAccumulator_;
	using ElementCompute	 = ElementCompute_;
	using ElementScalar		 = ElementCompute;
	using ElementC			 = ElementSource_;
	using ElementD			 = ElementOutput_;

	static constexpr int kCount			= Count;
	static const ScaleType::Kind kScale = Scale;
	using FragmentOutput				= Array<ElementOutput, kCount>;
	using FragmentSource				= Array<ElementSource, kCount>;
	using FragmentAccumulator			= Array<ElementAccumulator, kCount>;
	using FragmentCompute				= Array<ElementCompute, kCount>;

	static constexpr FloatRoundStyle kRound = Round;

	struct Params {
		ElementCompute alpha;
		ElementCompute beta;
		ElementCompute const* alpha_ptr;
		ElementCompute const* beta_ptr;
		ElementCompute const* const* alpha_ptr_array;
		ElementCompute const* const* beta_ptr_array;
		CUTLASS_HOST_DEVICE Params() : alpha(ElementCompute(1)), beta(ElementCompute(0)), alpha_ptr(nullptr), beta_ptr(nullptr), alpha_ptr_array(nullptr), beta_ptr_array(nullptr) {
		}

		CUTLASS_HOST_DEVICE Params(ElementCompute alpha, ElementCompute beta)
			: alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr), alpha_ptr_array(nullptr), beta_ptr_array(nullptr) {
		}

		CUTLASS_HOST_DEVICE Params(ElementCompute alpha) : alpha(alpha), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr), alpha_ptr_array(nullptr), beta_ptr_array(nullptr) {
		}

		CUTLASS_HOST_DEVICE Params(ElementCompute const* alpha_ptr, ElementCompute const* beta_ptr)
			: alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr), alpha_ptr_array(nullptr), beta_ptr_array(nullptr) {
		}

		CUTLASS_HOST_DEVICE Params(ElementCompute const* alpha_ptr)
			: alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(nullptr), alpha_ptr_array(nullptr), beta_ptr_array(nullptr) {
		}

		CUTLASS_HOST_DEVICE Params(ElementCompute const* const* alpha_ptr_array, ElementCompute const* const* beta_ptr_array)
			: alpha(0), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr), alpha_ptr_array(alpha_ptr_array), beta_ptr_array(beta_ptr_array) {
		}

		CUTLASS_HOST_DEVICE Params(ElementCompute const* const* alpha_ptr_array)
			: alpha(0), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr), alpha_ptr_array(alpha_ptr_array), beta_ptr_array(nullptr) {
		}
	};

  private:
	ElementCompute alpha_;
	ElementCompute beta_;

  public:
	CUTLASS_HOST_DEVICE explicit LinearCombination(Params const& params, int group_idx) {
		if (params.alpha_ptr_array != nullptr && params.alpha_ptr_array[group_idx] != nullptr) {
			alpha_ = *(params.alpha_ptr_array[group_idx]);
		} else if (params.alpha_ptr != nullptr) {
			alpha_ = *params.alpha_ptr;
		} else {
			alpha_ = params.alpha;
		}
		if (params.beta_ptr_array != nullptr && params.beta_ptr_array[group_idx] != nullptr) {
			beta_ = *(params.beta_ptr_array[group_idx]);
		} else if (params.beta_ptr != nullptr) {
			beta_ = *params.beta_ptr;
		} else {
			beta_ = params.beta;
		}
	}

	CUTLASS_HOST_DEVICE explicit LinearCombination(const Params& params) : LinearCombination(params, 0) {
	}

	CUTLASS_HOST_DEVICE bool is_source_needed() const {
		if (Scale == ScaleType::NoBetaScaling)
			return true;

		if (Scale == ScaleType::OnlyAlphaScaling)
			return false;

		if (Scale == ScaleType::Nothing)
			return false;

		return beta_ != ElementCompute(0);
	}

	CUTLASS_HOST_DEVICE void set_k_partition(int k_partition, int k_partition_count) {
		if (k_partition) {
			beta_ = ElementCompute(1);
		}
	}

	CUTLASS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const& accumulator, FragmentSource const& source) const {
		NumericArrayConverter<ElementCompute, ElementSource, kCount, Round> source_converter;
		NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

		NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

		FragmentCompute converted_source	  = source_converter(source);
		FragmentCompute converted_accumulator = accumulator_converter(accumulator);

		if (Scale == ScaleType::Nothing)
			return destination_converter(converted_accumulator);

		FragmentCompute intermediate;

		multiplies<FragmentCompute> mul_add_source;
		multiply_add<FragmentCompute> mul_add_accumulator;

		if (Scale == ScaleType::NoBetaScaling)
			intermediate = converted_source;
		else
			intermediate = mul_add_source(beta_, converted_source);
		intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);
		return destination_converter(intermediate);
	}

	CUTLASS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
		NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

		NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

		FragmentCompute converted_accumulator = accumulator_converter(accumulator);

		if (Scale == ScaleType::Nothing)
			return destination_converter(converted_accumulator);

		FragmentCompute intermediate;
		multiplies<FragmentCompute> mul_accumulator;

		intermediate = mul_accumulator(alpha_, converted_accumulator);
		return destination_converter(intermediate);
	}

	CUTLASS_HOST_DEVICE ElementD operator()(ElementAccumulator const accumulator, ElementC const source) const {
		NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
		[[maybe_unused]] NumericConverter<ElementCompute, ElementC, Round> source_converter;
		NumericConverter<ElementD, ElementCompute, Round> destination_converter;


		ElementCompute converted_accumulator = accumulator_converter(accumulator);
		if constexpr (Scale == ScaleType::Nothing) {
			return destination_converter(converted_accumulator);
		}

		ElementCompute intermediate;
		multiplies<ElementCompute> multiply;
		multiply_add<ElementCompute> madd;

		if constexpr (Scale == ScaleType::NoBetaScaling) {
			intermediate = source_converter(source);
		} else {
			intermediate = multiply(beta_, source);
		}

		intermediate = madd(alpha_, converted_accumulator, intermediate);
		return destination_converter(intermediate);
	}

	CUTLASS_HOST_DEVICE ElementD operator()(ElementAccumulator const accumulator) const {
		NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
		NumericConverter<ElementD, ElementCompute, Round> destination_converter;
		ElementCompute converted_accumulator = accumulator_converter(accumulator);

		if constexpr (Scale == ScaleType::Nothing) {
			return destination_converter(converted_accumulator);
		}

		ElementCompute intermediate;
		multiplies<ElementCompute> multiply;

		intermediate = multiply(alpha_, accumulator);
		return destination_converter(intermediate);
	}
};

CUTLASS_DEVICE
int RematerializeThreadIdxX() {
	return threadIdx.x;
}

CUTLASS_DEVICE
int RematerializeThreadIdxY() {
	return threadIdx.y;
}

CUTLASS_DEVICE
int RematerializeThreadIdxZ() {
	return threadIdx.z;
}

CUTLASS_DEVICE
int RematerializeBlockIdxX() {
	return blockIdx.x;
}

CUTLASS_DEVICE
int RematerializeBlockIdxY() {
	return blockIdx.y;
}

CUTLASS_DEVICE
int RematerializeBlockIdxZ() {
	return blockIdx.z;
}

CUTLASS_DEVICE
int RematerializeBlockDimX() {
	return blockDim.x;
}

CUTLASS_DEVICE
int RematerializeBlockDimY() {
	return blockDim.y;
}

CUTLASS_DEVICE
int RematerializeBlockDimZ() {
	return blockDim.z;
}

template<typename OperatorClass, typename ArchTag, typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator> struct DefaultGemmConfiguration;

template<typename ArchTag, typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
struct DefaultGemmConfiguration<OpClassSimt, ArchTag, ElementA, ElementB, ElementC, ElementAccumulator> {
	static constexpr int kAlignmentA = 1;
	static constexpr int kAlignmentB = 1;
	using ThreadblockShape			 = GemmShape<128, 128, 8>;
	using WarpShape					 = GemmShape<32, 64, 8>;
	using InstructionShape			 = GemmShape<1, 1, 1>;
	static constexpr int kStages	 = 2;

	using EpilogueOutputOp = LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>;

	using Operator = OpMultiplyAdd;
};

struct Sm120 {
	static constexpr int kMinComputeCapability = 120;
};

template<int N = 1> struct GemmIdentityThreadblockSwizzle {
	CUTLASS_HOST_DEVICE GemmIdentityThreadblockSwizzle() {
	}

	CUTLASS_HOST_DEVICE static GemmCoord get_tiled_shape(GemmCoord problem_size, GemmCoord tile_size, int split_k_slices) {
		return GemmCoord((problem_size.m() + tile_size.m() - 1) / tile_size.m(), (problem_size.n() + tile_size.n() - 1) / tile_size.n(), split_k_slices);
	}

	CUTLASS_HOST_DEVICE static dim3 get_grid_shape(GemmCoord tiled_shape) {
		int tile = 1 << get_log_tile(tiled_shape);
		return dim3(tiled_shape.m() * tile, (tiled_shape.n() + tile - 1) / tile, tiled_shape.k());
	}

	CUTLASS_HOST_DEVICE static int get_log_tile(GemmCoord tiled_shape) {
		auto n = tiled_shape.n();
		if (N >= 8 && n >= 6)
			return 3;
		else if (N >= 4 && n >= 3)
			return 2;
		else if (N >= 2 && n >= 2)
			return 1;
		else
			return 0;
	}

	CUTLASS_DEVICE
	static GemmCoord get_tile_offset(int log_tile) {
		int block_idx_x = RematerializeBlockIdxX();
		int block_idx_y = RematerializeBlockIdxY();
		int block_idx_z = RematerializeBlockIdxZ();

		return GemmCoord{ (block_idx_x >> log_tile), (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)), block_idx_z };
	}

	CUTLASS_DEVICE
	static GemmCoord get_tile_offset(GemmCoord tiled_shape) {
		int const kTile = N;
		int block_idx_x = RematerializeBlockIdxX();
		int block_idx_y = RematerializeBlockIdxY();

		if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile))
			return GemmCoord{ block_idx_x, block_idx_y, RematerializeBlockIdxZ() };

		return GemmCoord{ (block_idx_x / kTile), (block_idx_y * kTile) + (block_idx_x % kTile), RematerializeBlockIdxZ() };
	}
};

struct MatrixCoord : public Coord<2, int> {
  public:
	using Index = int;

	using Base = Coord<2, Index>;

	using LongIndex = typename Base::LongIndex;

  private:
	static constexpr int kRow = 0;

	static constexpr int kColumn = 1;

  public:
	CUTLASS_HOST_DEVICE MatrixCoord() {
	}

	CUTLASS_HOST_DEVICE MatrixCoord(Coord<2, Index> const& coord) : Base(coord) {
	}

	CUTLASS_HOST_DEVICE MatrixCoord(Index row, Index column) : Base(make_Coord(row, column)) {
	}

	CUTLASS_HOST_DEVICE MatrixCoord(LongIndex row, LongIndex column) : Base(make_Coord(Index(row), Index(column))) {
	}

	CUTLASS_HOST_DEVICE Index const& row() const {
		return this->at(kRow);
	}

	CUTLASS_HOST_DEVICE Index& row() {
		return this->at(kRow);
	}

	CUTLASS_HOST_DEVICE Index const& column() const {
		return this->at(kColumn);
	}

	CUTLASS_HOST_DEVICE Index& column() {
		return this->at(kColumn);
	}

	CUTLASS_HOST_DEVICE MatrixCoord operator+(Base const& b) const {
		return MatrixCoord(Base::operator+(b));
	}

	CUTLASS_HOST_DEVICE MatrixCoord operator-(Base const& b) const {
		return MatrixCoord(Base::operator-(b));
	}

	CUTLASS_HOST_DEVICE MatrixCoord operator*(Base const& b) const {
		return MatrixCoord(Base::operator*(b));
	}

	CUTLASS_HOST_DEVICE MatrixCoord operator/(Base const& b) const {
		return MatrixCoord(Base::operator/(b));
	}

	CUTLASS_HOST_DEVICE MatrixCoord& operator+=(Base const& b) {
		Base::operator+=(b);
		return *this;
	}

	CUTLASS_HOST_DEVICE MatrixCoord& operator-=(Base const& b) {
		Base::operator-=(b);
		return *this;
	}

	CUTLASS_HOST_DEVICE MatrixCoord& operator*=(Base const& b) {
		Base::operator*=(b);
		return *this;
	}

	CUTLASS_HOST_DEVICE MatrixCoord& operator/=(Base const& b) {
		Base::operator/=(b);
		return *this;
	}
};

template<int Contiguous, int Strided> struct PitchLinearShape {
	static constexpr int kContiguous = Contiguous;
	static constexpr int kStrided	 = Strided;
	static constexpr int kCount		 = Contiguous * Strided;
};


struct PitchLinearCoord : public Coord<2, int> {
  public:
	using Index = int;

	using Base = Coord<2, Index>;

	using LongIndex = typename Base::LongIndex;

  private:
	static constexpr int kContiguous = 0;

	static constexpr int kStrided = 1;

  public:
	CUTLASS_HOST_DEVICE PitchLinearCoord() {
	}

	CUTLASS_HOST_DEVICE PitchLinearCoord(Coord<2, Index> const& coord) : Base(coord) {
	}

	CUTLASS_HOST_DEVICE PitchLinearCoord(Index contiguous_, Index strided_) : Base(make_Coord(contiguous_, strided_)) {
	}

	CUTLASS_HOST_DEVICE PitchLinearCoord(LongIndex contiguous_, LongIndex strided_) : Base(make_Coord(Index(contiguous_), Index(strided_))) {
	}

	CUTLASS_HOST_DEVICE Index const& contiguous() const {
		return this->at(kContiguous);
	}

	CUTLASS_HOST_DEVICE Index& contiguous() {
		return this->at(kContiguous);
	}

	CUTLASS_HOST_DEVICE Index const& strided() const {
		return this->at(kStrided);
	}

	CUTLASS_HOST_DEVICE Index& strided() {
		return this->at(kStrided);
	}

	CUTLASS_HOST_DEVICE PitchLinearCoord operator+(Base const& b) const {
		return PitchLinearCoord(Base::operator+(b));
	}

	CUTLASS_HOST_DEVICE PitchLinearCoord operator-(Base const& b) const {
		return PitchLinearCoord(Base::operator-(b));
	}

	CUTLASS_HOST_DEVICE PitchLinearCoord operator-() const {
		return PitchLinearCoord(-at(0), -at(1));
	}

	CUTLASS_HOST_DEVICE PitchLinearCoord operator*(Base const& b) const {
		return PitchLinearCoord(Base::operator*(b));
	}

	CUTLASS_HOST_DEVICE PitchLinearCoord operator/(Base const& b) const {
		return PitchLinearCoord(Base::operator/(b));
	}

	CUTLASS_HOST_DEVICE PitchLinearCoord& operator+=(Base const& b) {
		Base::operator+=(b);
		return *this;
	}

	CUTLASS_HOST_DEVICE PitchLinearCoord& operator-=(Base const& b) {
		Base::operator-=(b);
		return *this;
	}

	CUTLASS_HOST_DEVICE PitchLinearCoord& operator*=(Base const& b) {
		Base::operator*=(b);
		return *this;
	}

	CUTLASS_HOST_DEVICE PitchLinearCoord& operator/=(Base const& b) {
		Base::operator/=(b);
		return *this;
	}
};

class PermuteBase {
  public:
	using Index = int32_t;

	using LongIndex = int64_t;
};

class NoPermute : public PermuteBase {
  public:
	CUTLASS_HOST_DEVICE NoPermute(MatrixCoord extent, Index stride) {};

	CUTLASS_HOST_DEVICE NoPermute(PitchLinearCoord extent, Index stride) {};

	CUTLASS_HOST_DEVICE LongIndex operator()(MatrixCoord coord) const {
		return 0;
	}
	CUTLASS_HOST_DEVICE LongIndex operator()(PitchLinearCoord coord) const {
		return 0;
	}
};

template<typename Permute> struct InversePermute {
	static_assert(!std::is_same<Permute, Permute>::value,
		"To apply permutation to a GEMM input operand (A or B), an inverse permutation for the desired "
		"permute class must be defined and enabled by specializing cutlass::layout::InversePermute trait.");
};

template<> struct InversePermute<NoPermute> {
	using type = NoPermute;
};

template<typename Permute> inline bool constexpr is_trivial_permute = std::is_same<Permute, NoPermute>::value;

template<int D1, int D2> class Tensor4DPermute0213RowMajor : public PermuteBase {
  private:
	Index D3_;

	Index stride_;

  public:
	CUTLASS_HOST_DEVICE Tensor4DPermute0213RowMajor(MatrixCoord extent, Index stride) {
		assert(extent.row() % D1 == 0);
		assert(extent.column() % D2 == 0);

		D3_ = extent.column() / D2;

		stride_ = stride * D1 / D2;
	}

	CUTLASS_HOST_DEVICE Tensor4DPermute0213RowMajor(PitchLinearCoord extent, Index stride)
		: Tensor4DPermute0213RowMajor(MatrixCoord(extent.strided(), extent.contiguous()), stride) {
	}

	CUTLASS_HOST_DEVICE LongIndex operator()(MatrixCoord coord) const {
		Index l = coord.column() % D3_;
		Index k = coord.column() / D3_;
		Index j = coord.row() % D1;
		Index i = coord.row() / D1;

		MatrixCoord permuted{ k + i * D2, l + j * D3_ };

		return LongIndex(permuted.row()) * LongIndex(stride_) + LongIndex(permuted.column());
	}

	CUTLASS_HOST_DEVICE LongIndex operator()(PitchLinearCoord coord) const {
		return operator()(MatrixCoord(coord.strided(), coord.contiguous()));
	}
};

template<typename Element_,/// CUTLASS numeric element type.
	typename Storage_ = uint8_t,/// Underlying storage type. Must be able to hold an integer
	///   number of objects of type Element.
	class = void>
class ConstSubbyteReference {
  public:
	using Element		 = Element_;
	using Storage		 = Storage_;
	using StoragePointer = Storage const*;

	static_assert(sizeof_bits<Element>::value <= sizeof_bits<Storage>::value, "Size of Element must not be greater than Storage.");

	static_assert(!(sizeof_bits<Storage>::value % sizeof_bits<Element>::value), "Storage must be divisible by Element");

  private:
	///! Number of elements per storage vector
	int const kElementsPerVector = sizeof_bits<Storage>::value / sizeof_bits<Element>::value;

	///! Bit mask
	Storage const kMask = ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value) ? (Storage(1) << sizeof_bits<Element>::value) - Storage(1) : ~Storage(0));

  private:
	/// Pointer to array containing element
	StoragePointer ptr_;

	/// Offset (in units of elements) from pointer.
	///
	/// Invariant: must always be in range [0, kElementsPerVector)
	int offset_;

  public:
	CUTLASS_HOST_DEVICE
	ConstSubbyteReference() : ptr_(nullptr), offset_(0) {
	}

	/// Constructor
	CUTLASS_HOST_DEVICE
	ConstSubbyteReference(Element const* ptr,/// pointer to memory
		int64_t offset/// logical offset in units of Element
		)
		: ptr_(reinterpret_cast<StoragePointer>(ptr)), offset_(0) {
		int64_t offset_in_vectors  = offset / kElementsPerVector;
		int64_t offset_in_elements = offset % kElementsPerVector;

		ptr_ += offset_in_vectors;
		offset_ = int(offset_in_elements);
	}

	/// Constructor
	CUTLASS_HOST_DEVICE
	ConstSubbyteReference(Element* ptr = nullptr) : ConstSubbyteReference(ptr, 0) {
	}

	/// Gets storage pointer
	CUTLASS_HOST_DEVICE
	StoragePointer storage_pointer() const {
		return ptr_;
	}

	/// Gets element offset within storage vector
	CUTLASS_HOST_DEVICE
	int element_offset() const {
		return offset_;
	}

	/// Unpacks an element from memory
	CUTLASS_HOST_DEVICE
	Element get() const {
		Storage item = Storage((*ptr_ >> (offset_ * sizeof_bits<Element>::value)) & kMask);
		return reinterpret_cast<Element const&>(item);
	}

	/// Unpacks an element from memory
	CUTLASS_HOST_DEVICE
	operator Element() const {
		return get();
	}

	/// Adds an offset in units of elements to the reference
	CUTLASS_HOST_DEVICE
	ConstSubbyteReference& operator+=(int offset) {
		offset += offset_;

		int offset_in_vectors  = offset / kElementsPerVector;
		int offset_in_elements = offset % kElementsPerVector;

		ptr_ += offset_in_vectors;
		offset_ = offset_in_elements;

		return *this;
	}

	/// Adds an offset in units of elements to the reference
	CUTLASS_HOST_DEVICE
	ConstSubbyteReference& operator+=(long long offset) {
		offset += offset_;

		long long offset_in_vectors = offset / kElementsPerVector;
		int offset_in_elements		= int(offset % kElementsPerVector);

		ptr_ += offset_in_vectors;
		offset_ = offset_in_elements;

		return *this;
	}

	/// Adds an offset in units of elements to the reference
	CUTLASS_HOST_DEVICE
	ConstSubbyteReference& operator-=(int offset) {
		int offset_in_vectors  = offset / kElementsPerVector;
		int offset_in_elements = offset % kElementsPerVector;

		ptr_ -= offset_in_vectors;
		offset_ -= offset_in_elements;

		if (offset_ < 0) {
			offset_ += kElementsPerVector;
			--ptr_;
		}

		return *this;
	}

	/// Adds an offset in units of elements to the reference
	CUTLASS_HOST_DEVICE
	ConstSubbyteReference& operator-=(long long offset) {
		long long offset_in_vectors = offset / kElementsPerVector;
		int offset_in_elements		= int(offset % kElementsPerVector);

		ptr_ -= offset_in_vectors;
		offset_ -= offset_in_elements;

		if (offset_ < 0) {
			offset_ += kElementsPerVector;
			--ptr_;
		}

		return *this;
	}

	/// Returns a reference to an element with a given offset from the current reference
	CUTLASS_HOST_DEVICE
	ConstSubbyteReference operator+(int offset) const {
		ConstSubbyteReference ref(ptr_, offset_);
		ref += offset;

		return ref;
	}

	/// Returns a reference to an element with a given offset from the current reference
	CUTLASS_HOST_DEVICE
	ConstSubbyteReference operator+(long long offset) const {
		ConstSubbyteReference ref(ptr_, offset_);
		ref += offset;

		return ref;
	}

	/// Returns a reference to an element with a given offset from the current reference
	CUTLASS_HOST_DEVICE
	ConstSubbyteReference operator-(int offset) const {
		ConstSubbyteReference ref(ptr_, offset_);
		ref -= offset;

		return ref;
	}

	/// Returns a reference to an element with a given offset from the current reference
	CUTLASS_HOST_DEVICE
	ConstSubbyteReference operator-=(long long offset) const {
		ConstSubbyteReference ref(ptr_, offset_);
		ref -= offset;

		return ref;
	}

	/// Computes the difference in elements between references
	CUTLASS_HOST_DEVICE
	ptrdiff_t operator-(ConstSubbyteReference ref) const {
		return (ptr_ - ref.ptr_) * kElementsPerVector + (offset_ - ref.offset_);
	}

	/// Explicit cast to int
	CUTLASS_HOST_DEVICE
	explicit operator int() const {
		return int(get());
	}

	/// Explicit cast to signed 64-bit integer
	CUTLASS_HOST_DEVICE
	explicit operator int64_t() const {
		return int64_t(get());
	}

	/// Explicit cast to unsigned 64-bit integer
	CUTLASS_HOST_DEVICE
	explicit operator uint64_t() const {
		return uint64_t(get());
	}

	/// Explicit cast to float
	CUTLASS_HOST_DEVICE
	explicit operator float() const {
		return float(get());
	}

	/// Explicit cast to double
	CUTLASS_HOST_DEVICE
	explicit operator double() const {
		return double(get());
	}
};

template<typename Element, bool subbyte = (sizeof_bits<Element>::value < 8)> struct ReferenceFactory;

template<typename Element> struct ReferenceFactory<Element, false> {
	///! Number of elements per storage vector
	static constexpr int kElementsPerVector = 1;

	CUTLASS_HOST_DEVICE
	static Element& get(Element* ptr, int64_t offset) {
		return ptr[offset];
	}

	CUTLASS_HOST_DEVICE
	static constexpr const Element& get(Element const* ptr, int64_t offset) {
		return ptr[offset];
	}

	CUTLASS_HOST_DEVICE
	static Element* add_pointer_offset(Element* ptr, int64_t offset) {
		return ptr + offset;
	}

	CUTLASS_HOST_DEVICE
	static constexpr Element* add_pointer_offset(Element const* ptr, int64_t offset) {
		return ptr + offset;
	}
};

template<typename Element_,/// CUTLASS numeric element type.
	typename Storage_ =/// Underlying storage type. Must be able to hold an integer
///   number of objects of type Element.
#if defined(__CUDA_ARCH__)/// Default size depends on width of atomicCas() overloads.
	#if (__CUDA_ARCH__ >= 700)///
	uint16_t
	#else
	uint32_t
	#endif
#else
	uint8_t
#endif
	,
	class = void>
class SubbyteReference {
  public:
	using Element		 = Element_;
	using Storage		 = Storage_;
	using StoragePointer = Storage*;

	static_assert(sizeof_bits<Element>::value <= sizeof_bits<Storage>::value, "Size of Element must not be greater than Storage.");

	static_assert(!(sizeof_bits<Storage>::value % sizeof_bits<Element>::value), "Storage must be divisible by Element");

  private:
	///! Number of elements per storage vector
	int const kElementsPerVector = sizeof_bits<Storage>::value / sizeof_bits<Element>::value;

	///! Bit mask
	Storage const kMask = ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value) ? (Storage(1) << sizeof_bits<Element>::value) - Storage(1) : ~Storage(0));

  private:
	/// Pointer to array containing element
	StoragePointer ptr_;

	/// Offset (in units of elements) from pointer.
	///
	/// Invariant: must always be in range [0, kElementsPerVector)
	int offset_;

  public:
	CUTLASS_HOST_DEVICE
	SubbyteReference() : ptr_(nullptr), offset_(0) {
	}

	/// Constructor
	CUTLASS_HOST_DEVICE
	SubbyteReference(Element* ptr,/// pointer to memory
		int64_t offset/// logical offset in units of Element
		)
		: ptr_(reinterpret_cast<StoragePointer>(ptr)), offset_(0) {
		int64_t offset_in_vectors  = offset / kElementsPerVector;
		int64_t offset_in_elements = offset % kElementsPerVector;

		ptr_ += offset_in_vectors;
		offset_ = int(offset_in_elements);
	}

	/// Constructor
	CUTLASS_HOST_DEVICE
	SubbyteReference(Element* ptr = nullptr) : SubbyteReference(ptr, 0) {
	}

	/// Gets storage pointer
	CUTLASS_HOST_DEVICE
	StoragePointer storage_pointer() const {
		return ptr_;
	}

	/// Gets storage pointer
	CUTLASS_HOST_DEVICE
	Element* operator&() const {
		return reinterpret_cast<Element*>(ptr_);
	}

	/// Gets element offset within storage vector
	CUTLASS_HOST_DEVICE
	int element_offset() const {
		return offset_;
	}

	/// Unpacks an element from memory
	CUTLASS_HOST_DEVICE
	Element get() const {
		uint8_t const* byte_ptr = reinterpret_cast<uint8_t const*>(ptr_);
		// Convert offset in elements to offset in bytes
		constexpr int elements_per_byte = sizeof_bits<uint8_t>::value / sizeof_bits<Element>::value;
		byte_ptr += offset_ / elements_per_byte;
		// Offset of element within a byte
		int byte_offset = offset_ % elements_per_byte;
		uint8_t item	= uint8_t((*byte_ptr >> (byte_offset * sizeof_bits<Element>::value)) & kMask);
		return reinterpret_cast<Element const&>(item);
	}

	/// Stores an element to memory
	CUTLASS_HOST_DEVICE
	SubbyteReference& set(Element const& x) {
		Storage item		= (reinterpret_cast<Storage const&>(x) & kMask);
		Storage kUpdateMask = Storage(~(kMask << (offset_ * sizeof_bits<Element>::value)));
		Storage new_bits	= Storage(item << (offset_ * sizeof_bits<Element>::value));

#if defined(__CUDA_ARCH__)

		//
		// Homebrew read-modify-write
		//
		Storage original;
		Storage updated;

		do {
			original = (*ptr_);

			updated = Storage((original & kUpdateMask) | new_bits);

			original = atomicCAS(ptr_, original, updated);

		} while (updated != original);

#else

		Storage original = (*ptr_);
		Storage updated	 = Storage((original & kUpdateMask) | new_bits);
		*ptr_			 = updated;

#endif

		return *this;
	}

	////

	/// Unpacks an element from memory
	CUTLASS_HOST_DEVICE
	operator Element() const {
		return get();
	}

	/// Stores an element to memory
	CUTLASS_HOST_DEVICE
	SubbyteReference& operator=(Element const& x) {
		return set(x);
	}

	/// Stores an element to memory
	CUTLASS_HOST_DEVICE
	SubbyteReference& operator=(SubbyteReference const& x) {
		return set(x.get());
	}

	/// Stores an element to memory
	CUTLASS_HOST_DEVICE
	SubbyteReference& operator=(ConstSubbyteReference<Element, Storage> const& x) {
		return set(x.get());
	}

	/// Adds an offset in units of elements to the reference
	CUTLASS_HOST_DEVICE
	SubbyteReference& operator+=(int offset) {
		offset += offset_;

		int offset_in_vectors  = offset / kElementsPerVector;
		int offset_in_elements = offset % kElementsPerVector;

		ptr_ += offset_in_vectors;
		offset_ = offset_in_elements;

		return *this;
	}

	/// Adds an offset in units of elements to the reference
	CUTLASS_HOST_DEVICE
	SubbyteReference& operator+=(long long offset) {
		offset += offset_;

		long long offset_in_vectors = offset / kElementsPerVector;
		int offset_in_elements		= int(offset % kElementsPerVector);

		ptr_ += offset_in_vectors;
		offset_ = offset_in_elements;

		return *this;
	}

	/// Adds an offset in units of elements to the reference
	CUTLASS_HOST_DEVICE
	SubbyteReference& operator-=(int offset) {
		int offset_in_vectors  = offset / kElementsPerVector;
		int offset_in_elements = offset % kElementsPerVector;

		ptr_ -= offset_in_vectors;
		offset_ -= offset_in_elements;

		if (offset_ < 0) {
			offset_ += kElementsPerVector;
			--ptr_;
		}

		return *this;
	}

	/// Adds an offset in units of elements to the reference
	CUTLASS_HOST_DEVICE
	SubbyteReference& operator-=(long long offset) {
		long long offset_in_vectors = offset / kElementsPerVector;
		int offset_in_elements		= int(offset % kElementsPerVector);

		ptr_ -= offset_in_vectors;
		offset_ -= offset_in_elements;

		if (offset_ < 0) {
			offset_ += kElementsPerVector;
			--ptr_;
		}

		return *this;
	}

	/// Returns a reference to an element with a given offset from the current reference
	CUTLASS_HOST_DEVICE
	SubbyteReference operator+(int offset) const {
		SubbyteReference ref(ptr_, offset_);
		ref += offset;

		return ref;
	}

	/// Returns a reference to an element with a given offset from the current reference
	CUTLASS_HOST_DEVICE
	SubbyteReference operator+(long long offset) const {
		SubbyteReference ref(ptr_, offset_);
		ref += offset;

		return ref;
	}

	/// Returns a reference to an element with a given offset from the current reference
	CUTLASS_HOST_DEVICE
	SubbyteReference operator-(int offset) const {
		SubbyteReference ref(ptr_, offset_);
		ref -= offset;

		return ref;
	}

	/// Returns a reference to an element with a given offset from the current reference
	CUTLASS_HOST_DEVICE
	SubbyteReference operator-=(long long offset) const {
		SubbyteReference ref(ptr_, offset_);
		ref -= offset;

		return ref;
	}

	/// Computes the difference in elements between references
	CUTLASS_HOST_DEVICE
	ptrdiff_t operator-(SubbyteReference ref) const {
		return (ptr_ - ref.ptr_) * kElementsPerVector + (offset_ - ref.offset_);
	}

	/// Explicit cast to int
	CUTLASS_HOST_DEVICE
	explicit operator int() const {
		return int(get());
	}

	/// Explicit cast to signed 64-bit integer
	CUTLASS_HOST_DEVICE
	explicit operator int64_t() const {
		return int64_t(get());
	}

	/// Explicit cast to unsigned 64-bit integer
	CUTLASS_HOST_DEVICE
	explicit operator uint64_t() const {
		return uint64_t(get());
	}

	/// Explicit cast to float
	CUTLASS_HOST_DEVICE
	explicit operator float() const {
		return float(get());
	}

	/// Explicit cast to double
	CUTLASS_HOST_DEVICE
	explicit operator double() const {
		return double(get());
	}
};

template<
	/// Data type of element stored within tensor (concept: NumericType)
	typename Element_,
	/// Defines a mapping from logical coordinate to linear memory (concept: Layout)
	typename Layout_>
class TensorRef {
  public:
	/// Data type of individual access
	using Element = Element_;

	/// Mapping function from logical coordinate to linear memory
	using Layout = Layout_;

	/// Reference type to an element
	using Reference = typename std::conditional<sizeof_bits<Element>::value >= 8, Element&, SubbyteReference<Element>>::type;

	/// Logical rank of tensor index space
	static int const kRank = Layout::kRank;

	/// Index type
	using Index = typename Layout::Index;

	/// Long index used for pointer offsets
	using LongIndex = typename Layout::LongIndex;

	/// Coordinate in logical tensor space
	using TensorCoord = typename Layout::TensorCoord;

	/// Layout's stride vector
	using Stride = typename Layout::Stride;

	/// TensorRef to constant data
	using ConstTensorRef = TensorRef<typename std::remove_const<Element>::type const, Layout>;

	/// TensorRef to non-constant data
	using NonConstTensorRef = TensorRef<typename std::remove_const<Element>::type, Layout>;

	/// Require at least rank=1. Mathematically, a rank=0 tensor would be considered to be a
	/// scalar, but degenerate cases such as these are difficult to accommodate without
	/// extensive C++ metaprogramming or support for zero-length arrays.
	static_assert(kRank > 0, "Cannot define a zero-rank TensorRef");

  private:
	/// Pointer
	Element* ptr_;

	/// Layout object maps logical coordinates to linear offsets
	Layout layout_;

  public:
	//
	// Methods
	//

	/// Constructs a TensorRef with a pointer and layout object.
	CUTLASS_HOST_DEVICE
	TensorRef() : ptr_(nullptr) {
	}

	/// Constructs a TensorRef with a pointer and layout object.
	CUTLASS_HOST_DEVICE
	TensorRef(Element* ptr,///< pointer to start of tensor
		Layout const& layout///< layout object containing stride and mapping function
		)
		: ptr_(ptr), layout_(layout) {
	}

	/// Converting constructor from TensorRef to non-constant data.
	template<typename _Magic = int> CUTLASS_HOST_DEVICE TensorRef(NonConstTensorRef const& ref,///< TensorRef to non-const data
		///SFINAE trick to avoid creating a copy-constructor when Element_ is already non-const
		_Magic magic = ( typename std::enable_if<!std::is_same<NonConstTensorRef, TensorRef<Element_, Layout_>>::value, _Magic>::type )0)
		: ptr_(ref.data()), layout_(ref.layout()) {
	}

	/// Returns a reference to constant-valued tensor.
	CUTLASS_HOST_DEVICE
	ConstTensorRef const_ref() const {
		return ConstTensorRef(ptr_, layout_);
	}

	CUTLASS_HOST_DEVICE
	NonConstTensorRef non_const_ref() const {
		return NonConstTensorRef(const_cast<typename std::remove_const<Element>::type*>(ptr_), layout_);
	}

	/// Updates only the pointer
	CUTLASS_HOST_DEVICE
	void reset(Element* ptr = nullptr) {
		ptr_ = ptr;
	}

	/// Updates the pointer and layout object
	CUTLASS_HOST_DEVICE
	void reset(Element* ptr, Layout const& layout) {
		ptr_	= ptr;
		layout_ = layout;
	}

	/// Returns true if the TensorRef is non-null
	CUTLASS_HOST_DEVICE
	bool good() const {
		return ptr_ != nullptr;
	}

	/// Returns the pointer to referenced data
	CUTLASS_HOST_DEVICE
	Element* data() const {
		return ptr_;
	}

	/// Returns a reference to the element at a given linear index
	CUTLASS_HOST_DEVICE
	Reference data(LongIndex idx) const {
		return ReferenceFactory<typename std::remove_const<Element>::type, (sizeof_bits<Element>::value < 8)>::get(ptr_, idx);
	}

	/// Returns the layout object
	CUTLASS_HOST_DEVICE
	Layout& layout() {
		return layout_;
	}

	/// Returns the layout object
	CUTLASS_HOST_DEVICE
	Layout layout() const {
		return layout_;
	}

	/// Returns the layout object's stride vector
	CUTLASS_HOST_DEVICE
	Stride stride() const {
		return layout_.stride();
	}

	/// Returns the layout object's stride vector
	CUTLASS_HOST_DEVICE
	Stride& stride() {
		return layout_.stride();
	}

	/// Returns the layout object's stride in a given physical dimension
	CUTLASS_HOST_DEVICE
	typename Layout::Stride::Index stride(int dim) const {
		return layout_.stride().at(dim);
	}

	/// Returns the layout object's stride in a given physical dimension
	CUTLASS_HOST_DEVICE
	typename Layout::Stride::Index& stride(int dim) {
		return layout_.stride().at(dim);
	}

	/// Computes the offset of an index from the origin of the tensor
	CUTLASS_HOST_DEVICE
	LongIndex offset(TensorCoord const& coord) const {
		return layout_(coord);
	}

	/// Returns a reference to the element at a given Coord
	CUTLASS_HOST_DEVICE
	Reference at(TensorCoord const& coord) const {
		return data(offset(coord));
	}

	/// Returns a reference to the element at a given Coord
	CUTLASS_HOST_DEVICE
	Reference operator[](TensorCoord const& coord) const {
		return data(offset(coord));
	}

	/// Adds an offset to each pointer
	CUTLASS_HOST_DEVICE
	TensorRef& add_pointer_offset(LongIndex offset_) {
		ptr_ = ReferenceFactory<typename std::remove_const<Element>::type, (sizeof_bits<Element>::value < 8)>::add_pointer_offset(ptr_, offset_);
		return *this;
	}

	/// Adds an offset to each pointer
	CUTLASS_HOST_DEVICE
	TensorRef& add_coord_offset(TensorCoord const& coord) {
		add_pointer_offset(offset(coord));
		return *this;
	}

	/// Returns a TensorRef offset by a given amount
	CUTLASS_HOST_DEVICE
	TensorRef operator+(TensorCoord const& b) const {
		TensorRef result(*this);
		result.add_coord_offset(b);
		return result;
	}

	/// Returns a TensorRef offset by a given amount
	CUTLASS_HOST_DEVICE
	TensorRef& operator+=(TensorCoord const& b) {
		add_coord_offset(b);
		return *this;
	}

	/// Returns a TensorRef offset by a given amount
	CUTLASS_HOST_DEVICE
	TensorRef operator-(TensorCoord const& b) const {
		TensorRef result(*this);
		result.add_pointer_offset(-offset(b));
		return result;
	}

	/// Returns a TensorRef offset by a given amount
	CUTLASS_HOST_DEVICE
	TensorRef& operator-=(TensorCoord const& b) {
		add_pointer_offset(-offset(b));
		return *this;
	}
};

template<int M_, int K_, typename ElementA_, typename LayoutA_, typename ElementB_, typename LayoutB_, typename ElementC_, typename LayoutC_,
	typename ElementAccumulator_ = ElementC_, typename OperatorClass_ = OpClassSimt, typename ArchTag_ = Sm120,
	typename ThreadblockShape_	 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::ThreadblockShape,
	typename WarpShape_			 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::WarpShape,
	typename InstructionShape_	 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::InstructionShape,
	typename EpilogueOutputOp_	 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::EpilogueOutputOp,
	typename ThreadblockSwizzle_ = GemmIdentityThreadblockSwizzle<>,
	int Stages					 = DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kStages,
	int AlignmentA				 = DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kAlignmentA,
	int AlignmentB = DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kAlignmentB, bool SplitKSerial = false,
	typename Operator_ = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::Operator, bool GatherA = false,
	bool GatherB = false, bool ScatterD = false, typename PermuteDLayout = NoPermute>
class Gemm {
  public:
	static constexpr int kM				= M_;
	static constexpr int kK				= K_;
	using ElementA						= ElementA_;
	using LayoutA						= LayoutA_;
	using TensorRefA					= TensorRef<ElementA const, LayoutA>;
	using ElementB						= ElementB_;
	using LayoutB						= LayoutB_;
	using TensorRefB					= TensorRef<ElementB const, LayoutB>;
	using ElementC						= ElementC_;
	using LayoutC						= LayoutC_;
	using TensorRefC					= TensorRef<ElementC const, LayoutC>;
	using TensorRefD					= TensorRef<ElementC, LayoutC>;
	using ElementAccumulator			= ElementAccumulator_;
	using OperatorClass					= OperatorClass_;
	using ArchTag						= ArchTag_;
	using ThreadblockShape				= ThreadblockShape_;
	using WarpShape						= WarpShape_;
	using InstructionShape				= InstructionShape_;
	using EpilogueOutputOp				= EpilogueOutputOp_;
	using ThreadblockSwizzle			= ThreadblockSwizzle_;
	using Operator						= Operator_;
	static constexpr int kStages		= Stages;
	static constexpr int kAlignmentA	= AlignmentA;
	static constexpr int kAlignmentB	= AlignmentB;
	static constexpr int kAlignmentC	= EpilogueOutputOp::kCount;
	static constexpr bool kSplitKSerial = SplitKSerial;
	static constexpr int kTiledM		= (kM + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
	static constexpr int kTiledK		= (kK + ThreadblockShape::kK - 1) / ThreadblockShape::kK;
	/*
		using GemmKernel = typename kernel::DefaultGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC, LayoutC, ElementAccumulator, OperatorClass, ArchTag,
		ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, kStages, kSplitKSerial, Operator, SharedMemoryClearOption::kNone, GatherA, GatherB,
		ScatterD, PermuteDLayout>::GemmKernel;

		struct Arguments {
						
		int N;		TensorRef<ElementA const, LayoutA> ref_A;
		TensorRef<ElementB const, LayoutB> ref_B;
		TensorRef<ElementC const, LayoutC> ref_C;
		TensorRef<ElementC, LayoutC> ref_D;
		typename EpilogueOutputOp::Params epilogue;
		int split_k_slices;
				int const* gather_A_indices;
		int const* gather_B_indices;
		int const* scatter_D_indices;

						
				CUTLASS_HOST_DEVICE 		Arguments() : N(0), split_k_slices(1) {
		}

				CUTLASS_HOST_DEVICE 		Arguments(int N_, TensorRef<ElementA const, LayoutA> ref_A_, TensorRef<ElementB const, LayoutB> ref_B_, TensorRef<ElementC const, LayoutC> ref_C_,
			TensorRef<ElementC, LayoutC> ref_D_, typename EpilogueOutputOp::Params epilogue_ = typename EpilogueOutputOp::Params(), int split_k_slices = 1,
			int const* gather_A_indices_ = nullptr, int const* gather_B_indices_ = nullptr, int const* scatter_D_indices_ = nullptr)
			: N(N_), ref_A(ref_A_), ref_B(ref_B_), ref_C(ref_C_), ref_D(ref_D_), epilogue(epilogue_), split_k_slices(split_k_slices), gather_A_indices(gather_A_indices_),
			  gather_B_indices(gather_B_indices_), scatter_D_indices(scatter_D_indices_) {
		}

				CUTLASS_HOST_DEVICE 		GemmCoord problem_size() const {
			return GemmCoord(kM, N, kK);
		}
	};

  private:
		typename GemmKernel::Params params_;

  public:
		Gemm() {
	}

		static Status can_implement(Arguments const& args) {
		if (!kSplitKSerial && args.split_k_slices > 1) {
			return Status::kErrorInvalidProblem;
		}

				GemmCoord problem_size(kM, args.N, kK);

		Status status = GemmKernel::can_implement(problem_size, args.ref_A.non_const_ref(), args.ref_B.non_const_ref(), args.ref_C.non_const_ref(), args.ref_D);

		if (status != Status::kSuccess) {
			return status;
		}

		return Status::kSuccess;
	}

		static size_t get_workspace_size(Arguments const& args) {
		size_t bytes = 0;

				ThreadblockSwizzle threadblock_swizzle;

				int tiled_n = (args.N + ThreadblockShape::kN - 1) / ThreadblockShape::kN;

				if (kSplitKSerial && args.split_k_slices > 1) {
						bytes += sizeof(int) * size_t(kTiledM) * size_t(tiled_n);
		}

		return bytes;
	}

		Status initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
				GemmCoord problem_size(kM, args.N, kK);

				ThreadblockSwizzle threadblock_swizzle;

		cutlass::gemm::GemmCoord grid_shape =
			threadblock_swizzle.get_tiled_shape(problem_size, { ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK }, args.split_k_slices);

		if (kSplitKSerial) {
			if (args.split_k_slices > 1) {
				if (!workspace) {
					return Status::kErrorWorkspaceNull;
				}

				size_t bytes = get_workspace_size(args);

				cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);

				if (result != cudaSuccess) {
					return Status::kErrorInternal;
				}
			}
		} else {
			if (args.split_k_slices > 1) {
				return Status::kErrorInvalidProblem;
			}
		}

				params_ = typename GemmKernel::Params{ problem_size, grid_shape, args.ref_A.non_const_ref(), args.ref_B.non_const_ref(), args.ref_C.non_const_ref(), args.ref_D,
			args.epilogue, static_cast<int*>(workspace), args.gather_A_indices, args.gather_B_indices, args.scatter_D_indices };

		return Status::kSuccess;
	}

		Status update(Arguments const& args, void* workspace = nullptr) {
		if (kSplitKSerial && args.split_k_slices > 1) {
			if (!workspace) {
				return Status::kErrorWorkspaceNull;
			}
		}

		params_.ref_A.reset(args.ref_A.non_const_ref().data());
		params_.ref_B.reset(args.ref_B.non_const_ref().data());
		params_.ref_C.reset(args.ref_C.non_const_ref().data());
		params_.ref_D.reset(args.ref_D.data());
		params_.output_op = args.epilogue;
		params_.semaphore = static_cast<int*>(workspace);

		return Status::kSuccess;
	}

		Status run(cudaStream_t stream = nullptr) {
		ThreadblockSwizzle threadblock_swizzle;

		dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
		dim3 block(GemmKernel::kThreadCount, 1, 1);

		cudaError_t result;

		int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

		if (smem_size >= (48 << 10)) {
			result = cudaFuncSetAttribute(Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

			if (result != cudaSuccess) {
				return Status::kErrorInternal;
			}
		}

		cutlass::arch::synclog_setup();
		cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);

		result = cudaGetLastError();

		return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
	}

		Status operator()(cudaStream_t stream = nullptr) {
		return run(stream);
	}

		Status operator()(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
		Status status = initialize(args, workspace, stream);

		if (status == Status::kSuccess) {
			status = run(stream);
		}

		return status;
	}*/
};

template<auto multiple, typename value_01_type = decltype(multiple)> BNCH_SWT_INLINE constexpr value_01_type round_up_to_multiple(value_01_type value) noexcept {
	if constexpr ((multiple & (multiple - 1)) == 0) {
		constexpr value_01_type mulSub1{ multiple - 1 };
		constexpr value_01_type notMulSub1{ static_cast<value_01_type>(~mulSub1) };
		return (value + (mulSub1)) & notMulSub1;
	} else {
		const value_01_type remainder = value % multiple;
		return remainder == 0 ? value : value + (multiple - remainder);
	}
}

struct cuda_buffer {
	using value_type													= uint8_t;
	using pointer														= value_type*;
	using size_type														= uint64_t;
	BNCH_SWT_INLINE cuda_buffer() noexcept								= default;
	BNCH_SWT_INLINE cuda_buffer& operator=(const cuda_buffer&) noexcept = delete;
	BNCH_SWT_INLINE cuda_buffer(const cuda_buffer&) noexcept			= delete;

	BNCH_SWT_INLINE cuda_buffer& operator=(cuda_buffer&& other) noexcept {
		if (this != &other) {
			std::swap(data_val, other.data_val);
			std::swap(size_val, other.size_val);
		}
		return *this;
	}

	BNCH_SWT_INLINE cuda_buffer(cuda_buffer&& other) noexcept {
		*this = std::move(other);
	}

	BNCH_SWT_INLINE void init(uint64_t size) {
		if (data_val) {
			clear();
		}

		cudaError_t result = cudaMalloc(&data_val, size);
		if (result != cudaSuccess) {
			data_val = nullptr;
			throw std::runtime_error{ "cuda_buffer - failed to allocate GPU memory" };
		}

		size_val = size;
	}

	BNCH_SWT_INLINE void deinit() noexcept {
		clear();
	}

	BNCH_SWT_INLINE uint64_t size() noexcept {
		return size_val;
	}

	BNCH_SWT_INLINE pointer data() noexcept {
		return data_val;
	}

	BNCH_SWT_INLINE void* claim_memory(uint64_t offset_to_claim) {
		uint64_t aligned_amount = round_up_to_multiple<64>(offset_to_claim);
		if (aligned_amount > size_val) {
			throw std::runtime_error{ "cuda_buffer - not enough memory allocated!" };
		}
		pointer return_value = data_val + aligned_amount;
		return return_value;
	}

	BNCH_SWT_INLINE ~cuda_buffer() noexcept {
		clear();
	}

  protected:
	uint64_t size_val{};
	pointer data_val{};

	BNCH_SWT_INLINE void clear() noexcept {
		if (data_val) {
			cudaError_t result = cudaFree(data_val);
			data_val		   = nullptr;
			size_val		   = 0;
		}
	}
};

using q8_quant = int8_t;

inline static uint16_t fp32_to_fp16(float f) {
	return static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0));
}

struct block_q8_0 {
	static constexpr uint64_t quant_count{ 32 };
	int16_t scale;
	int8_t quants[quant_count];
};

inline block_q8_0 generate_block(const float* x) {
	block_q8_0 return_values{};

	float amax = 0.0f;

	for (int32_t j = 0; j < 32; j++) {
		const float v = x[j];
		amax		  = std::max(amax, fabsf(v));
	}

	const float d  = amax / ((1 << 7) - 1);
	const float id = d ? 1.0f / d : 0.0f;

	return_values.scale = fp32_to_fp16(d);

	for (int32_t j = 0; j < 32; ++j) {
		const float x0 = x[j] * id;

		return_values.quants[j] = roundf(x0);
	}
	return return_values;
}

inline std::vector<std::vector<block_q8_0>> generate_blocks(const std::vector<std::vector<float>>& floats) {
	std::vector<std::vector<block_q8_0>> result;
	result.reserve(floats.size());

	for (const auto& row: floats) {
		const uint64_t row_elements	 = row.size();
		const uint64_t blocks_needed = (row_elements + 31) / 32;

		std::vector<block_q8_0> row_blocks;
		row_blocks.reserve(blocks_needed);
		for (uint64_t x = 0; x < row_elements / 32; ++x) {
			row_blocks.emplace_back(generate_block(row.data() + x * 32));
		}

		result.emplace_back(std::move(row_blocks));
	}

	return result;
}

inline std::vector<std::vector<std::vector<block_q8_0>>> generate_blocks_final(const std::vector<std::vector<std::vector<float>>>& floats) {
	std::vector<std::vector<std::vector<block_q8_0>>> result;
	result.reserve(floats.size());

	for (const auto& values: floats) {
		result.emplace_back(generate_blocks(values));
	}

	return result;
}

inline float generate_llm_float() {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::normal_distribution<float> dist(0.0f, 0.02f);
	float value = dist(gen);
	return std::clamp(value, -0.5f, 0.5f);
}

template<uint64_t dim_00, uint64_t dim_01> inline std::vector<std::vector<float>> generate_floats() {
	std::vector<std::vector<float>> result;
	result.resize(dim_00);
	for (uint64_t x = 0; x < dim_00; ++x) {
		result[x].reserve(dim_01);
	}
	for (uint64_t i = 0; i < dim_00; ++i) {
		for (uint64_t j = 0; j < dim_01; ++j) {
			result[i].emplace_back(generate_llm_float());
		}
	}
	return result;
}

template<uint64_t iteration_count, uint64_t dim_00, uint64_t dim_01> inline std::vector<std::vector<std::vector<float>>> generate_floats_final() {
	std::vector<std::vector<std::vector<float>>> result;
	result.reserve(iteration_count);
	for (uint64_t x = 0; x < iteration_count; ++x) {
		result.emplace_back(generate_floats<dim_00, dim_01>());
	}
	return result;
}

template<typename value_type> inline std::vector<value_type> linearize_values(const std::vector<std::vector<value_type>>& values) {
	std::vector<value_type> return_values{};
	return_values.reserve(values.size() * values[0].size());
	for (uint64_t x = 0; x < values.size(); ++x) {
		for (uint64_t y = 0; y < values[x].size(); ++y) {
			return_values.emplace_back(values[x][y]);
		}
	}
	return return_values;
}

template<typename value_type> inline std::vector<std::vector<value_type>> transpose_values(const std::vector<std::vector<value_type>>& floats) {
	const uint64_t rows = floats.size();
	const uint64_t cols = floats.empty() ? 0 : floats[0].size();

	std::vector<std::vector<value_type>> result;
	result.resize(cols);
	for (uint64_t x = 0; x < cols; ++x) {
		result[x].reserve(rows);
	}

	for (uint64_t i = 0; i < rows; ++i) {
		for (uint64_t j = 0; j < cols; ++j) {
			result[j].emplace_back(floats[i][j]);
		}
	}
	return result;
}

template<typename value_type> inline std::vector<std::vector<std::vector<value_type>>> transpose_values_final(const std::vector<std::vector<std::vector<value_type>>>& floats) {
	std::vector<std::vector<std::vector<value_type>>> result;
	result.reserve(floats.size());
	for (uint64_t x = 0; x < floats.size(); ++x) {
		result.emplace_back(transpose_values(floats[x]));
	}
	return result;
}

template<typename value_type> inline std::vector<std::vector<value_type>> generate_values_final(const std::vector<std::vector<std::vector<value_type>>>& values) {
	std::vector<std::vector<value_type>> return_values{};
	for (uint64_t x = 0; x < values.size(); ++x) {
		return_values.emplace_back(linearize_values(values[x]));
	}
	return return_values;
}

template<uint64_t M, uint64_t K> struct reference_mul_mat_float {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& inputs_a, std::vector<std::vector<float>>& inputs_b,
		std::vector<std::vector<float>>& outputs, uint64_t N) {
		const auto& current_inputs_a = inputs_a[current_index];
		const auto& current_inputs_b = inputs_b[current_index];
		auto& current_outputs		 = outputs[current_index];

		for (uint64_t row = 0; row < M; ++row) {
			for (uint64_t col = 0; col < N; ++col) {
				float sum = 0.0f;
				for (uint64_t k = 0; k < K; ++k) {
					const float a_elem = current_inputs_a[row * K + k];
					const float b_elem = current_inputs_b[k * N + col];
					sum += a_elem * b_elem;
				}
				current_outputs[row * N + col] = sum;
			}
		}
		++current_index;
		return current_outputs.size() * sizeof(float);
	}
};

template<uint64_t M, uint64_t K> struct reference_mul_mat_q8_0 {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<block_q8_0>>& blocks, std::vector<std::vector<float>>& floats,
		std::vector<std::vector<float>>& outputs, uint64_t N) {
		const auto& current_blocks = blocks[current_index];
		const auto& current_floats = floats[current_index];
		auto& current_outputs	   = outputs[current_index];

		for (uint64_t row = 0; row < M; ++row) {
			for (uint64_t col = 0; col < N; ++col) {
				float sum = 0.0f;
				for (uint64_t k = 0; k < K; ++k) {
					const uint64_t block_idx	 = (row * K + k) / 32;
					const uint64_t elem_in_block = (row * K + k) % 32;
					const auto& block			 = current_blocks[block_idx];
					const float scale			 = __half2float(*reinterpret_cast<const __half*>(&block.scale));
					const float a_elem			 = scale * static_cast<float>(block.quants[elem_in_block]);
					const float b_elem			 = current_floats[k * N + col];
					sum += a_elem * b_elem;
				}
				current_outputs[row * N + col] = sum;
			}
		}
		++current_index;
		return current_outputs.size() * sizeof(float);
	}
};

template<typename block_type>
	requires(std::is_same_v<block_q8_0, block_type>)
BNCH_SWT_INLINE constexpr size_t get_byte_size_from_element_count(size_t element_count) {
	constexpr size_t elements_per_block = block_type::quant_count;
	const size_t total_blocks			= (element_count + elements_per_block - 1) / elements_per_block;
	return total_blocks * sizeof(block_type);
}
template<typename block_type>
	requires(std::is_same_v<float, block_type>)
BNCH_SWT_INLINE constexpr size_t get_byte_size_from_element_count(size_t element_count) {
	return element_count * sizeof(block_type);
}

/*
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <vector>
#include <iostream>
#include <memory>

static ggml_backend_t get_ggml_backend() {
	static ggml_backend_t backend = nullptr;
	if (!backend) {
		backend = ggml_backend_cuda_init(0);
		if (!backend) {
			backend = ggml_backend_cpu_init();
		}
	}
	return backend;
}

static ggml_context* get_ggml_context() {
	static ggml_context* ctx = nullptr;
	if (!ctx) {
		struct ggml_init_params params = {
			.mem_size	= 512 * 1024 * 1024,
			.mem_buffer = nullptr,
			.no_alloc	= true,
		};
		ctx = ggml_init(params);
	}
	return ctx;
}

template<uint64_t M, uint64_t K, typename input_type_01, typename input_type_02, typename output_type> struct ggml_cuda_mul_mat {
	inline static ggml_tensor* g_tensor_A = nullptr;
	inline static ggml_tensor* g_tensor_B = nullptr;
	inline static ggml_tensor* g_tensor_C = nullptr;
	inline static ggml_gallocr_t g_allocr = nullptr;
	BNCH_SWT_INLINE static uint64_t impl_prep(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<input_type_01>>& inputs_a,
		std::vector<std::vector<input_type_02>>& inputs_b, std::vector<std::vector<output_type>>& outputs, uint64_t N) {
		ggml_context* ctx	   = get_ggml_context();
		ggml_backend_t backend = get_ggml_backend();

		g_tensor_A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
		g_tensor_B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
		g_tensor_C = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);

		g_allocr				 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
		ggml_cgraph* alloc_graph = ggml_new_graph(ctx);
		ggml_build_forward_expand(alloc_graph, g_tensor_A);
		ggml_build_forward_expand(alloc_graph, g_tensor_B);
		ggml_build_forward_expand(alloc_graph, g_tensor_C);
		ggml_gallocr_alloc_graph(g_allocr, alloc_graph);

		const uint64_t inputs_a_size = get_byte_size_from_element_count<input_type_01>(M * K);
		const uint64_t inputs_b_size = get_byte_size_from_element_count<input_type_02>(K * N);

		uint64_t offset	 = 0;
		g_tensor_A->data = reinterpret_cast<input_type_01*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset			 = round_up_to_multiple<64>(offset + inputs_a_size);

		g_tensor_B->data = reinterpret_cast<input_type_02*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset			 = round_up_to_multiple<64>(offset + inputs_b_size);

		g_tensor_C->data = reinterpret_cast<output_type*>(static_cast<uint8_t*>(buffer.data()) + offset);

		const auto& current_inputs_a = inputs_a[current_index];
		const auto& current_inputs_b = inputs_b[current_index];

		cudaMemcpy(g_tensor_A->data, current_inputs_a.data(), ggml_nbytes(g_tensor_A), cudaMemcpyHostToDevice);
		cudaMemcpy(g_tensor_B->data, current_inputs_b.data(), ggml_nbytes(g_tensor_B), cudaMemcpyHostToDevice);

		return 0;
	}

	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<input_type_01>>& inputs_a,
		std::vector<std::vector<input_type_02>>& inputs_b, std::vector<std::vector<output_type>>& outputs, uint64_t N) {
		ggml_context* ctx	   = get_ggml_context();
		ggml_backend_t backend = get_ggml_backend();

		ggml_cgraph* gf = ggml_new_graph(ctx);

		ggml_tensor* result = ggml_mul_mat(ctx, g_tensor_A, g_tensor_B);

		ggml_build_forward_expand(gf, result);

		ggml_gallocr_alloc_graph(g_allocr, gf);

		ggml_backend_graph_compute(backend, gf);

		ggml_backend_tensor_get(result, g_tensor_C->data, 0, ggml_nbytes(g_tensor_C));
		auto err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "GGML CUDA q8_0 kernel execution failed: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		return ggml_nbytes(g_tensor_A) + ggml_nbytes(g_tensor_B) + ggml_nbytes(g_tensor_C);
	}

	BNCH_SWT_INLINE static uint64_t impl_post(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<input_type_01>>& inputs_a,
		std::vector<std::vector<input_type_02>>& inputs_b, std::vector<std::vector<output_type>>& outputs, uint64_t N) {
		const uint64_t inputs_a_size  = get_byte_size_from_element_count<input_type_01>(M * K);
		const uint64_t inputs_b_size  = get_byte_size_from_element_count<input_type_02>(K * N);
		const uint64_t outputs_C_size = get_byte_size_from_element_count<output_type>(M * N);

		uint64_t offset	   = round_up_to_multiple<64>(inputs_a_size);
		offset			   = round_up_to_multiple<64>(offset + inputs_b_size);
		output_type* C_ptr = reinterpret_cast<output_type*>(buffer.data() + offset);

		auto& previous_outputs = outputs[current_index];
		cudaError_t err		   = cudaMemcpy(previous_outputs.data(), C_ptr, outputs_C_size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::cerr << "Failed to copy previous outputs from device: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		err = cudaMemset(C_ptr, 0, outputs_C_size);
		if (err != cudaSuccess) {
			std::cerr << "Failed to zero output buffer: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		++current_index;
		return 0;
	}
};
*/
template<typename value_type> using base_type = std::remove_cvref_t<value_type>;

template<typename value_type> using x_type = decltype(base_type<value_type>::x);

template<typename value_type>
concept uint_cuda_types = std::is_unsigned_v<x_type<value_type>> && std::is_integral_v<x_type<value_type>>;

template<typename value_type>
concept int_cuda_types = std::is_signed_v<x_type<value_type>> && std::is_integral_v<x_type<value_type>> && !uint_cuda_types<value_type>;

template<typename value_type>
concept int8_cuda_types = int_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

template<typename value_type>
concept int16_cuda_types = int_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

template<typename value_type>
concept int32_cuda_types = int_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept int64_cuda_types = int_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept uint8_cuda_types = uint_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

template<typename value_type>
concept uint16_cuda_types = uint_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

template<typename value_type>
concept uint32_cuda_types = uint_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept uint64_cuda_types = uint_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept float_cuda_types = std::floating_point<x_type<value_type>>;

template<typename value_type>
concept float32_cuda_types = float_cuda_types<value_type> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept float64_cuda_types = float_cuda_types<value_type> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept r_value_reference_types = std::is_rvalue_reference_v<value_type>;

template<typename value_type> BNCH_SWT_INLINE __device__ constexpr value_type&& device_forward(value_type& arg) noexcept {
	return static_cast<value_type&&>(arg);
}

template<r_value_reference_types value_type> BNCH_SWT_INLINE __device__ constexpr value_type device_forward(value_type arg) noexcept {
	return static_cast<value_type&&>(arg);
}

enum class get_value_type_errors {
	invalid_type,
};

template<typename value_type>
concept dim04_types = requires() { base_type<value_type>::w; };

template<typename value_type>
concept dim03_types = requires() { base_type<value_type>::z; } && !dim04_types<value_type>;

template<typename value_type>
concept dim02_types = requires() { base_type<value_type>::y; } && !dim03_types<value_type> && !dim04_types<value_type>;

template<typename value_type>
concept dim01_types = requires() { base_type<value_type>::x; } && !dim02_types<value_type> && !dim03_types<value_type> && !dim04_types<value_type>;

template<typename value_type>
concept dim_types = requires() { base_type<value_type>::x; };

template<auto enum_error, typename... types> struct error_printer_impl;

template<bool value, auto enum_error, typename... value_to_test> struct static_assert_printer {
	static constexpr bool impl{ [] {
		if constexpr (!value) {
			error_printer_impl<enum_error, value_to_test...>::failure_value;
			return false;
		} else {
			return true;
		}
	}() };
};

template<auto enum_error, auto... values> struct error_printer_impl_val;

template<bool value, auto enum_error, auto... values> struct static_assert_printer_val {
	static constexpr bool impl{ [] {
		if constexpr (!value) {
			error_printer_impl_val<enum_error, values...>::failure_value;
			return false;
		} else {
			return true;
		}
	}() };
};

template<typename value_type> struct get_value_type {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr decltype(auto) impl(value_types&&... args) noexcept {};
};

template<int8_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr decltype(auto) impl(value_types&&... args) noexcept {
		if constexpr (dim01_types<value_type>) {
			return make_char1(device_forward<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_char2(device_forward<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_char3(device_forward<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_char4(device_forward<value_types>(args)...);
		}
	}
};

template<int16_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr decltype(auto) impl(value_types&&... args) noexcept {
		if constexpr (dim01_types<value_type>) {
			return make_short1(device_forward<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_short2(device_forward<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_short3(device_forward<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_short4(device_forward<value_types>(args)...);
		}
	}
};

template<int32_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr decltype(auto) impl(value_types&&... args) noexcept {
		if constexpr (dim01_types<value_type>) {
			return make_int1(device_forward<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_int2(device_forward<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_int3(device_forward<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_int4(device_forward<value_types>(args)...);
		}
	}
};

template<int64_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr decltype(auto) impl(value_types&&... args) noexcept {
		if constexpr (dim01_types<value_type>) {
			return make_long1(device_forward<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_long2(device_forward<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_long3(device_forward<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_long4(device_forward<value_types>(args)...);
		}
	}
};

template<uint8_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr decltype(auto) impl(value_types&&... args) noexcept {
		if constexpr (dim01_types<value_type>) {
			return make_uchar1(device_forward<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_uchar2(device_forward<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_uchar3(device_forward<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_uchar4(device_forward<value_types>(args)...);
		}
	}
};

template<uint16_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr decltype(auto) impl(value_types&&... args) noexcept {
		if constexpr (dim01_types<value_type>) {
			return make_ushort1(device_forward<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_ushort2(device_forward<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_ushort3(device_forward<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_ushort4(device_forward<value_types>(args)...);
		}
	}
};

template<uint32_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr decltype(auto) impl(value_types&&... args) noexcept {
		if constexpr (dim01_types<value_type>) {
			return make_uint1(device_forward<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_uint2(device_forward<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_uint3(device_forward<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_uint4(device_forward<value_types>(args)...);
		}
	}
};

template<uint64_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr decltype(auto) impl(value_types&&... args) noexcept {
		if constexpr (dim01_types<value_type>) {
			return make_ulong1(device_forward<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_ulong2(device_forward<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_ulong3(device_forward<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_ulong4(device_forward<value_types>(args)...);
		}
	}
};

template<float32_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr decltype(auto) impl(value_types&&... args) noexcept {
		if constexpr (dim01_types<value_type>) {
			return make_float1(device_forward<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_float2(device_forward<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_float3(device_forward<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_float4(device_forward<value_types>(args)...);
		}
	}
};

template<float64_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr decltype(auto) impl(value_types&&... args) noexcept {
		if constexpr (dim01_types<value_type>) {
			return make_double1(device_forward<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_double2(device_forward<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_double3(device_forward<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_double4(device_forward<value_types>(args)...);
		}
	}
};

enum class binary_op_types {
	add,
	mul,
	sub,
	div,
};

template<binary_op_types> struct binary_op_core;

template<> struct binary_op_core<binary_op_types::add> {
	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ decltype(auto) impl(value_type01&& val01, value_type02&& val02) noexcept {
		return device_forward<value_type01>(val01) + static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ void impl_in_place(value_type01& val01, value_type02&& val02) noexcept {
		val01 += static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
	}
};

template<> struct binary_op_core<binary_op_types::mul> {
	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ decltype(auto) impl(value_type01&& val01, value_type02&& val02) noexcept {
		return device_forward<value_type01>(val01) * static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ void impl_in_place(value_type01& val01, value_type02&& val02) noexcept {
		val01 *= static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
	}
};

template<> struct binary_op_core<binary_op_types::sub> {
	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ decltype(auto) impl(value_type01&& val01, value_type02&& val02) noexcept {
		return device_forward<value_type01>(val01) - static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ void impl_in_place(value_type01& val01, value_type02&& val02) noexcept {
		val01 -= static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
	}
};

template<> struct binary_op_core<binary_op_types::div> {
	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ decltype(auto) impl(value_type01&& val01, value_type02&& val02) noexcept {
		return device_forward<value_type01>(val01) / static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ void impl_in_place(value_type01& val01, value_type02&& val02) noexcept {
		val01 /= static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
	}
};

template<typename value_type, binary_op_types binary_op_type> struct binary_op_base;

template<dim01_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ decltype(auto) impl(value_type01&& val01, value_type02&& val02) noexcept {
		using op_core_type = binary_op_core<binary_op_type>;
		return get_value_type<value_type01>::impl(op_core_type::impl(device_forward<value_type01>(val01).x, device_forward<value_type02>(val02).x));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ void impl_in_place(value_type01& val01, value_type02&& val02) noexcept {
		using op_core_type = binary_op_core<binary_op_type>;
		op_core_type::impl_in_place(val01.x, device_forward<value_type02>(val02).x);
	}
};

template<dim02_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ decltype(auto) impl(value_type01&& val01, value_type02&& val02) noexcept {
		using op_core_type = binary_op_core<binary_op_type>;
		return get_value_type<value_type01>::impl(op_core_type::impl(device_forward<value_type01>(val01).x, device_forward<value_type02>(val02).x),
			op_core_type::impl(device_forward<value_type01>(val01).y, device_forward<value_type02>(val02).y));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ void impl_in_place(value_type01& val01, value_type02&& val02) noexcept {
		using op_core_type = binary_op_core<binary_op_type>;
		op_core_type::impl_in_place(val01.x, device_forward<value_type02>(val02).x);
		op_core_type::impl_in_place(val01.y, device_forward<value_type02>(val02).y);
	}
};

template<dim03_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ decltype(auto) impl(value_type01&& val01, value_type02&& val02) noexcept {
		using op_core_type = binary_op_core<binary_op_type>;
		return get_value_type<value_type01>::impl(op_core_type::impl(device_forward<value_type01>(val01).x, device_forward<value_type02>(val02).x),
			op_core_type::impl(device_forward<value_type01>(val01).y, device_forward<value_type02>(val02).y),
			op_core_type::impl(device_forward<value_type01>(val01).z, device_forward<value_type02>(val02).z));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ void impl_in_place(value_type01& val01, value_type02&& val02) noexcept {
		using op_core_type = binary_op_core<binary_op_type>;
		op_core_type::impl_in_place(val01.x, device_forward<value_type02>(val02).x);
		op_core_type::impl_in_place(val01.y, device_forward<value_type02>(val02).y);
		op_core_type::impl_in_place(val01.z, device_forward<value_type02>(val02).z);
	}
};

template<dim04_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ decltype(auto) impl(value_type01&& val01, value_type02&& val02) noexcept {
		using op_core_type = binary_op_core<binary_op_type>;
		return get_value_type<value_type01>::impl(op_core_type::impl(device_forward<value_type01>(val01).x, device_forward<value_type02>(val02).x),
			op_core_type::impl(device_forward<value_type01>(val01).y, device_forward<value_type02>(val02).y),
			op_core_type::impl(device_forward<value_type01>(val01).z, device_forward<value_type02>(val02).z),
			op_core_type::impl(device_forward<value_type01>(val01).w, device_forward<value_type02>(val02).w));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ void impl_in_place(value_type01& val01, value_type02&& val02) noexcept {
		using op_core_type = binary_op_core<binary_op_type>;
		op_core_type::impl_in_place(val01.x, device_forward<value_type02>(val02).x);
		op_core_type::impl_in_place(val01.y, device_forward<value_type02>(val02).y);
		op_core_type::impl_in_place(val01.z, device_forward<value_type02>(val02).z);
		op_core_type::impl_in_place(val01.w, device_forward<value_type02>(val02).w);
	}
};

template<binary_op_types binary_op_type> struct binary_op {
	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ decltype(auto) impl(value_type01&& val01, value_type02&& val02) noexcept {
		return binary_op_base<value_type01, binary_op_type>::impl(device_forward<value_type01>(val01), device_forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_INLINE static __device__ decltype(auto) impl_in_place(value_type01& val01, value_type02&& val02) noexcept {
		return binary_op_base<value_type01, binary_op_type>::impl_in_place(val01, device_forward<value_type02>(val02));
	}
};

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_INLINE __device__ decltype(auto) operator+=(value_type01& val01, value_type02&& val02) noexcept {
	return binary_op<binary_op_types::add>::impl_in_place(val01, device_forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_INLINE __device__ decltype(auto) operator+(value_type01&& val01, value_type02&& val02) noexcept {
	return binary_op<binary_op_types::add>::impl(device_forward<value_type01>(val01), device_forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_INLINE __device__ decltype(auto) operator*=(value_type01& val01, value_type02&& val02) noexcept {
	return binary_op<binary_op_types::mul>::impl_in_place(val01, device_forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_INLINE __device__ decltype(auto) operator*(value_type01&& val01, value_type02&& val02) noexcept {
	return binary_op<binary_op_types::mul>::impl(device_forward<value_type01>(val01), device_forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_INLINE __device__ decltype(auto) operator-=(value_type01& val01, value_type02&& val02) noexcept {
	return binary_op<binary_op_types::sub>::impl_in_place(val01, device_forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_INLINE __device__ decltype(auto) operator-(value_type01&& val01, value_type02&& val02) noexcept {
	return binary_op<binary_op_types::sub>::impl(device_forward<value_type01>(val01), device_forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_INLINE __device__ decltype(auto) operator/=(value_type01& val01, value_type02&& val02) noexcept {
	return binary_op<binary_op_types::div>::impl_in_place(val01, device_forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_INLINE __device__ decltype(auto) operator/(value_type01&& val01, value_type02&& val02) noexcept {
	return binary_op<binary_op_types::div>::impl(device_forward<value_type01>(val01), device_forward<value_type02>(val02));
}

struct gpu_properties {
	static constexpr uint64_t sm_count{ 70ull };
	static constexpr uint64_t max_threads_per_sm{ 1536ull };
	static constexpr uint64_t max_threads_per_block{ 1024ull };
	static constexpr uint64_t warp_size{ 32ull };
	static constexpr uint64_t l2_cache_size{ 50331648ull };
	static constexpr uint64_t shared_mem_per_block{ 49152ull };
	static constexpr uint64_t memory_bus_width{ 256ull };
	static constexpr uint64_t memory_clock_rate{ 14001000ull };
	static constexpr uint64_t major_compute_capability{ 12ull };
	static constexpr uint64_t minor_compute_capability{ 0ull };
	static constexpr uint64_t max_grid_size_x{ 2147483647ull };
	static constexpr uint64_t gpu_arch_index{ 4ull };
	static constexpr uint64_t total_threads{ 107520ull };
	static constexpr uint64_t optimal_block_size{ 512ull };
	static constexpr uint64_t optimal_grid_size{ 210ull };
};

template<uint64_t block_m, uint64_t block_n, uint64_t block_k, uint64_t warp_m_new, uint64_t warp_n_new, uint64_t thread_m_new, uint64_t thread_n_new> struct cuda_kernel_traits {
	static constexpr uint64_t block_tile_m		= block_m;
	static constexpr uint64_t block_tile_n		= block_n;
	static constexpr uint64_t block_tile_k		= block_k;
	static constexpr uint64_t warp_tile_m		= warp_m_new;
	static constexpr uint64_t warp_tile_n		= warp_n_new;
	static constexpr uint64_t thread_tile_m		= thread_m_new;
	static constexpr uint64_t thread_tile_n		= thread_n_new;
	static constexpr uint64_t warps_m			= block_m / warp_m_new;
	static constexpr uint64_t warps_n			= block_n / warp_n_new;
	static constexpr uint64_t threads_per_warp	= gpu_properties::warp_size;
	static constexpr uint64_t threads_per_block = warps_m * warps_n * threads_per_warp;

	static_assert(block_m > 0, "block_m must be greater than 0");
	static_assert(block_n > 0, "block_n must be greater than 0");
	static_assert(block_k > 0, "block_k must be greater than 0");
	static_assert(warp_m_new > 0, "warp_m must be greater than 0");
	static_assert(warp_n_new > 0, "warp_n must be greater than 0");
	static_assert(thread_m_new > 0, "thread_m must be greater than 0");
	static_assert(thread_n_new > 0, "thread_n must be greater than 0");

	static_assert(block_m % warp_m_new == 0, "block_m must be evenly divisible by warp_m");
	static_assert(block_n % warp_n_new == 0, "block_n must be evenly divisible by warp_n");

	static_assert(warp_m_new % thread_m_new == 0, "warp_m must be evenly divisible by thread_m");
	static_assert(warp_n_new % thread_n_new == 0, "warp_n must be evenly divisible by thread_n");

	static_assert((warp_m_new / thread_m_new) * (warp_n_new / thread_n_new) == gpu_properties::warp_size, "Warp configuration must result in exactly warp_size threads per warp");

	static_assert(threads_per_block <= gpu_properties::max_threads_per_block, "threads_per_block cannot exceed max_threads_per_block");
	static_assert(threads_per_block >= gpu_properties::warp_size, "threads_per_block must be at least warp_size");

	static_assert(block_m <= 512, "block_m should not exceed 512 for reasonable shared memory usage");
	static_assert(block_n <= 512, "block_n should not exceed 512 for reasonable shared memory usage");
	static_assert(block_k <= 64, "block_k should not exceed 64 for reasonable register usage");

	static_assert(block_k % 4 == 0, "block_k should be a multiple of 4 for vectorized loads");

	static_assert(thread_m_new <= 8, "thread_m should not exceed 8 for reasonable register usage");
	static_assert(thread_n_new <= 8, "thread_n should not exceed 8 for reasonable register usage");

	static_assert(warps_m > 0 && warps_n > 0, "Must have at least one warp in each dimension");
	static_assert(warps_m * warps_n <= 32, "Total warps per block should not exceed 32");

	static constexpr uint64_t shared_mem_usage = 2 * (block_m * block_k + block_k * block_n) * sizeof(float);
	static_assert(shared_mem_usage <= gpu_properties::shared_mem_per_block, "Estimated shared memory usage exceeds shared_mem_per_block limit");

	static_assert(threads_per_block % gpu_properties::warp_size == 0, "threads_per_block must be a multiple of warp_size");
};

template<uint64_t M, uint64_t K, typename traits>
__device__ __forceinline__ void load_smem_tile_A(float* smem_A, const block_q8_0* A_global, uint64_t N, uint64_t k_offset, uint64_t block_row) noexcept {
	constexpr uint64_t block_m			 = traits::block_tile_m;
	constexpr uint64_t block_k			 = traits::block_tile_k;
	constexpr uint64_t threads_per_block = traits::threads_per_block;
	const uint64_t tid					 = threadIdx.x;
	const uint64_t k_blocks				 = (K + 31) / 32;
	const uint64_t elements_per_block	 = block_m * block_k;
	const uint64_t vec4_elements		 = elements_per_block / 4;
	const uint64_t vec4_per_thread		 = (vec4_elements + threads_per_block - 1) / threads_per_block;
	for (uint64_t i = 0; i < vec4_per_thread; ++i) {
		const uint64_t vec4_idx							 = tid + i * threads_per_block;
		const uint64_t linear_idx						 = vec4_idx * 4;
		const uint64_t row								 = linear_idx / block_k;
		const uint64_t col								 = linear_idx % block_k;
		const uint64_t global_row						 = block_row + row;
		const uint64_t global_col						 = k_offset + col;
		const uint64_t q8_block_row						 = global_row;
		const uint64_t q8_block_col						 = global_col / 32;
		const uint64_t q8_elem_idx						 = global_col % 32;
		const uint64_t q8_block_idx						 = q8_block_row * k_blocks + q8_block_col;
		const block_q8_0& q8_block						 = A_global[q8_block_idx];
		const float scale_raw							 = __half2float(*reinterpret_cast<const __half*>(&q8_block.scale));
		const uint64_t smem_offset						 = row * block_k + col;
		*reinterpret_cast<float4*>(&smem_A[smem_offset]) = make_float4(static_cast<float>(q8_block.quants[q8_elem_idx]), static_cast<float>(q8_block.quants[q8_elem_idx + 1]),
															   static_cast<float>(q8_block.quants[q8_elem_idx + 2]), static_cast<float>(q8_block.quants[q8_elem_idx + 3])) *
			make_float4(scale_raw, scale_raw, scale_raw, scale_raw);
	}
}

template<uint64_t M, uint64_t K, typename traits>
__device__ __forceinline__ void load_smem_tile_B(float* smem_B, const float* B_global, uint64_t N, uint64_t k_offset, uint64_t block_col) noexcept {
	constexpr uint64_t block_n			 = traits::block_tile_n;
	constexpr uint64_t block_k			 = traits::block_tile_k;
	constexpr uint64_t threads_per_block = traits::threads_per_block;

	const uint64_t tid					 = threadIdx.x;
	const uint64_t vec4_cols_per_row	 = block_n / 4;
	const uint64_t total_vec4_loads		 = block_k * vec4_cols_per_row;
	const uint64_t vec4_loads_per_thread = (total_vec4_loads + threads_per_block - 1) / threads_per_block;

	for (uint64_t i = 0; i < vec4_loads_per_thread; ++i) {
		const uint64_t vec4_idx = tid + i * threads_per_block;
		if (vec4_idx < total_vec4_loads) {
			const uint64_t row		= vec4_idx / vec4_cols_per_row;
			const uint64_t vec4_col = vec4_idx % vec4_cols_per_row;
			const uint64_t col		= vec4_col * 4;

			const uint64_t global_row = k_offset + row;
			const uint64_t global_col = block_col + col;

			if (global_row < K && global_col + 3 < N) {
				const uint64_t global_offset					 = global_row * N + global_col;
				const uint64_t smem_offset						 = row * block_n + col;
				*reinterpret_cast<float4*>(&smem_B[smem_offset]) = *reinterpret_cast<const float4*>(&B_global[global_offset]);
			} else {
				for (uint64_t elem = 0; elem < 4; ++elem) {
					const uint64_t elem_global_col = global_col + elem;
					const uint64_t elem_col		   = col + elem;
					if (global_row < K && elem_global_col < N && elem_col < block_n) {
						smem_B[row * block_n + elem_col] = B_global[global_row * N + elem_global_col];
					}
				}
			}
		}
	}
}

template<typename traits> __device__ __forceinline__ void compute_warp_tile(float* smem_A, float* smem_B, float accumulator[traits::thread_tile_m][traits::thread_tile_n],
	uint64_t warp_row, uint64_t warp_col) noexcept {
	constexpr uint64_t warp_m	= traits::warp_tile_m;
	constexpr uint64_t warp_n	= traits::warp_tile_n;
	constexpr uint64_t thread_m = traits::thread_tile_m;
	constexpr uint64_t thread_n = traits::thread_tile_n;
	constexpr uint64_t block_k	= traits::block_tile_k;
	constexpr uint64_t block_n	= traits::block_tile_n;
	constexpr uint64_t block_m	= traits::block_tile_m;

	const uint64_t lane_id		   = threadIdx.x % 32;
	const uint64_t threads_per_row = warp_n / thread_n;
	const uint64_t thread_row	   = lane_id / threads_per_row;
	const uint64_t thread_col	   = lane_id % threads_per_row;

	if constexpr (thread_m % 4 == 0 && thread_n % 4 == 0) {
		float4 frag_A[thread_m / 4];
		float4 frag_B[thread_n / 4];

		for (uint64_t k = 0; k < block_k; ++k) {
			for (uint64_t tm_vec = 0; tm_vec < thread_m / 4; ++tm_vec) {
				const uint64_t base_row	   = warp_row + thread_row * thread_m + tm_vec * 4;
				const uint64_t smem_offset = base_row * block_k + k;

				frag_A[tm_vec] = make_float4(smem_A[smem_offset], smem_A[smem_offset + block_k], smem_A[smem_offset + 2 * block_k], smem_A[smem_offset + 3 * block_k]);
			}

			for (uint64_t tn_vec = 0; tn_vec < thread_n / 4; ++tn_vec) {
				const uint64_t base_col	   = warp_col + thread_col * thread_n + tn_vec * 4;
				const uint64_t smem_offset = k * block_n + base_col;

				frag_B[tn_vec] = *reinterpret_cast<const float4*>(&smem_B[smem_offset]);
			}

			for (uint64_t tm_vec = 0; tm_vec < thread_m / 4; ++tm_vec) {
				for (uint64_t tn_vec = 0; tn_vec < thread_n / 4; ++tn_vec) {
					const float4& a_vec = frag_A[tm_vec];
					const float4& b_vec = frag_B[tn_vec];

					accumulator[tm_vec * 4][tn_vec * 4] += a_vec.x * b_vec.x;
					accumulator[tm_vec * 4][tn_vec * 4 + 1] += a_vec.x * b_vec.y;
					accumulator[tm_vec * 4][tn_vec * 4 + 2] += a_vec.x * b_vec.z;
					accumulator[tm_vec * 4][tn_vec * 4 + 3] += a_vec.x * b_vec.w;

					accumulator[tm_vec * 4 + 1][tn_vec * 4] += a_vec.y * b_vec.x;
					accumulator[tm_vec * 4 + 1][tn_vec * 4 + 1] += a_vec.y * b_vec.y;
					accumulator[tm_vec * 4 + 1][tn_vec * 4 + 2] += a_vec.y * b_vec.z;
					accumulator[tm_vec * 4 + 1][tn_vec * 4 + 3] += a_vec.y * b_vec.w;

					accumulator[tm_vec * 4 + 2][tn_vec * 4] += a_vec.z * b_vec.x;
					accumulator[tm_vec * 4 + 2][tn_vec * 4 + 1] += a_vec.z * b_vec.y;
					accumulator[tm_vec * 4 + 2][tn_vec * 4 + 2] += a_vec.z * b_vec.z;
					accumulator[tm_vec * 4 + 2][tn_vec * 4 + 3] += a_vec.z * b_vec.w;

					accumulator[tm_vec * 4 + 3][tn_vec * 4] += a_vec.w * b_vec.x;
					accumulator[tm_vec * 4 + 3][tn_vec * 4 + 1] += a_vec.w * b_vec.y;
					accumulator[tm_vec * 4 + 3][tn_vec * 4 + 2] += a_vec.w * b_vec.z;
					accumulator[tm_vec * 4 + 3][tn_vec * 4 + 3] += a_vec.w * b_vec.w;
				}
			}
		}
	} else {
		float frag_A[thread_m];
		float frag_B[thread_n];

		for (uint64_t k = 0; k < block_k; ++k) {
			for (uint64_t tm = 0; tm < thread_m; ++tm) {
				const uint64_t smem_row = warp_row + thread_row * thread_m + tm;
				if (smem_row < block_m) {
					frag_A[tm] = smem_A[smem_row * block_k + k];
				}
			}

			for (uint64_t tn = 0; tn < thread_n; ++tn) {
				const uint64_t smem_col = warp_col + thread_col * thread_n + tn;
				if (smem_col < block_n) {
					frag_B[tn] = smem_B[k * block_n + smem_col];
				}
			}

			for (uint64_t tm = 0; tm < thread_m; ++tm) {
				for (uint64_t tn = 0; tn < thread_n; ++tn) {
					accumulator[tm][tn] += frag_A[tm] * frag_B[tn];
				}
			}
		}
	}
}

template<typename traits> __device__ __forceinline__ void store_output_tile(float* C_global, float accumulator[traits::thread_tile_m][traits::thread_tile_n], uint64_t M,
	uint64_t N, uint64_t block_row, uint64_t block_col, uint64_t warp_row, uint64_t warp_col) noexcept {
	constexpr uint64_t thread_m = traits::thread_tile_m;
	constexpr uint64_t thread_n = traits::thread_tile_n;
	constexpr uint64_t warp_n	= traits::warp_tile_n;

	const uint64_t lane_id		   = threadIdx.x % 32;
	const uint64_t threads_per_row = warp_n / thread_n;
	const uint64_t thread_row	   = lane_id / threads_per_row;
	const uint64_t thread_col	   = lane_id % threads_per_row;

#pragma unroll
	for (uint64_t tm = 0; tm < thread_m; ++tm) {
#pragma unroll
		for (uint64_t tn = 0; tn < thread_n; ++tn) {
			const uint64_t global_row = block_row + warp_row + thread_row * thread_m + tm;
			const uint64_t global_col = block_col + warp_col + thread_col * thread_n + tn;

			if (global_row < M && global_col < N) {
				C_global[global_row * N + global_col] = accumulator[tm][tn];
			}
		}
	}
}

template<uint64_t M, uint64_t K, typename traits> __global__ void nihilus_gemm_kernel(const block_q8_0* A, const float* B, float* C, uint64_t N) {
	constexpr uint64_t block_m	= traits::block_tile_m;
	constexpr uint64_t block_n	= traits::block_tile_n;
	constexpr uint64_t block_k	= traits::block_tile_k;
	constexpr uint64_t warp_m	= traits::warp_tile_m;
	constexpr uint64_t warp_n	= traits::warp_tile_n;
	constexpr uint64_t thread_m = traits::thread_tile_m;
	constexpr uint64_t thread_n = traits::thread_tile_n;
	constexpr uint64_t warps_m	= traits::warps_m;
	constexpr uint64_t warps_n	= traits::warps_n;

	__shared__ float smem_A[2][block_m * block_k];
	__shared__ float smem_B[2][block_k * block_n];

	const uint64_t block_row = blockIdx.y * block_m;
	const uint64_t block_col = blockIdx.x * block_n;

	const uint64_t warp_id	= threadIdx.x / 32;
	const uint64_t warp_row = (warp_id / warps_n) * warp_m;
	const uint64_t warp_col = (warp_id % warps_n) * warp_n;

	float accumulator[thread_m][thread_n];
#pragma unroll
	for (uint64_t tm = 0; tm < thread_m; ++tm) {
#pragma unroll
		for (uint64_t tn = 0; tn < thread_n; ++tn) {
			accumulator[tm][tn] = 0.0f;
		}
	}

	uint64_t smem_write_stage = 0;
	uint64_t smem_read_stage  = 0;

	load_smem_tile_A<M, K, traits>(smem_A[smem_write_stage], A, N, 0, block_row);
	load_smem_tile_B<M, K, traits>(smem_B[smem_write_stage], B, N, 0, block_col);
	__syncthreads();

	for (uint64_t k_tile = 0; k_tile < K; k_tile += block_k) {
		smem_read_stage	 = smem_write_stage;
		smem_write_stage = 1 - smem_write_stage;

		if (k_tile + block_k < K) {
			load_smem_tile_A<M, K, traits>(smem_A[smem_write_stage], A, N, k_tile + block_k, block_row);
			load_smem_tile_B<M, K, traits>(smem_B[smem_write_stage], B, N, k_tile + block_k, block_col);
		}

		compute_warp_tile<traits>(smem_A[smem_read_stage], smem_B[smem_read_stage], accumulator, warp_row, warp_col);

		__syncthreads();
	}

	store_output_tile<traits>(C, accumulator, M, N, block_row, block_col, warp_row, warp_col);
}

using mul_mat_1_to_128 = cuda_kernel_traits<32, 64, 16, 16, 32, 4, 4>;

__device__ constexpr uint64_t log2_constexpr(uint64_t value) noexcept {
	static_assert(sizeof(uint64_t) <= 8, "Only up to 64-bit supported");
	return (value < 2) ? 0 : 1 + log2_constexpr(value >> 1);
}

__device__ constexpr bool is_power_of_two(unsigned long long value) noexcept {
	return value != 0 && (value & (value - 1)) == 0;
}

template<uint64_t quants_per_block> __forceinline__ __device__ uint64_t get_block_index(uint64_t index) noexcept {
	if constexpr (is_power_of_two(quants_per_block)) {
		static constexpr uint64_t power{ log2_constexpr(quants_per_block) };
		return index >> power;
	} else {
		return index / quants_per_block;
	}
}

template<uint64_t quants_per_block> __forceinline__ __device__ uint64_t get_elem_in_block(uint64_t index) noexcept {
	if constexpr (is_power_of_two(quants_per_block)) {
		static constexpr uint64_t mask{ quants_per_block - 1 };
		return index & mask;
	} else {
		return index % quants_per_block;
	}
}

template<typename value_type>
	requires(std::is_same_v<std::remove_cvref_t<value_type>, float>)
__forceinline__ __device__ decltype(auto) convert_scale(value_type&& value) noexcept {
	return std::forward<value_type>(value);
};

template<typename value_type>
	requires(std::is_same_v<std::remove_cvref_t<value_type>, int16_t>)
__forceinline__ __device__ decltype(auto) convert_scale(value_type&& value) noexcept {
	return __half2float(*reinterpret_cast<const __half*>(&std::forward<value_type>(value)));
};

template<typename blocks_type> __global__ void dequantize_blocks(const blocks_type* input_blocks, float* output, uint64_t total_elements) {
	const uint64_t idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t stride = blockDim.x * gridDim.x;

	for (uint64_t i = idx; i < total_elements; i += stride) {
		const uint64_t block_idx	 = get_block_index<blocks_type::quant_count>(i);
		const uint64_t elem_in_block = get_elem_in_block<blocks_type::quant_count>(i);
		const blocks_type& block	 = input_blocks[block_idx];
		const float scale			 = convert_scale(block.scale);
		output[i]					 = scale * static_cast<float>(block.quants[elem_in_block]);
	}
}

template<typename blocks_type> __global__ void cutlass_dequantize_blocks(const blocks_type* input_blocks, float* output, uint64_t total_elements) {
	const uint64_t idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t stride = blockDim.x * gridDim.x;

	for (uint64_t i = idx; i < total_elements; i += stride) {
		const uint64_t block_idx	 = i / 32;
		const uint64_t elem_in_block = i % 32;
		const blocks_type& block	 = input_blocks[block_idx];
		const float scale			 = convert_scale(block.scale);
		output[i]					 = scale * static_cast<float>(block.quants[elem_in_block]);
	}
}

#include <cutlass_base/gemm/device/gemm.h>

template<typename blocks_type> __global__ void dequantize_block(const blocks_type& input_blocks, float* output, uint64_t total_elements) {
	for (uint64_t i = 0; i < blocks_type::quant_count; ++i) {
		const uint64_t block_idx	 = get_block_index<blocks_type::quant_count>(i);
		const uint64_t elem_in_block = get_elem_in_block<blocks_type::quant_count>(i);
		const blocks_type& block	 = input_blocks[block_idx];
		const float scale			 = convert_scale(block.scale);
		output[i]					 = scale * static_cast<float>(block.quants[elem_in_block]);
	}
}

template<uint64_t M, uint64_t K, typename input_type_01, typename input_type_02, typename output_type> struct cutlass_base_mul_mat {
	using element_a = float;
	using element_b = float;
	using element_c = float;
	using layout_a	= cutlass_base::layout::RowMajor;
	using layout_b	= cutlass_base::layout::RowMajor;
	using layout_c	= cutlass_base::layout::RowMajor;

	BNCH_SWT_INLINE static uint64_t impl_prep(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<input_type_01>>& inputs_a,
		std::vector<std::vector<input_type_02>>& inputs_b, std::vector<std::vector<output_type>>& outputs, uint64_t N) noexcept {
		const uint64_t inputs_a_size = get_byte_size_from_element_count<input_type_01>(M * K);
		const uint64_t inputs_b_size = get_byte_size_from_element_count<input_type_02>(K * N);

		uint64_t offset		 = 0;
		input_type_01* A_ptr = reinterpret_cast<input_type_01*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset				 = round_up_to_multiple<64>(offset + inputs_a_size);

		input_type_02* B_ptr = reinterpret_cast<input_type_02*>(static_cast<uint8_t*>(buffer.data()) + offset);

		const auto& current_inputs_a = inputs_a[current_index];
		const auto& current_inputs_b = inputs_b[current_index];

		cudaError_t err = cudaMemcpy(A_ptr, current_inputs_a.data(), inputs_a_size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "Failed to copy Matrix A to device: " << cudaGetErrorString(err) << std::endl;
		}

		err = cudaMemcpy(B_ptr, current_inputs_b.data(), inputs_b_size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "Failed to copy Matrix B to device: " << cudaGetErrorString(err) << std::endl;
		}

		return 0;
	}

	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<input_type_01>>& inputs_a,
		std::vector<std::vector<input_type_02>>& inputs_b, std::vector<std::vector<output_type>>& outputs, uint64_t N) noexcept {
		const uint64_t inputs_a_size  = get_byte_size_from_element_count<input_type_01>(M * K);
		const uint64_t inputs_b_size  = get_byte_size_from_element_count<input_type_02>(K * N);
		const uint64_t outputs_C_size = get_byte_size_from_element_count<output_type>(M * N);

		uint64_t offset	   = 0;
		offset			   = round_up_to_multiple<64>(offset + inputs_a_size);
		offset			   = round_up_to_multiple<64>(offset + inputs_b_size);
		output_type* C_ptr = reinterpret_cast<output_type*>(static_cast<uint8_t*>(buffer.data()) + offset);

		using index_type			 = cutlass_base::gemm::GemmCoord::Index;
		using cutlass_base_gemm_type = cutlass_base::gemm::device::Gemm<element_a, layout_a, element_b, layout_b, element_c, layout_c, element_c>;

		if constexpr (std::is_same_v<input_type_01, float>) {
			offset			   = 0;
			const float* A_ptr = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
			offset			   = round_up_to_multiple<64>(offset + inputs_a_size);

			const float* B_ptr = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);

			cutlass_base_gemm_type gemm_op;
			cutlass_base::Status status = gemm_op({ { static_cast<index_type>(M), static_cast<index_type>(N), static_cast<index_type>(K) }, { A_ptr, static_cast<index_type>(K) },
				{ B_ptr, static_cast<index_type>(N) }, { C_ptr, static_cast<index_type>(N) }, { C_ptr, static_cast<index_type>(N) }, { 1.0f, 0.0f } });

			if (status != cutlass_base::Status::kSuccess) {
				std::cerr << "Cutlass float32 Gemm failed: " << cutlass_base::cutlassGetStatusString(status) << std::endl;
			}
		} else if constexpr (std::is_same_v<input_type_01, block_q8_0>) {
			constexpr uint64_t blocks_per_row = K / block_q8_0::quant_count;
			constexpr uint64_t total_blocks_A = M * blocks_per_row;
			const uint64_t dequant_A_size	  = (M * K) * sizeof(float);

			offset						  = 0;
			const block_q8_0* A_quant_ptr = reinterpret_cast<const block_q8_0*>(static_cast<uint8_t*>(buffer.data()) + offset);
			offset						  = round_up_to_multiple<64>(offset + inputs_a_size);

			const float* B_ptr = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
			offset			   = round_up_to_multiple<64>(offset + inputs_b_size);
			offset			   = round_up_to_multiple<64>(offset + outputs_C_size);

			float* A_dequant_ptr = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);

			const uint64_t total_elements = M * K;

			cutlass_dequantize_blocks<<<(total_elements + 255) / 256, 256>>>(A_quant_ptr, A_dequant_ptr, total_elements);
			auto err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cerr << "GGML CUDA q8_0 kernel execution failed: " + std::string(cudaGetErrorString(err)) << std::endl;
			}

			cutlass_base_gemm_type gemm_op;
			cutlass_base::Status status =
				gemm_op({ { static_cast<index_type>(M), static_cast<index_type>(N), static_cast<index_type>(K) }, { A_dequant_ptr, static_cast<index_type>(K) },
					{ B_ptr, static_cast<index_type>(N) }, { C_ptr, static_cast<index_type>(N) }, { C_ptr, static_cast<index_type>(N) }, { 1.0f, 0.0f } });

			if (status != cutlass_base::Status::kSuccess) {
				std::cerr << "Cutlass Q8_0 Gemm failed: " << cutlass_base::cutlassGetStatusString(status) << std::endl;
			}
		}

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error after Cutlass Gemm: " << cudaGetErrorString(err) << std::endl;
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "GGML CUDA q8_0 kernel execution failed: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		return inputs_a_size + inputs_b_size + outputs_C_size;
	}

	BNCH_SWT_INLINE static uint64_t impl_post(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<input_type_01>>& inputs_a,
		std::vector<std::vector<input_type_02>>& inputs_b, std::vector<std::vector<output_type>>& outputs, uint64_t N) noexcept {
		const uint64_t inputs_a_size  = get_byte_size_from_element_count<input_type_01>(M * K);
		const uint64_t inputs_b_size  = get_byte_size_from_element_count<input_type_02>(K * N);
		const uint64_t outputs_C_size = get_byte_size_from_element_count<output_type>(M * N);

		uint64_t offset	   = round_up_to_multiple<64>(inputs_a_size);
		offset			   = round_up_to_multiple<64>(offset + inputs_b_size);
		output_type* C_ptr = reinterpret_cast<output_type*>(buffer.data() + offset);

		auto& current_outputs = outputs[current_index];
		cudaError_t err		  = cudaMemcpy(current_outputs.data(), C_ptr, outputs_C_size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::cerr << "Failed to copy outputs from device: " << cudaGetErrorString(err) << std::endl;
		}

		err = cudaMemset(C_ptr, 0, outputs_C_size);
		if (err != cudaSuccess) {
			std::cerr << "Failed to zero output buffer: " << cudaGetErrorString(err) << std::endl;
		}

		++current_index;
		return 0;
	}
};

#include <cutlass_new/gemm/device/gemm.h>

template<uint64_t M, uint64_t K> __global__ void nihilus_custom_cuda_kernel(const float* input_A, const float* input_B, float* output, uint64_t N) {
	const uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= M || col >= N)
		return;
	float sum = 0.0f;

	const uint64_t k_end = K & ~3;
	uint64_t k			 = 0;

	for (; k < k_end; k += 4) {
#pragma unroll
		for (uint64_t i = 0; i < 4; ++i) {
			const uint64_t k_idx = k + i;
			const float a_elem	 = input_A[row * K + k_idx];
			const float b_elem	 = input_B[k_idx * N + col];
			sum += a_elem * b_elem;
		}
	}

	output[row * N + col] = sum;
}

template<uint64_t M, uint64_t K, typename input_type_01, typename input_type_02, typename output_type> struct nihilus_mul_mat {
	using element_a = float;
	using element_b = float;
	using element_c = float;
	using layout_a	= cutlass::layout::RowMajor;
	using layout_b	= cutlass::layout::RowMajor;
	using layout_c	= cutlass::layout::RowMajor;

	BNCH_SWT_INLINE static uint64_t impl_prep(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<input_type_01>>& inputs_a,
		std::vector<std::vector<input_type_02>>& inputs_b, std::vector<std::vector<output_type>>& outputs, uint64_t N) {
		const uint64_t inputs_a_size = get_byte_size_from_element_count<input_type_01>(M * K);
		const uint64_t inputs_b_size = get_byte_size_from_element_count<input_type_02>(K * N);

		uint64_t offset		 = 0;
		input_type_01* A_ptr = reinterpret_cast<input_type_01*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset				 = round_up_to_multiple<64>(offset + inputs_a_size);

		input_type_02* B_ptr = reinterpret_cast<input_type_02*>(static_cast<uint8_t*>(buffer.data()) + offset);

		const auto& current_inputs_a = inputs_a[current_index];
		const auto& current_inputs_b = inputs_b[current_index];

		cudaError_t err = cudaMemcpy(A_ptr, current_inputs_a.data(), inputs_a_size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "Failed to copy Matrix A to device: " << cudaGetErrorString(err) << std::endl;
		}

		err = cudaMemcpy(B_ptr, current_inputs_b.data(), inputs_b_size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "Failed to copy Matrix B to device: " << cudaGetErrorString(err) << std::endl;
		}

		return 0;
	}

	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<input_type_01>>& inputs_a,
		std::vector<std::vector<input_type_02>>& inputs_b, std::vector<std::vector<output_type>>& outputs, uint64_t N) {
		const uint64_t inputs_a_size  = get_byte_size_from_element_count<input_type_01>(M * K);
		const uint64_t inputs_b_size  = get_byte_size_from_element_count<input_type_02>(K * N);
		const uint64_t outputs_C_size = get_byte_size_from_element_count<output_type>(M * N);

		uint64_t offset	   = 0;
		offset			   = round_up_to_multiple<64>(offset + inputs_a_size);
		offset			   = round_up_to_multiple<64>(offset + inputs_b_size);
		output_type* C_ptr = reinterpret_cast<output_type*>(static_cast<uint8_t*>(buffer.data()) + offset);

		using index_type = cutlass::gemm::GemmCoord::Index;


		if constexpr (std::is_same_v<input_type_01, float>) {
			offset			   = 0;
			const float* A_ptr = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
			offset			   = round_up_to_multiple<64>(offset + inputs_a_size);

			const float* B_ptr		= reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
			using nihilus_gemm_type = cutlass::gemm::device::Gemm<M, K, element_a, layout_a, element_b, layout_b, element_c, layout_c>;
			nihilus_gemm_type op;
			cutlass::Status status = op({ static_cast<index_type>(N), { A_ptr, static_cast<index_type>(K) }, { B_ptr, static_cast<index_type>(N) },
				{ C_ptr, static_cast<index_type>(N) }, { C_ptr, static_cast<index_type>(N) }, { 1.0f, 0.0f } });

			if (status != cutlass::Status::kSuccess) {
				std::cerr << "Nihilus float32 Gemm failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
			}
		} else if constexpr (std::is_same_v<input_type_01, block_q8_0>) {
			const uint64_t dequant_A_size = get_byte_size_from_element_count<input_type_02>(M * K);
			offset						  = 0;
			const block_q8_0* A_quant_ptr = reinterpret_cast<const block_q8_0*>(static_cast<uint8_t*>(buffer.data()) + offset);
			offset						  = round_up_to_multiple<64>(offset + inputs_a_size);

			const float* B_ptr = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
			offset			   = round_up_to_multiple<64>(offset + inputs_b_size);
			offset			   = round_up_to_multiple<64>(offset + outputs_C_size);

			float* A_dequant_ptr = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);

			const uint64_t total_elements = M * K;

			dequantize_blocks<<<(total_elements + 255) / 256, 256>>>(A_quant_ptr, A_dequant_ptr, total_elements);

			using nihilus_gemm_type = cutlass::gemm::device::Gemm<M, K, element_a, layout_a, element_b, layout_b, element_c, layout_c>;
			nihilus_gemm_type op;
			cutlass::Status status = op({ static_cast<index_type>(N), { A_dequant_ptr, static_cast<index_type>(K) }, { B_ptr, static_cast<index_type>(N) },
				{ C_ptr, static_cast<index_type>(N) }, { C_ptr, static_cast<index_type>(N) }, { 1.0f, 0.0f } });

			if (status != cutlass::Status::kSuccess) {
				std::cerr << "Nihilus Q8_0 Gemm failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
			}
		}

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error after Nihilus Gemm: " << cudaGetErrorString(err) << std::endl;
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "GGML CUDA q8_0 kernel execution failed: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		return inputs_a_size + inputs_b_size + outputs_C_size;
	}

	BNCH_SWT_INLINE static uint64_t impl_post(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<input_type_01>>& inputs_a,
		std::vector<std::vector<input_type_02>>& inputs_b, std::vector<std::vector<output_type>>& outputs, uint64_t N) {
		const uint64_t inputs_a_size  = get_byte_size_from_element_count<input_type_01>(M * K);
		const uint64_t inputs_b_size  = get_byte_size_from_element_count<input_type_02>(K * N);
		const uint64_t outputs_C_size = get_byte_size_from_element_count<output_type>(M * N);

		uint64_t offset	   = round_up_to_multiple<64>(inputs_a_size);
		offset			   = round_up_to_multiple<64>(offset + inputs_b_size);
		output_type* C_ptr = reinterpret_cast<output_type*>(buffer.data() + offset);

		auto& current_outputs = outputs[current_index];
		cudaError_t err		  = cudaMemcpy(current_outputs.data(), C_ptr, outputs_C_size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::cerr << "Failed to copy outputs from device: " << cudaGetErrorString(err) << std::endl;
		}

		err = cudaMemset(C_ptr, 0, outputs_C_size);
		if (err != cudaSuccess) {
			std::cerr << "Failed to zero output buffer: " << cudaGetErrorString(err) << std::endl;
		}

		++current_index;
		return 0;
	}
};

template<bnch_swt::string_literal rhs> inline bool compare_floats(float val1, float val2, uint64_t row, uint64_t col) {
	static constexpr float relative_tolerance	= 0.15f;
	static constexpr float absolute_tolerance	= 1e-7f;
	static constexpr float tiny_value_threshold = 1e-6f;
	if (val1 == val2)
		return true;

	const float abs_val1 = std::abs(val1);
	const float abs_val2 = std::abs(val2);
	const float abs_diff = std::abs(val1 - val2);

	if (abs_val1 < tiny_value_threshold && abs_val2 < tiny_value_threshold) {
		return abs_diff <= absolute_tolerance;
	}

	const float max_val = std::max(abs_val1, abs_val2);
	if (abs_diff <= relative_tolerance * max_val) {
		return true;
	}

	std::cerr << rhs.operator std::string_view() << ": Mismatch at 0[" << 0 << "] position[" << row << "," << col << "]: Ref Val: " << val1 << " vs Incorrect Val: " << val2
			  << std::endl;
	std::cerr << "Absolute difference: " << abs_diff << ", Relative difference: " << (abs_diff / max_val) * 100.0f << "%" << std::endl;
	return false;
}

template<uint64_t M, uint64_t K, uint64_t matB_dim_00, uint64_t N, bnch_swt::string_literal rhs>
inline void compare_outputs(const std::vector<std::vector<float>>& outputs01, const std::vector<std::vector<float>>& outputs02) {
	static_assert(matB_dim_00 == K, "matB_dim_00 should equal K for matrix multiplication");

	if (outputs01.size() != outputs02.size()) {
		std::cerr << rhs.operator std::string_view() << ": Unequal 0 count! " << outputs01.size() << " vs " << outputs02.size() << std::endl;
		return;
	}

	constexpr uint64_t expected_size = M * N;

	if (outputs01[0].size() != outputs02[0].size()) {
		std::cerr << rhs.operator std::string_view() << ": Unequal matrix sizes at 0 " << 0 << "! " << outputs01[0].size() << " vs " << outputs02[0].size() << std::endl;
		return;
	}

	if (outputs01[0].size() != expected_size) {
		std::cerr << rhs.operator std::string_view() << ": Unexpected matrix size at 0 " << 0 << "! Expected " << expected_size << ", got " << outputs01[0].size() << std::endl;
		return;
	}

	for (uint64_t row = 0; row < M; ++row) {
		for (uint64_t col = 0; col < N; ++col) {
			const uint64_t idx = row * N + col;

			const float val1 = outputs01[0][idx];
			const float val2 = outputs02[0][idx];

			if (!compare_floats<rhs>(val1, val2, row, col)) {
				std::cerr << "\n--- Additional Diagnostic: Checking Last Element ---" << std::endl;

				const uint64_t last_row = M - 1;
				const uint64_t last_col = N - 1;
				const uint64_t last_idx = last_row * N + last_col;

				const float last_val1 = outputs01[0][last_idx];
				const float last_val2 = outputs02[0][last_idx];

				if (compare_floats<rhs>(last_val1, last_val2, last_row, last_col)) {
					std::cerr << "Last element comparison PASSED - suggests localized error" << std::endl;
				} else {
					std::cerr << "Last element comparison FAILED - suggests systematic error" << std::endl;
				}

				std::cerr << "\n--- Additional Diagnostic Elements ---" << std::endl;

				const uint64_t mid_row = M / 2;
				const uint64_t mid_col = N / 2;
				const uint64_t mid_idx = mid_row * N + mid_col;

				const float mid_val1 = outputs01[0][mid_idx];
				const float mid_val2 = outputs02[0][mid_idx];

				if (compare_floats<rhs>(mid_val1, mid_val2, mid_row, mid_col)) {
					std::cerr << "Middle element [" << mid_row << "," << mid_col << "] comparison PASSED" << std::endl;
				} else {
					std::cerr << "Middle element [" << mid_row << "," << mid_col << "] comparison FAILED" << std::endl;
				}

				const uint64_t last_row_first_col_idx = last_row * N + 0;
				const float last_row_first_val1		  = outputs01[0][last_row_first_col_idx];
				const float last_row_first_val2		  = outputs02[0][last_row_first_col_idx];

				if (compare_floats<rhs>(last_row_first_val1, last_row_first_val2, last_row, 0)) {
					std::cerr << "Last row, first column [" << last_row << ",0] comparison PASSED" << std::endl;
				} else {
					std::cerr << "Last row, first column [" << last_row << ",0] comparison FAILED" << std::endl;
				}

				return;
			}
		}
	}

	std::cout << rhs.operator std::string_view() << ": All output comparisons passed!" << std::endl;
}

template<typename input_type_01, uint64_t M, uint64_t K, uint64_t mat_b_dim_00, uint64_t N>
	requires(std::is_same_v<input_type_01, block_q8_0>)
[[msvc::noinline]] void test_function() {
	static constexpr uint64_t total_elements_C{ M * N };
	std::vector<std::vector<block_q8_0>> inputs_a{ generate_values_final(generate_blocks_final(generate_floats_final<total_iterations, M, K>())) };
	std::vector<std::vector<float>> inputs_b{ generate_values_final(generate_floats_final<total_iterations, K, N>()) };
	std::vector<std::vector<float>> outputs01{};
	std::vector<std::vector<float>> outputs02{};
	std::vector<std::vector<float>> outputs03{};
	std::vector<std::vector<float>> outputs04{};
	outputs01.resize(total_iterations);
	outputs02.resize(total_iterations);
	outputs03.resize(total_iterations);
	outputs04.resize(total_iterations);
	for (uint64_t x = 0; x < total_iterations; ++x) {
		outputs01[x].resize(total_elements_C);
		outputs02[x].resize(total_elements_C);
		outputs03[x].resize(total_elements_C);
		outputs04[x].resize(total_elements_C);
	}
	static constexpr bnch_swt::string_literal stage_name{ "(Q8_0 * F32) mul_mat: [" + bnch_swt::internal::toStringLiteral<M>() + "x" + bnch_swt::internal::toStringLiteral<K>() +
		" * " + bnch_swt::internal::toStringLiteral<mat_b_dim_00>() + "x" + bnch_swt::internal::toStringLiteral<N>() + "]" };
	constexpr uint64_t total_elements_A = M * K;
	constexpr uint64_t total_elements_B = K * N;
	constexpr uint64_t blocks_per_row	= K / block_q8_0::quant_count;
	constexpr uint64_t total_blocks_A	= M * blocks_per_row;
	constexpr uint64_t quantized_A_size = total_blocks_A * sizeof(block_q8_0);
	constexpr uint64_t inputs_b_size	= total_elements_B * sizeof(float);
	constexpr uint64_t floats_C_size	= total_elements_C * sizeof(float);
	constexpr uint64_t dequant_A_size	= total_elements_A * sizeof(float);
	uint64_t total_buffer_size			= 0;
	total_buffer_size += round_up_to_multiple<64>(quantized_A_size);
	total_buffer_size += round_up_to_multiple<64>(inputs_b_size);
	total_buffer_size += round_up_to_multiple<64>(floats_C_size);
	total_buffer_size += round_up_to_multiple<64>(dequant_A_size);
	cuda_buffer buffer{};
	buffer.init(total_buffer_size);
	uint64_t current_index{};

	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrepAndPost<"cutlass_base_mul_mat_q8_0",
		cutlass_base_mul_mat<M, K, block_q8_0, float, float>>(buffer, current_index, inputs_a, inputs_b, outputs01, N);

	current_index = 0;

	current_index = 0;
	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrepAndPost<"nihilus_mul_mat_q8_0",
		nihilus_mul_mat<M, K, block_q8_0, float, float>>(buffer, current_index, inputs_a, inputs_b, outputs03, N);

	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::printResults();
	compare_outputs<M, K, mat_b_dim_00, N, "nihilus_mul_mat_float">(outputs01, outputs03);
}

template<typename input_type_01, uint64_t M, uint64_t K, uint64_t mat_b_dim_00, uint64_t N>
	requires(std::is_same_v<input_type_01, float>)
[[msvc::noinline]] void test_function() {
	static constexpr uint64_t total_elements_C{ M * N };
	std::vector<std::vector<float>> inputs_a{ generate_values_final(generate_floats_final<total_iterations, M, K>()) };
	std::vector<std::vector<float>> inputs_b{ generate_values_final(generate_floats_final<total_iterations, K, N>()) };
	std::vector<std::vector<float>> outputs01{};
	std::vector<std::vector<float>> outputs02{};
	std::vector<std::vector<float>> outputs03{};
	std::vector<std::vector<float>> outputs04{};
	outputs01.resize(total_iterations);
	outputs02.resize(total_iterations);
	outputs03.resize(total_iterations);
	outputs04.resize(total_iterations);
	for (uint64_t x = 0; x < total_iterations; ++x) {
		outputs01[x].resize(total_elements_C);
		outputs02[x].resize(total_elements_C);
		outputs03[x].resize(total_elements_C);
		outputs04[x].resize(total_elements_C);
	}

	static constexpr bnch_swt::string_literal stage_name{ "(F32 * F32) mul_mat: [" + bnch_swt::internal::toStringLiteral<M>() + "x" + bnch_swt::internal::toStringLiteral<K>() +
		" * " + bnch_swt::internal::toStringLiteral<mat_b_dim_00>() + "x" + bnch_swt::internal::toStringLiteral<N>() + "]" };
	constexpr uint64_t total_elements_A = M * K;
	constexpr uint64_t total_elements_B = K * N;
	constexpr uint64_t inputs_a_size	= total_elements_A * sizeof(float);
	constexpr uint64_t inputs_b_size	= total_elements_B * sizeof(float);
	constexpr uint64_t floats_C_size	= total_elements_C * sizeof(float);

	uint64_t total_buffer_size = 0;
	total_buffer_size += round_up_to_multiple<64>(inputs_a_size);
	total_buffer_size += round_up_to_multiple<64>(inputs_b_size);
	total_buffer_size += round_up_to_multiple<64>(floats_C_size);

	cuda_buffer buffer{};
	buffer.init(total_buffer_size);

	uint64_t current_index{};
	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrepAndPost<"cutlass_base_mul_mat_float",
		cutlass_base_mul_mat<M, K, float, float, float>>(buffer, current_index, inputs_a, inputs_b, outputs01, N);

	current_index = 0;

	current_index = 0;
	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrepAndPost<"nihilus_mul_mat_float",
		nihilus_mul_mat<M, K, float, float, float>>(buffer, current_index, inputs_a, inputs_b, outputs03, N);

	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::printResults();
	compare_outputs<M, K, mat_b_dim_00, N, "nihilus_mul_mat_float">(outputs01, outputs03);
};

int32_t main() {
	/*
	test_function<float, 4096, 4096, 4096, 1>();
	test_function<float, 4096, 4096, 4096, 2>();
	test_function<float, 4096, 4096, 4096, 3>();
	test_function<float, 4096, 4096, 4096, 4>();
	test_function<float, 4096, 4096, 4096, 5>();
	test_function<float, 4096, 4096, 4096, 8>();
	test_function<float, 4096, 4096, 4096, 13>();
	test_function<float, 4096, 4096, 4096, 16>();
	test_function<float, 4096, 4096, 4096, 25>();
	test_function<float, 4096, 4096, 4096, 32>();
	test_function<float, 4096, 4096, 4096, 49>();
	test_function<float, 4096, 4096, 4096, 64>();
	test_function<float, 4096, 4096, 4096, 97>();
	test_function<float, 4096, 4096, 4096, 128>();
	test_function<float, 4096, 4096, 4096, 193>();
	test_function<float, 4096, 4096, 4096, 256>();
	test_function<float, 4096, 4096, 4096, 385>();
	test_function<float, 4096, 4096, 4096, 512>();
	test_function<float, 4096, 4096, 4096, 769>();
	test_function<float, 4096, 4096, 4096, 1024>();
	test_function<float, 4096, 4096, 4096, 1537>();
	test_function<float, 4096, 4096, 4096, 2048>();
	test_function<float, 4096, 4096, 4096, 3073>();
	test_function<float, 4096, 4096, 4096, 4096>();
	test_function<float, 4096, 4096, 4096, 6145>();
	test_function<float, 4096, 4096, 4096, 8192>();
	test_function<float, 4096, 4096, 4096, 16384>();
	test_function<float, 14336, 4096, 4096, 1>();
	test_function<float, 14336, 4096, 4096, 2>();
	test_function<float, 14336, 4096, 4096, 4>();
	test_function<float, 14336, 4096, 4096, 8>();
	test_function<float, 14336, 4096, 4096, 16>();
	test_function<float, 14336, 4096, 4096, 32>();
	test_function<float, 14336, 4096, 4096, 64>();
	test_function<float, 14336, 4096, 4096, 128>();
	test_function<float, 14336, 4096, 4096, 256>();
	test_function<float, 14336, 4096, 4096, 512>();
	test_function<float, 14336, 4096, 4096, 1024>();
	test_function<float, 14336, 4096, 4096, 2048>();
	test_function<float, 14336, 4096, 4096, 4096>();
	test_function<float, 14336, 4096, 4096, 8192>();
	test_function<float, 14336, 4096, 4096, 16384>();
	test_function<block_q8_0, 4096, 4096, 4096, 1>();
	test_function<block_q8_0, 4096, 4096, 4096, 2>();
	test_function<block_q8_0, 4096, 4096, 4096, 4>();
	test_function<block_q8_0, 4096, 4096, 4096, 8>();
	test_function<block_q8_0, 4096, 4096, 4096, 16>();
	test_function<block_q8_0, 4096, 4096, 4096, 32>();
	test_function<block_q8_0, 4096, 4096, 4096, 64>();
	test_function<block_q8_0, 4096, 4096, 4096, 128>();
	test_function<block_q8_0, 4096, 4096, 4096, 256>();
	test_function<block_q8_0, 4096, 4096, 4096, 512>();
	test_function<block_q8_0, 4096, 4096, 4096, 1024>();
	test_function<block_q8_0, 4096, 4096, 4096, 2048>();
	test_function<block_q8_0, 4096, 4096, 4096, 4096>();
	test_function<block_q8_0, 4096, 4096, 4096, 8192>();
	test_function<block_q8_0, 4096, 4096, 4096, 16384>();
	test_function<block_q8_0, 14336, 4096, 4096, 1>();
	test_function<block_q8_0, 14336, 4096, 4096, 2>();
	test_function<block_q8_0, 14336, 4096, 4096, 4>();
	test_function<block_q8_0, 14336, 4096, 4096, 8>();
	test_function<block_q8_0, 14336, 4096, 4096, 16>();
	test_function<block_q8_0, 14336, 4096, 4096, 32>();
	test_function<block_q8_0, 14336, 4096, 4096, 64>();
	test_function<block_q8_0, 14336, 4096, 4096, 128>();
	*/
	test_function<float, 4096, 4096, 4096, 32>();
	using layout_a = cutlass::layout::RowMajor;
	using layout_b = cutlass::layout::RowMajor;
	using layout_c = cutlass::layout::RowMajor;
	test_function<block_q8_0, 4096, 4096, 4096, 32>();
	using nihilus_gemm_type = Gemm<4096, 4096, float, layout_a, float, layout_b, float, layout_c>;
	nihilus_gemm_type op;
	/*
	test_function<block_q8_0, 14336, 4096, 4096, 512>();
	test_function<block_q8_0, 14336, 4096, 4096, 1024>();
	test_function<block_q8_0, 14336, 4096, 4096, 2048>();
	test_function<block_q8_0, 14336, 4096, 4096, 4096>();
	test_function<block_q8_0, 14336, 4096, 4096, 8192>();
	test_function<block_q8_0, 14336, 4096, 4096, 16384>();*/
	return 0;
}