#include <BnchSwt/BenchmarkSuite.hpp>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>
#include <array>
#include <queue>
#include <latch>
#include <bit>
#include <cstring>
#include <algorithm>
#include <cmath>

struct alignas(64) atomic_flag_wrapper {
	BNCH_SWT_INLINE atomic_flag_wrapper() noexcept = default;
	BNCH_SWT_INLINE atomic_flag_wrapper& operator=(const atomic_flag_wrapper&) noexcept {
		return *this;
	}

	BNCH_SWT_INLINE atomic_flag_wrapper(const atomic_flag_wrapper&) noexcept {
	}

	BNCH_SWT_INLINE void store(int64_t value_new) {
		flag.store(value_new, std::memory_order_release);
	}

	BNCH_SWT_INLINE int64_t load() {
		return flag.load(std::memory_order_release);
	}

	BNCH_SWT_INLINE void clear() {
		flag.store(0, std::memory_order_release);
	}

	BNCH_SWT_INLINE void test_and_set() {
		flag.store(1, std::memory_order_release);
	}

	BNCH_SWT_INLINE void notify_one() {
		flag.notify_one();
	}

	BNCH_SWT_INLINE void notify_all() {
		flag.notify_all();
	}

	BNCH_SWT_INLINE int64_t fetch_add(int64_t value) {
		return flag.fetch_add(value, std::memory_order_acquire);
	}

	BNCH_SWT_INLINE int64_t fetch_sub(int64_t value) {
		return flag.fetch_sub(value, std::memory_order_acquire);
	}

	BNCH_SWT_INLINE bool test() {
		return flag.load(std::memory_order_acquire) == 1;
	}

	BNCH_SWT_INLINE void wait(int64_t value) {
		flag.wait(value, std::memory_order_acquire);
	}

  protected:
	alignas(64) std::atomic<int64_t> flag{};
	char padding[56]{};
};


template<typename derived_type>
struct print_stuff {
	BNCH_SWT_INLINE static void impl() {
		std::cout << "Printing stuff: " << typeid(derived_type).name() << std::endl;
	}
};


struct alignas(64) op_latch {
	BNCH_SWT_INLINE op_latch()							 = default;
	BNCH_SWT_INLINE op_latch& operator=(const op_latch&) = delete;
	BNCH_SWT_INLINE op_latch(const op_latch&)			 = delete;
	alignas(64) atomic_flag_wrapper flag{};
	alignas(64) uint64_t thread_count{};

	BNCH_SWT_INLINE void init(uint64_t thread_count_new) {
		thread_count = thread_count_new;
		flag.store(0);
	}

	BNCH_SWT_INLINE void arrive_and_wait() {
		auto new_value = flag.fetch_add(1);
		bool wait{ !(new_value == thread_count - 1) };
		((wait) && (flag.wait(new_value + 1), print_stuff<op_latch>::impl(), true) || (flag.notify_all(), flag.store(0), print_stuff<op_latch>::impl(), false));
	}
};

struct alignas(64) op_latch_branched {
	BNCH_SWT_INLINE op_latch_branched()									   = default;
	BNCH_SWT_INLINE op_latch_branched& operator=(const op_latch_branched&) = delete;
	BNCH_SWT_INLINE op_latch_branched(const op_latch_branched&)			   = delete;
	alignas(64) atomic_flag_wrapper flag{};
	alignas(64) uint64_t thread_count{};

	BNCH_SWT_INLINE void init(uint64_t thread_count_new) {
		thread_count = thread_count_new;
		flag.store(0);
	}

	BNCH_SWT_INLINE void arrive_and_wait() {
		auto new_value = flag.fetch_add(1);
		bool wait{ !(new_value == thread_count - 1) };
		if (wait) {
			flag.wait(new_value + 1);
			print_stuff<op_latch_branched>::impl();
		} else {
			flag.notify_all(), flag.store(0);
			print_stuff<op_latch_branched>::impl();
		}
	}
};

struct alignas(64) op_latch_ternary {
	BNCH_SWT_INLINE op_latch_ternary()									 = default;
	BNCH_SWT_INLINE op_latch_ternary& operator=(const op_latch_ternary&) = delete;
	BNCH_SWT_INLINE op_latch_ternary(const op_latch_ternary&)			 = delete;
	alignas(64) atomic_flag_wrapper flag{};
	alignas(64) uint64_t thread_count{};

	BNCH_SWT_INLINE void init(uint64_t thread_count_new) {
		thread_count = thread_count_new;
		flag.store(0);
	}

	BNCH_SWT_INLINE void arrive_and_wait() {
		auto new_value = flag.fetch_add(1);
		bool wait{ !(new_value == thread_count - 1) };
		wait ? (flag.wait(new_value + 1), print_stuff<op_latch_ternary>::impl()) : (flag.notify_all(), flag.store(0), print_stuff<op_latch_ternary>::impl());
	}
};

struct alignas(64) op_latch_fn_ptrs {
	BNCH_SWT_INLINE op_latch_fn_ptrs()									 = default;
	BNCH_SWT_INLINE op_latch_fn_ptrs& operator=(const op_latch_fn_ptrs&) = delete;
	BNCH_SWT_INLINE op_latch_fn_ptrs y(const op_latch_fn_ptrs&)			 = delete;
	alignas(64) atomic_flag_wrapper flag{};
	alignas(64) uint64_t thread_count{};
	
	template<size_t index> BNCH_SWT_INLINE void call_function(int64_t new_value) {
		if constexpr (index == 0) {
			(flag.wait(new_value + 1), print_stuff<op_latch_fn_ptrs>::impl());
		} else {
			flag.notify_all(), flag.store(0), print_stuff<op_latch_fn_ptrs>::impl();
		}
	}

	using function_type = decltype(&op_latch_fn_ptrs::call_function<0>);

	static constexpr std::array<function_type, 2> func_ptrs{ [] {
		std::array<function_type, 2> return_values{};
		return_values[0] = &op_latch_fn_ptrs::call_function<0>;
		return_values[1] = &op_latch_fn_ptrs::call_function<1>;
		return return_values;
	}() };

	BNCH_SWT_INLINE void init(uint64_t thread_count_new) {
		thread_count = thread_count_new;
		flag.store(0);
	}

	BNCH_SWT_INLINE void arrive_and_wait() {
		auto new_value = flag.fetch_add(1);
		bool wait{ (new_value == thread_count - 1) };
		(*this.*func_ptrs[wait])(new_value);
	}
};

int main(int argc, char** argv) {
	op_latch latch01{};
	latch01.init(1);
	latch01.arrive_and_wait();
	op_latch_branched latch02{};
	latch02.init(1);
	latch02.arrive_and_wait();

	op_latch_ternary latch03{};
	latch03.init(1);
	latch03.arrive_and_wait();

	op_latch_fn_ptrs latch04{};
	latch04.init(1);
	latch04.arrive_and_wait();
	return 0;
}