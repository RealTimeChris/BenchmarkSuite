#include <BnchSwt/BenchmarkSuite.hpp>
#include<source_location>
#include <iostream>
#include <vector>
#include <array>
#include <bit>
#include <thread>
#include <latch>
#include <mutex>


enum class oiml_type {
	float_32	   = 0,
	float_16	   = 1,
	q8_0		   = 8,
	int_8		   = 24,
	int_32		   = 26,
	brain_float_16 = 30,
	i8x32xf32x1	   = 36,
	count		   = 39,
};

enum class oiml_op {
	BNCH_SWT_OP_NONE = 0,
	BNCH_SWT_OP_MUL,
	BNCH_SWT_OP_RMS_NORM,
	BNCH_SWT_OP_GET_ROWS,
	BNCH_SWT_OP_DUP,
	BNCH_SWT_OP_ADD,
	BNCH_SWT_OP_ADD1,
	BNCH_SWT_OP_ACC,
	BNCH_SWT_OP_SUB,
	BNCH_SWT_OP_DIV,
	BNCH_SWT_OP_SQR,
	BNCH_SWT_OP_SQRT,
	BNCH_SWT_OP_LOG,
	BNCH_SWT_OP_SIN,
	BNCH_SWT_OP_COS,
	BNCH_SWT_OP_SUM,
	BNCH_SWT_OP_SUM_ROWS,
	BNCH_SWT_OP_MEAN,
	BNCH_SWT_OP_ARGMAX,
	BNCH_SWT_OP_COUNT_EQUAL,
	BNCH_SWT_OP_REPEAT,
	BNCH_SWT_OP_REPEAT_BACK,
	BNCH_SWT_OP_CONCAT,
	BNCH_SWT_OP_SILU_BACK,
	BNCH_SWT_OP_NORM,// normalize
	BNCH_SWT_OP_RMS_NORM_BACK,
	BNCH_SWT_OP_GROUP_NORM,

	BNCH_SWT_OP_MUL_MAT,
	BNCH_SWT_OP_MUL_MAT_ID,
	BNCH_SWT_OP_OUT_PROD,

	BNCH_SWT_OP_SCALE,
	BNCH_SWT_OP_SET,
	BNCH_SWT_OP_CPY,
	BNCH_SWT_OP_CONT,
	BNCH_SWT_OP_RESHAPE,
	BNCH_SWT_OP_VIEW,
	BNCH_SWT_OP_PERMUTE,
	BNCH_SWT_OP_TRANSPOSE,
	BNCH_SWT_OP_GET_ROWS_BACK,
	BNCH_SWT_OP_DIAG,
	BNCH_SWT_OP_DIAG_MASK_INF,
	BNCH_SWT_OP_DIAG_MASK_ZERO,
	BNCH_SWT_OP_SOFT_MAX,
	BNCH_SWT_OP_SOFT_MAX_BACK,
	BNCH_SWT_OP_ROPE,
	BNCH_SWT_OP_ROPE_BACK,
	BNCH_SWT_OP_CLAMP,
	BNCH_SWT_OP_CONV_TRANSPOSE_1D,
	BNCH_SWT_OP_IM2COL,
	BNCH_SWT_OP_IM2COL_BACK,
	BNCH_SWT_OP_CONV_TRANSPOSE_2D,
	BNCH_SWT_OP_POOL_1D,
	BNCH_SWT_OP_POOL_2D,
	BNCH_SWT_OP_POOL_2D_BACK,
	BNCH_SWT_OP_UPSCALE,// nearest interpolate
	BNCH_SWT_OP_PAD,
	BNCH_SWT_OP_PAD_REFLECT_1D,
	BNCH_SWT_OP_ARANGE,
	BNCH_SWT_OP_TIMESTEP_EMBEDDING,
	BNCH_SWT_OP_ARGSORT,
	BNCH_SWT_OP_LEAKY_RELU,

	BNCH_SWT_OP_FLASH_ATTN_EXT,
	BNCH_SWT_OP_FLASH_ATTN_BACK,
	BNCH_SWT_OP_SSM_CONV,
	BNCH_SWT_OP_SSM_SCAN,
	BNCH_SWT_OP_WIN_PART,
	BNCH_SWT_OP_WIN_UNPART,
	BNCH_SWT_OP_GET_REL_POS,
	BNCH_SWT_OP_ADD_REL_POS,
	BNCH_SWT_OP_RWKV_WKV6,
	BNCH_SWT_OP_GATED_LINEAR_ATTN,

	BNCH_SWT_OP_UNARY,

	BNCH_SWT_OP_MAP_UNARY,
	BNCH_SWT_OP_MAP_BINARY,

	BNCH_SWT_OP_MAP_CUSTOM1_F32,
	BNCH_SWT_OP_MAP_CUSTOM2_F32,
	BNCH_SWT_OP_MAP_CUSTOM3_F32,

	BNCH_SWT_OP_MAP_CUSTOM1,
	BNCH_SWT_OP_MAP_CUSTOM2,
	BNCH_SWT_OP_MAP_CUSTOM3,

	BNCH_SWT_OP_CROSS_ENTROPY_LOSS,
	BNCH_SWT_OP_CROSS_ENTROPY_LOSS_BACK,
	BNCH_SWT_OP_OPT_STEP_ADAMW,
	BNCH_SWT_OP_DEQUANTIZE,
	BNCH_SWT_OP_QUANTIZE,
	BNCH_SWT_OP_SAVE,

	BNCH_SWT_OP_FP32_TO_FP16,
	BNCH_SWT_OP_FP16_TO_FP32,
	BNCH_SWT_OP_COUNT,
};

enum class oiml_backend_device_types { cpu, gpu, BNCH_SWT_BACKEND_DEVICE_TYPE_ACCEL };

struct oiml_op_params {
	oiml_backend_device_types dev_type{};
	oiml_op type{};
	oiml_type src02{};
	oiml_type dst{};
	oiml_type src01{};
};

struct pointers {
	const void* src01{};
	const void* src02{};
	void* dst{};
};

struct impl_indices {
	size_t cpu_index{};
	size_t gpu_index{};
};

template<impl_indices indices> struct oiml_sched_task_base;

struct oiml_op_params;

template<impl_indices indices> struct oiml_sched_task_base {
	BNCH_SWT_INLINE oiml_sched_task_base() noexcept = default;
	BNCH_SWT_INLINE oiml_sched_task_base& operator=(const oiml_sched_task_base& other) {
		active.store(other.active.load(std::memory_order_acquire), std::memory_order_release);
		thread_index = other.thread_index;
		thread_count = other.thread_count;
		ptrs		 = other.ptrs;
		op			 = other.op;
		return *this;
	}
	BNCH_SWT_INLINE oiml_sched_task_base(const oiml_sched_task_base& other) {
		*this = other;
	}
	std::atomic_bool active{ false };
	std::vector<pointers> ptrs{};
	size_t thread_index{};
	size_t thread_count{};
	oiml_op_params op{};
};

template<impl_indices indices> struct oiml_unary_sched_task : public oiml_sched_task_base<indices> {};

template<impl_indices indices> struct oiml_binary_sched_task : public oiml_sched_task_base<indices> {
	std::vector<void*> ptrs_src02{};
};

struct oiml_sched_job_base {};

template<impl_indices indices> struct oiml_sched_job : public oiml_sched_job_base {
	BNCH_SWT_INLINE oiml_sched_job& operator=(oiml_sched_job&& other) noexcept {
		completion_signal = std::move(other.completion_signal);
		tasks			  = std::move(other.tasks);
		return *this;
	}

	BNCH_SWT_INLINE oiml_sched_job(oiml_sched_job&& other) noexcept {
		*this = std::move(other);
	}

	BNCH_SWT_INLINE oiml_sched_task_base<indices>* get_next_task() {
		size_t current_index{ index.load(std::memory_order_acquire) };
		if (current_index < tasks.size() && !tasks[current_index]->active.load(std::memory_order_acquire)) {
			tasks[current_index]->active.store(true, std::memory_order_release);
			index.fetch_add(1, std::memory_order_release);
			return tasks[current_index].get();
		} else {
			return nullptr;
		}
	}

	BNCH_SWT_INLINE bool have_task() {
		return index.load(std::memory_order_acquire) < tasks.size();
	}

	BNCH_SWT_INLINE size_t get_total_required_memory() {
		size_t return_value{};
		for (auto& value: tasks) {
			return_value += value->op.src01.get_total_byte_size();
			return_value += value->op.src02.get_total_byte_size();
			return_value += value->op.dst.get_total_byte_size();
		}
		return return_value;
	}

	BNCH_SWT_INLINE oiml_sched_job() = default;

	std::vector<std::unique_ptr<oiml_sched_task_base<indices>>> tasks{};
	std::unique_ptr<std::latch> completion_signal{};
	std::atomic_uint64_t index{};
	std::mutex task_mutex{};

	BNCH_SWT_INLINE void reset() {
		std::unique_lock lock{ task_mutex };
		completion_signal = std::make_unique<std::latch>(tasks.size());
		index.store(0, std::memory_order_release);
		for (auto& value: tasks) {
			value->active.store(false, std::memory_order_release);
		}
	}
};

BNCH_SWT_NOINLINE void test_function() {
}

template<impl_indices indices, typename value_type>
concept binary_sched_task_type = std::is_same_v<std::remove_cvref_t<value_type>, oiml_binary_sched_task<indices>>;

template<impl_indices indices, typename value_type>
concept unary_sched_task_type = std::is_same_v<std::remove_cvref_t<value_type>, oiml_unary_sched_task<indices>>;

struct oiml_core_config {
	size_t thread_count{ std::thread::hardware_concurrency() };
};

template<impl_indices indices> struct type_trio {
	oiml_type type01{};
	oiml_type type02{};
	oiml_type type03{};
	template<typename value_type> BNCH_SWT_INLINE bool operator==(value_type* params) const {
		return params->op.src01 == type01 && params->op.src02 == type02 && params->op.dst == type03;
	}
};

template<impl_indices indices, oiml_type type01_new, oiml_type type02_new, oiml_type type03_new> struct type_trio_impl {
	inline static constexpr oiml_type type01{ type01_new };
	inline static constexpr oiml_type type02{ type02_new };
	inline static constexpr oiml_type type03{ type03_new };
	template<typename value_type> BNCH_SWT_INLINE bool operator==(value_type* params) const {
		return params->op.src01 == type01 && params->op.src02 == type02 && params->op.dst == type03;
	}
};

template<impl_indices indices, oiml_op> struct op_entity;

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_NONE> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_NONE };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_MUL> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_MUL };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_RMS_NORM> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_RMS_NORM };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_GET_ROWS> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_GET_ROWS };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_DUP> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_DUP };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_ADD> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_ADD };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_ADD1> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_ADD1 };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_ACC> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_ACC };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SUB> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SUB };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_DIV> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_DIV };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SQR> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SQR };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SQRT> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SQRT };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_LOG> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_LOG };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SIN> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SIN };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_COS> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_COS };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SUM> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SUM };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SUM_ROWS> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SUM_ROWS };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_MEAN> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_MEAN };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_ARGMAX> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_ARGMAX };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_COUNT_EQUAL> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_COUNT_EQUAL };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_REPEAT> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_REPEAT };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_REPEAT_BACK> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_REPEAT_BACK };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_CONCAT> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_CONCAT };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SILU_BACK> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SILU_BACK };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_NORM> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_NORM };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_RMS_NORM_BACK> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_RMS_NORM_BACK };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_GROUP_NORM> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_GROUP_NORM };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_MUL_MAT> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_MUL_MAT };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_MUL_MAT_ID> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_MUL_MAT_ID };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_OUT_PROD> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_OUT_PROD };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SCALE> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SCALE };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SET> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SET };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_CPY> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_CPY };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_CONT> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_CONT };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_RESHAPE> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_RESHAPE };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_VIEW> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_VIEW };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_PERMUTE> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_PERMUTE };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_TRANSPOSE> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_TRANSPOSE };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_GET_ROWS_BACK> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_GET_ROWS_BACK };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_DIAG> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_DIAG };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_DIAG_MASK_INF> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_DIAG_MASK_INF };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_DIAG_MASK_ZERO> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_DIAG_MASK_ZERO };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SOFT_MAX> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SOFT_MAX };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SOFT_MAX_BACK> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SOFT_MAX_BACK };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_ROPE> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_ROPE };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_ROPE_BACK> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_ROPE_BACK };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_CLAMP> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_CLAMP };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_CONV_TRANSPOSE_1D> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_CONV_TRANSPOSE_1D };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_IM2COL> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_IM2COL };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_IM2COL_BACK> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_IM2COL_BACK };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_CONV_TRANSPOSE_2D> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_CONV_TRANSPOSE_2D };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_POOL_1D> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_POOL_1D };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_POOL_2D> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_POOL_2D };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_POOL_2D_BACK> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_POOL_2D_BACK };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_UPSCALE> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_UPSCALE };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_PAD> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_PAD };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_PAD_REFLECT_1D> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_PAD_REFLECT_1D };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_ARANGE> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_ARANGE };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_TIMESTEP_EMBEDDING> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_TIMESTEP_EMBEDDING };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_ARGSORT> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_ARGSORT };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_LEAKY_RELU> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_LEAKY_RELU };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_FLASH_ATTN_EXT> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_FLASH_ATTN_EXT };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_FLASH_ATTN_BACK> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_FLASH_ATTN_BACK };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SSM_CONV> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SSM_CONV };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SSM_SCAN> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SSM_SCAN };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_WIN_PART> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_WIN_PART };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_WIN_UNPART> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_WIN_UNPART };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_GET_REL_POS> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_GET_REL_POS };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_ADD_REL_POS> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_ADD_REL_POS };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_RWKV_WKV6> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_RWKV_WKV6 };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_GATED_LINEAR_ATTN> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_GATED_LINEAR_ATTN };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_UNARY> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_UNARY };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_MAP_UNARY> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_MAP_UNARY };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_MAP_BINARY> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_MAP_BINARY };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_MAP_CUSTOM1_F32> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_MAP_CUSTOM1_F32 };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_MAP_CUSTOM2_F32> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_MAP_CUSTOM2_F32 };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_MAP_CUSTOM3_F32> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_MAP_CUSTOM3_F32 };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_MAP_CUSTOM1> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_MAP_CUSTOM1 };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_MAP_CUSTOM2> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_MAP_CUSTOM2 };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_MAP_CUSTOM3> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_MAP_CUSTOM3 };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_CROSS_ENTROPY_LOSS> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_CROSS_ENTROPY_LOSS };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_CROSS_ENTROPY_LOSS_BACK> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_CROSS_ENTROPY_LOSS_BACK };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_OPT_STEP_ADAMW> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_OPT_STEP_ADAMW };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_DEQUANTIZE> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_DEQUANTIZE };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_QUANTIZE> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_QUANTIZE };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_SAVE> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_SAVE };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_FP32_TO_FP16> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_FP32_TO_FP16 };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_FP16_TO_FP32> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_FP16_TO_FP32 };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices> struct op_entity<indices, oiml_op::BNCH_SWT_OP_COUNT> {
	inline static constexpr oiml_op type{ oiml_op::BNCH_SWT_OP_COUNT };
	static constexpr std::array<type_trio<indices>, 1> trios{ [] {
		std::array<type_trio<indices>, 1> return_values{};
		return_values[0] = type_trio<indices>{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<oiml_backend_device_types dev_type, impl_indices indices_new, oiml_op op_type, typename op_entity_type> struct op_entities_internal : public op_entity_type {
	constexpr op_entities_internal() noexcept = default;

	BNCH_SWT_INLINE static void processIndex(oiml_sched_task_base<indices_new>* params) {
		static constexpr op_entity_type op_entity{};
		if (params == op_entity) {
			params->op.type = static_cast<oiml_op>((static_cast<uint64_t>(params->op.type) + static_cast<uint64_t>(params->op.type)));
		}
	};
};

template<impl_indices indices_new, typename... bases> struct op_map : public bases... {
	template<typename op_entity_type, typename... arg_types> BNCH_SWT_INLINE static void iterateValuesImpl(arg_types&&... args) {
		test_function();
		op_entity_type::processIndex(args...);
	}

	template<typename... arg_types> static constexpr void iterateValues(arg_types&&... args) {
		(iterateValuesImpl<bases>(args...), ...);
	}
};

template<impl_indices indices_new, oiml_op op_type, size_t index> struct get_trio_types {
	static constexpr auto trio = op_entity<indices_new, op_type>::trios[index];
	using type				   = type_trio_impl<indices_new, trio.type01, trio.type02, trio.type03>;
};

template<oiml_backend_device_types dev_type, impl_indices indices_new, oiml_op op_type, typename index_sequence, typename... value_types> struct get_op_entity_base_internal;

template<oiml_backend_device_types dev_type, impl_indices indices_new, oiml_op op_type, size_t... index>
struct get_op_entity_base_internal<dev_type, indices_new, op_type, std::index_sequence<index...>> {
	using type = op_map<indices_new, op_entities_internal<dev_type, indices_new, op_type, typename get_trio_types<indices_new, op_type, index>::type...>>;
};

template<oiml_backend_device_types dev_type, impl_indices indices_new, oiml_op op_type> using op_entity_base_internal_t =
	typename get_op_entity_base_internal<dev_type, indices_new, op_type, std::make_index_sequence<static_cast<size_t>(op_entity<indices_new, op_type>::trios.size())>>::type;

template<oiml_backend_device_types dev_type, impl_indices indices_new, oiml_op op_type> struct dispatch_op {
	BNCH_SWT_INLINE static void impl(oiml_sched_task_base<indices_new>* params) {
		op_entity_base_internal_t<dev_type, indices_new, op_type>::iterateValues(params);
	}
};

template<oiml_backend_device_types dev_type, impl_indices indices_new, typename op_entity_type> struct op_entities : public op_entity_type {
	constexpr op_entities() noexcept = default;

	BNCH_SWT_INLINE static void processIndex(oiml_sched_task_base<indices_new>* params, oiml_op op_type) {
		if (op_type == op_entity_type::type) {
			dispatch_op<dev_type, indices_new, op_entity_type::type>::impl(static_cast<oiml_binary_sched_task<indices_new>*>(params));
		}
	};
};

template<oiml_backend_device_types dev_type, impl_indices indices_new, typename index_sequence, typename... value_types> struct get_op_entity_base;

template<oiml_backend_device_types dev_type, impl_indices indices_new, size_t... index> struct get_op_entity_base<dev_type, indices_new, std::index_sequence<index...>> {
	using type = op_map<indices_new, op_entities<dev_type, indices_new, op_entity<indices_new, static_cast<oiml_op>(index)>>...>;
};

template<oiml_backend_device_types dev_type, impl_indices indices_new> using op_entity_base_t =
	typename get_op_entity_base<dev_type, indices_new, std::make_index_sequence<static_cast<size_t>(oiml_op::BNCH_SWT_OP_COUNT)>>::type;

template<oiml_backend_device_types dev_type, impl_indices indices_new> struct op_dispatcher_final {
	BNCH_SWT_INLINE static void impl(oiml_sched_task_base<indices_new>* params) {
		op_entity_base_t<dev_type, indices_new>::iterateValues(params, params->op.type);
	}
};

int main() {
	srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	oiml_op type{ static_cast<oiml_op>(rand()) };
	oiml_sched_task_base<impl_indices{}> params{};
	params.op.type = type;
	op_dispatcher_final<oiml_backend_device_types::cpu, impl_indices{}>::impl(&params);
	bnch_swt::doNotOptimizeAway(params);
	return 0;
}
