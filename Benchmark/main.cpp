#include <BnchSwt/BenchmarkSuite.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <latch>
#include <array>
#include <bit>

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
	OIML_OP_NONE = 0,
	OIML_OP_MUL,
	OIML_OP_RMS_NORM,
	OIML_OP_GET_ROWS,
	OIML_OP_DUP,
	OIML_OP_ADD,
	OIML_OP_ADD1,
	OIML_OP_ACC,
	OIML_OP_SUB,
	OIML_OP_DIV,
	OIML_OP_SQR,
	OIML_OP_SQRT,
	OIML_OP_LOG,
	OIML_OP_SIN,
	OIML_OP_COS,
	OIML_OP_SUM,
	OIML_OP_SUM_ROWS,
	OIML_OP_MEAN,
	OIML_OP_ARGMAX,
	OIML_OP_COUNT_EQUAL,
	OIML_OP_REPEAT,
	OIML_OP_REPEAT_BACK,
	OIML_OP_CONCAT,
	OIML_OP_SILU_BACK,
	OIML_OP_NORM,// normalize
	OIML_OP_RMS_NORM_BACK,
	OIML_OP_GROUP_NORM,

	OIML_OP_MUL_MAT,
	OIML_OP_MUL_MAT_ID,
	OIML_OP_OUT_PROD,

	OIML_OP_SCALE,
	OIML_OP_SET,
	OIML_OP_CPY,
	OIML_OP_CONT,
	OIML_OP_RESHAPE,
	OIML_OP_VIEW,
	OIML_OP_PERMUTE,
	OIML_OP_TRANSPOSE,
	OIML_OP_GET_ROWS_BACK,
	OIML_OP_DIAG,
	OIML_OP_DIAG_MASK_INF,
	OIML_OP_DIAG_MASK_ZERO,
	OIML_OP_SOFT_MAX,
	OIML_OP_SOFT_MAX_BACK,
	OIML_OP_ROPE,
	OIML_OP_ROPE_BACK,
	OIML_OP_CLAMP,
	OIML_OP_CONV_TRANSPOSE_1D,
	OIML_OP_IM2COL,
	OIML_OP_IM2COL_BACK,
	OIML_OP_CONV_TRANSPOSE_2D,
	OIML_OP_POOL_1D,
	OIML_OP_POOL_2D,
	OIML_OP_POOL_2D_BACK,
	OIML_OP_UPSCALE,// nearest interpolate
	OIML_OP_PAD,
	OIML_OP_PAD_REFLECT_1D,
	OIML_OP_ARANGE,
	OIML_OP_TIMESTEP_EMBEDDING,
	OIML_OP_ARGSORT,
	OIML_OP_LEAKY_RELU,

	OIML_OP_FLASH_ATTN_EXT,
	OIML_OP_FLASH_ATTN_BACK,
	OIML_OP_SSM_CONV,
	OIML_OP_SSM_SCAN,
	OIML_OP_WIN_PART,
	OIML_OP_WIN_UNPART,
	OIML_OP_GET_REL_POS,
	OIML_OP_ADD_REL_POS,
	OIML_OP_RWKV_WKV6,
	OIML_OP_GATED_LINEAR_ATTN,

	OIML_OP_UNARY,

	OIML_OP_MAP_UNARY,
	OIML_OP_MAP_BINARY,

	OIML_OP_MAP_CUSTOM1_F32,
	OIML_OP_MAP_CUSTOM2_F32,
	OIML_OP_MAP_CUSTOM3_F32,

	OIML_OP_MAP_CUSTOM1,
	OIML_OP_MAP_CUSTOM2,
	OIML_OP_MAP_CUSTOM3,

	OIML_OP_CROSS_ENTROPY_LOSS,
	OIML_OP_CROSS_ENTROPY_LOSS_BACK,
	OIML_OP_OPT_STEP_ADAMW,
	OIML_OP_DEQUANTIZE,
	OIML_OP_QUANTIZE,
	OIML_OP_SAVE,

	OIML_OP_FP32_TO_FP16,
	OIML_OP_FP16_TO_FP32,
	OIML_OP_COUNT,
};

struct pointers {
	void* src01{};
	void* src02{};
	void* dst{};
	size_t dst_stride{};
	size_t inner_loops_per_outer_loop{};
	size_t src01_stride{};
	size_t src02_stride{};
	size_t index_count{};
};

struct impl_indices {
	size_t cpu_index{};
	size_t gpu_index{};
};

template<impl_indices indices> struct oiml_sched_task_base;

struct oiml_op_params;

template<impl_indices indices> struct oiml_sched_task_base {
	std::vector<std::vector<void*>> ptrs_src01{};
	std::vector<void*> ptrs_dst{};
	std::vector<pointers> ptrs{};
	oiml_op_params* op_ptr{};
	size_t thread_index{};
	size_t thread_count{};
	pointers ptrs_new{};
};

template<impl_indices indices> struct oiml_unary_sched_task : public oiml_sched_task_base<indices> {};

template<impl_indices indices> struct oiml_binary_sched_task : public oiml_sched_task_base<indices> {
	std::vector<void*> ptrs_src02{};
};

struct oiml_sched_job_base {};

template<impl_indices indices> struct oiml_sched_job : public oiml_sched_job_base {
	BNCH_SWT_INLINE oiml_sched_job& operator=(oiml_sched_job&& other) noexcept {
		completion_signal.swap(other.completion_signal);
		tasks = std::move(other.tasks);
		return *this;
	}

	BNCH_SWT_INLINE oiml_sched_job(oiml_sched_job&& other) noexcept {
		*this = std::move(other);
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
	}
};

struct oiml_op_params {
	oiml_op type{};
};

template<impl_indices indices, typename value_type>
concept binary_sched_task_type = std::is_same_v<std::remove_cvref_t<value_type>, oiml_binary_sched_task<indices>>;

template<impl_indices indices, typename value_type>
concept unary_sched_task_type = std::is_same_v<std::remove_cvref_t<value_type>, oiml_unary_sched_task<indices>>;

struct oiml_core_config {
	size_t thread_count{ std::thread::hardware_concurrency() };
};

struct type_trio {
	oiml_type type01{};
	oiml_type type02{};
	oiml_type type03{};
	template<impl_indices indices> BNCH_SWT_INLINE bool operator==(oiml_binary_sched_task<indices>* params) const {
		return params->op_ptr->src01.type == type01 && params->op_ptr->src02.type == type02 && params->op_ptr->dst.type == type03;
	}
};

struct type_duo {
	oiml_type type01{};
	oiml_type type02{};
	template<impl_indices indices> BNCH_SWT_INLINE bool operator==(oiml_unary_sched_task<indices>* params) const {
		return params->op_ptr->src01.type == type01 && params->op_ptr->dst.type == type02;
	}
};

template<oiml_op op_type, uint64_t index, oiml_type... types> struct function_dispatcher_impl;

template<impl_indices indices> struct function_dispatcher_base {
	BNCH_SWT_INLINE virtual void impl(oiml_binary_sched_task<indices>*) const = 0;
};

// Specialization 0: q8_0, q8_0, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::q8_0, oiml_type::q8_0, oiml_type::float_32>
	: public function_dispatcher_base<impl_indices{ .cpu_index = 0 }> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) const {
		std::cout << "We are here - q8_0, q8_0, float_32" << std::endl;
	}
};
static constexpr function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::q8_0, oiml_type::q8_0, oiml_type::float_32> value01{};

// Specialization 1: i8x32xf32x1, i8x32xf32x1, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::i8x32xf32x1, oiml_type::i8x32xf32x1, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - i8x32xf32x1, i8x32xf32x1, float_32" << std::endl;
	}
};

// Specialization 2: float_32, float_32, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::float_32, oiml_type::float_32, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - float_32, float_32, float_32" << std::endl;
	}
};

// Specialization 3: float_16, float_16, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::float_16, oiml_type::float_16, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - float_16, float_16, float_32" << std::endl;
	}
};

// Specialization 4: brain_float_16, brain_float_16, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::brain_float_16, oiml_type::brain_float_16, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - brain_float_16, brain_float_16, float_32" << std::endl;
	}
};

// Specialization 5: int_8, int_8, int_32 (already provided, but included for completeness)
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::int_8, oiml_type::int_8, oiml_type::int_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - int_8, int_8, int_32" << std::endl;
	}
};

// Specialization 6: int_32, int_32, int_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::int_32, oiml_type::int_32, oiml_type::int_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - int_32, int_32, int_32" << std::endl;
	}
};

// Specialization 7: q8_0, float_16, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::q8_0, oiml_type::float_16, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - q8_0, float_16, float_32" << std::endl;
	}
};

// Specialization 8: float_16, q8_0, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::float_16, oiml_type::q8_0, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - float_16, q8_0, float_32" << std::endl;
	}
};

// Specialization 9: brain_float_16, float_16, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::brain_float_16, oiml_type::float_16, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - brain_float_16, float_16, float_32" << std::endl;
	}
};

// Specialization 10: float_16, brain_float_16, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::float_16, oiml_type::brain_float_16, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - float_16, brain_float_16, float_32" << std::endl;
	}
};

// Specialization 11: int_8, q8_0, int_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::int_8, oiml_type::q8_0, oiml_type::int_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - int_8, q8_0, int_32" << std::endl;
	}
};

// Specialization 12: q8_0, int_8, int_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::q8_0, oiml_type::int_8, oiml_type::int_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - q8_0, int_8, int_32" << std::endl;
	}
};

// Specialization 13: int_32, q8_0, int_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::int_32, oiml_type::q8_0, oiml_type::int_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - int_32, q8_0, int_32" << std::endl;
	}
};

// Specialization 14: float_32, brain_float_16, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::float_32, oiml_type::brain_float_16, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - float_32, brain_float_16, float_32" << std::endl;
	}
};

// Specialization 15: brain_float_16, float_32, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::brain_float_16, oiml_type::float_32, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - brain_float_16, float_32, float_32" << std::endl;
	}
};

// Specialization 16: i8x32xf32x1, q8_0, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::i8x32xf32x1, oiml_type::q8_0, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - i8x32xf32x1, q8_0, float_32" << std::endl;
	}
};

// Specialization 17: q8_0, i8x32xf32x1, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::q8_0, oiml_type::i8x32xf32x1, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - q8_0, i8x32xf32x1, float_32" << std::endl;
	}
};

// Specialization 18: float_32, i8x32xf32x1, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::float_32, oiml_type::i8x32xf32x1, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - float_32, i8x32xf32x1, float_32" << std::endl;
	}
};

// Specialization 19: i8x32xf32x1, float_32, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::i8x32xf32x1, oiml_type::float_32, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - i8x32xf32x1, float_32, float_32" << std::endl;
	}
};

// Specialization 20: float_32, int_8, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::float_32, oiml_type::int_8, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - float_32, int_8, float_32" << std::endl;
	}
};

// Specialization 21: int_8, float_32, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::int_8, oiml_type::float_32, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - int_8, float_32, float_32" << std::endl;
	}
};

// Specialization 22: float_16, int_8, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::float_16, oiml_type::int_8, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - float_16, int_8, float_32" << std::endl;
	}
};

// Specialization 23: int_8, float_16, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::int_8, oiml_type::float_16, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - int_8, float_16, float_32" << std::endl;
	}
};

// Specialization 24: brain_float_16, int_8, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::brain_float_16, oiml_type::int_8, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - brain_float_16, int_8, float_32" << std::endl;
	}
};

// Specialization 25: int_8, brain_float_16, float_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::int_8, oiml_type::brain_float_16, oiml_type::float_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - int_8, brain_float_16, float_32" << std::endl;
	}
};

// Specialization 26: int_32, float_32, int_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::int_32, oiml_type::float_32, oiml_type::int_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - int_32, float_32, int_32" << std::endl;
	}
};

// Specialization 27: float_32, int_32, int_32
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::float_32, oiml_type::int_32, oiml_type::int_32> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - float_32, int_32, int_32" << std::endl;
	}
};

// Specialization 28: brain_float_16, brain_float_16, brain_float_16
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::brain_float_16, oiml_type::brain_float_16, oiml_type::brain_float_16> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - brain_float_16, brain_float_16, brain_float_16" << std::endl;
	}
};

// Specialization 29: float_16, float_16, float_16
template<> struct function_dispatcher_impl<oiml_op::OIML_OP_MUL_MAT, 0, oiml_type::float_16, oiml_type::float_16, oiml_type::float_16> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<impl_indices{ .cpu_index = 0 }>*) {
		std::cout << "We are here - float_16, float_16, float_16" << std::endl;
	}
};

template<impl_indices indices, oiml_op op_type, oiml_type... types> struct function_dispatcher_new;

template<impl_indices indices, oiml_op op_type, oiml_type type01, oiml_type type02, oiml_type type03> struct function_dispatcher_new<indices, op_type, type01, type02, type03> {
	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<indices>* params) {
		return function_dispatcher_impl<op_type, indices.cpu_index, type01, type02, type03>::impl(params);
	}
};

template<impl_indices indices> using function_type = function_dispatcher_base<indices>;

template<impl_indices indices, oiml_op op_type, oiml_type type01, oiml_type type02> struct function_dispatcher_new<indices, op_type, type01, type02> {
	BNCH_SWT_INLINE void impl(oiml_unary_sched_task<indices>* params) {
		return function_dispatcher_impl<op_type, indices.cpu_index, type01, type02>::impl(params);
	}
};

template<oiml_op op_type> struct type_trio_holder {
	static_assert(false, "Sorry, but you need to specialize type_trio_holder for this oiml_op type.");
};

template<oiml_op op_type> struct type_duo_holder {
	static_assert(false, "Sorry, but you need to specialize type_duo_holder for this oiml_op type.");
};

template<> struct type_trio_holder<oiml_op::OIML_OP_MUL_MAT> {
	static constexpr std::array<type_trio, 2> trios{ [] {
		std::array<type_trio, 2> return_values{};
		return_values[0] = type_trio{ .type01 = oiml_type::q8_0, .type02 = oiml_type::q8_0, .type03 = oiml_type::float_32 };
		return_values[1] = type_trio{ .type01 = oiml_type::i8x32xf32x1, .type02 = oiml_type::i8x32xf32x1, .type03 = oiml_type::float_32 };
		return return_values;
	}() };
};

template<impl_indices indices_new, auto type_trios, oiml_op op_type> struct dispatch_binary_op {
	template<size_t... indices> BNCH_SWT_INLINE void impl(oiml_binary_sched_task<indices_new>* params, std::index_sequence<indices...>) {
		((type_trios[indices] == params
				 ? (function_dispatcher_new<indices_new, op_type, type_trios[indices].type01, type_trios[indices].type02, type_trios[indices].type03>::impl(params), false)
				 : (true)) &&
			...);
	}

	BNCH_SWT_INLINE void impl(oiml_binary_sched_task<indices_new>* params) {
		return impl(params, std::make_index_sequence<type_trios.size()>{});
	}
};

template<impl_indices indices_new, auto type_duos, oiml_op op_type> struct dispatch_unary_op {
	template<size_t... indices> BNCH_SWT_INLINE void impl(oiml_unary_sched_task<indices_new>* params, std::index_sequence<indices...>) {
		((type_duos[indices] == params ? (function_dispatcher_new<indices_new, op_type, type_duos[indices].type01, type_duos[indices].type02>::impl(params), false) : (true)) &&
			...);
	}

	BNCH_SWT_INLINE void impl(oiml_unary_sched_task<indices_new>* params) {
		return impl(params, std::make_index_sequence<type_duos.size()>{});
	}
};

template<impl_indices indices_new, oiml_op op_type> struct oiml_op_dispatcher_impl {
	BNCH_SWT_INLINE void impl(oiml_sched_task_base<indices_new>* op_params) noexcept {
		if constexpr (op_type == oiml_op::OIML_OP_ACC) {
		} else if constexpr (op_type == oiml_op::OIML_OP_DUP) {
		} else if constexpr (op_type == oiml_op::OIML_OP_ADD) {
		} else if constexpr (op_type == oiml_op::OIML_OP_ADD1) {
		} else if constexpr (op_type == oiml_op::OIML_OP_ACC) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SUB) {
		} else if constexpr (op_type == oiml_op::OIML_OP_MUL) {
		} else if constexpr (op_type == oiml_op::OIML_OP_DIV) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SQR) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SQRT) {
		} else if constexpr (op_type == oiml_op::OIML_OP_LOG) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SIN) {
		} else if constexpr (op_type == oiml_op::OIML_OP_COS) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SUM) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SUM_ROWS) {
		} else if constexpr (op_type == oiml_op::OIML_OP_MEAN) {
		} else if constexpr (op_type == oiml_op::OIML_OP_ARGMAX) {
		} else if constexpr (op_type == oiml_op::OIML_OP_COUNT_EQUAL) {
		} else if constexpr (op_type == oiml_op::OIML_OP_REPEAT) {
		} else if constexpr (op_type == oiml_op::OIML_OP_REPEAT_BACK) {
		} else if constexpr (op_type == oiml_op::OIML_OP_CONCAT) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SILU_BACK) {
		} else if constexpr (op_type == oiml_op::OIML_OP_NORM) {
		} else if constexpr (op_type == oiml_op::OIML_OP_RMS_NORM) {
		} else if constexpr (op_type == oiml_op::OIML_OP_RMS_NORM_BACK) {
		} else if constexpr (op_type == oiml_op::OIML_OP_GROUP_NORM) {
		} else if constexpr (op_type == oiml_op::OIML_OP_MUL_MAT) {
			dispatch_binary_op<indices_new, type_trio_holder<oiml_op::OIML_OP_MUL_MAT>::trios, op_type>::impl(static_cast<oiml_binary_sched_task<indices_new>*>(op_params));
		} else if constexpr (op_type == oiml_op::OIML_OP_MUL_MAT_ID) {
		} else if constexpr (op_type == oiml_op::OIML_OP_OUT_PROD) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SCALE) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SET) {
		} else if constexpr (op_type == oiml_op::OIML_OP_CPY) {
		} else if constexpr (op_type == oiml_op::OIML_OP_CONT) {
		} else if constexpr (op_type == oiml_op::OIML_OP_RESHAPE) {
		} else if constexpr (op_type == oiml_op::OIML_OP_VIEW) {
		} else if constexpr (op_type == oiml_op::OIML_OP_PERMUTE) {
		} else if constexpr (op_type == oiml_op::OIML_OP_TRANSPOSE) {
		} else if constexpr (op_type == oiml_op::OIML_OP_GET_ROWS) {
		} else if constexpr (op_type == oiml_op::OIML_OP_GET_ROWS_BACK) {
		} else if constexpr (op_type == oiml_op::OIML_OP_DIAG) {
		} else if constexpr (op_type == oiml_op::OIML_OP_DIAG_MASK_INF) {
		} else if constexpr (op_type == oiml_op::OIML_OP_DIAG_MASK_ZERO) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SOFT_MAX) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SOFT_MAX_BACK) {
		} else if constexpr (op_type == oiml_op::OIML_OP_ROPE) {
		} else if constexpr (op_type == oiml_op::OIML_OP_ROPE_BACK) {
		} else if constexpr (op_type == oiml_op::OIML_OP_CLAMP) {
		} else if constexpr (op_type == oiml_op::OIML_OP_CONV_TRANSPOSE_1D) {
		} else if constexpr (op_type == oiml_op::OIML_OP_IM2COL) {
		} else if constexpr (op_type == oiml_op::OIML_OP_IM2COL_BACK) {
		} else if constexpr (op_type == oiml_op::OIML_OP_CONV_TRANSPOSE_2D) {
		} else if constexpr (op_type == oiml_op::OIML_OP_POOL_1D) {
		} else if constexpr (op_type == oiml_op::OIML_OP_POOL_2D) {
		} else if constexpr (op_type == oiml_op::OIML_OP_POOL_2D_BACK) {
		} else if constexpr (op_type == oiml_op::OIML_OP_UPSCALE) {
		} else if constexpr (op_type == oiml_op::OIML_OP_PAD) {
		} else if constexpr (op_type == oiml_op::OIML_OP_PAD_REFLECT_1D) {
		} else if constexpr (op_type == oiml_op::OIML_OP_ARANGE) {
		} else if constexpr (op_type == oiml_op::OIML_OP_TIMESTEP_EMBEDDING) {
		} else if constexpr (op_type == oiml_op::OIML_OP_ARGSORT) {
		} else if constexpr (op_type == oiml_op::OIML_OP_LEAKY_RELU) {
		} else if constexpr (op_type == oiml_op::OIML_OP_FLASH_ATTN_EXT) {
		} else if constexpr (op_type == oiml_op::OIML_OP_FLASH_ATTN_BACK) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SSM_CONV) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SSM_SCAN) {
		} else if constexpr (op_type == oiml_op::OIML_OP_WIN_PART) {
		} else if constexpr (op_type == oiml_op::OIML_OP_WIN_UNPART) {
		} else if constexpr (op_type == oiml_op::OIML_OP_GET_REL_POS) {
		} else if constexpr (op_type == oiml_op::OIML_OP_ADD_REL_POS) {
		} else if constexpr (op_type == oiml_op::OIML_OP_RWKV_WKV6) {
		} else if constexpr (op_type == oiml_op::OIML_OP_GATED_LINEAR_ATTN) {
		} else if constexpr (op_type == oiml_op::OIML_OP_UNARY) {
		} else if constexpr (op_type == oiml_op::OIML_OP_MAP_UNARY) {
		} else if constexpr (op_type == oiml_op::OIML_OP_MAP_BINARY) {
		} else if constexpr (op_type == oiml_op::OIML_OP_MAP_CUSTOM1_F32) {
		} else if constexpr (op_type == oiml_op::OIML_OP_MAP_CUSTOM2_F32) {
		} else if constexpr (op_type == oiml_op::OIML_OP_MAP_CUSTOM3_F32) {
		} else if constexpr (op_type == oiml_op::OIML_OP_MAP_CUSTOM1) {
		} else if constexpr (op_type == oiml_op::OIML_OP_MAP_CUSTOM2) {
		} else if constexpr (op_type == oiml_op::OIML_OP_MAP_CUSTOM3) {
		} else if constexpr (op_type == oiml_op::OIML_OP_CROSS_ENTROPY_LOSS) {
		} else if constexpr (op_type == oiml_op::OIML_OP_CROSS_ENTROPY_LOSS_BACK) {
		} else if constexpr (op_type == oiml_op::OIML_OP_OPT_STEP_ADAMW) {
		} else if constexpr (op_type == oiml_op::OIML_OP_DEQUANTIZE) {
		} else if constexpr (op_type == oiml_op::OIML_OP_QUANTIZE) {
		} else if constexpr (op_type == oiml_op::OIML_OP_SAVE) {
		} else if constexpr (op_type == oiml_op::OIML_OP_FP32_TO_FP16) {
		} else if constexpr (op_type == oiml_op::OIML_OP_FP16_TO_FP32) {
		} else {
		};
	}
};

template<impl_indices indices> static constexpr std::array<const function_type<indices>*, 6> functions{ [] {
	std::array<const function_type<indices>*, 6> return_values{};

	return_values[0] = &value01;
	return_values[1] = &value01;
	return_values[2] = &value01;
	return_values[3] = &value01;
	return_values[4] = &value01;
	return_values[5] = &value01;

	return return_values;
}() };

template<impl_indices indices_new> struct op_dispatcher_final {
	template<size_t... indices> BNCH_SWT_INLINE void impl_internal(oiml_sched_task_base<indices_new>* params, oiml_op index, std::index_sequence<indices...>) {
		((indices == static_cast<size_t>(index) ? (oiml_op_dispatcher_impl<indices_new, static_cast<oiml_op>(indices)>::impl(params), false) : true) && ...);
	}

	BNCH_SWT_INLINE void impl(oiml_sched_task_base<indices_new>* params) {
		return impl_internal(params, params->op_ptr->type, std::make_index_sequence<static_cast<size_t>(oiml_op::OIML_OP_COUNT)>{});
	}
};

int main() {
	srand(static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
	static constexpr impl_indices indices{};
	size_t index{ static_cast<size_t>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f) };
	oiml_binary_sched_task<indices> params{};
	functions<indices>[index] -> impl(&params);
	return 0;
}