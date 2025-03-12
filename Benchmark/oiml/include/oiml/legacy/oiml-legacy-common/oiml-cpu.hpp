#pragma once

#include <oiml/legacy/oiml-legacy-common/oiml-final.hpp>
#include <oiml/common/config.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-backend.hpp>
#include <memory>


// the compute plan that needs to be prepared for oiml_graph_compute()
// since https://github.com/ggerganov/oiml/issues/287
struct oiml_cplan {
	size_t work_size;// size of work buffer, calculated by `oiml_graph_plan()`
	int8_t* work_data;// work buffer, to be allocated by caller before calling to `oiml_graph_compute()`

	int n_threads;
	oiml_threadpool* threadpool;

	// abort oiml_graph_compute when true
	oiml_abort_callback abort_callback;
	void* abort_callback_data;
};

// numa strategies
enum oiml_numa_strategy {
	OIML_NUMA_STRATEGY_DISABLED	  = 0,
	OIML_NUMA_STRATEGY_DISTRIBUTE = 1,
	OIML_NUMA_STRATEGY_ISOLATE	  = 2,
	OIML_NUMA_STRATEGY_NUMACTL	  = 3,
	OIML_NUMA_STRATEGY_MIRROR	  = 4,
	OIML_NUMA_STRATEGY_COUNT
};

void oiml_numa_init(oiml_numa_strategy numa);// call once for better performance on NUMA systems
bool oiml_is_numa();// true if init detected that system has >1 NUMA node

oiml_tensor* oiml_new_i32(oiml_context* ctx, int32_t value);
oiml_tensor* oiml_new_f32(oiml_context* ctx, float value);

oiml_tensor* oiml_set_i32(oiml_tensor* tensor, int32_t value);
oiml_tensor* oiml_set_f32(oiml_tensor* tensor, float value);

int32_t oiml_get_i32_1d(const oiml_tensor* tensor, int i);
void oiml_set_i32_1d(const oiml_tensor* tensor, int i, int32_t value);

int32_t oiml_get_i32_nd(const oiml_tensor* tensor, int i0, int i1, int i2, int i3);
void oiml_set_i32_nd(const oiml_tensor* tensor, int i0, int i1, int i2, int i3, int32_t value);

float oiml_get_f32_1d(const oiml_tensor* tensor, int i);
void oiml_set_f32_1d(const oiml_tensor* tensor, int i, float value);

float oiml_get_f32_nd(const oiml_tensor* tensor, int i0, int i1, int i2, int i3);
void oiml_set_f32_nd(const oiml_tensor* tensor, int i0, int i1, int i2, int i3, float value);

oiml_threadpool* oiml_threadpool_new(oiml_threadpool_params* params);
void oiml_threadpool_free(oiml_threadpool* threadpool);
int oiml_threadpool_get_n_threads(oiml_threadpool* threadpool);
void oiml_threadpool_pause(oiml_threadpool* threadpool);
void oiml_threadpool_resume(oiml_threadpool* threadpool);

// oiml_graph_plan() has to be called before oiml_graph_compute()
// when plan.work_size > 0, caller must allocate memory for plan.work_data
oiml_cplan oiml_graph_plan(const oiml_cgraph* cgraph, int n_threads, /* = OIML_DEFAULT_N_THREADS */
	oiml_threadpool* threadpool /* = NULL */);
oiml_status oiml_graph_compute(oiml_cgraph* cgraph, oiml_cplan* cplan);

// same as oiml_graph_compute() but the work data is allocated as a part of the context
// note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
oiml_status oiml_graph_compute_with_ctx(oiml_context* ctx, oiml_cgraph* cgraph, int n_threads);

//
// system info
//

// x86
OIML_INLINE consteval int oiml_cpu_has_avx() {
#if defined(__AVX__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_avx_vnni() {
#if defined(__AVXVNNI__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_avx2() {
#if defined(__AVX2__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_avx512() {
#if defined(__AVX512F__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_avx512_vbmi() {
#if defined(__AVX512VBMI__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_avx512_vnni() {
#if defined(__AVX512VNNI__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_avx512_bf16() {
#if defined(__AVX512BF16__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_amx_int8() {
#if defined(__AMX_INT8__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_fma() {
#if defined(__FMA__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_arm_fma() {
#if defined(__ARM_FEATURE_FMA)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_riscv_v() {
#if defined(__riscv_v_intrinsic)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_f16c() {
#if defined(__F16C__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_fp16_va() {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_wasm_simd() {
#if defined(__wasm_simd128__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_llamafile() {
#if defined(true)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_sse3() {
#if defined(__SSE3__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_ssse3() {
#if defined(__SSSE3__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_vsx() {
#if defined(__POWER9_VECTOR__)
	return 1;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_neon() {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
	return 1;//oiml_arm_arch_features.has_neon; // implied from defined(__ARM_ARCH) && defined(__ARM_NEON)
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_dotprod() {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
	return 1;//oiml_arm_arch_features.has_dotprod;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_sve() {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
	return 1;//oiml_arm_arch_features.has_sve;
#else
	return 0;
#endif
}

OIML_INLINE consteval int oiml_cpu_has_matmul_int8() {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
	return 1;//oiml_arm_arch_features.has_i8mm;
#else
	return 0;
#endif
}

// Internal types and functions exposed for tests and benchmarks



const oiml_type_traits* oiml_get_type_traits(oiml::oiml_representation_types type);

void oiml_cpu_init();

//
// CPU backend
//

oiml_backend_t oiml_backend_cpu_init();

bool oiml_backend_is_cpu(oiml_backend_t backend);
void oiml_backend_cpu_set_n_threads(oiml_backend_t backend_cpu, int n_threads);
void oiml_backend_cpu_set_threadpool(oiml_backend_t backend_cpu, oiml_threadpool_t threadpool);
void oiml_backend_cpu_set_abort_callback(oiml_backend_t backend_cpu, oiml_abort_callback abort_callback, void* abort_callback_data);

oiml_backend_reg_t oiml_backend_cpu_reg();
