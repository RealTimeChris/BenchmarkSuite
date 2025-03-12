#pragma once
#include <oiml/legacy/oiml-legacy-common/oiml-backend.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-backend-impl.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-cpu.hpp>
#include <oiml/legacy/oiml-legacy-cpu/oiml-cpu-traits.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-impl.hpp>

#include <cctype>
#include <string>
#include <vector>

#if defined(__APPLE__)
	#include <sys/types.h>
	#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
	#define WIN32_LEAN_AND_MEAN
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <windows.h>
#endif

// oiml-backend interface

OIML_INLINE std::vector<oiml_backend_buffer_type_t>& oiml_backend_cpu_get_extra_buffers_type() {
	static std::vector<oiml_backend_buffer_type_t> bufts = []() {
		std::vector<oiml_backend_buffer_type_t> bufts;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
		if (oiml_backend_amx_buffer_type()) {
			bufts.push_back(oiml_backend_amx_buffer_type());
		}
#endif

#ifdef OIML_USE_CPU_AARCH64
		if (oiml_backend_cpu_aarch64_buffer_type()) {
			bufts.push_back(oiml_backend_cpu_aarch64_buffer_type());
		}
#endif

		bufts.push_back(NULL);

		return bufts;
	}();

	return bufts;
}

OIML_INLINE static oiml_backend_buffer_type_t* oiml_backend_cpu_device_get_extra_buffers_type(oiml_backend_dev_t device) {
	return oiml_backend_cpu_get_extra_buffers_type().data();

	OIML_UNUSED(device);
}

OIML_INLINE static bool oiml_backend_cpu_is_extra_buffer_type(oiml_backend_buffer_type_t buft) {
	for (auto extra: oiml_backend_cpu_get_extra_buffers_type()) {
		if (extra && extra == buft)
			return true;
	}
	return false;
}

// CPU backend - backend (stream)

struct oiml_backend_cpu_context {
	int n_threads;
	oiml_threadpool_t threadpool;

	int8_t* work_data;
	size_t work_size;

	oiml_abort_callback abort_callback;
	void* abort_callback_data;
};

OIML_INLINE static constexpr const char* oiml_backend_cpu_get_name(oiml_backend_t) {
	return "CPU";
}

OIML_INLINE static void oiml_backend_cpu_free(oiml_backend_t backend) {
	struct oiml_backend_cpu_context* cpu_ctx = ( struct oiml_backend_cpu_context* )backend->context;
	delete[] cpu_ctx->work_data;
	delete cpu_ctx;
	delete backend;
}

struct oiml_backend_plan_cpu {
	oiml_cplan cplan;
	struct oiml_cgraph cgraph;
};

OIML_INLINE static oiml_backend_graph_plan_t oiml_backend_cpu_graph_plan_create(oiml_backend_t backend, const oiml_cgraph* cgraph) {
	struct oiml_backend_cpu_context* cpu_ctx = ( struct oiml_backend_cpu_context* )backend->context;

	struct oiml_backend_plan_cpu* cpu_plan = new oiml_backend_plan_cpu;

	cpu_plan->cplan	 = oiml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
	cpu_plan->cgraph = *cgraph;// FIXME: deep copy

	if (cpu_plan->cplan.work_size > 0) {
		cpu_plan->cplan.work_data = new int8_t[cpu_plan->cplan.work_size];
		if (cpu_plan->cplan.work_data == NULL) {
			delete cpu_plan;
			return NULL;
		}
	}

	cpu_plan->cplan.abort_callback		= cpu_ctx->abort_callback;
	cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;

	return cpu_plan;
}

OIML_INLINE static void oiml_backend_cpu_graph_plan_free(oiml_backend_t backend, oiml_backend_graph_plan_t plan) {
	struct oiml_backend_plan_cpu* cpu_plan = ( struct oiml_backend_plan_cpu* )plan;

	delete[] cpu_plan->cplan.work_data;
	delete cpu_plan;

	OIML_UNUSED(backend);
}

OIML_INLINE static oiml_status oiml_backend_cpu_graph_plan_compute(oiml_backend_t backend, oiml_backend_graph_plan_t plan) {
	struct oiml_backend_plan_cpu* cpu_plan = ( struct oiml_backend_plan_cpu* )plan;

	return oiml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

	OIML_UNUSED(backend);
}

OIML_INLINE static oiml_status oiml_backend_cpu_graph_compute(oiml_backend_t backend, oiml_cgraph* cgraph) {
	struct oiml_backend_cpu_context* cpu_ctx = ( struct oiml_backend_cpu_context* )backend->context;

	oiml_cplan cplan = oiml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

	if (cpu_ctx->work_size < cplan.work_size) {
		delete[] cpu_ctx->work_data;
		cpu_ctx->work_data = new int8_t[cplan.work_size];
		if (cpu_ctx->work_data == NULL) {
			cpu_ctx->work_size = 0;
			return OIML_STATUS_ALLOC_FAILED;
		}
		cpu_ctx->work_size = cplan.work_size;
	}
	cplan.work_data = ( int8_t* )cpu_ctx->work_data;

	cplan.abort_callback	  = cpu_ctx->abort_callback;
	cplan.abort_callback_data = cpu_ctx->abort_callback_data;

	return oiml_graph_compute(cgraph, &cplan);
}

static constexpr oiml_backend_i oiml_backend_cpu_i = {
	/* .get_name                = */ oiml_backend_cpu_get_name,
	/* .free                    = */ oiml_backend_cpu_free,
	/* .set_tensor_async        = */ NULL,
	/* .get_tensor_async        = */ NULL,
	/* .cpy_tensor_async        = */ NULL,
	/* .synchronize             = */ NULL,
	/* .graph_plan_create       = */ oiml_backend_cpu_graph_plan_create,
	/* .graph_plan_free         = */ oiml_backend_cpu_graph_plan_free,
	/* .graph_plan_update       = */ NULL,
	/* .graph_plan_compute      = */ oiml_backend_cpu_graph_plan_compute,
	/* .graph_compute           = */ oiml_backend_cpu_graph_compute,
	/* .event_record            = */ NULL,
	/* .event_wait              = */ NULL,
};

OIML_INLINE static oiml_guid_t oiml_backend_cpu_guid() {
	static oiml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
	return &guid;
}

OIML_INLINE oiml_backend_t oiml_backend_cpu_init() {
	// initialize CPU backend now to avoid slowing the first graph computation
	oiml_cpu_init();

	struct oiml_backend_cpu_context* ctx = new oiml_backend_cpu_context;
	if (ctx == NULL) {
		return NULL;
	}

	ctx->n_threads			 = OIML_DEFAULT_N_THREADS;
	ctx->threadpool			 = NULL;
	ctx->work_data			 = NULL;
	ctx->work_size			 = 0;
	ctx->abort_callback		 = NULL;
	ctx->abort_callback_data = NULL;

	oiml_backend_t cpu_backend = new oiml_backend{
		/* .guid      = */ oiml_backend_cpu_guid(),
		/* .interface = */ oiml_backend_cpu_i,
		/* .device    = */ oiml_backend_reg_dev_get(oiml_backend_cpu_reg(), 0),
		/* .context   = */ ctx,
	};

	if (cpu_backend == NULL) {
		delete ctx;
		return NULL;
	}

	return cpu_backend;
}

OIML_INLINE bool oiml_backend_is_cpu(oiml_backend_t backend) {
	return backend != NULL && oiml_guid_matches(backend->guid, oiml_backend_cpu_guid());
}

OIML_INLINE void oiml_backend_cpu_set_n_threads(oiml_backend_t backend_cpu, int n_threads) {
	OIML_ASSERT(oiml_backend_is_cpu(backend_cpu));

	struct oiml_backend_cpu_context* ctx = ( struct oiml_backend_cpu_context* )backend_cpu->context;
	ctx->n_threads						 = n_threads;
}

OIML_INLINE void oiml_backend_cpu_set_threadpool(oiml_backend_t backend_cpu, oiml_threadpool_t threadpool) {
	OIML_ASSERT(oiml_backend_is_cpu(backend_cpu));

	struct oiml_backend_cpu_context* ctx = ( struct oiml_backend_cpu_context* )backend_cpu->context;

	if (ctx->threadpool && ctx->threadpool != threadpool) {
		// already had a different threadpool, pause/suspend it before switching
		oiml_threadpool_pause(ctx->threadpool);
	}
	ctx->threadpool = threadpool;
}

OIML_INLINE void oiml_backend_cpu_set_abort_callback(oiml_backend_t backend_cpu, oiml_abort_callback abort_callback, void* abort_callback_data) {
	OIML_ASSERT(oiml_backend_is_cpu(backend_cpu));

	struct oiml_backend_cpu_context* ctx = ( struct oiml_backend_cpu_context* )backend_cpu->context;
	ctx->abort_callback					 = abort_callback;
	ctx->abort_callback_data			 = abort_callback_data;
}

// CPU backend - device

struct oiml_backend_cpu_device_context {
	std::string description = "CPU";

	oiml_backend_cpu_device_context() {
#ifdef __APPLE__
		size_t len = 0;
		if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
			description.resize(len);
			sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0);// NOLINT
		}
#elif defined(__linux__)
		FILE* f = fopen("/proc/cpuinfo", "r");
		if (f) {
			char buf[1024];
			while (fgets(buf, sizeof(buf), f)) {
				if (strncmp(buf, "model name", 10) == 0) {
					char* p = strchr(buf, ':');
					if (p) {
						p++;
						while (std::isspace(*p)) {
							p++;
						}
						while (std::isspace(p[strlen(p) - 1])) {
							p[strlen(p) - 1] = '\0';
						}
						description = p;
						break;
					}
				}
			}
			fclose(f);
		}
#elif defined(_WIN32)
		HKEY hKey;
		if (RegOpenKeyEx(HKEY_LOCAL_MACHINE, TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"), 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
			DWORD cpu_brand_size = 0;
			if (RegQueryValueExA(hKey, TEXT("ProcessorNameString"), NULL, NULL, NULL, &cpu_brand_size) == ERROR_SUCCESS) {
				description.resize(cpu_brand_size);
				if (RegQueryValueExA(hKey, TEXT("ProcessorNameString"), NULL, NULL,
						( LPBYTE )&description[0],// NOLINT
						&cpu_brand_size) == ERROR_SUCCESS) {
					if (description.find('\0') != std::string::npos) {
						description.resize(description.find('\0'));
					}
				}
			}
			RegCloseKey(hKey);
		}
#endif
	}
};

OIML_INLINE static constexpr const char* oiml_backend_cpu_device_get_name(oiml_backend_dev_t) {
	return "CPU";
}

OIML_INLINE static constexpr const char* oiml_backend_cpu_device_get_description(oiml_backend_dev_t dev) {
	struct oiml_backend_cpu_device_context* ctx = ( struct oiml_backend_cpu_device_context* )dev->context;

	return ctx->description.c_str();
}

OIML_INLINE static void oiml_backend_cpu_device_get_memory(oiml_backend_dev_t dev, size_t* free, size_t* total) {
	// TODO
	*free  = 0;
	*total = 0;

	OIML_UNUSED(dev);
}

OIML_INLINE static enum oiml_backend_device_types oiml_backend_cpu_device_get_type(oiml_backend_dev_t dev) {
	return cpu;

	OIML_UNUSED(dev);
}

OIML_INLINE static void oiml_backend_cpu_device_get_props(oiml_backend_dev_t dev, struct oiml_backend_dev_props* props) {
	props->name		   = oiml_backend_cpu_device_get_name(nullptr);
	props->description = oiml_backend_cpu_device_get_description(dev);
	props->type		   = oiml_backend_cpu_device_get_type(dev);
	oiml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
	props->caps = {
		/* .async                 = */ false,
		/* .host_buffer           = */ false,
		/* .buffer_from_host_ptr  = */ true,
		/* .events                = */ false,
	};
}

OIML_INLINE static oiml_backend_t oiml_backend_cpu_device_init_backend(oiml_backend_dev_t dev, const char* params) {
	return oiml_backend_cpu_init();

	OIML_UNUSED(dev);
	OIML_UNUSED(params);
}

OIML_INLINE static oiml_backend_buffer_type_t oiml_backend_cpu_device_get_buffer_type(oiml_backend_dev_t dev) {
	return oiml_backend_cpu_buffer_type();

	OIML_UNUSED(dev);
}

OIML_INLINE static oiml_backend_buffer_t oiml_backend_cpu_device_buffer_from_host_ptr(oiml_backend_dev_t dev, void* ptr, size_t size, size_t max_tensor_size) {
	return oiml_backend_cpu_buffer_from_ptr(ptr, size);

	OIML_UNUSED(dev);
	OIML_UNUSED(max_tensor_size);
}

OIML_INLINE static bool oiml_backend_cpu_device_supports_op(oiml_backend_dev_t dev, const oiml_tensor* op) {
	const oiml_tensor* src0 = op->src[0];
	const oiml_tensor* src1 = op->src[1];

	if (op->op == OIML_OP_NONE || op->op == OIML_OP_RESHAPE || op->op == OIML_OP_VIEW || op->op == OIML_OP_PERMUTE || op->op == OIML_OP_TRANSPOSE) {
		return true;
	}

	// extra_buffer_op?
	for (auto extra: oiml_backend_cpu_get_extra_buffers_type()) {
		if (extra) {
			auto buf_extra = ( oiml_legacy::cpu::extra_buffer_type* )extra->context;
			if (buf_extra && buf_extra->supports_op(op)) {
				return true;
			}
		}
	}

	// the other case need host buffer.
	for (int i = 0; i < OIML_MAX_SRC; i++) {
		if (op->src[i] && op->src[i]->buffer && !oiml_backend_buft_is_host(op->src[i]->buffer->buft)) {
			return false;
		}
	}

	switch (op->op) {
		// case OIML_OP_CPY:
		// 	return op->type != OIML_TYPE_IQ3_XXS && op->type != OIML_TYPE_IQ3_S && op->type != OIML_TYPE_IQ2_XXS && op->type != OIML_TYPE_IQ2_XS && op->type != OIML_TYPE_IQ2_S &&
		// 		op->type != OIML_TYPE_IQ1_S && op->type != OIML_TYPE_IQ1_M;// missing type_traits.from_float
		case OIML_OP_MUL_MAT:
			return src1->type == oiml::oiml_representation_types::float_32 || src1->type == oiml_get_type_traits(src0->type)->vec_dot_type;
		case OIML_OP_SOFT_MAX_BACK: {
			if (op->src[0]->type != oiml::oiml_representation_types::float_32 || op->src[1]->type != oiml::oiml_representation_types::float_32) {
				return false;
			}
			float max_bias = 0.0f;

			memcpy(&max_bias, ( const float* )op->op_params + 1, sizeof(float));

			return max_bias == 0.0f;
		}
		case OIML_OP_IM2COL_BACK:
			return src0->type == oiml::oiml_representation_types::float_32 && src1->type == oiml::oiml_representation_types::float_32;
		case OIML_OP_OUT_PROD:
			return (src0->type == oiml::oiml_representation_types::float_32 || (oiml_is_quantized(src0->type) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) && src1->type == oiml::oiml_representation_types::float_32 &&
				op->type == oiml::oiml_representation_types::float_32;
		default:
			return true;
	}
}

OIML_INLINE static bool oiml_backend_cpu_device_supports_buft(oiml_backend_dev_t dev, oiml_backend_buffer_type_t buft) {
	return oiml_backend_buft_is_host(buft) || oiml_backend_cpu_is_extra_buffer_type(buft);
	OIML_UNUSED(dev);
}

static constexpr oiml_backend_device_i oiml_backend_cpu_device_i = {
	/* .get_name             = */ oiml_backend_cpu_device_get_name,
	/* .get_description      = */ oiml_backend_cpu_device_get_description,
	/* .get_memory           = */ oiml_backend_cpu_device_get_memory,
	/* .get_type             = */ oiml_backend_cpu_device_get_type,
	/* .get_props            = */ oiml_backend_cpu_device_get_props,
	/* .init_backend         = */ oiml_backend_cpu_device_init_backend,
	/* .get_buffer_type      = */ oiml_backend_cpu_device_get_buffer_type,
	/* .get_host_buffer_type = */ NULL,
	/* .buffer_from_host_ptr = */ oiml_backend_cpu_device_buffer_from_host_ptr,
	/* .supports_op          = */ oiml_backend_cpu_device_supports_op,
	/* .supports_buft        = */ oiml_backend_cpu_device_supports_buft,
	/* .offload_op           = */ NULL,
	/* .event_new            = */ NULL,
	/* .event_free           = */ NULL,
	/* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

OIML_INLINE static constexpr const char* oiml_backend_cpu_reg_get_name(oiml_backend_reg_t reg) {
	return "CPU";

	OIML_UNUSED(reg);
}

OIML_INLINE static size_t oiml_backend_cpu_reg_get_device_count(oiml_backend_reg_t reg) {
	return 1;

	OIML_UNUSED(reg);
}

OIML_INLINE static oiml_backend_dev_t oiml_backend_cpu_reg_get_device(oiml_backend_reg_t reg, size_t index) {
	OIML_ASSERT(index == 0);

	static oiml_backend_cpu_device_context ctx;
	static oiml_backend_device oiml_backend_cpu_device = {
		/* .iface   = */ oiml_backend_cpu_device_i,
		/* .reg     = */ reg,
		/* .context = */ &ctx,
	};

	return &oiml_backend_cpu_device;
}

OIML_FORCE_INLINE consteval int oiml_cpu_get_sve_cnt() {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
	return 1;//oiml_arm_arch_features.sve_cnt;
#else
	return 0;
#endif
}

// This is intended to replace the the oiml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
OIML_INLINE static oiml_backend_feature* oiml_backend_cpu_get_features(oiml_backend_reg_t reg) {
	static std::vector<oiml_backend_feature> features = []() {
		oiml_cpu_init();

		std::vector<oiml_backend_feature> features;
		if (oiml_cpu_has_sse3()) {
			features.push_back({ "SSE3", "1" });
		}
		if (oiml_cpu_has_ssse3()) {
			features.push_back({ "SSSE3", "1" });
		}
		if (oiml_cpu_has_avx()) {
			features.push_back({ "AVX", "1" });
		}
		if (oiml_cpu_has_avx_vnni()) {
			features.push_back({ "AVX_VNNI", "1" });
		}
		if (oiml_cpu_has_avx2()) {
			features.push_back({ "AVX2", "1" });
		}
		if (oiml_cpu_has_f16c()) {
			features.push_back({ "F16C", "1" });
		}
		if (oiml_cpu_has_fma()) {
			features.push_back({ "FMA", "1" });
		}
		if (oiml_cpu_has_avx512()) {
			features.push_back({ "AVX512", "1" });
		}
		if (oiml_cpu_has_avx512_vbmi()) {
			features.push_back({ "AVX512_VBMI", "1" });
		}
		if (oiml_cpu_has_avx512_vnni()) {
			features.push_back({ "AVX512_VNNI", "1" });
		}
		if (oiml_cpu_has_avx512_bf16()) {
			features.push_back({ "AVX512_BF16", "1" });
		}
		if (oiml_cpu_has_amx_int8()) {
			features.push_back({ "AMX_INT8", "1" });
		}
		if (oiml_cpu_has_neon()) {
			features.push_back({ "NEON", "1" });
		}
		if (oiml_cpu_has_arm_fma()) {
			features.push_back({ "ARM_FMA", "1" });
		}
		if (oiml_cpu_has_fp16_va()) {
			features.push_back({ "FP16_VA", "1" });
		}
		if (oiml_cpu_has_matmul_int8()) {
			features.push_back({ "MATMUL_INT8", "1" });
		}
		if (oiml_cpu_has_sve()) {
			features.push_back({ "SVE", "1" });
		}
		if (oiml_cpu_has_dotprod()) {
			features.push_back({ "DOTPROD", "1" });
		}
		if (oiml_cpu_has_matmul_int8()) {
			features.push_back({ "MATMUL_INT8", "1" });
		}
		if (oiml_cpu_get_sve_cnt() > 0) {
			static std::string sve_cnt = std::to_string(oiml_cpu_get_sve_cnt());
			features.push_back({ "SVE_CNT", sve_cnt.c_str() });
		}
		if (oiml_cpu_has_riscv_v()) {
			features.push_back({ "RISCV_V", "1" });
		}
		if (oiml_cpu_has_vsx()) {
			features.push_back({ "VSX", "1" });
		}
		if (oiml_cpu_has_wasm_simd()) {
			features.push_back({ "WASM_SIMD", "1" });
		}
		if (oiml_cpu_has_llamafile()) {
			features.push_back({ "LLAMAFILE", "1" });
		}
#ifdef OIML_USE_ACCELERATE
		features.push_back({ "ACCELERATE", "1" });
#endif
#ifdef OIML_USE_CPU_HBM
		features.push_back({ "CPU_HBM", "1" });
#endif
#ifdef OIML_USE_OPENMP
		features.push_back({ "OPENMP", "1" });
#endif
#ifdef OIML_USE_CPU_AARCH64
		features.push_back({ "AARCH64_REPACK", "1" });
#endif

		features.push_back({ nullptr, nullptr });

		return features;
	}();

	return features.data();

	OIML_UNUSED(reg);
}

OIML_INLINE static void* oiml_backend_cpu_get_proc_address(oiml_backend_reg_t reg, const char* name) {
	if (strcmp(name, "oiml_backend_set_n_threads") == 0) {
		oiml_backend_set_n_threads_t fct = oiml_backend_cpu_set_n_threads;
		return ( void* )fct;
	}
	if (strcmp(name, "oiml_backend_dev_get_extra_bufts") == 0) {
		oiml_backend_dev_get_extra_bufts_t fct = oiml_backend_cpu_device_get_extra_buffers_type;
		return ( void* )fct;
	}
	if (strcmp(name, "oiml_backend_get_features") == 0) {
		return ( void* )oiml_backend_cpu_get_features;
	}
	if (strcmp(name, "oiml_backend_set_abort_callback") == 0) {
		return ( void* )oiml_backend_cpu_set_abort_callback;
	}
	if (strcmp(name, "oiml_backend_cpu_numa_init") == 0) {
		return ( void* )oiml_numa_init;
	}
	if (strcmp(name, "oiml_backend_cpu_is_numa") == 0) {
		return ( void* )oiml_is_numa;
	}

	// threadpool - TODO:  move to oiml-base
	if (strcmp(name, "oiml_threadpool_new") == 0) {
		return ( void* )oiml_threadpool_new;
	}
	if (strcmp(name, "oiml_threadpool_free") == 0) {
		return ( void* )oiml_threadpool_free;
	}
	if (strcmp(name, "oiml_backend_cpu_set_threadpool") == 0) {
		return ( void* )oiml_backend_cpu_set_threadpool;
	}

	return NULL;

	OIML_UNUSED(reg);
}

static constexpr oiml_backend_reg_i oiml_backend_cpu_reg_i = {
	/* .get_name         = */ oiml_backend_cpu_reg_get_name,
	/* .get_device_count = */ oiml_backend_cpu_reg_get_device_count,
	/* .get_device       = */ oiml_backend_cpu_reg_get_device,
	/* .get_proc_address = */ oiml_backend_cpu_get_proc_address,
};

OIML_INLINE oiml_backend_reg_t oiml_backend_cpu_reg() {
	// init CPU feature detection
	oiml_cpu_init();

	static struct oiml_backend_reg oiml_backend_cpu_reg = {
		/* .api_version = */ OIML_API_VERSION,
		/* .iface       = */ oiml_backend_cpu_reg_i,
		/* .context     = */ NULL,
	};

	return &oiml_backend_cpu_reg;
}

OIML_BACKEND_DL_IMPL(oiml_backend_cpu_reg)
