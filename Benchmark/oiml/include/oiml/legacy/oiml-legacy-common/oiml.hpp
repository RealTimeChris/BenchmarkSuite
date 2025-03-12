#pragma once
#define _CRT_SECURE_NO_DEPRECATE// Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES// For M_PI on MSVC

#include <oiml/legacy/oiml-legacy-common/oiml-backend-impl-final.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-backend.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-impl.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-opt.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-quants-impl.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-threading.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-final.hpp>

// FIXME: required here for quantization functions
#include <oiml/legacy/oiml-legacy-common/oiml-quants.hpp>

#ifdef OIML_USE_CPU_HBM
	#include <hbwmalloc.h>
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
	#include <malloc.h>// using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
	#include <alloca.h>
#endif
#include <chrono>
#include <unordered_map>
#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
	#include <syscall.h>
#endif

#if defined(__APPLE__)
	#include <unistd.h>
	#include <mach/mach.h>
	#include <TargetConditionals.h>
#endif

#if defined(_WIN32)
	#define WIN32_LEAN_AND_MEAN
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <windows.h>
#endif

#define UNUSED OIML_UNUSED

#if defined(_MSC_VER)
	#define m512bh(p) p
	#define m512i(p) p
#else
	#define m512bh(p) (__m512bh)(p)
	#define m512i(p) (__m512i)(p)
#endif

#if (defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)) && (!defined(TARGET_OS_TV) && !defined(TARGET_OS_WATCH))
	#include <unistd.h>
	#include <sys/types.h>
	#include <sys/stat.h>
	#include <sys/wait.h>

	#if defined(__ANDROID__)
		#include <unwind.h>
		#include <dlfcn.h>
		#include <stdio.h>

struct backtrace_state {
	void** current;
	void** end;
};

OIML_INLINE _Unwind_Reason_Code unwind_callback(_Unwind_Context* context, void* arg) {
	backtrace_state* state = ( backtrace_state* )arg;
	uintptr_t pc		   = _Unwind_GetIP(context);
	if (pc) {
		if (state->current == state->end) {
			return _URC_END_OF_STACK;
		} else {
			*state->current++ = ( void* )pc;
		}
	}
	return _URC_NO_REASON;
}

OIML_INLINE void oiml_print_backtrace_symbols() {
	const int max = 100;
	void* buffer[max];

	struct backtrace_state state = { buffer, buffer + max };
	_Unwind_Backtrace(unwind_callback, &state);

	int count = state.current - buffer;

	for (int idx = 0; idx < count; ++idx) {
		const void* addr   = buffer[idx];
		const char* symbol = "";

		Dl_info info;
		if (dladdr(addr, &info) && info.dli_sname) {
			symbol = info.dli_sname;
		}

		fprintf(stderr, "%d: %p %s\n", idx, addr, symbol);
	}
}
	#elif defined(__linux__) && defined(__GLIBC__)
		#include <execinfo.h>
OIML_INLINE void oiml_print_backtrace_symbols() {
	void* trace[100];
	int nptrs = backtrace(trace, sizeof(trace) / sizeof(trace[0]));
	backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
	#else
OIML_INLINE void oiml_print_backtrace_symbols() {
	// platform not supported
}
	#endif

OIML_INLINE void oiml_print_backtrace() {
	char attach[32];
	snprintf(attach, sizeof(attach), "attach %d", getpid());
	int pid = fork();
	if (pid == 0) {
		// try gdb
		execlp("gdb", "gdb", "--batch", "-ex", "set style enabled on", "-ex", attach, "-ex", "bt -frame-info source-and-location", "-ex", "detach", "-ex", "quit", ( char* )NULL);
		// try lldb
		execlp("lldb", "lldb", "--batch", "-o", "bt", "-o", "quit", "-p", attach, ( char* )NULL);
		exit(EXIT_FAILURE);
	} else {
		int wstatus;
		waitpid(pid, &wstatus, 0);
		if (WIFEXITED(wstatus)) {
			if (WEXITSTATUS(wstatus) == EXIT_FAILURE) {
				// gdb failed, fallback to backtrace_symbols
				oiml_print_backtrace_symbols();
			}
		}
	}
}
#else
OIML_INLINE void oiml_print_backtrace() {
	// platform not supported
}
#endif

OIML_INLINE void oiml_abort(const char* file, int line, const char* fmt, ...) {
	fflush(stdout);

	fprintf(stderr, "%s:%d: ", file, line);

	va_list args;
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);

	fprintf(stderr, "\n");

	oiml_print_backtrace();
	abort();
}

//
// logging
//

struct oiml_logger_state {
	oiml_log_callback log_callback;
	void* log_callback_user_data;
};
inline oiml_logger_state g_oiml_logger_state = { oiml_log_callback_default, NULL };

OIML_INLINE void oiml_log_internal_v(enum oiml_log_level level, const char* format, va_list args) {
	if (format == NULL) {
		return;
	}
	va_list args_copy;
	va_copy(args_copy, args);
	char buffer[128];
	int len = vsnprintf(buffer, 128, format, args);
	if (len < 128) {
		g_oiml_logger_state.log_callback(level, buffer, g_oiml_logger_state.log_callback_user_data);
	} else {
		char* buffer2 = ( char* )calloc(len + 1, sizeof(char));
		vsnprintf(buffer2, len + 1, format, args_copy);
		buffer2[len] = 0;
		g_oiml_logger_state.log_callback(level, buffer2, g_oiml_logger_state.log_callback_user_data);
		free(buffer2);
	}
	va_end(args_copy);
}

OIML_INLINE void oiml_log_internal(enum oiml_log_level level, const char* format, ...) {
	va_list args;
	va_start(args, format);
	oiml_log_internal_v(level, format, args);
	va_end(args);
}

OIML_INLINE void oiml_log_callback_default(enum oiml_log_level level, const char* text, void* user_data) {
	( void )level;
	( void )user_data;
	fputs(text, stderr);
	fflush(stderr);
}

//
// end of logging oiml::block
//

#ifdef OIML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define OIML_SOFT_MAX_ACCELERATE
#endif


OIML_INLINE void* oiml_aligned_malloc(size_t size) {
	const int alignment = 64;

#if defined(_MSC_VER) || defined(__MINGW32__)
	return _aligned_malloc(size, alignment);
#else
	if (size == 0) {
		OIML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for oiml_aligned_malloc!\n");
		return NULL;
	}
	void* aligned_memory = NULL;
	#ifdef OIML_USE_CPU_HBM
	int result = hbw_posix_memalign(&aligned_memory, alignment, size);
	#elif TARGET_OS_OSX
	OIML_UNUSED(alignment);
	kern_return_t alloc_status = vm_allocate(( vm_map_t )mach_task_self(), ( vm_address_t* )&aligned_memory, size, VM_FLAGS_ANYWHERE);
	int result				   = EFAULT;
	switch (alloc_status) {
		case KERN_SUCCESS:
			result = 0;
			break;
		case KERN_INVALID_ADDRESS:
			result = EINVAL;
			break;
		case KERN_NO_SPACE:
			result = ENOMEM;
			break;
		default:
			result = EFAULT;
			break;
	}
	#else
	int result = posix_memalign(&aligned_memory, alignment, size);
	#endif
	if (result != 0) {
		// Handle allocation failure
		const char* error_desc = "unknown allocation error";
		switch (result) {
			case EINVAL:
				error_desc = "invalid alignment value";
				break;
			case ENOMEM:
				error_desc = "insufficient memory";
				break;
		}
		OIML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size / (1024.0 * 1024.0));
		return NULL;
	}
	return aligned_memory;
#endif
}

OIML_INLINE void oiml_aligned_free(void* ptr, size_t size) {
	OIML_UNUSED(size);
#if defined(_MSC_VER) || defined(__MINGW32__)
	_aligned_free(ptr);
#elif OIML_USE_CPU_HBM
	if (ptr != NULL) {
		hbw_free(ptr);
	}
#elif TARGET_OS_OSX
	if (ptr != NULL) {
		vm_deallocate(( vm_map_t )mach_task_self(), ( vm_address_t )ptr, size);
	}
#else
	free(ptr);
#endif
}


OIML_INLINE void* oiml_malloc(size_t size) {
	if (size == 0) {
		OIML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for oiml_malloc!\n");
		return NULL;
	}
	void* result = malloc(size);
	if (result == NULL) {
		OIML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size / (1024.0 * 1024.0));
		OIML_ABORT("fatal error");
	}
	return result;
}

// calloc
OIML_INLINE void* oiml_calloc(size_t num, size_t size) {
	if (num == 0 || size == 0) {
		OIML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for oiml_calloc!\n");
		return NULL;
	}
	void* result = calloc(num, size);
	if (result == NULL) {
		OIML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size / (1024.0 * 1024.0));
		OIML_ABORT("fatal error");
	}
	return result;
}

#define OIML_MALLOC(size) oiml_malloc(size)
#define OIML_CALLOC(num, size) oiml_calloc(num, size)

#define OIML_FREE(ptr) free(ptr)

OIML_INLINE const char* oiml_status_to_string(oiml_status status) {
	switch (status) {
		case OIML_STATUS_ALLOC_FAILED:
			return "OIML status: error (failed to allocate memory)";
		case OIML_STATUS_FAILED:
			return "OIML status: error (operation failed)";
		case OIML_STATUS_SUCCESS:
			return "OIML status: success";
		case OIML_STATUS_ABORTED:
			return "OIML status: warning (operation aborted)";
	}

	return "OIML status: unknown";
}

OIML_INLINE float oiml_fp16_to_fp32(oiml_fp16_t x) {
	return oiml::oiml_lookup_fp16_to_fp32(x);
}

OIML_INLINE oiml_fp16_t oiml_fp32_to_fp16(float x) {
	return oiml::oiml_fp32_to_fp16(x);
}

OIML_INLINE float oiml_bf16_to_fp32(oiml_bf16_t x) {
	return oiml::oiml_bf16_to_fp32(x);// it just left shifts
}

OIML_INLINE oiml_bf16_t oiml_fp32_to_bf16(float x) {
	return oiml::oiml_fp32_to_bf16(x);
}

OIML_INLINE void oiml_fp16_to_fp32_row(const oiml_fp16_t* x, float* y, int64_t n) {
	for (int64_t i = 0; i < n; i++) {
		y[i] = oiml::oiml_lookup_fp16_to_fp32(x[i]);
	}
}

OIML_INLINE void oiml_fp16_to_fp32_row_bindings(const oiml_tensor_binding* vx, float* y, int64_t offset, int64_t n) {
	OIML_ASSERT(vx->type == oiml::oiml_representation_types::float_16);
	OIML_ASSERT(vx->num_channels == 1);
	OIML_ASSERT(vx->data_channels[0].type == oiml_data_channel_type::value);
	OIML_ASSERT(vx->data_channels[0].data_type == oiml::oiml_representation_types::float_16);

	oiml_fp16_t* x = reinterpret_cast<oiml_fp16_t*>(reinterpret_cast<uint8_t*>(vx->data_channels[0].data) + offset);
	oiml_fp16_to_fp32_row(x, y, n);
}

// FIXME: these functions must detect the instruction set at runtime, since they are part of the core oiml library
//        currently, the oiml_cpu_has_* functions are entirely compile-time
OIML_INLINE void oiml_fp32_to_fp16_row(const float* x, oiml_fp16_t* y, int64_t n) {
	int64_t i = 0;
#if defined(__F16C__)
	//if (oiml_cpu_has_f16c()) {
	for (; i + 7 < n; i += 8) {
		__m256 x_vec  = _mm256_loadu_ps(x + i);
		__m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
		_mm_storeu_si128(( __m128i* )(y + i), y_vec);
	}
	for (; i + 3 < n; i += 4) {
		__m128 x_vec  = _mm_loadu_ps(x + i);
		__m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
		_mm_storel_epi64(( __m128i* )(y + i), y_vec);
	}
	//}
#endif
	for (; i < n; i++) {
		y[i] = oiml::oiml_fp32_to_fp16(x[i]);
	}
}

OIML_INLINE void oiml_fp32_to_fp16_row_bindings(const float* x, oiml_tensor_binding* vy, int64_t offset, int64_t n) {
	OIML_ASSERT(vy->type == oiml::oiml_representation_types::float_16);
	OIML_ASSERT(vy->num_channels == 1);
	OIML_ASSERT(vy->data_channels[0].type == oiml_data_channel_type::value);
	OIML_ASSERT(vy->data_channels[0].data_type == oiml::oiml_representation_types::float_16);

	oiml_fp16_t* y = reinterpret_cast<oiml_fp16_t*>(reinterpret_cast<uint8_t*>(vy->data_channels[0].data) + offset);
	oiml_fp32_to_fp16_row(x, y, n);
}

OIML_INLINE void oiml_bf16_to_fp32_row(const oiml_bf16_t* x, float* y, int64_t n) {
	int64_t i = 0;
#if defined(__AVX512F__)
	//if (oiml_cpu_has_avx512()) {
	for (; i + 16 <= n; i += 16) {
		_mm512_storeu_ps(y + i, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(( const __m256i* )(x + i))), 16)));
	}
	//}
#endif
#if defined(__AVX2__)
	//if (oiml_cpu_has_avx2()) {
	for (; i + 8 <= n; i += 8) {
		_mm256_storeu_ps(y + i, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(( const __m128i* )(x + i))), 16)));
	}
	//}
#endif
	for (; i < n; i++) {
		y[i] = oiml::oiml_bf16_to_fp32(x[i]);
	}
}

OIML_INLINE void oiml_bf16_to_fp32_row_bindings(const oiml_tensor_binding* vx, float* y, int64_t offset, int64_t n) {
	OIML_ASSERT(vx->type == oiml::oiml_representation_types::brain_float_16);
	OIML_ASSERT(vx->num_channels == 1);
	OIML_ASSERT(vx->data_channels[0].type == oiml_data_channel_type::value);
	OIML_ASSERT(vx->data_channels[0].data_type == oiml::oiml_representation_types::brain_float_16);

	oiml_bf16_t* x = reinterpret_cast<oiml_bf16_t*>(reinterpret_cast<uint8_t*>(vx->data_channels[0].data) + offset);
	oiml_bf16_to_fp32_row(x, y, n);
}

OIML_INLINE void oiml_fp32_to_bf16_row_ref(const float* x, oiml_bf16_t* y, int64_t n) {
	for (int i = 0; i < n; i++) {
		y[i] = oiml::oiml_compute_fp32_to_bf16(x[i]);
	}
}

OIML_INLINE void oiml_fp32_to_bf16_row_bindings_ref(const float* x, oiml_tensor_binding* vy, int64_t offset, int64_t n) {
	OIML_ASSERT(vy->type == oiml::oiml_representation_types::brain_float_16);
	OIML_ASSERT(vy->num_channels == 1);
	OIML_ASSERT(vy->data_channels[0].type == oiml_data_channel_type::value);
	OIML_ASSERT(vy->data_channels[0].data_type == oiml::oiml_representation_types::brain_float_16);

	oiml_bf16_t* y = reinterpret_cast<oiml_bf16_t*>(reinterpret_cast<uint8_t*>(vy->data_channels[0].data) + offset);
	oiml_fp32_to_bf16_row_ref(x, y, n);
}

OIML_INLINE void oiml_fp32_to_bf16_row(const float* x, oiml_bf16_t* y, int64_t n) {
	int i = 0;
#if defined(__AVX512BF16__)
	// subnormals are flushed to zero on this platform
	for (; i + 32 <= n; i += 32) {
		_mm512_storeu_si512(( __m512i* )(y + i), m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16), _mm512_loadu_ps(x + i))));
	}
#endif
	for (; i < n; i++) {
		y[i] = oiml::oiml_fp32_to_bf16(x[i]);
	}
}

OIML_INLINE void oiml_fp32_to_bf16_row_bindings(const float* x, oiml_tensor_binding* vy, int64_t offset, int64_t n) {
	OIML_ASSERT(vy->type == oiml::oiml_representation_types::brain_float_16);
	OIML_ASSERT(vy->num_channels == 1);
	OIML_ASSERT(vy->data_channels[0].type == oiml_data_channel_type::value);
	OIML_ASSERT(vy->data_channels[0].data_type == oiml::oiml_representation_types::brain_float_16);

	oiml_bf16_t* y = reinterpret_cast<oiml_bf16_t*>(reinterpret_cast<uint8_t*>(vy->data_channels[0].data) + offset);
	oiml_fp32_to_bf16_row(x, y, n);
}

OIML_INLINE bool oiml_guid_matches(oiml_guid_t guid_a, oiml_guid_t guid_b) {
	return memcmp(guid_a, guid_b, sizeof(oiml_guid)) == 0;
}

//
// timing
//

OIML_INLINE int64_t oiml_time_ms() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
OIML_INLINE int64_t oiml_time_us() {
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

OIML_INLINE int64_t oiml_cycles() {
	return clock();
}

OIML_INLINE int64_t oiml_cycles_per_ms() {
	return CLOCKS_PER_SEC / 1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
OIML_INLINE wchar_t* oiml_mbstowcs(const char* mbs) {
	int wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, NULL, 0);
	if (!wlen) {
		errno = EINVAL;
		return NULL;
	}

	wchar_t* wbuf = ( wchar_t* )OIML_MALLOC(wlen * sizeof(wchar_t));
	wlen		  = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
	if (!wlen) {
		OIML_FREE(wbuf);
		errno = EINVAL;
		return NULL;
	}

	return wbuf;
}
#endif

OIML_INLINE FILE* oiml_fopen(const char* fname, const char* mode) {
#ifdef _WIN32
	FILE* file = NULL;

	// convert fname (UTF-8)
	wchar_t* wfname = oiml_mbstowcs(fname);
	if (wfname) {
		// convert mode (ANSI)
		wchar_t* wmode	 = ( wchar_t* )OIML_MALLOC((strlen(mode) + 1) * sizeof(wchar_t));
		wchar_t* wmode_p = wmode;
		do {
			*wmode_p++ = ( wchar_t )*mode;
		} while (*mode++);

		// open file
		file = _wfopen(wfname, wmode);

		OIML_FREE(wfname);
		OIML_FREE(wmode);
	}

	return file;
#else
	return fopen(fname, mode);
#endif
}

OIML_INLINE void oiml_fp16_to_fp32_row_bindings(const oiml_tensor_binding* vx, float* y, int64_t offset, int64_t n);
OIML_INLINE void oiml_bf16_to_fp32_row_bindings(const oiml_tensor_binding* vx, float* y, int64_t offset, int64_t n);

OIML_INLINE void oiml_vec_dot_f32_bindings(int n, float* __restrict s, size_t bs, const oiml_tensor_binding* __restrict x, size_t x_off, size_t bx, const oiml_tensor_binding* __restrict y, size_t y_off, size_t by, int nrc);
OIML_INLINE void oiml_vec_dot_f16_bindings(int n, float* __restrict s, size_t bs, const oiml_tensor_binding* __restrict x, size_t x_off, size_t bx, const oiml_tensor_binding* __restrict y, size_t y_off, size_t by, int nrc);
OIML_INLINE void oiml_vec_dot_bf16_bindings(int n, float* __restrict s, size_t bs, const oiml_tensor_binding* __restrict x, size_t x_off, size_t bx, const oiml_tensor_binding* __restrict y, size_t y_off, size_t by, int nrc);

OIML_INLINE void oiml_vec_dot_f32(int n, float* __restrict s, size_t bs, const float* __restrict x, size_t bx, const float* __restrict y, size_t by, int nrc);
OIML_INLINE void oiml_vec_dot_f16(int n, float* __restrict s, size_t bs, oiml_fp16_t* __restrict x, size_t bx, oiml_fp16_t* __restrict y, size_t by, int nrc);
OIML_INLINE void oiml_vec_dot_bf16(int n, float* __restrict s, size_t bs, oiml_bf16_t* __restrict x, size_t bx, oiml_bf16_t* __restrict y, size_t by, int nrc);

const struct std::unordered_map<oiml::oiml_representation_types, oiml_type_traits> type_traits{ {
	std::make_pair(oiml::oiml_representation_types::int_32,
		oiml_type_traits{
			.type_name	  = "i32",
			.blck_size	  = 1,
			.type_size	  = sizeof(int32_t),
			.is_quantized = false,
		}),
	std::make_pair(oiml::oiml_representation_types::float_32,
		oiml_type_traits{
			.vec_dot	  = ( oiml_vec_dot_t )oiml_vec_dot_f32,
			.vec_dot_type = oiml::oiml_representation_types::float_32,
			.nrows = 1,
			.type_name	  = "f32",
			.blck_size	  = 1,
			.type_size	  = sizeof(float),
			.is_quantized			 = false
		}),
	std::make_pair(oiml::oiml_representation_types::float_16,
		oiml_type_traits{
			.from_float	  = ( oiml_from_float_t )oiml::oiml_fp32_to_fp16_row,
			.vec_dot	  = ( oiml_vec_dot_t )oiml_vec_dot_f16,
			.vec_dot_type = oiml::oiml_representation_types::float_16,
			.nrows = 1,
			.type_name		= "f16",
			.blck_size		= 1,
			.type_size		= sizeof(oiml_fp16_t),
			.is_quantized	= false,
			.to_float		= oiml_fp16_to_fp32_row_bindings,
			.from_float_ref			 = oiml_fp32_to_fp16_row_bindings
		}),
	std::make_pair(oiml::oiml_representation_types::q8_0,
		oiml_type_traits{
			.vec_dot_type = oiml::oiml_representation_types::q8_0,
#if defined(__ARM_FEATURE_MATMUL_INT8)
			.nrows = 2,
#else
			.nrows = 1,
#endif
			.type_name		= "q8_0",
			.blck_size		= oiml::Q_SIZE,
			.type_size		= sizeof(oiml::block_q8_0<oiml_half>),
			.is_quantized	= true,
			.to_float		= dequantize_row_q8_0_bindings,
			.from_float_ref = quantize_row_q8_0_bindings_ref
		}),
	std::make_pair(oiml::oiml_representation_types::brain_float_16,
		oiml_type_traits{
			.from_float	  = ( oiml_from_float_t )oiml::oiml_fp32_to_bf16_row,
			.vec_dot	  = ( oiml_vec_dot_t )oiml_vec_dot_bf16,
			.vec_dot_type = oiml::oiml_representation_types::brain_float_16,
			.nrows = 1,
			.type_name		= "bf16",
			.blck_size		= 1,
			.type_size		= sizeof(oiml_bf16_t),
			.is_quantized	= false,
			.to_float		= oiml_bf16_to_fp32_row_bindings,
			.from_float_ref			 = oiml_fp32_to_bf16_row_bindings_ref,
		}),
} };

OIML_INLINE const oiml_type_traits* oiml_get_type_traits(oiml::oiml_representation_types type) {
	OIML_ASSERT(type < oiml::oiml_representation_types::count);
	return &type_traits.at(type);
}

//
// oiml object
//

struct oiml_object {
	size_t offs;
	size_t size;

	oiml_object* next;

	enum oiml_object_type type;

	char padding[4];
};

constexpr size_t OIML_OBJECT_SIZE = sizeof(struct oiml_object);

//
// oiml context
//

struct oiml_context {
	size_t mem_size;
	void* mem_buffer;
	bool mem_buffer_owned;
	bool no_alloc;

	int n_objects;

	oiml_object* objects_begin;
	oiml_object* objects_end;
};

struct oiml_context_container {
	bool used;

	struct oiml_context context;
};

//
// data types
//

constexpr const char* OIML_OP_NAME[OIML_OP_COUNT] = {
	"NONE",

	"DUP",
	"ADD",
	"ADD1",
	"ACC",
	"SUB",
	"MUL",
	"DIV",
	"SQR",
	"SQRT",
	"LOG",
	"SIN",
	"COS",
	"SUM",
	"SUM_ROWS",
	"MEAN",
	"ARGMAX",
	"COUNT_EQUAL",
	"REPEAT",
	"REPEAT_BACK",
	"CONCAT",
	"SILU_BACK",
	"NORM",
	"RMS_NORM",
	"RMS_NORM_BACK",
	"GROUP_NORM",

	"MUL_MAT",
	"MUL_MAT_ID",
	"OUT_PROD",

	"SCALE",
	"SET",
	"CPY",
	"CONT",
	"RESHAPE",
	"VIEW",
	"PERMUTE",
	"TRANSPOSE",
	"GET_ROWS",
	"GET_ROWS_BACK",
	"DIAG",
	"DIAG_MASK_INF",
	"DIAG_MASK_ZERO",
	"SOFT_MAX",
	"SOFT_MAX_BACK",
	"ROPE",
	"ROPE_BACK",
	"CLAMP",
	"CONV_TRANSPOSE_1D",
	"IM2COL",
	"IM2COL_BACK",
	"CONV_TRANSPOSE_2D",
	"POOL_1D",
	"POOL_2D",
	"POOL_2D_BACK",
	"UPSCALE",
	"PAD",
	"PAD_REFLECT_1D",
	"ARANGE",
	"TIMESTEP_EMBEDDING",
	"ARGSORT",
	"LEAKY_RELU",

	"FLASH_ATTN_EXT",
	"FLASH_ATTN_BACK",
	"SSM_CONV",
	"SSM_SCAN",
	"WIN_PART",
	"WIN_UNPART",
	"GET_REL_POS",
	"ADD_REL_POS",
	"RWKV_WKV6",
	"GATED_LINEAR_ATTN",

	"UNARY",

	"MAP_UNARY",
	"MAP_BINARY",

	"MAP_CUSTOM1_F32",
	"MAP_CUSTOM2_F32",
	"MAP_CUSTOM3_F32",

	"MAP_CUSTOM1",
	"MAP_CUSTOM2",
	"MAP_CUSTOM3",

	"CROSS_ENTROPY_LOSS",
	"CROSS_ENTROPY_LOSS_BACK",
	"OPT_STEP_ADAMW",
};

static_assert(OIML_OP_COUNT == 83, "OIML_OP_COUNT != 83");

constexpr const char* OIML_OP_SYMBOL[OIML_OP_COUNT] = {
	"none",

	"x",
	"x+y",
	"x+y",
	"view(x,nb,offset)+=y->x",
	"x-y",
	"x*y",
	"x/y",
	"x^2",
	"√x",
	"log(x)",
	"sin(x)",
	"cos(x)",
	"Σx",
	"Σx_k",
	"Σx/n",
	"argmax(x)",
	"count_equal(x)",
	"repeat(x)",
	"repeat_back(x)",
	"concat(x, y)",
	"silu_back(x)",
	"norm(x)",
	"rms_norm(x)",
	"rms_norm_back(x)",
	"group_norm(x)",

	"X*Y",
	"X[i]*Y",
	"X*Y",

	"x*v",
	"y-\\>view(x)",
	"x-\\>y",
	"cont(x)",
	"reshape(x)",
	"view(x)",
	"permute(x)",
	"transpose(x)",
	"get_rows(x)",
	"get_rows_back(x)",
	"diag(x)",
	"diag_mask_inf(x)",
	"diag_mask_zero(x)",
	"soft_max(x)",
	"soft_max_back(x)",
	"rope(x)",
	"rope_back(x)",
	"clamp(x)",
	"conv_transpose_1d(x)",
	"im2col(x)",
	"im2col_back(x)",
	"conv_transpose_2d(x)",
	"pool_1d(x)",
	"pool_2d(x)",
	"pool_2d_back(x)",
	"upscale(x)",
	"pad(x)",
	"pad_reflect_1d(x)",
	"arange(start, stop, step)",
	"timestep_embedding(timesteps, dim, max_period)",
	"argsort(x)",
	"leaky_relu(x)",

	"flash_attn_ext(x)",
	"flash_attn_back(x)",
	"ssm_conv(x)",
	"ssm_scan(x)",
	"win_part(x)",
	"win_unpart(x)",
	"get_rel_pos(x)",
	"add_rel_pos(x)",
	"rwkv_wkv6(k, v, r, tf, td, s)",
	"gated_linear_attn(k, v, q, gate, s)",

	"unary(x)",

	"f(x)",
	"f(x,y)",

	"custom_f32(x)",
	"custom_f32(x,y)",
	"custom_f32(x,y,z)",

	"custom(x)",
	"custom(x,y)",
	"custom(x,y,z)",

	"cross_entropy_loss(x,y)",
	"cross_entropy_loss_back(x,y)",
	"adamw(x)",
};

static_assert(OIML_OP_COUNT == 83, "OIML_OP_COUNT != 83");

static_assert(OIML_OP_POOL_COUNT == 2, "OIML_OP_POOL_COUNT != 2");


constexpr const char* OIML_UNARY_OP_NAME[OIML_UNARY_OP_COUNT] = {
	"ABS",
	"SGN",
	"NEG",
	"STEP",
	"TANH",
	"ELU",
	"RELU",
	"SIGMOID",
	"GELU",
	"GELU_QUICK",
	"SILU",
	"HARDSWISH",
	"HARDSIGMOID",
	"EXP",
};

static_assert(OIML_UNARY_OP_COUNT == 14, "OIML_UNARY_OP_COUNT != 14");


static_assert(sizeof(struct oiml_object) % OIML_MEM_ALIGN == 0, "oiml_object size must be a multiple of OIML_MEM_ALIGN");
static_assert(sizeof(struct oiml_tensor) % OIML_MEM_ALIGN == 0, "oiml_tensor size must be a multiple of OIML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

OIML_INLINE void oiml_print_object(const oiml_object* obj) {
	OIML_LOG_INFO(" - oiml_object: type = %d, offset = %zu, size = %zu, next = %p\n", obj->type, obj->offs, obj->size, ( const void* )obj->next);
}

OIML_INLINE void oiml_print_objects(const oiml_context* ctx) {
	oiml_object* obj = ctx->objects_begin;

	OIML_LOG_INFO("%s: objects in context %p:\n", __func__, ( const void* )ctx);

	while (obj != NULL) {
		oiml_print_object(obj);
		obj = obj->next;
	}

	OIML_LOG_INFO("%s: --- end ---\n", __func__);
}

OIML_INLINE int64_t oiml_nelements(const oiml_tensor* tensor) {
	static_assert(OIML_MAX_DIMS == 4, "OIML_MAX_DIMS is not 4 - update this function");

	return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

OIML_INLINE int64_t oiml_nrows(const oiml_tensor* tensor) {
	static_assert(OIML_MAX_DIMS == 4, "OIML_MAX_DIMS is not 4 - update this function");

	return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

OIML_INLINE size_t oiml_nbytes(const oiml_tensor* tensor) {
	size_t nbytes;
	const size_t blck_size = oiml_blck_size(tensor->type);
	if (blck_size == 1) {
		nbytes = oiml_type_size(tensor->type);
		for (int i = 0; i < OIML_MAX_DIMS; ++i) {
			nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
		}
	} else {
		nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;
		for (int i = 1; i < OIML_MAX_DIMS; ++i) {
			nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
		}
	}

	return nbytes;
}

OIML_INLINE size_t oiml_nbytes_pad(const oiml_tensor* tensor) {
	return OIML_PAD(oiml_nbytes(tensor), OIML_MEM_ALIGN);
}

OIML_INLINE int64_t oiml_blck_size(oiml::oiml_representation_types type) {
	return type_traits.at(type).blck_size;
}

OIML_INLINE size_t oiml_type_size(oiml::oiml_representation_types type) {
	return type_traits.at(type).type_size;
}

OIML_INLINE size_t oiml_row_size(oiml::oiml_representation_types type, int64_t ne) {
	assert(ne % oiml_blck_size(type) == 0);
	return oiml_type_size(type) * ne / oiml_blck_size(type);
}

OIML_INLINE double oiml_type_sizef(oiml::oiml_representation_types type) {
	return (( double )(type_traits.at(type).type_size)) / type_traits.at(type).blck_size;
}

OIML_INLINE const char* oiml_type_name(oiml::oiml_representation_types type) {
	return type < oiml::oiml_representation_types::count ? type_traits.at(type).type_name : "NONE";
}

OIML_INLINE bool oiml_is_quantized(oiml::oiml_representation_types type) {
	return type_traits.at(type).is_quantized;
}

OIML_INLINE const char* oiml_op_name(enum oiml_op op) {
	return OIML_OP_NAME[op];
}

OIML_INLINE const char* oiml_op_symbol(enum oiml_op op) {
	return OIML_OP_SYMBOL[op];
}

OIML_INLINE const char* oiml_unary_op_name(enum oiml_unary_op op) {
	return OIML_UNARY_OP_NAME[op];
}

OIML_INLINE const char* oiml_op_desc(const oiml_tensor* t) {
	if (t->op == OIML_OP_UNARY) {
		enum oiml_unary_op uop = oiml_get_unary_op(t);
		return oiml_unary_op_name(uop);
	}
	return oiml_op_name(t->op);
}

OIML_INLINE size_t oiml_element_size(const oiml_tensor* tensor) {
	return oiml_type_size(tensor->type);
}

OIML_INLINE bool oiml_is_scalar(const oiml_tensor* tensor) {
	static_assert(OIML_MAX_DIMS == 4, "OIML_MAX_DIMS is not 4 - update this function");

	return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

OIML_INLINE bool oiml_is_vector(const oiml_tensor* tensor) {
	static_assert(OIML_MAX_DIMS == 4, "OIML_MAX_DIMS is not 4 - update this function");

	return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

OIML_INLINE bool oiml_is_matrix(const oiml_tensor* tensor) {
	static_assert(OIML_MAX_DIMS == 4, "OIML_MAX_DIMS is not 4 - update this function");

	return tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

OIML_INLINE bool oiml_is_3d(const oiml_tensor* tensor) {
	return tensor->ne[3] == 1;
}

OIML_INLINE int oiml_n_dims(const oiml_tensor* tensor) {
	for (int i = OIML_MAX_DIMS - 1; i >= 1; --i) {
		if (tensor->ne[i] > 1) {
			return i + 1;
		}
	}
	return 1;
}

OIML_INLINE oiml::oiml_representation_types oiml_ftype_to_oiml_type(enum oiml_ftype ftype) {
	oiml::oiml_representation_types wtype = oiml::oiml_representation_types::count;

	switch (ftype) {
		case OIML_FTYPE_ALL_F32:
			wtype = oiml::oiml_representation_types::float_32;
			break;
		case OIML_FTYPE_MOSTLY_F16:
			wtype = oiml::oiml_representation_types::float_16;
			break;
		case OIML_FTYPE_MOSTLY_BF16:
			wtype = oiml::oiml_representation_types::brain_float_16;
			break;
		case OIML_FTYPE_MOSTLY_Q8_0:
			wtype = oiml::oiml_representation_types::q8_0;
			break;
		case OIML_FTYPE_UNKNOWN:
			wtype = oiml::oiml_representation_types::count;
			break;
	}

	OIML_ASSERT(wtype != oiml::oiml_representation_types::count);

	return wtype;
}

OIML_INLINE size_t oiml_tensor_overhead() {
	return OIML_OBJECT_SIZE + OIML_TENSOR_SIZE;
}

OIML_INLINE bool oiml_is_transposed(const oiml_tensor* tensor) {
	return tensor->nb[0] > tensor->nb[1];
}

OIML_INLINE bool oiml_is_contiguous_n(const oiml_tensor* tensor, int n) {
	size_t next_nb = oiml_type_size(tensor->type);
	if (tensor->ne[0] != oiml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
		return false;
	}
	next_nb *= tensor->ne[0] / oiml_blck_size(tensor->type);
	for (int i = 1; i < OIML_MAX_DIMS; i++) {
		if (tensor->ne[i] != 1) {
			if (i > n) {
				if (tensor->nb[i] != next_nb) {
					return false;
				}
				next_nb *= tensor->ne[i];
			} else {
				// this dimension does not need to be contiguous
				next_nb = tensor->ne[i] * tensor->nb[i];
			}
		}
	}
	return true;
}

OIML_INLINE bool oiml_is_contiguous(const oiml_tensor* tensor) {
	return oiml_is_contiguous_0(tensor);
}

OIML_INLINE bool oiml_is_contiguous_0(const oiml_tensor* tensor) {
	return oiml_is_contiguous_n(tensor, 0);
}

OIML_INLINE bool oiml_is_contiguous_1(const oiml_tensor* tensor) {
	return oiml_is_contiguous_n(tensor, 1);
}

OIML_INLINE bool oiml_is_contiguous_2(const oiml_tensor* tensor) {
	return oiml_is_contiguous_n(tensor, 2);
}

OIML_INLINE bool oiml_is_permuted(const oiml_tensor* tensor) {
	static_assert(OIML_MAX_DIMS == 4, "OIML_MAX_DIMS is not 4 - update this function");

	return tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

OIML_INLINE bool oiml_is_padded_1d(const oiml_tensor* tensor) {
	static_assert(OIML_MAX_DIMS == 4, "OIML_MAX_DIMS is not 4 - update this function");

	return tensor->nb[0] == oiml_type_size(tensor->type) && tensor->nb[2] == tensor->nb[1] * tensor->ne[1] && tensor->nb[3] == tensor->nb[2] * tensor->ne[2];
}

OIML_INLINE bool oiml_is_empty(const oiml_tensor* tensor) {
	for (int i = 0; i < OIML_MAX_DIMS; ++i) {
		if (tensor->ne[i] == 0) {
			// empty if any dimension has no elements
			return true;
		}
	}
	return false;
}

OIML_INLINE bool oiml_are_same_shape(const oiml_tensor* t0, const oiml_tensor* t1) {
	static_assert(OIML_MAX_DIMS == 4, "OIML_MAX_DIMS is not 4 - update this function");

	return (t0->ne[0] == t1->ne[0]) && (t0->ne[1] == t1->ne[1]) && (t0->ne[2] == t1->ne[2]) && (t0->ne[3] == t1->ne[3]);
}

OIML_INLINE bool oiml_are_same_stride(const oiml_tensor* t0, const oiml_tensor* t1) {
	static_assert(OIML_MAX_DIMS == 4, "OIML_MAX_DIMS is not 4 - update this function");

	return (t0->nb[0] == t1->nb[0]) && (t0->nb[1] == t1->nb[1]) && (t0->nb[2] == t1->nb[2]) && (t0->nb[3] == t1->nb[3]);
}

// check if t1 can be represented as a repeatition of t0
OIML_INLINE bool oiml_can_repeat(const oiml_tensor* t0, const oiml_tensor* t1) {
	static_assert(OIML_MAX_DIMS == 4, "OIML_MAX_DIMS is not 4 - update this function");

	return oiml_is_empty(t0) ? oiml_is_empty(t1) : (t1->ne[0] % t0->ne[0] == 0) && (t1->ne[1] % t0->ne[1] == 0) && (t1->ne[2] % t0->ne[2] == 0) && (t1->ne[3] % t0->ne[3] == 0);
}

OIML_INLINE bool oiml_can_repeat_rows(const oiml_tensor* t0, const oiml_tensor* t1) {
	static_assert(OIML_MAX_DIMS == 4, "OIML_MAX_DIMS is not 4 - update this function");

	return (t0->ne[0] == t1->ne[0]) && oiml_can_repeat(t0, t1);
}

// assert that pointer is aligned to OIML_MEM_ALIGN
#define OIML_ASSERT_ALIGNED(ptr) OIML_ASSERT((( uintptr_t )(ptr)) % OIML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

OIML_INLINE oiml_context* oiml_init(struct oiml_init_params params) {
	bool is_first_call = true;

	oiml_critical_section_start();

	oiml_critical_section_end();

	oiml_context* ctx = ( oiml_context* )OIML_MALLOC(sizeof(struct oiml_context));

	// allow to call oiml_init with 0 size
	if (params.mem_size == 0) {
		params.mem_size = OIML_MEM_ALIGN;
	}

	const size_t mem_size = params.mem_buffer ? params.mem_size : OIML_PAD(params.mem_size, OIML_MEM_ALIGN);

	*ctx = oiml_context{
		/*.mem_size           =*/mem_size,
		/*.mem_buffer         =*/params.mem_buffer ? params.mem_buffer : oiml_aligned_malloc(mem_size),
		/*.mem_buffer_owned   =*/params.mem_buffer ? false : true,
		/*.no_alloc           =*/params.no_alloc,
		/*.n_objects          =*/0,
		/*.objects_begin      =*/NULL,
		/*.objects_end        =*/NULL,
	};

	OIML_ASSERT(ctx->mem_buffer != NULL);

	OIML_ASSERT_ALIGNED(ctx->mem_buffer);

	OIML_PRINT_DEBUG("%s: context initialized\n", __func__);

	return ctx;
}

OIML_INLINE void oiml_reset(oiml_context* ctx) {
	if (ctx == NULL) {
		return;
	}

	ctx->n_objects	   = 0;
	ctx->objects_begin = NULL;
	ctx->objects_end   = NULL;
}

OIML_INLINE void oiml_free(oiml_context* ctx) {
	if (ctx == NULL) {
		return;
	}

	if (ctx->mem_buffer_owned) {
		oiml_aligned_free(ctx->mem_buffer, ctx->mem_size);
	}

	OIML_FREE(ctx);
}

OIML_INLINE size_t oiml_used_mem(const oiml_context* ctx) {
	return ctx->objects_end == NULL ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

OIML_INLINE bool oiml_get_no_alloc(oiml_context* ctx) {
	return ctx->no_alloc;
}

OIML_INLINE void oiml_set_no_alloc(oiml_context* ctx, bool no_alloc) {
	ctx->no_alloc = no_alloc;
}

OIML_INLINE void* oiml_get_mem_buffer(const oiml_context* ctx) {
	return ctx->mem_buffer;
}

OIML_INLINE size_t oiml_get_mem_size(const oiml_context* ctx) {
	return ctx->mem_size;
}

OIML_INLINE size_t oiml_get_max_tensor_size(const oiml_context* ctx) {
	size_t max_size = 0;

	for (oiml_tensor* tensor = oiml_get_first_tensor(ctx); tensor != NULL; tensor = oiml_get_next_tensor(ctx, tensor)) {
		size_t bytes = oiml_nbytes(tensor);
		max_size	 = MAX(max_size, bytes);
	}

	return max_size;
}

////////////////////////////////////////////////////////////////////////////////

OIML_INLINE oiml_object* oiml_new_object(oiml_context* ctx, enum oiml_object_type type, size_t size) {
	// always insert objects at the end of the context's memory pool
	oiml_object* obj_cur = ctx->objects_end;

	const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
	const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
	const size_t cur_end  = cur_offs + cur_size;

	// align to OIML_MEM_ALIGN
	size_t size_needed = OIML_PAD(size, OIML_MEM_ALIGN);

	char* const mem_buffer	   = ( char* const )ctx->mem_buffer;
	oiml_object* const obj_new = ( oiml_object* )(mem_buffer + cur_end);

	if (cur_end + size_needed + OIML_OBJECT_SIZE > ctx->mem_size) {
		OIML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n", __func__, cur_end + size_needed + OIML_OBJECT_SIZE, ctx->mem_size);
#ifndef NDEBUG
		OIML_ABORT("not enough space in the context's memory pool");
#endif
		return NULL;
	}

	*obj_new = oiml_object{
		.offs = cur_end + OIML_OBJECT_SIZE,
		.size = size_needed,
		.next = NULL,
		.type = type,
	};

	OIML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

	if (obj_cur != NULL) {
		obj_cur->next = obj_new;
	} else {
		// this is the first object in this context
		ctx->objects_begin = obj_new;
	}

	ctx->objects_end = obj_new;

	//printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

	return obj_new;
}

OIML_INLINE oiml_tensor* oiml_new_tensor_impl(oiml_context* ctx, oiml::oiml_representation_types type, int n_dims, const int64_t* ne, oiml_tensor* view_src, size_t view_offs) {
	OIML_ASSERT(static_cast<int32_t>(type) >= 0 && type < oiml::oiml_representation_types::count);
	OIML_ASSERT(n_dims >= 1 && n_dims <= OIML_MAX_DIMS);

	// find the base tensor and absolute offset
	if (view_src != NULL && view_src->view_src != NULL) {
		view_offs += view_src->view_offs;
		view_src = view_src->view_src;
	}

	size_t data_size = oiml_row_size(type, ne[0]);
	for (int i = 1; i < n_dims; i++) {
		data_size *= ne[i];
	}

	OIML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= oiml_nbytes(view_src));

	void* data = view_src != NULL ? view_src->data : NULL;
	if (data != NULL) {
		data = ( char* )data + view_offs;
	}

	size_t obj_alloc_size = 0;

	if (view_src == NULL && !ctx->no_alloc) {
		// allocate tensor data in the context's memory pool
		obj_alloc_size = data_size;
	}

	oiml_object* const obj_new = oiml_new_object(ctx, OIML_OBJECT_TYPE_TENSOR, OIML_TENSOR_SIZE + obj_alloc_size);
	OIML_ASSERT(obj_new);

	oiml_tensor* const result = ( oiml_tensor* )(( char* )ctx->mem_buffer + obj_new->offs);

	*result = oiml_tensor{
		/*.type         =*/type,
		/*.buffer       =*/NULL,
		/*.ne           =*/{ 1, 1, 1, 1 },
		/*.nb           =*/{ 0, 0, 0, 0 },
		/*.op           =*/OIML_OP_NONE,
		/*.op_params    =*/{ 0 },
		/*.flags        =*/0,
		/*.src          =*/{ NULL },
		/*.view_src     =*/view_src,
		/*.view_offs    =*/view_offs,
		/*.data         =*/obj_alloc_size > 0 ? ( void* )(result + 1) : data,
		/*.name         =*/{ 0 },
		/*.extra        =*/NULL,
		/*.padding      =*/{ 0 },
	};

	// TODO: this should not be needed as long as we don't rely on aligned SIMD loads
	//OIML_ASSERT_ALIGNED(result->data);

	for (int i = 0; i < n_dims; i++) {
		result->ne[i] = ne[i];
	}

	result->nb[0] = oiml_type_size(type);
	result->nb[1] = result->nb[0] * (result->ne[0] / oiml_blck_size(type));
	for (int i = 2; i < OIML_MAX_DIMS; i++) {
		result->nb[i] = result->nb[i - 1] * result->ne[i - 1];
	}

	ctx->n_objects++;

	return result;
}

OIML_INLINE oiml_tensor* oiml_new_tensor(oiml_context* ctx, oiml::oiml_representation_types type, int n_dims, const int64_t* ne) {
	return oiml_new_tensor_impl(ctx, type, n_dims, ne, NULL, 0);
}

OIML_INLINE oiml_tensor* oiml_new_tensor_1d(oiml_context* ctx, oiml::oiml_representation_types type, int64_t ne0) {
	return oiml_new_tensor(ctx, type, 1, &ne0);
}

OIML_INLINE oiml_tensor* oiml_new_tensor_2d(oiml_context* ctx, oiml::oiml_representation_types type, int64_t ne0, int64_t ne1) {
	const int64_t ne[2] = { ne0, ne1 };
	return oiml_new_tensor(ctx, type, 2, ne);
}

OIML_INLINE oiml_tensor* oiml_new_tensor_3d(oiml_context* ctx, oiml::oiml_representation_types type, int64_t ne0, int64_t ne1, int64_t ne2) {
	const int64_t ne[3] = { ne0, ne1, ne2 };
	return oiml_new_tensor(ctx, type, 3, ne);
}

OIML_INLINE oiml_tensor* oiml_new_tensor_4d(oiml_context* ctx, oiml::oiml_representation_types type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
	const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
	return oiml_new_tensor(ctx, type, 4, ne);
}

OIML_INLINE void* oiml_new_buffer(oiml_context* ctx, size_t nbytes) {
	oiml_object* obj = oiml_new_object(ctx, OIML_OBJECT_TYPE_WORK_BUFFER, nbytes);

	return ( int8_t* )ctx->mem_buffer + obj->offs;
}

OIML_INLINE oiml_tensor* oiml_dup_tensor(oiml_context* ctx, const oiml_tensor* src) {
	return oiml_new_tensor(ctx, src->type, OIML_MAX_DIMS, src->ne);
}

OIML_INLINE void oiml_unravel_index(const oiml_tensor* tensor, int64_t i, int64_t* i0, int64_t* i1, int64_t* i2, int64_t* i3) {
	const int64_t ne2 = tensor->ne[2];
	const int64_t ne1 = tensor->ne[1];
	const int64_t ne0 = tensor->ne[0];

	const int64_t i3_ = (i / (ne2 * ne1 * ne0));
	const int64_t i2_ = (i - i3_ * ne2 * ne1 * ne0) / (ne1 * ne0);
	const int64_t i1_ = (i - i3_ * ne2 * ne1 * ne0 - i2_ * ne1 * ne0) / ne0;
	const int64_t i0_ = (i - i3_ * ne2 * ne1 * ne0 - i2_ * ne1 * ne0 - i1_ * ne0);

	if (i0) {
		*i0 = i0_;
	}
	if (i1) {
		*i1 = i1_;
	}
	if (i2) {
		*i2 = i2_;
	}
	if (i3) {
		*i3 = i3_;
	}
}

OIML_INLINE void* oiml_get_data(const oiml_tensor* tensor) {
	return tensor->data;
}

OIML_INLINE float* oiml_get_data_f32(const oiml_tensor* tensor) {
	assert(tensor->type == oiml::oiml_representation_types::float_32);
	return ( float* )(tensor->data);
}

OIML_INLINE enum oiml_unary_op oiml_get_unary_op(const oiml_tensor* tensor) {
	OIML_ASSERT(tensor->op == OIML_OP_UNARY);
	return ( enum oiml_unary_op )oiml_get_op_params_i32(tensor, 0);
}

OIML_INLINE const char* oiml_get_name(const oiml_tensor* tensor) {
	return tensor->name;
}

OIML_INLINE oiml_tensor* oiml_set_name(oiml_tensor* tensor, const char* name) {
	size_t i;
	for (i = 0; i < sizeof(tensor->name) - 1 && name[i] != '\0'; i++) {
		tensor->name[i] = name[i];
	}
	tensor->name[i] = '\0';
	return tensor;
}

OIML_INLINE oiml_tensor* oiml_format_name(oiml_tensor* tensor, const char* fmt, ...) {
	va_list args;
	va_start(args, fmt);
	vsnprintf(tensor->name, sizeof(tensor->name), fmt, args);
	va_end(args);
	return tensor;
}

OIML_INLINE oiml_tensor* oiml_view_tensor(oiml_context* ctx, oiml_tensor* src) {
	oiml_tensor* result = oiml_new_tensor_impl(ctx, src->type, OIML_MAX_DIMS, src->ne, src, 0);
	oiml_format_name(result, "%s (view)", src->name);

	for (int i = 0; i < OIML_MAX_DIMS; i++) {
		result->nb[i] = src->nb[i];
	}

	return result;
}

OIML_INLINE oiml_tensor* oiml_get_first_tensor(const oiml_context* ctx) {
	oiml_object* obj = ctx->objects_begin;

	char* const mem_buffer = ( char* const )ctx->mem_buffer;

	while (obj != NULL) {
		if (obj->type == OIML_OBJECT_TYPE_TENSOR) {
			return ( oiml_tensor* )(mem_buffer + obj->offs);
		}

		obj = obj->next;
	}

	return NULL;
}

OIML_INLINE oiml_tensor* oiml_get_next_tensor(const oiml_context* ctx, oiml_tensor* tensor) {
	oiml_object* obj = ( oiml_object* )(( char* )tensor - OIML_OBJECT_SIZE);
	obj				 = obj->next;

	char* const mem_buffer = ( char* const )ctx->mem_buffer;

	while (obj != NULL) {
		if (obj->type == OIML_OBJECT_TYPE_TENSOR) {
			return ( oiml_tensor* )(mem_buffer + obj->offs);
		}

		obj = obj->next;
	}

	return NULL;
}

OIML_INLINE oiml_tensor* oiml_get_tensor(oiml_context* ctx, const char* name) {
	oiml_object* obj = ctx->objects_begin;

	char* const mem_buffer = ( char* const )ctx->mem_buffer;

	while (obj != NULL) {
		if (obj->type == OIML_OBJECT_TYPE_TENSOR) {
			oiml_tensor* cur = ( oiml_tensor* )(mem_buffer + obj->offs);
			if (strcmp(cur->name, name) == 0) {
				return cur;
			}
		}

		obj = obj->next;
	}

	return NULL;
}

////////////////////////////////////////////////////////////////////////////////

// oiml_dup

OIML_INLINE oiml_tensor* oiml_dup_impl(oiml_context* ctx, oiml_tensor* a, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	result->op	   = OIML_OP_DUP;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_dup(oiml_context* ctx, oiml_tensor* a) {
	return oiml_dup_impl(ctx, a, false);
}

OIML_INLINE oiml_tensor* oiml_dup_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_dup_impl(ctx, a, true);
}

// oiml_add

OIML_INLINE oiml_tensor* oiml_add_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, bool inplace) {
	OIML_ASSERT(oiml_can_repeat(b, a));

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	result->op	   = OIML_OP_ADD;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE oiml_tensor* oiml_add(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	return oiml_add_impl(ctx, a, b, false);
}

OIML_INLINE oiml_tensor* oiml_add_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	return oiml_add_impl(ctx, a, b, true);
}

// oiml_add_cast

OIML_INLINE oiml_tensor* oiml_add_cast_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml::oiml_representation_types type) {
	// TODO: support less-strict constraint
	//       OIML_ASSERT(oiml_can_repeat(b, a));
	OIML_ASSERT(oiml_can_repeat_rows(b, a));

	// currently only supported for quantized input and f16
	OIML_ASSERT(oiml_is_quantized(a->type) || a->type == oiml::oiml_representation_types::float_16 || a->type == oiml::oiml_representation_types::brain_float_16);

	oiml_tensor* result = oiml_new_tensor(ctx, type, OIML_MAX_DIMS, a->ne);

	result->op	   = OIML_OP_ADD;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE oiml_tensor* oiml_add_cast(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml::oiml_representation_types type) {
	return oiml_add_cast_impl(ctx, a, b, type);
}

// oiml_add1

OIML_INLINE oiml_tensor* oiml_add1_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, bool inplace) {
	OIML_ASSERT(oiml_is_scalar(b));
	OIML_ASSERT(oiml_is_padded_1d(a));

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	result->op	   = OIML_OP_ADD1;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE oiml_tensor* oiml_add1(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	return oiml_add1_impl(ctx, a, b, false);
}

OIML_INLINE oiml_tensor* oiml_add1_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	return oiml_add1_impl(ctx, a, b, true);
}

// oiml_acc

OIML_INLINE oiml_tensor* oiml_acc_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset, bool inplace) {
	OIML_ASSERT(oiml_nelements(b) <= oiml_nelements(a));
	OIML_ASSERT(oiml_is_contiguous(a));
	OIML_ASSERT(a->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(b->type == oiml::oiml_representation_types::float_32);

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	int32_t params[] = { static_cast<int32_t>(nb1), static_cast<int32_t>(nb2), static_cast<int32_t>(nb3), static_cast<int32_t>(offset), inplace ? 1 : 0 };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_ACC;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE oiml_tensor* oiml_acc(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset) {
	return oiml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

OIML_INLINE oiml_tensor* oiml_acc_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset) {
	return oiml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// oiml_sub

OIML_INLINE oiml_tensor* oiml_sub_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, bool inplace) {
	OIML_ASSERT(oiml_can_repeat(b, a));

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	result->op	   = OIML_OP_SUB;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE oiml_tensor* oiml_sub(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	return oiml_sub_impl(ctx, a, b, false);
}

OIML_INLINE oiml_tensor* oiml_sub_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	return oiml_sub_impl(ctx, a, b, true);
}

// oiml_mul

OIML_INLINE oiml_tensor* oiml_mul_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, bool inplace) {
	OIML_ASSERT(oiml_can_repeat(b, a));

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	result->op	   = OIML_OP_MUL;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE oiml_tensor* oiml_mul(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	return oiml_mul_impl(ctx, a, b, false);
}

OIML_INLINE oiml_tensor* oiml_mul_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	return oiml_mul_impl(ctx, a, b, true);
}

// oiml_div

OIML_INLINE oiml_tensor* oiml_div_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, bool inplace) {
	OIML_ASSERT(oiml_can_repeat(b, a));

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	result->op	   = OIML_OP_DIV;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE oiml_tensor* oiml_div(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	return oiml_div_impl(ctx, a, b, false);
}

OIML_INLINE oiml_tensor* oiml_div_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	return oiml_div_impl(ctx, a, b, true);
}

// oiml_sqr

OIML_INLINE oiml_tensor* oiml_sqr_impl(oiml_context* ctx, oiml_tensor* a, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	result->op	   = OIML_OP_SQR;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_sqr(oiml_context* ctx, oiml_tensor* a) {
	return oiml_sqr_impl(ctx, a, false);
}

OIML_INLINE oiml_tensor* oiml_sqr_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_sqr_impl(ctx, a, true);
}

// oiml_sqrt

OIML_INLINE oiml_tensor* oiml_sqrt_impl(oiml_context* ctx, oiml_tensor* a, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	result->op	   = OIML_OP_SQRT;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_sqrt(oiml_context* ctx, oiml_tensor* a) {
	return oiml_sqrt_impl(ctx, a, false);
}

OIML_INLINE oiml_tensor* oiml_sqrt_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_sqrt_impl(ctx, a, true);
}

// oiml_log

OIML_INLINE oiml_tensor* oiml_log_impl(oiml_context* ctx, oiml_tensor* a, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	result->op	   = OIML_OP_LOG;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_log(oiml_context* ctx, oiml_tensor* a) {
	return oiml_log_impl(ctx, a, false);
}

OIML_INLINE oiml_tensor* oiml_log_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_log_impl(ctx, a, true);
}

// oiml_sin

OIML_INLINE oiml_tensor* oiml_sin_impl(oiml_context* ctx, oiml_tensor* a, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	result->op	   = OIML_OP_SIN;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_sin(oiml_context* ctx, oiml_tensor* a) {
	return oiml_sin_impl(ctx, a, false);
}

OIML_INLINE oiml_tensor* oiml_sin_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_sin_impl(ctx, a, true);
}

// oiml_cos

OIML_INLINE oiml_tensor* oiml_cos_impl(oiml_context* ctx, oiml_tensor* a, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	result->op	   = OIML_OP_COS;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_cos(oiml_context* ctx, oiml_tensor* a) {
	return oiml_cos_impl(ctx, a, false);
}

OIML_INLINE oiml_tensor* oiml_cos_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_cos_impl(ctx, a, true);
}

// oiml_sum

OIML_INLINE oiml_tensor* oiml_sum(oiml_context* ctx, oiml_tensor* a) {
	oiml_tensor* result = oiml_new_tensor_1d(ctx, a->type, 1);

	result->op	   = OIML_OP_SUM;
	result->src[0] = a;

	return result;
}

// oiml_sum_rows

OIML_INLINE oiml_tensor* oiml_sum_rows(oiml_context* ctx, oiml_tensor* a) {
	int64_t ne[OIML_MAX_DIMS] = { 1 };
	for (int i = 1; i < OIML_MAX_DIMS; ++i) {
		ne[i] = a->ne[i];
	}

	oiml_tensor* result = oiml_new_tensor(ctx, a->type, OIML_MAX_DIMS, ne);

	result->op	   = OIML_OP_SUM_ROWS;
	result->src[0] = a;

	return result;
}

// oiml_mean

OIML_INLINE oiml_tensor* oiml_mean(oiml_context* ctx, oiml_tensor* a) {
	int64_t ne[4]		= { 1, a->ne[1], a->ne[2], a->ne[3] };
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, ne);

	result->op	   = OIML_OP_MEAN;
	result->src[0] = a;

	return result;
}

// oiml_argmax

OIML_INLINE oiml_tensor* oiml_argmax(oiml_context* ctx, oiml_tensor* a) {
	OIML_ASSERT(oiml_is_matrix(a));
	OIML_ASSERT(a->ne[0] <= INT32_MAX);

	oiml_tensor* result = oiml_new_tensor_1d(ctx, oiml::oiml_representation_types::int_32, a->ne[1]);

	result->op	   = OIML_OP_ARGMAX;
	result->src[0] = a;

	return result;
}

// oiml_count_equal

OIML_INLINE oiml_tensor* oiml_count_equal(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	OIML_ASSERT(false && "remove or rewrite this, since it depends on I64 which is unsupported");
	return nullptr;

#if 0// [rnourai] TODO: remove
	OIML_ASSERT(oiml_are_same_shape(a, b));

	oiml_tensor* result = oiml_new_tensor_1d(ctx, OIML_TYPE_I64, 1);

	result->op	   = OIML_OP_COUNT_EQUAL;
	result->src[0] = a;
	result->src[1] = b;

	return result;
#endif
}

// oiml_repeat

OIML_INLINE oiml_tensor* oiml_repeat(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	OIML_ASSERT(oiml_can_repeat(a, b));

	oiml_tensor* result = oiml_new_tensor(ctx, a->type, OIML_MAX_DIMS, b->ne);

	result->op	   = OIML_OP_REPEAT;
	result->src[0] = a;

	return result;
}

// oiml_repeat_back

OIML_INLINE oiml_tensor* oiml_repeat_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	OIML_ASSERT(oiml_can_repeat(b, a));

	oiml_tensor* result = oiml_new_tensor(ctx, a->type, OIML_MAX_DIMS, b->ne);

	result->op	   = OIML_OP_REPEAT_BACK;
	result->src[0] = a;

	return result;
}

// oiml_concat

OIML_INLINE oiml_tensor* oiml_concat(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int dim) {
	OIML_ASSERT(dim >= 0 && dim < OIML_MAX_DIMS);

	int64_t ne[OIML_MAX_DIMS];
	for (int d = 0; d < OIML_MAX_DIMS; ++d) {
		if (d == dim) {
			ne[d] = a->ne[d] + b->ne[d];
			continue;
		}
		OIML_ASSERT(a->ne[d] == b->ne[d]);
		ne[d] = a->ne[d];
	}

	oiml_tensor* result = oiml_new_tensor(ctx, a->type, OIML_MAX_DIMS, ne);

	oiml_set_op_params_i32(result, 0, dim);

	result->op	   = OIML_OP_CONCAT;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

// oiml_abs

OIML_INLINE oiml_tensor* oiml_abs(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_ABS);
}

OIML_INLINE oiml_tensor* oiml_abs_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary_inplace(ctx, a, OIML_UNARY_OP_ABS);
}

// oiml_sgn

OIML_INLINE oiml_tensor* oiml_sgn(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_SGN);
}

OIML_INLINE oiml_tensor* oiml_sgn_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary_inplace(ctx, a, OIML_UNARY_OP_SGN);
}

// oiml_neg

OIML_INLINE oiml_tensor* oiml_neg(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_NEG);
}

OIML_INLINE oiml_tensor* oiml_neg_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary_inplace(ctx, a, OIML_UNARY_OP_NEG);
}

// oiml_step

OIML_INLINE oiml_tensor* oiml_step(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_STEP);
}

OIML_INLINE oiml_tensor* oiml_step_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary_inplace(ctx, a, OIML_UNARY_OP_STEP);
}

// oiml_tanh

OIML_INLINE oiml_tensor* oiml_tanh(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_TANH);
}

OIML_INLINE oiml_tensor* oiml_tanh_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary_inplace(ctx, a, OIML_UNARY_OP_TANH);
}

// oiml_elu

OIML_INLINE oiml_tensor* oiml_elu(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_ELU);
}

OIML_INLINE oiml_tensor* oiml_elu_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary_inplace(ctx, a, OIML_UNARY_OP_ELU);
}

// oiml_relu

OIML_INLINE oiml_tensor* oiml_relu(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_RELU);
}

OIML_INLINE oiml_tensor* oiml_relu_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary_inplace(ctx, a, OIML_UNARY_OP_RELU);
}

// oiml_leaky_relu

OIML_INLINE oiml_tensor* oiml_leaky_relu(oiml_context* ctx, oiml_tensor* a, float negative_slope, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	oiml_set_op_params(result, &negative_slope, sizeof(negative_slope));

	result->op	   = OIML_OP_LEAKY_RELU;
	result->src[0] = a;

	return result;
}

// oiml_sigmoid

OIML_INLINE oiml_tensor* oiml_sigmoid(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_SIGMOID);
}

OIML_INLINE oiml_tensor* oiml_sigmoid_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary_inplace(ctx, a, OIML_UNARY_OP_SIGMOID);
}

// oiml_gelu

OIML_INLINE oiml_tensor* oiml_gelu(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_GELU);
}

OIML_INLINE oiml_tensor* oiml_gelu_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary_inplace(ctx, a, OIML_UNARY_OP_GELU);
}

// oiml_gelu_quick

OIML_INLINE oiml_tensor* oiml_gelu_quick(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_GELU_QUICK);
}

OIML_INLINE oiml_tensor* oiml_gelu_quick_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary_inplace(ctx, a, OIML_UNARY_OP_GELU_QUICK);
}

// oiml_silu

OIML_INLINE oiml_tensor* oiml_silu(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_SILU);
}

OIML_INLINE oiml_tensor* oiml_silu_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary_inplace(ctx, a, OIML_UNARY_OP_SILU);
}

// oiml_silu_back

OIML_INLINE oiml_tensor* oiml_silu_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	oiml_tensor* result = oiml_dup_tensor(ctx, a);

	result->op	   = OIML_OP_SILU_BACK;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

// oiml hardswish

OIML_INLINE oiml_tensor* oiml_hardswish(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_HARDSWISH);
}

// oiml hardsigmoid

OIML_INLINE oiml_tensor* oiml_hardsigmoid(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_HARDSIGMOID);
}

// oiml exp

OIML_INLINE oiml_tensor* oiml_exp(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary(ctx, a, OIML_UNARY_OP_EXP);
}

OIML_INLINE oiml_tensor* oiml_exp_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_unary_inplace(ctx, a, OIML_UNARY_OP_EXP);
}

// oiml_norm

OIML_INLINE oiml_tensor* oiml_norm_impl(oiml_context* ctx, oiml_tensor* a, float eps, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	oiml_set_op_params(result, &eps, sizeof(eps));

	result->op	   = OIML_OP_NORM;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_norm(oiml_context* ctx, oiml_tensor* a, float eps) {
	return oiml_norm_impl(ctx, a, eps, false);
}

OIML_INLINE oiml_tensor* oiml_norm_inplace(oiml_context* ctx, oiml_tensor* a, float eps) {
	return oiml_norm_impl(ctx, a, eps, true);
}

// oiml_rms_norm

OIML_INLINE oiml_tensor* oiml_rms_norm_impl(oiml_context* ctx, oiml_tensor* a, float eps, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	oiml_set_op_params(result, &eps, sizeof(eps));

	result->op	   = OIML_OP_RMS_NORM;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_rms_norm(oiml_context* ctx, oiml_tensor* a, float eps) {
	return oiml_rms_norm_impl(ctx, a, eps, false);
}

OIML_INLINE oiml_tensor* oiml_rms_norm_inplace(oiml_context* ctx, oiml_tensor* a, float eps) {
	return oiml_rms_norm_impl(ctx, a, eps, true);
}

// oiml_rms_norm_back

OIML_INLINE oiml_tensor* oiml_rms_norm_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, float eps) {
	oiml_tensor* result = oiml_dup_tensor(ctx, a);

	oiml_set_op_params(result, &eps, sizeof(eps));

	result->op	   = OIML_OP_RMS_NORM_BACK;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

// oiml_group_norm

OIML_INLINE oiml_tensor* oiml_group_norm_impl(oiml_context* ctx, oiml_tensor* a, int n_groups, float eps, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	oiml_set_op_params_i32(result, 0, n_groups);
	oiml_set_op_params_f32(result, 1, eps);

	result->op	   = OIML_OP_GROUP_NORM;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_group_norm(oiml_context* ctx, oiml_tensor* a, int n_groups, float eps) {
	return oiml_group_norm_impl(ctx, a, n_groups, eps, false);
}

OIML_INLINE oiml_tensor* oiml_group_norm_inplace(oiml_context* ctx, oiml_tensor* a, int n_groups, float eps) {
	return oiml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// oiml_mul_mat

OIML_INLINE bool oiml_can_mul_mat(const oiml_tensor* t0, const oiml_tensor* t1) {
	static_assert(OIML_MAX_DIMS == 4, "OIML_MAX_DIMS is not 4 - update this function");

	return (t0->ne[0] == t1->ne[0]) && (t1->ne[2] % t0->ne[2] == 0) &&// verify t0 is broadcastable
		(t1->ne[3] % t0->ne[3] == 0);
}

OIML_INLINE oiml_tensor* oiml_mul_mat(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	OIML_ASSERT(oiml_can_mul_mat(a, b));
	OIML_ASSERT(!oiml_is_transposed(a));

	const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, ne);

	result->op	   = OIML_OP_MUL_MAT;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE void oiml_mul_mat_set_prec(oiml_tensor* a, enum oiml_prec prec) {
	OIML_ASSERT(a->op == OIML_OP_MUL_MAT);

	const int32_t prec_i32 = ( int32_t )prec;

	oiml_set_op_params_i32(a, 0, prec_i32);
}

// oiml_mul_mat_id

/*
    c = oiml_mul_mat_id(ctx, as, b, ids);

    as  -> [cols, rows, n_expert]
    ids -> [n_experts_used, n_tokens] (i32)
    b   -> [cols, n_expert_used, n_tokens]
    c   -> [rows, n_expert_used, n_tokens]

    in b, n_experts_used can be broadcasted to match the n_expert_used of ids

    c ~= as[:,:,i] @ b[:,i%r,t], i = ids[e,t] for all e,t in ids
*/
OIML_INLINE oiml_tensor* oiml_mul_mat_id(oiml_context* ctx, oiml_tensor* as, oiml_tensor* b, oiml_tensor* ids) {
	OIML_ASSERT(!oiml_is_transposed(as));
	OIML_ASSERT(ids->type == oiml::oiml_representation_types::int_32);

	OIML_ASSERT(as->ne[3] == 1);// as is 3d (one matrix per expert)
	OIML_ASSERT(b->ne[3] == 1);// b is 3d
	OIML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1);// ids is 2d
	OIML_ASSERT(ids->ne[1] == b->ne[2]);// must have an expert list per b row
	OIML_ASSERT(as->ne[0] == b->ne[0]);// can_mul_mat
	OIML_ASSERT(ids->ne[0] % b->ne[1] == 0);// can broadcast

	const int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, ne);

	result->op	   = OIML_OP_MUL_MAT_ID;
	result->src[0] = as;
	result->src[1] = b;
	result->src[2] = ids;

	return result;
}

// oiml_out_prod

OIML_INLINE bool oiml_can_out_prod(const oiml_tensor* t0, const oiml_tensor* t1) {
	static_assert(OIML_MAX_DIMS == 4, "OIML_MAX_DIMS is not 4 - update this function");

	return (t0->ne[1] == t1->ne[1]) && (t1->ne[2] % t0->ne[2] == 0) &&// verify t0 is broadcastable
		(t1->ne[3] % t0->ne[3] == 0);
}

OIML_INLINE oiml_tensor* oiml_out_prod(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	OIML_ASSERT(oiml_can_out_prod(a, b));
	OIML_ASSERT(!oiml_is_transposed(a));

	// a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
	const int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, ne);

	result->op	   = OIML_OP_OUT_PROD;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

// oiml_scale

OIML_INLINE oiml_tensor* oiml_scale_impl(oiml_context* ctx, oiml_tensor* a, float s, bool inplace) {
	OIML_ASSERT(oiml_is_padded_1d(a));

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	oiml_set_op_params(result, &s, sizeof(s));

	result->op	   = OIML_OP_SCALE;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_scale(oiml_context* ctx, oiml_tensor* a, float s) {
	return oiml_scale_impl(ctx, a, s, false);
}

OIML_INLINE oiml_tensor* oiml_scale_inplace(oiml_context* ctx, oiml_tensor* a, float s) {
	return oiml_scale_impl(ctx, a, s, true);
}

// oiml_set

OIML_INLINE oiml_tensor* oiml_set_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset, bool inplace) {
	OIML_ASSERT(oiml_nelements(a) >= oiml_nelements(b));

	// make a view of the destination
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	OIML_ASSERT(offset < ( size_t )(1 << 30));
	int32_t params[] = { static_cast<int32_t>(nb1), static_cast<int32_t>(nb2), static_cast<int32_t>(nb3), static_cast<int32_t>(offset), inplace ? 1 : 0 };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_SET;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE oiml_tensor* oiml_set(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset) {
	return oiml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

OIML_INLINE oiml_tensor* oiml_set_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset) {
	return oiml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

OIML_INLINE oiml_tensor* oiml_set_1d(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t offset) {
	return oiml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

OIML_INLINE oiml_tensor* oiml_set_1d_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t offset) {
	return oiml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

OIML_INLINE oiml_tensor* oiml_set_2d(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1, size_t offset) {
	return oiml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

OIML_INLINE oiml_tensor* oiml_set_2d_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1, size_t offset) {
	return oiml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// oiml_cpy

OIML_INLINE oiml_tensor* oiml_cpy_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	OIML_ASSERT(oiml_nelements(a) == oiml_nelements(b));

	// make a view of the destination
	oiml_tensor* result = oiml_view_tensor(ctx, b);
	if (strlen(b->name) > 0) {
		oiml_format_name(result, "%s (copy of %s)", b->name, a->name);
	} else {
		oiml_format_name(result, "%s (copy)", a->name);
	}

	result->op	   = OIML_OP_CPY;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE oiml_tensor* oiml_cpy(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	return oiml_cpy_impl(ctx, a, b);
}

OIML_INLINE oiml_tensor* oiml_cast(oiml_context* ctx, oiml_tensor* a, oiml::oiml_representation_types type) {
	oiml_tensor* result = oiml_new_tensor(ctx, type, OIML_MAX_DIMS, a->ne);
	oiml_format_name(result, "%s (copy)", a->name);

	result->op	   = OIML_OP_CPY;
	result->src[0] = a;
	result->src[1] = result;

	return result;
}

// oiml_cont

OIML_INLINE oiml_tensor* oiml_cont_impl(oiml_context* ctx, oiml_tensor* a) {
	oiml_tensor* result = oiml_dup_tensor(ctx, a);
	oiml_format_name(result, "%s (cont)", a->name);

	result->op	   = OIML_OP_CONT;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_cont(oiml_context* ctx, oiml_tensor* a) {
	return oiml_cont_impl(ctx, a);
}

// make contiguous, with new shape
OIML_INLINE oiml_tensor* oiml_cont_1d(oiml_context* ctx, oiml_tensor* a, int64_t ne0) {
	return oiml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

OIML_INLINE oiml_tensor* oiml_cont_2d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1) {
	return oiml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

OIML_INLINE oiml_tensor* oiml_cont_3d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2) {
	return oiml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

OIML_INLINE oiml_tensor* oiml_cont_4d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
	OIML_ASSERT(oiml_nelements(a) == (ne0 * ne1 * ne2 * ne3));

	oiml_tensor* result = oiml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
	oiml_format_name(result, "%s (cont)", a->name);

	result->op	   = OIML_OP_CONT;
	result->src[0] = a;

	return result;
}

// oiml_reshape

OIML_INLINE oiml_tensor* oiml_reshape(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	OIML_ASSERT(oiml_is_contiguous(a));
	// as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
	OIML_ASSERT(oiml_nelements(a) == oiml_nelements(b));

	oiml_tensor* result = oiml_new_tensor_impl(ctx, a->type, OIML_MAX_DIMS, b->ne, a, 0);
	oiml_format_name(result, "%s (reshaped)", a->name);

	result->op	   = OIML_OP_RESHAPE;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_reshape_1d(oiml_context* ctx, oiml_tensor* a, int64_t ne0) {
	OIML_ASSERT(oiml_is_contiguous(a));
	OIML_ASSERT(oiml_nelements(a) == ne0);

	const int64_t ne[1] = { ne0 };
	oiml_tensor* result = oiml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
	oiml_format_name(result, "%s (reshaped)", a->name);

	result->op	   = OIML_OP_RESHAPE;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_reshape_2d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1) {
	OIML_ASSERT(oiml_is_contiguous(a));
	OIML_ASSERT(oiml_nelements(a) == ne0 * ne1);

	const int64_t ne[2] = { ne0, ne1 };
	oiml_tensor* result = oiml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
	oiml_format_name(result, "%s (reshaped)", a->name);

	result->op	   = OIML_OP_RESHAPE;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_reshape_3d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2) {
	OIML_ASSERT(oiml_is_contiguous(a));
	OIML_ASSERT(oiml_nelements(a) == ne0 * ne1 * ne2);

	const int64_t ne[3] = { ne0, ne1, ne2 };
	oiml_tensor* result = oiml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
	oiml_format_name(result, "%s (reshaped)", a->name);

	result->op	   = OIML_OP_RESHAPE;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_reshape_4d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
	OIML_ASSERT(oiml_is_contiguous(a));
	OIML_ASSERT(oiml_nelements(a) == ne0 * ne1 * ne2 * ne3);

	const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
	oiml_tensor* result = oiml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
	oiml_format_name(result, "%s (reshaped)", a->name);

	result->op	   = OIML_OP_RESHAPE;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_view_impl(oiml_context* ctx, oiml_tensor* a, int n_dims, const int64_t* ne, size_t offset) {
	oiml_tensor* result = oiml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
	oiml_format_name(result, "%s (view)", a->name);

	oiml_set_op_params(result, &offset, sizeof(offset));

	result->op	   = OIML_OP_VIEW;
	result->src[0] = a;

	return result;
}

// oiml_view_1d

OIML_INLINE oiml_tensor* oiml_view_1d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, size_t offset) {
	oiml_tensor* result = oiml_view_impl(ctx, a, 1, &ne0, offset);

	return result;
}

// oiml_view_2d

OIML_INLINE oiml_tensor* oiml_view_2d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset) {
	const int64_t ne[2] = { ne0, ne1 };

	oiml_tensor* result = oiml_view_impl(ctx, a, 2, ne, offset);
	result->nb[1]		= nb1;
	result->nb[2]		= result->nb[1] * ne1;
	result->nb[3]		= result->nb[2];

	return result;
}

// oiml_view_3d

OIML_INLINE oiml_tensor* oiml_view_3d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset) {
	const int64_t ne[3] = { ne0, ne1, ne2 };

	oiml_tensor* result = oiml_view_impl(ctx, a, 3, ne, offset);
	result->nb[1]		= nb1;
	result->nb[2]		= nb2;
	result->nb[3]		= result->nb[2] * ne2;

	return result;
}

// oiml_view_4d

OIML_INLINE oiml_tensor* oiml_view_4d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset) {
	const int64_t ne[4] = { ne0, ne1, ne2, ne3 };

	oiml_tensor* result = oiml_view_impl(ctx, a, 4, ne, offset);
	result->nb[1]		= nb1;
	result->nb[2]		= nb2;
	result->nb[3]		= nb3;

	return result;
}

// oiml_permute

OIML_INLINE oiml_tensor* oiml_permute(oiml_context* ctx, oiml_tensor* a, int axis0, int axis1, int axis2, int axis3) {
	OIML_ASSERT(axis0 >= 0 && axis0 < OIML_MAX_DIMS);
	OIML_ASSERT(axis1 >= 0 && axis1 < OIML_MAX_DIMS);
	OIML_ASSERT(axis2 >= 0 && axis2 < OIML_MAX_DIMS);
	OIML_ASSERT(axis3 >= 0 && axis3 < OIML_MAX_DIMS);

	OIML_ASSERT(axis0 != axis1);
	OIML_ASSERT(axis0 != axis2);
	OIML_ASSERT(axis0 != axis3);
	OIML_ASSERT(axis1 != axis2);
	OIML_ASSERT(axis1 != axis3);
	OIML_ASSERT(axis2 != axis3);

	oiml_tensor* result = oiml_view_tensor(ctx, a);
	oiml_format_name(result, "%s (permuted)", a->name);

	int ne[OIML_MAX_DIMS];
	int nb[OIML_MAX_DIMS];

	ne[axis0] = a->ne[0];
	ne[axis1] = a->ne[1];
	ne[axis2] = a->ne[2];
	ne[axis3] = a->ne[3];

	nb[axis0] = a->nb[0];
	nb[axis1] = a->nb[1];
	nb[axis2] = a->nb[2];
	nb[axis3] = a->nb[3];

	result->ne[0] = ne[0];
	result->ne[1] = ne[1];
	result->ne[2] = ne[2];
	result->ne[3] = ne[3];
	result->nb[0] = nb[0];
	result->nb[1] = nb[1];
	result->nb[2] = nb[2];
	result->nb[3] = nb[3];

	result->op	   = OIML_OP_PERMUTE;
	result->src[0] = a;

	int32_t params[] = { axis0, axis1, axis2, axis3 };
	oiml_set_op_params(result, params, sizeof(params));

	return result;
}

// oiml_transpose

OIML_INLINE oiml_tensor* oiml_transpose(oiml_context* ctx, oiml_tensor* a) {
	oiml_tensor* result = oiml_view_tensor(ctx, a);
	oiml_format_name(result, "%s (transposed)", a->name);

	result->ne[0] = a->ne[1];
	result->ne[1] = a->ne[0];

	result->nb[0] = a->nb[1];
	result->nb[1] = a->nb[0];

	result->op	   = OIML_OP_TRANSPOSE;
	result->src[0] = a;

	return result;
}

// oiml_get_rows

OIML_INLINE oiml_tensor* oiml_get_rows(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	OIML_ASSERT(a->ne[2] == b->ne[1]);
	OIML_ASSERT(b->ne[3] == 1);
	OIML_ASSERT(b->type == oiml::oiml_representation_types::int_32);

	// TODO: implement non F32 return
	oiml::oiml_representation_types type = oiml::oiml_representation_types::float_32;
	if (a->type == oiml::oiml_representation_types::int_32) {
		type = a->type;
	}
	oiml_tensor* result = oiml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

	result->op	   = OIML_OP_GET_ROWS;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

// oiml_get_rows_back

OIML_INLINE oiml_tensor* oiml_get_rows_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c) {
	OIML_ASSERT(oiml_is_matrix(a) && oiml_is_vector(b) && b->type == oiml::oiml_representation_types::int_32);
	OIML_ASSERT(oiml_is_matrix(c) && (a->ne[0] == c->ne[0]));

	// TODO: implement non F32 return
	//struct oiml_tensor * result = oiml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
	oiml_tensor* result = oiml_new_tensor_2d(ctx, oiml::oiml_representation_types::float_32, c->ne[0], c->ne[1]);

	result->op	   = OIML_OP_GET_ROWS_BACK;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

// oiml_diag

OIML_INLINE oiml_tensor* oiml_diag(oiml_context* ctx, oiml_tensor* a) {
	OIML_ASSERT(a->ne[1] == 1);

	const int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
	oiml_tensor* result = oiml_new_tensor(ctx, a->type, 4, ne);

	result->op	   = OIML_OP_DIAG;
	result->src[0] = a;

	return result;
}

// oiml_diag_mask_inf

OIML_INLINE oiml_tensor* oiml_diag_mask_inf_impl(oiml_context* ctx, oiml_tensor* a, int n_past, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	int32_t params[] = { n_past };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_DIAG_MASK_INF;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_diag_mask_inf(oiml_context* ctx, oiml_tensor* a, int n_past) {
	return oiml_diag_mask_inf_impl(ctx, a, n_past, false);
}

OIML_INLINE oiml_tensor* oiml_diag_mask_inf_inplace(oiml_context* ctx, oiml_tensor* a, int n_past) {
	return oiml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// oiml_diag_mask_zero

OIML_INLINE oiml_tensor* oiml_diag_mask_zero_impl(oiml_context* ctx, oiml_tensor* a, int n_past, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	int32_t params[] = { n_past };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_DIAG_MASK_ZERO;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_diag_mask_zero(oiml_context* ctx, oiml_tensor* a, int n_past) {
	return oiml_diag_mask_zero_impl(ctx, a, n_past, false);
}

OIML_INLINE oiml_tensor* oiml_diag_mask_zero_inplace(oiml_context* ctx, oiml_tensor* a, int n_past) {
	return oiml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// oiml_soft_max

OIML_INLINE oiml_tensor* oiml_soft_max_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* mask, float scale, float max_bias, bool inplace) {
	OIML_ASSERT(oiml_is_contiguous(a));

	if (mask) {
		OIML_ASSERT(mask->type == oiml::oiml_representation_types::float_16 || mask->type == oiml::oiml_representation_types::float_32);
		OIML_ASSERT(oiml_is_contiguous(mask));
		OIML_ASSERT(oiml_is_matrix(mask));
		OIML_ASSERT(mask->ne[0] == a->ne[0]);
		OIML_ASSERT(mask->ne[1] >= a->ne[1]);
	}

	if (max_bias > 0.0f) {
		OIML_ASSERT(mask);
	}

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	float params[] = { scale, max_bias };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_SOFT_MAX;
	result->src[0] = a;
	result->src[1] = mask;

	return result;
}

OIML_INLINE oiml_tensor* oiml_soft_max(oiml_context* ctx, oiml_tensor* a) {
	return oiml_soft_max_impl(ctx, a, NULL, 1.0f, 0.0f, false);
}

OIML_INLINE oiml_tensor* oiml_soft_max_inplace(oiml_context* ctx, oiml_tensor* a) {
	return oiml_soft_max_impl(ctx, a, NULL, 1.0f, 0.0f, true);
}

OIML_INLINE oiml_tensor* oiml_soft_max_ext(oiml_context* ctx, oiml_tensor* a, oiml_tensor* mask, float scale, float max_bias) {
	return oiml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

// oiml_soft_max_ext_back

OIML_INLINE oiml_tensor* oiml_soft_max_ext_back_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, float scale, float max_bias, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	result->op	   = OIML_OP_SOFT_MAX_BACK;
	result->src[0] = a;
	result->src[1] = b;

	memcpy(( float* )result->op_params + 0, &scale, sizeof(float));
	memcpy(( float* )result->op_params + 1, &max_bias, sizeof(float));

	return result;
}

OIML_INLINE oiml_tensor* oiml_soft_max_ext_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, float scale, float max_bias) {
	return oiml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

OIML_INLINE oiml_tensor* oiml_soft_max_ext_back_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, float scale, float max_bias) {
	return oiml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, true);
}

// oiml_rope

OIML_INLINE oiml_tensor* oiml_rope_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale,
	float ext_factor, float attn_factor, float beta_fast, float beta_slow, bool inplace) {
	OIML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

	OIML_ASSERT(oiml_is_vector(b));
	OIML_ASSERT(b->type == oiml::oiml_representation_types::int_32);
	OIML_ASSERT(a->ne[2] == b->ne[0]);

	if (c) {
		OIML_ASSERT(c->type == oiml::oiml_representation_types::float_32);
		OIML_ASSERT(c->ne[0] >= n_dims / 2);
	}

	int sections[4] = { 0, 0, 0, 0 };

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	int32_t params[15] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
	memcpy(params + 5, &freq_base, sizeof(float));
	memcpy(params + 6, &freq_scale, sizeof(float));
	memcpy(params + 7, &ext_factor, sizeof(float));
	memcpy(params + 8, &attn_factor, sizeof(float));
	memcpy(params + 9, &beta_fast, sizeof(float));
	memcpy(params + 10, &beta_slow, sizeof(float));
	memcpy(params + 11, &sections, sizeof(int) * 4);
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_ROPE;
	result->src[0] = a;
	result->src[1] = b;
	result->src[2] = c;

	return result;
}

OIML_INLINE oiml_tensor* oiml_rope(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int n_dims, int mode) {
	return oiml_rope_impl(ctx, a, b, NULL, n_dims, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false);
}

OIML_INLINE oiml_tensor* oiml_rope_multi(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, int n_dims, int sections[4], int mode, int n_ctx_orig, float freq_base,
	float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
	// Multimodal Rotary Position Embedding
	OIML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

	OIML_ASSERT(oiml_is_vector(b));
	OIML_ASSERT(b->type == oiml::oiml_representation_types::int_32);
	OIML_ASSERT(a->ne[2] * 4 == b->ne[0]);// mrope expecting 4 position ids per token

	if (c) {
		OIML_ASSERT(c->type == oiml::oiml_representation_types::float_32);
		OIML_ASSERT(c->ne[0] >= n_dims / 2);
	}

	oiml_tensor* result = oiml_dup_tensor(ctx, a);

	int32_t params[11 + 4] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
	memcpy(params + 5, &freq_base, sizeof(float));
	memcpy(params + 6, &freq_scale, sizeof(float));
	memcpy(params + 7, &ext_factor, sizeof(float));
	memcpy(params + 8, &attn_factor, sizeof(float));
	memcpy(params + 9, &beta_fast, sizeof(float));
	memcpy(params + 10, &beta_slow, sizeof(float));
	memcpy(&params[11], sections, sizeof(int) * 4);
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_ROPE;
	result->src[0] = a;
	result->src[1] = b;
	result->src[2] = c;

	return result;
}

OIML_INLINE oiml_tensor* oiml_rope_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int n_dims, int mode) {
	return oiml_rope_impl(ctx, a, b, NULL, n_dims, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true);
}

OIML_INLINE oiml_tensor* oiml_rope_ext(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale,
	float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
	return oiml_rope_impl(ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, false);
}

OIML_INLINE oiml_tensor* oiml_rope_ext_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, int n_dims, int mode, int n_ctx_orig, float freq_base,
	float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
	return oiml_rope_impl(ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, true);
}

OIML_INLINE oiml_tensor* oiml_rope_custom(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale,
	float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
	return oiml_rope_impl(ctx, a, b, NULL, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, false);
}

OIML_INLINE oiml_tensor* oiml_rope_custom_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale,
	float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
	return oiml_rope_impl(ctx, a, b, NULL, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, true);
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
OIML_INLINE float oiml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
	return n_dims * logf(n_ctx_orig / (n_rot * 2 * ( float )M_PI)) / (2 * logf(base));
}

OIML_INLINE void oiml_rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
	// start and end correction dims
	float start = floorf(oiml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
	float end	= ceilf(oiml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
	dims[0]		= MAX(0, start);
	dims[1]		= MIN(n_dims - 1, end);
}

// oiml_rope_back

OIML_INLINE oiml_tensor* oiml_rope_ext_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, int n_dims, int mode, int n_ctx_orig, float freq_base,
	float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
	oiml_tensor* result = oiml_rope_ext(ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
	result->op			= OIML_OP_ROPE_BACK;
	return result;
}

OIML_INLINE oiml_tensor* oiml_rope_multi_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, int n_dims, int sections[4], int mode, int n_ctx_orig,
	float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
	oiml_tensor* result = oiml_rope_multi(ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
	result->op			= OIML_OP_ROPE_BACK;
	return result;
}
// oiml_clamp

OIML_INLINE oiml_tensor* oiml_clamp(oiml_context* ctx, oiml_tensor* a, float min, float max) {
	// TODO: when implement backward, fix this:
	oiml_tensor* result = oiml_view_tensor(ctx, a);

	float params[] = { min, max };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_CLAMP;
	result->src[0] = a;

	return result;
}

OIML_INLINE int64_t oiml_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
	return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
OIML_INLINE oiml_tensor* oiml_im2col(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int s0, int s1, int p0, int p1, int d0, int d1, bool is_2D, oiml::oiml_representation_types dst_type) {
	if (is_2D) {
		OIML_ASSERT(a->ne[2] == b->ne[2]);
	} else {
		//OIML_ASSERT(b->ne[1] % a->ne[1] == 0);
		OIML_ASSERT(b->ne[1] == a->ne[1]);
		OIML_ASSERT(b->ne[3] == 1);
	}

	const int64_t OH = is_2D ? oiml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
	const int64_t OW = oiml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

	OIML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
	OIML_ASSERT((OW > 0) && "b too small compared to a");

	const int64_t ne[4] = {
		is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
		OW,
		is_2D ? OH : b->ne[2],
		is_2D ? b->ne[3] : 1,
	};

	oiml_tensor* result = oiml_new_tensor(ctx, dst_type, 4, ne);
	int32_t params[]	= { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_IM2COL;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE oiml_tensor* oiml_im2col_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int64_t* ne, int s0, int s1, int p0, int p1, int d0, int d1, bool is_2D) {
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, ne);
	int32_t params[]	= { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_IM2COL_BACK;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

// oiml_conv_1d

OIML_INLINE oiml_tensor* oiml_conv_1d(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int s0, int p0, int d0) {
	oiml_tensor* im2col = oiml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, oiml::oiml_representation_types::float_16);// [N, OL, IC * K]

	oiml_tensor* result = oiml_mul_mat(ctx, oiml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])),// [N, OL, IC * K] => [N*OL, IC * K]
		oiml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));// [OC，IC, K] => [OC, IC * K]

	result = oiml_reshape_3d(ctx, result, im2col->ne[1], a->ne[2], im2col->ne[2]);// [N, OC, OL]

	return result;
}

// oiml_conv_1d_ph

OIML_INLINE oiml_tensor* oiml_conv_1d_ph(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int s, int d) {
	return oiml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// oiml_conv_1d_dw

OIML_INLINE oiml_tensor* oiml_conv_1d_dw(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int s0, int p0, int d0) {
	oiml_tensor* new_a = oiml_reshape_4d(ctx, a, a->ne[0], 1, a->ne[1], a->ne[2]);
	oiml_tensor* new_b = oiml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

	oiml_tensor* im2col = oiml_im2col(ctx, new_a, new_b, s0, 0, p0, 0, d0, 0, false, oiml::oiml_representation_types::float_16);

	oiml_tensor* result = oiml_mul_mat(ctx, im2col, a);

	result = oiml_reshape_3d(ctx, result, b->ne[0], b->ne[1], 1);

	return result;
}

// oiml_conv_1d_dw_ph

OIML_INLINE oiml_tensor* oiml_conv_1d_dw_ph(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int s0, int d0) {
	return oiml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// oiml_conv_transpose_1d

OIML_INLINE int64_t oiml_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
	return (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

OIML_INLINE oiml_tensor* oiml_conv_transpose_1d(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int s0, int p0, int d0) {
	OIML_ASSERT(oiml_is_matrix(b));
	OIML_ASSERT(a->ne[2] == b->ne[1]);
	OIML_ASSERT(a->ne[3] == 1);

	OIML_ASSERT(p0 == 0);
	OIML_ASSERT(d0 == 1);

	const int64_t ne[4] = {
		oiml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
		a->ne[1],
		b->ne[2],
		1,
	};
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, ne);

	int32_t params[] = { s0, p0, d0 };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_CONV_TRANSPOSE_1D;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

// oiml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
OIML_INLINE oiml_tensor* oiml_conv_2d(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int s0, int s1, int p0, int p1, int d0, int d1) {
	oiml_tensor* im2col = oiml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type);// [N, OH, OW, IC * KH * KW]

	oiml_tensor* result =
		oiml_mul_mat(ctx, oiml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),// [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
			oiml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]), a->ne[3]));// [OC，IC, KH, KW] => [OC, IC * KH * KW]

	result = oiml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]);// [OC, N, OH, OW]
	result = oiml_cont(ctx, oiml_permute(ctx, result, 0, 1, 3, 2));// [N, OC, OH, OW]


	return result;
}

// oiml_conv_2d_sk_p0

OIML_INLINE oiml_tensor* oiml_conv_2d_sk_p0(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	return oiml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// oiml_conv_2d_s1_ph

OIML_INLINE oiml_tensor* oiml_conv_2d_s1_ph(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	return oiml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// oiml_conv_2d_dw

OIML_INLINE oiml_tensor* oiml_conv_2d_dw(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int s0, int s1, int p0, int p1, int d0, int d1) {
	oiml_tensor* new_a = oiml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
	oiml_tensor* im2col =
		oiml_im2col(ctx, new_a, oiml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]), s0, s1, p0, p1, d0, d1, true, oiml::oiml_representation_types::float_16);// [N * IC, OH, OW, KH * KW]
	oiml_tensor* new_b = oiml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]);// [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

	new_a				= oiml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2], new_a->ne[3], 1);// [OC，1, KH, KW] => [1, OC, 1, KH * KW]
	oiml_tensor* result = oiml_mul_mat(ctx, new_a, new_b);
	result				= oiml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]);// [N, OC, OH, OW]

	return result;
}

// oiml_conv_transpose_2d_p0

OIML_INLINE int64_t oiml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, int s, int p) {
	return (ins - 1) * s - 2 * p + ks;
}

OIML_INLINE oiml_tensor* oiml_conv_transpose_2d_p0(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int stride) {
	OIML_ASSERT(a->ne[3] == b->ne[2]);

	const int64_t ne[4] = {
		oiml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
		oiml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
		a->ne[2],
		b->ne[3],
	};

	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, ne);

	oiml_set_op_params_i32(result, 0, stride);

	result->op	   = OIML_OP_CONV_TRANSPOSE_2D;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

// oiml_pool_*

OIML_INLINE int64_t oiml_calc_pool_output_size(int64_t ins, int ks, int s, float p) {
	return (ins + 2 * p - ks) / s + 1;
}

// oiml_pool_1d

OIML_INLINE oiml_tensor* oiml_pool_1d(oiml_context* ctx, oiml_tensor* a, enum oiml_op_pool op, int k0, int s0, int p0) {
	const int64_t ne[4] = {
		oiml_calc_pool_output_size(a->ne[0], k0, s0, p0),
		a->ne[1],
		a->ne[2],
		a->ne[3],
	};
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, ne);

	int32_t params[] = { op, k0, s0, p0 };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_POOL_1D;
	result->src[0] = a;

	return result;
}

// oiml_pool_2d

OIML_INLINE oiml_tensor* oiml_pool_2d(oiml_context* ctx, oiml_tensor* a, enum oiml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1) {
	oiml_tensor* result;
	const int64_t ne[4] = {
		oiml_calc_pool_output_size(a->ne[0], k0, s0, p0),
		oiml_calc_pool_output_size(a->ne[1], k1, s1, p1),
		a->ne[2],
		a->ne[3],
	};
	result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, ne);

	int32_t params[] = { op, k0, k1, s0, s1, static_cast<int32_t>(p0), static_cast<int32_t>(p1) };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_POOL_2D;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_pool_2d_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* af, enum oiml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1) {
	oiml_tensor* result;
	result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, af->ne);

	int32_t params[] = { op, k0, k1, s0, s1, static_cast<int32_t>(p0), static_cast<int32_t>(p1) };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_POOL_2D_BACK;
	result->src[0] = a;
	result->src[1] = af;

	return result;
}

// oiml_upscale

OIML_INLINE oiml_tensor* oiml_upscale_impl(oiml_context* ctx, oiml_tensor* a, int ne0, int ne1, int ne2, int ne3) {
	OIML_ASSERT(a->ne[0] <= ne0);
	OIML_ASSERT(a->ne[1] <= ne1);
	OIML_ASSERT(a->ne[2] <= ne2);
	OIML_ASSERT(a->ne[3] <= ne3);

	oiml_tensor* result = oiml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

	result->op	   = OIML_OP_UPSCALE;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_upscale(oiml_context* ctx, oiml_tensor* a, int scale_factor) {
	return oiml_upscale_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3]);
}

OIML_INLINE oiml_tensor* oiml_upscale_ext(oiml_context* ctx, oiml_tensor* a, int ne0, int ne1, int ne2, int ne3) {
	return oiml_upscale_impl(ctx, a, ne0, ne1, ne2, ne3);
}

// oiml_pad

OIML_INLINE oiml_tensor* oiml_pad(oiml_context* ctx, oiml_tensor* a, int p0, int p1, int p2, int p3) {
	oiml_tensor* result = oiml_new_tensor_4d(ctx, a->type, a->ne[0] + p0, a->ne[1] + p1, a->ne[2] + p2, a->ne[3] + p3);

	result->op	   = OIML_OP_PAD;
	result->src[0] = a;

	return result;
}

// oiml_pad_reflect_1d

OIML_INLINE oiml_tensor* oiml_pad_reflect_1d(oiml_context* ctx, oiml_tensor* a, int p0, int p1) {
	OIML_ASSERT(p0 >= 0);
	OIML_ASSERT(p1 >= 0);

	OIML_ASSERT(p0 < a->ne[0]);// padding length on each size must be less than the
	OIML_ASSERT(p1 < a->ne[0]);// existing length of the dimension being padded

	OIML_ASSERT(oiml_is_contiguous(a));
	OIML_ASSERT(a->type == oiml::oiml_representation_types::float_32);

	oiml_tensor* result = oiml_new_tensor_4d(ctx, a->type, a->ne[0] + p0 + p1, a->ne[1], a->ne[2], a->ne[3]);

	int32_t params[] = { p0, p1 };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_PAD_REFLECT_1D;
	result->src[0] = a;

	return result;
}

// oiml_arange

OIML_INLINE oiml_tensor* oiml_arange(oiml_context* ctx, float start, float stop, float step) {
	OIML_ASSERT(stop > start);

	const int64_t steps = ( int64_t )ceilf((stop - start) / step);

	oiml_tensor* result = oiml_new_tensor_1d(ctx, oiml::oiml_representation_types::float_32, steps);

	oiml_set_op_params_f32(result, 0, start);
	oiml_set_op_params_f32(result, 1, stop);
	oiml_set_op_params_f32(result, 2, step);

	result->op = OIML_OP_ARANGE;

	return result;
}

// oiml_timestep_embedding

OIML_INLINE oiml_tensor* oiml_timestep_embedding(oiml_context* ctx, oiml_tensor* timesteps, int dim, int max_period) {
	int actual_dim = dim;
	if (dim % 2 != 0) {
		actual_dim = dim + 1;
	}

	oiml_tensor* result = oiml_new_tensor_2d(ctx, oiml::oiml_representation_types::float_32, actual_dim, timesteps->ne[0]);

	oiml_set_op_params_i32(result, 0, dim);
	oiml_set_op_params_i32(result, 1, max_period);

	result->op	   = OIML_OP_TIMESTEP_EMBEDDING;
	result->src[0] = timesteps;

	return result;
}

// oiml_argsort

OIML_INLINE oiml_tensor* oiml_argsort(oiml_context* ctx, oiml_tensor* a, enum oiml_sort_order order) {
	OIML_ASSERT(a->ne[0] <= INT32_MAX);
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::int_32, OIML_MAX_DIMS, a->ne);

	oiml_set_op_params_i32(result, 0, ( int32_t )order);

	result->op	   = OIML_OP_ARGSORT;
	result->src[0] = a;

	return result;
}

// oiml_top_k

OIML_INLINE oiml_tensor* oiml_top_k(oiml_context* ctx, oiml_tensor* a, int k) {
	OIML_ASSERT(a->ne[0] >= k);

	oiml_tensor* result = oiml_argsort(ctx, a, OIML_SORT_ORDER_DESC);

	result = oiml_view_4d(ctx, result, k, result->ne[1], result->ne[2], result->ne[3], result->nb[1], result->nb[2], result->nb[3], 0);

	return result;
}

// oiml_flash_attn_ext

OIML_INLINE oiml_tensor* oiml_flash_attn_ext(oiml_context* ctx, oiml_tensor* q, oiml_tensor* k, oiml_tensor* v, oiml_tensor* mask, float scale, float max_bias,
	float logit_softcap) {
	OIML_ASSERT(oiml_can_mul_mat(k, q));
	// TODO: check if vT can be multiplied by (k*qT)

	if (mask) {
		OIML_ASSERT(oiml_is_contiguous(mask));
		OIML_ASSERT(mask->ne[2] == 1);
		OIML_ASSERT(mask->ne[3] == 1);
		OIML_ASSERT(
			mask->ne[1] >= OIML_PAD(q->ne[1], OIML_KQ_MASK_PAD) && "the Flash-Attention kernel requires the mask to be padded to OIML_KQ_MASK_PAD and at least n_queries big");
		//OIML_ASSERT(oiml_can_repeat_rows(mask, qk));
	}

	if (max_bias > 0.0f) {
		OIML_ASSERT(mask);
	}

	// permute(0, 2, 1, 3)
	int64_t ne[4]		= { q->ne[0], q->ne[2], q->ne[1], q->ne[3] };
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, ne);

	float params[] = { scale, max_bias, logit_softcap };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_FLASH_ATTN_EXT;
	result->src[0] = q;
	result->src[1] = k;
	result->src[2] = v;
	result->src[3] = mask;

	return result;
}

OIML_INLINE void oiml_flash_attn_ext_set_prec(oiml_tensor* a, enum oiml_prec prec) {
	OIML_ASSERT(a->op == OIML_OP_FLASH_ATTN_EXT);

	const int32_t prec_i32 = ( int32_t )prec;

	oiml_set_op_params_i32(a, 3, prec_i32);// scale is on first pos, max_bias on second
}

OIML_INLINE enum oiml_prec oiml_flash_attn_ext_get_prec(const oiml_tensor* a) {
	OIML_ASSERT(a->op == OIML_OP_FLASH_ATTN_EXT);

	const int32_t prec_i32 = oiml_get_op_params_i32(a, 3);

	return ( enum oiml_prec )prec_i32;
}

// oiml_flash_attn_back

OIML_INLINE oiml_tensor* oiml_flash_attn_back(oiml_context* ctx, oiml_tensor* q, oiml_tensor* k, oiml_tensor* v, oiml_tensor* d, bool masked) {
	OIML_ABORT("TODO: adapt to oiml_flash_attn_ext() changes");

	OIML_ASSERT(oiml_can_mul_mat(k, q));
	// TODO: check if vT can be multiplied by (k*qT)

	// d shape [D,N,ne2,ne3]
	// q shape [D,N,ne2,ne3]
	// k shape [D,M,kvne2,ne3]
	// v shape [M,D,kvne2,ne3]

	const int64_t D		= q->ne[0];
	const int64_t N		= q->ne[1];
	const int64_t M		= k->ne[1];
	const int64_t ne2	= q->ne[2];
	const int64_t ne3	= q->ne[3];
	const int64_t kvne2 = k->ne[2];

	OIML_ASSERT(k->ne[0] == D);
	OIML_ASSERT(v->ne[0] == M);
	OIML_ASSERT(v->ne[1] == D);
	OIML_ASSERT(d->ne[0] == D);
	OIML_ASSERT(d->ne[1] == N);
	OIML_ASSERT(k->ne[2] == kvne2);
	OIML_ASSERT(k->ne[3] == ne3);
	OIML_ASSERT(v->ne[2] == kvne2);
	OIML_ASSERT(v->ne[3] == ne3);
	OIML_ASSERT(d->ne[2] == ne2);
	OIML_ASSERT(d->ne[3] == ne3);

	OIML_ASSERT(ne2 % kvne2 == 0);

	// store gradients of q, k and v as continuous tensors concatenated in result.
	// note: v and gradv are actually transposed, i.e. v->ne[0] != D.
	const int64_t elem_q = oiml_nelements(q);
	const int64_t elem_k = oiml_nelements(k);
	const int64_t elem_v = oiml_nelements(v);

	oiml::oiml_representation_types result_type = oiml::oiml_representation_types::float_32;
	OIML_ASSERT(oiml_blck_size(result_type) == 1);
	const size_t tsize = oiml_type_size(result_type);

	const size_t offs_q = 0;
	const size_t offs_k = offs_q + OIML_PAD(elem_q * tsize, OIML_MEM_ALIGN);
	const size_t offs_v = offs_k + OIML_PAD(elem_k * tsize, OIML_MEM_ALIGN);
	const size_t end	= offs_v + OIML_PAD(elem_v * tsize, OIML_MEM_ALIGN);

	const size_t nelements = (end + tsize - 1) / tsize;

	oiml_tensor* result = oiml_new_tensor_1d(ctx, oiml::oiml_representation_types::float_32, nelements);

	int32_t masked_i = masked ? 1 : 0;
	oiml_set_op_params(result, &masked_i, sizeof(masked_i));

	result->op	   = OIML_OP_FLASH_ATTN_BACK;
	result->src[0] = q;
	result->src[1] = k;
	result->src[2] = v;
	result->src[3] = d;

	return result;
}

// oiml_ssm_conv

OIML_INLINE oiml_tensor* oiml_ssm_conv(oiml_context* ctx, oiml_tensor* sx, oiml_tensor* c) {
	OIML_ASSERT(oiml_is_3d(sx));
	OIML_ASSERT(oiml_is_matrix(c));

	const int64_t d_conv  = c->ne[0];
	const int64_t d_inner = c->ne[1];
	const int64_t n_t	  = sx->ne[0] - d_conv + 1;// tokens per sequence
	const int64_t n_s	  = sx->ne[2];

	// TODO: maybe support other strides than 1?
	// FIXME: this is always true?
	OIML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
	OIML_ASSERT(sx->ne[1] == d_inner);
	OIML_ASSERT(n_t >= 0);

	oiml_tensor* result = oiml_new_tensor_3d(ctx, oiml::oiml_representation_types::float_32, d_inner, n_t, n_s);

	result->op	   = OIML_OP_SSM_CONV;
	result->src[0] = sx;
	result->src[1] = c;

	return result;
}

// oiml_ssm_scan

OIML_INLINE oiml_tensor* oiml_ssm_scan(oiml_context* ctx, oiml_tensor* s, oiml_tensor* x, oiml_tensor* dt, oiml_tensor* A, oiml_tensor* B, oiml_tensor* C) {
	OIML_ASSERT(oiml_is_contiguous(s));
	OIML_ASSERT(oiml_is_contiguous(x));
	OIML_ASSERT(oiml_is_contiguous(dt));
	OIML_ASSERT(oiml_is_contiguous(A));
	OIML_ASSERT(oiml_is_matrix(A));
	OIML_ASSERT(oiml_is_3d(B));
	OIML_ASSERT(oiml_is_3d(s));
	OIML_ASSERT(B->nb[0] == oiml_type_size(B->type));
	OIML_ASSERT(C->nb[0] == oiml_type_size(C->type));
	OIML_ASSERT(oiml_are_same_shape(x, dt));
	OIML_ASSERT(oiml_are_same_shape(B, C));

	{
		const int64_t d_state	   = s->ne[0];
		const int64_t d_inner	   = s->ne[1];
		const int64_t n_seq_tokens = x->ne[1];
		const int64_t n_seqs	   = x->ne[2];

		OIML_ASSERT(s->ne[2] == n_seqs);
		OIML_ASSERT(x->ne[0] == d_inner);
		OIML_ASSERT(A->ne[0] == d_state);
		OIML_ASSERT(A->ne[1] == d_inner);
		OIML_ASSERT(B->ne[0] == d_state);
		OIML_ASSERT(B->ne[1] == n_seq_tokens);
		OIML_ASSERT(B->ne[2] == n_seqs);
	}

	// concatenated y + ssm_states
	oiml_tensor* result = oiml_new_tensor_1d(ctx, oiml::oiml_representation_types::float_32, oiml_nelements(x) + oiml_nelements(s));

	result->op	   = OIML_OP_SSM_SCAN;
	result->src[0] = s;
	result->src[1] = x;
	result->src[2] = dt;
	result->src[3] = A;
	result->src[4] = B;
	result->src[5] = C;

	return result;
}

// oiml_win_part

OIML_INLINE oiml_tensor* oiml_win_part(oiml_context* ctx, oiml_tensor* a, int w) {
	OIML_ASSERT(a->ne[3] == 1);
	OIML_ASSERT(a->type == oiml::oiml_representation_types::float_32);

	// padding
	const int px = (w - a->ne[1] % w) % w;
	const int py = (w - a->ne[2] % w) % w;

	const int npx = (px + a->ne[1]) / w;
	const int npy = (py + a->ne[2]) / w;
	const int np  = npx * npy;

	const int64_t ne[4] = {
		a->ne[0],
		w,
		w,
		np,
	};
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, ne);

	int32_t params[] = { npx, npy, w };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_WIN_PART;
	result->src[0] = a;

	return result;
}

// oiml_win_unpart

OIML_INLINE oiml_tensor* oiml_win_unpart(oiml_context* ctx, oiml_tensor* a, int w0, int h0, int w) {
	OIML_ASSERT(a->type == oiml::oiml_representation_types::float_32);

	const int64_t ne[4] = {
		a->ne[0],
		w0,
		h0,
		1,
	};
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 3, ne);

	int32_t params[] = { w };
	oiml_set_op_params(result, params, sizeof(params));

	result->op	   = OIML_OP_WIN_UNPART;
	result->src[0] = a;

	return result;
}

// oiml_get_rel_pos

OIML_INLINE oiml_tensor* oiml_get_rel_pos(oiml_context* ctx, oiml_tensor* a, int qh, int kh) {
	OIML_ASSERT(qh == kh);
	OIML_ASSERT(2 * MAX(qh, kh) - 1 == a->ne[1]);

	const int64_t ne[4] = {
		a->ne[0],
		kh,
		qh,
		1,
	};
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_16, 3, ne);

	result->op	   = OIML_OP_GET_REL_POS;
	result->src[0] = a;

	return result;
}

// oiml_add_rel_pos

OIML_INLINE oiml_tensor* oiml_add_rel_pos_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* pw, oiml_tensor* ph, bool inplace) {
	OIML_ASSERT(oiml_are_same_shape(pw, ph));
	OIML_ASSERT(oiml_is_contiguous(a));
	OIML_ASSERT(oiml_is_contiguous(pw));
	OIML_ASSERT(oiml_is_contiguous(ph));
	OIML_ASSERT(ph->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(pw->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(pw->ne[3] == a->ne[2]);
	OIML_ASSERT(pw->ne[0] * pw->ne[0] == a->ne[0]);
	OIML_ASSERT(pw->ne[1] * pw->ne[2] == a->ne[1]);

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);
	oiml_set_op_params_i32(result, 0, inplace ? 1 : 0);

	result->op	   = OIML_OP_ADD_REL_POS;
	result->src[0] = a;
	result->src[1] = pw;
	result->src[2] = ph;

	return result;
}

OIML_INLINE oiml_tensor* oiml_add_rel_pos(oiml_context* ctx, oiml_tensor* a, oiml_tensor* pw, oiml_tensor* ph) {
	return oiml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

OIML_INLINE oiml_tensor* oiml_add_rel_pos_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* pw, oiml_tensor* ph) {
	return oiml_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// oiml_rwkv_wkv6

OIML_INLINE oiml_tensor* oiml_rwkv_wkv6(oiml_context* ctx, oiml_tensor* k, oiml_tensor* v, oiml_tensor* r, oiml_tensor* tf, oiml_tensor* td, oiml_tensor* state) {
	OIML_ASSERT(oiml_is_contiguous(k));
	OIML_ASSERT(oiml_is_contiguous(v));
	OIML_ASSERT(oiml_is_contiguous(r));
	OIML_ASSERT(oiml_is_contiguous(tf));
	OIML_ASSERT(oiml_is_contiguous(td));
	OIML_ASSERT(oiml_is_contiguous(state));

	const int64_t S		   = k->ne[0];
	const int64_t H		   = k->ne[1];
	const int64_t n_tokens = k->ne[2];
	const int64_t n_seqs   = state->ne[1];
	{
		OIML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
		OIML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens);
		OIML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens);
		OIML_ASSERT(oiml_nelements(state) == S * S * H * n_seqs);
	}

	// concat output and new_state
	const int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, ne);

	result->op	   = OIML_OP_RWKV_WKV6;
	result->src[0] = k;
	result->src[1] = v;
	result->src[2] = r;
	result->src[3] = tf;
	result->src[4] = td;
	result->src[5] = state;

	return result;
}

// oiml_gated_linear_attn

OIML_INLINE oiml_tensor* oiml_gated_linear_attn(oiml_context* ctx, oiml_tensor* k, oiml_tensor* v, oiml_tensor* q, oiml_tensor* g, oiml_tensor* state, float scale) {
	OIML_ASSERT(oiml_is_contiguous(k));
	OIML_ASSERT(oiml_is_contiguous(v));
	OIML_ASSERT(oiml_is_contiguous(q));
	OIML_ASSERT(oiml_is_contiguous(g));
	OIML_ASSERT(oiml_is_contiguous(state));

	const int64_t S		   = k->ne[0];
	const int64_t H		   = k->ne[1];
	const int64_t n_tokens = k->ne[2];
	const int64_t n_seqs   = state->ne[1];
	{
		OIML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
		OIML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
		OIML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
		OIML_ASSERT(oiml_nelements(state) == S * S * H * n_seqs);
	}

	// concat output and new_state
	const int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
	oiml_tensor* result = oiml_new_tensor(ctx, oiml::oiml_representation_types::float_32, 4, ne);

	oiml_set_op_params_f32(result, 0, scale);

	result->op	   = OIML_OP_GATED_LINEAR_ATTN;
	result->src[0] = k;
	result->src[1] = v;
	result->src[2] = q;
	result->src[3] = g;
	result->src[4] = state;

	return result;
}

// oiml_unary

OIML_INLINE oiml_tensor* oiml_unary_impl(oiml_context* ctx, oiml_tensor* a, enum oiml_unary_op op, bool inplace) {
	OIML_ASSERT(oiml_is_contiguous_1(a));

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	oiml_set_op_params_i32(result, 0, ( int32_t )op);

	result->op	   = OIML_OP_UNARY;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_unary(oiml_context* ctx, oiml_tensor* a, enum oiml_unary_op op) {
	return oiml_unary_impl(ctx, a, op, false);
}

OIML_INLINE oiml_tensor* oiml_unary_inplace(oiml_context* ctx, oiml_tensor* a, enum oiml_unary_op op) {
	return oiml_unary_impl(ctx, a, op, true);
}

// oiml_map_unary

OIML_INLINE oiml_tensor* oiml_map_unary_impl_f32(oiml_context* ctx, oiml_tensor* a, const oiml_unary_op_f32_t fun, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	oiml_set_op_params(result, ( const void* )&fun, sizeof(fun));

	result->op	   = OIML_OP_MAP_UNARY;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_map_unary_f32(oiml_context* ctx, oiml_tensor* a, const oiml_unary_op_f32_t fun) {
	return oiml_map_unary_impl_f32(ctx, a, fun, false);
}

OIML_INLINE oiml_tensor* oiml_map_unary_inplace_f32(oiml_context* ctx, oiml_tensor* a, const oiml_unary_op_f32_t fun) {
	return oiml_map_unary_impl_f32(ctx, a, fun, true);
}

// oiml_map_binary

OIML_INLINE oiml_tensor* oiml_map_binary_impl_f32(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, const oiml_binary_op_f32_t fun, bool inplace) {
	OIML_ASSERT(oiml_are_same_shape(a, b));

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	oiml_set_op_params(result, ( const void* )&fun, sizeof(fun));

	result->op	   = OIML_OP_MAP_BINARY;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE oiml_tensor* oiml_map_binary_f32(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, const oiml_binary_op_f32_t fun) {
	return oiml_map_binary_impl_f32(ctx, a, b, fun, false);
}

OIML_INLINE oiml_tensor* oiml_map_binary_inplace_f32(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, const oiml_binary_op_f32_t fun) {
	return oiml_map_binary_impl_f32(ctx, a, b, fun, true);
}

// oiml_map_custom1_f32

OIML_INLINE oiml_tensor* oiml_map_custom1_impl_f32(oiml_context* ctx, oiml_tensor* a, const oiml_custom1_op_f32_t fun, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	oiml_set_op_params(result, ( const void* )&fun, sizeof(fun));

	result->op	   = OIML_OP_MAP_CUSTOM1_F32;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_map_custom1_f32(oiml_context* ctx, oiml_tensor* a, const oiml_custom1_op_f32_t fun) {
	return oiml_map_custom1_impl_f32(ctx, a, fun, false);
}

OIML_INLINE oiml_tensor* oiml_map_custom1_inplace_f32(oiml_context* ctx, oiml_tensor* a, const oiml_custom1_op_f32_t fun) {
	return oiml_map_custom1_impl_f32(ctx, a, fun, true);
}

// oiml_map_custom2_f32

OIML_INLINE oiml_tensor* oiml_map_custom2_impl_f32(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, const oiml_custom2_op_f32_t fun, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	oiml_set_op_params(result, ( const void* )&fun, sizeof(fun));

	result->op	   = OIML_OP_MAP_CUSTOM2_F32;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE oiml_tensor* oiml_map_custom2_f32(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, const oiml_custom2_op_f32_t fun) {
	return oiml_map_custom2_impl_f32(ctx, a, b, fun, false);
}

OIML_INLINE oiml_tensor* oiml_map_custom2_inplace_f32(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, const oiml_custom2_op_f32_t fun) {
	return oiml_map_custom2_impl_f32(ctx, a, b, fun, true);
}

// oiml_map_custom3_f32

OIML_INLINE oiml_tensor* oiml_map_custom3_impl_f32(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, const oiml_custom3_op_f32_t fun, bool inplace) {
	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	oiml_set_op_params(result, ( const void* )&fun, sizeof(fun));

	result->op	   = OIML_OP_MAP_CUSTOM3_F32;
	result->src[0] = a;
	result->src[1] = b;
	result->src[2] = c;

	return result;
}

OIML_INLINE oiml_tensor* oiml_map_custom3_f32(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, const oiml_custom3_op_f32_t fun) {
	return oiml_map_custom3_impl_f32(ctx, a, b, c, fun, false);
}

OIML_INLINE oiml_tensor* oiml_map_custom3_inplace_f32(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, const oiml_custom3_op_f32_t fun) {
	return oiml_map_custom3_impl_f32(ctx, a, b, c, fun, true);
}

// oiml_map_custom1

OIML_INLINE oiml_tensor* oiml_map_custom1_impl(oiml_context* ctx, oiml_tensor* a, const oiml_custom1_op_t fun, int n_tasks, void* userdata, bool inplace) {
	OIML_ASSERT(n_tasks == OIML_N_TASKS_MAX || n_tasks > 0);

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	struct oiml_map_custom1_op_params params = { /*.fun      =*/fun,
		/*.n_tasks  =*/n_tasks,
		/*.userdata =*/userdata };
	oiml_set_op_params(result, ( const void* )&params, sizeof(params));

	result->op	   = OIML_OP_MAP_CUSTOM1;
	result->src[0] = a;

	return result;
}

OIML_INLINE oiml_tensor* oiml_map_custom1(oiml_context* ctx, oiml_tensor* a, const oiml_custom1_op_t fun, int n_tasks, void* userdata) {
	return oiml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

OIML_INLINE oiml_tensor* oiml_map_custom1_inplace(oiml_context* ctx, oiml_tensor* a, const oiml_custom1_op_t fun, int n_tasks, void* userdata) {
	return oiml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// oiml_map_custom2

OIML_INLINE oiml_tensor* oiml_map_custom2_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, const oiml_custom2_op_t fun, int n_tasks, void* userdata, bool inplace) {
	OIML_ASSERT(n_tasks == OIML_N_TASKS_MAX || n_tasks > 0);

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	struct oiml_map_custom2_op_params params = { /*.fun      =*/fun,
		/*.n_tasks  =*/n_tasks,
		/*.userdata =*/userdata };
	oiml_set_op_params(result, ( const void* )&params, sizeof(params));

	result->op	   = OIML_OP_MAP_CUSTOM2;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

OIML_INLINE oiml_tensor* oiml_map_custom2(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, const oiml_custom2_op_t fun, int n_tasks, void* userdata) {
	return oiml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

OIML_INLINE oiml_tensor* oiml_map_custom2_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, const oiml_custom2_op_t fun, int n_tasks, void* userdata) {
	return oiml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// oiml_map_custom3

OIML_INLINE oiml_tensor* oiml_map_custom3_impl(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, const oiml_custom3_op_t fun, int n_tasks, void* userdata,
	bool inplace) {
	OIML_ASSERT(n_tasks == OIML_N_TASKS_MAX || n_tasks > 0);

	oiml_tensor* result = inplace ? oiml_view_tensor(ctx, a) : oiml_dup_tensor(ctx, a);

	struct oiml_map_custom3_op_params params = { /*.fun      =*/fun,
		/*.n_tasks  =*/n_tasks,
		/*.userdata =*/userdata };
	oiml_set_op_params(result, ( const void* )&params, sizeof(params));

	result->op	   = OIML_OP_MAP_CUSTOM3;
	result->src[0] = a;
	result->src[1] = b;
	result->src[2] = c;

	return result;
}

OIML_INLINE oiml_tensor* oiml_map_custom3(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, const oiml_custom3_op_t fun, int n_tasks, void* userdata) {
	return oiml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

OIML_INLINE oiml_tensor* oiml_map_custom3_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, const oiml_custom3_op_t fun, int n_tasks, void* userdata) {
	return oiml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

// oiml_cross_entropy_loss

OIML_INLINE oiml_tensor* oiml_cross_entropy_loss(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b) {
	OIML_ASSERT(oiml_are_same_shape(a, b));

	oiml_tensor* result = oiml_new_tensor_1d(ctx, a->type, 1);

	result->op	   = OIML_OP_CROSS_ENTROPY_LOSS;
	result->src[0] = a;
	result->src[1] = b;

	return result;
}

// oiml_cross_entropy_loss_back

OIML_INLINE oiml_tensor* oiml_cross_entropy_loss_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c) {
	OIML_ASSERT(oiml_is_scalar(a));
	OIML_ASSERT(oiml_are_same_shape(b, c));

	oiml_tensor* result = oiml_dup_tensor(ctx, b);

	result->op	   = OIML_OP_CROSS_ENTROPY_LOSS_BACK;
	result->src[0] = a;
	result->src[1] = b;
	result->src[2] = c;

	return result;
}

// opt_step_adamw

OIML_INLINE oiml_tensor* oiml_opt_step_adamw(oiml_context* ctx, oiml_tensor* a, oiml_tensor* grad, oiml_tensor* m, oiml_tensor* v, oiml_tensor* adamw_params) {
	OIML_ASSERT(a->flags & OIML_TENSOR_FLAG_PARAM);
	OIML_ASSERT(oiml_are_same_shape(a, grad));
	OIML_ASSERT(oiml_are_same_shape(a, m));
	OIML_ASSERT(oiml_are_same_shape(a, v));
	OIML_ASSERT(adamw_params->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(oiml_nelements(adamw_params) == 7);

	oiml_tensor* result = oiml_view_tensor(ctx, a);

	result->op	   = OIML_OP_OPT_STEP_ADAMW;
	result->src[0] = a;
	result->src[1] = grad;
	result->src[2] = m;
	result->src[3] = v;
	result->src[4] = adamw_params;

	return result;
}

////////////////////////////////////////////////////////////////////////////////

OIML_INLINE struct oiml_hash_set oiml_hash_set_new(size_t size) {
	size = oiml_hash_size(size);
	struct oiml_hash_set result;
	result.size = size;
	result.keys = ( oiml_tensor** )OIML_MALLOC(sizeof(oiml_tensor*) * size);
	result.used = ( oiml_bitset_t* )OIML_CALLOC(oiml_bitset_size(size), sizeof(oiml_bitset_t));
	return result;
}

OIML_INLINE void oiml_hash_set_reset(oiml_hash_set* hash_set) {
	memset(hash_set->used, 0, sizeof(oiml_bitset_t) * oiml_bitset_size(hash_set->size));
}

OIML_INLINE void oiml_hash_set_free(oiml_hash_set* hash_set) {
	OIML_FREE(hash_set->used);
	OIML_FREE(hash_set->keys);
}

OIML_INLINE size_t oiml_hash_size(size_t min_sz) {
	// next primes after powers of two
	constexpr size_t primes[] = { 2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031, 2053, 4099, 8209, 16411, 32771, 65537, 131101, 262147, 524309, 1048583, 2097169, 4194319, 8388617,
		16777259, 33554467, 67108879, 134217757, 268435459, 536870923, 1073741827, 2147483659 };
	constexpr size_t n_primes = sizeof(primes) / sizeof(primes[0]);

	// find the smallest prime that is larger or equal than min_sz
	size_t l = 0;
	size_t r = n_primes;
	while (l < r) {
		size_t m = (l + r) / 2;
		if (primes[m] < min_sz) {
			l = m + 1;
		} else {
			r = m;
		}
	}
	size_t sz = l < n_primes ? primes[l] : min_sz | 1;
	return sz;
}

struct hash_map {
	struct oiml_hash_set set;
	oiml_tensor** vals;
};

OIML_INLINE hash_map* oiml_new_hash_map(size_t size) {
	hash_map* result = ( hash_map* )OIML_MALLOC(sizeof(struct hash_map));
	result->set		 = oiml_hash_set_new(size);
	result->vals	 = ( oiml_tensor** )OIML_CALLOC(result->set.size, sizeof(oiml_tensor*));
	return result;
}

OIML_INLINE void oiml_hash_map_free(hash_map* map) {
	oiml_hash_set_free(&map->set);
	OIML_FREE(map->vals);
	OIML_FREE(map);
}

// utility functions to change gradients
// isrc is the index of tensor in cgraph->visited_has_set.keys
// the corresponding gradient (accumulators) are also at position isrc
// if tensor has a gradient accumulator, modify that accumulator in-place
// else if there is no gradient for tensor, set the corresponding value
// else, just add/subtract/etc. the gradients

OIML_INLINE void oiml_add_or_set(oiml_context* ctx, oiml_cgraph* cgraph, size_t isrc, oiml_tensor* tensor) {
	oiml_tensor* src = cgraph->visited_hash_set.keys[isrc];
	OIML_ASSERT(src);
	if (cgraph->grads[isrc]) {
		cgraph->grads[isrc] = oiml_add_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/cgraph->grad_accs[isrc]);
	} else {
		cgraph->grads[isrc] = tensor;
	}
	oiml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
	oiml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

OIML_INLINE void oiml_acc_or_set(oiml_context* ctx, oiml_cgraph* cgraph, size_t isrc, oiml_tensor* tensor, const size_t nb1, const size_t nb2, const size_t nb3,
	const size_t offset) {
	oiml_tensor* src = cgraph->visited_hash_set.keys[isrc];
	OIML_ASSERT(src);
	if (cgraph->grads[isrc]) {
		cgraph->grads[isrc] = oiml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[isrc]);
	} else {
		oiml_tensor* a_zero = oiml_scale(ctx, src, 0.0f);// FIXME this is going to produce NaN if a contains inf/NaN
		cgraph->grads[isrc] = oiml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
	}
	oiml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.keys[isrc]->name);
	oiml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

OIML_INLINE void oiml_add1_or_set(oiml_context* ctx, oiml_cgraph* cgraph, size_t isrc, oiml_tensor* tensor) {
	oiml_tensor* src = cgraph->visited_hash_set.keys[isrc];
	OIML_ASSERT(src);
	if (cgraph->grads[isrc]) {
		cgraph->grads[isrc] = oiml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
	} else {
		cgraph->grads[isrc] = oiml_repeat(ctx, tensor, src);
	}
	oiml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
	oiml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

OIML_INLINE void oiml_sub_or_set(oiml_context* ctx, oiml_cgraph* cgraph, size_t isrc, oiml_tensor* tensor) {
	oiml_tensor* src = cgraph->visited_hash_set.keys[isrc];
	OIML_ASSERT(src);
	if (cgraph->grads[isrc]) {
		cgraph->grads[isrc] = oiml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
	} else {
		cgraph->grads[isrc] = oiml_neg(ctx, tensor);
	}
	oiml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
	oiml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

OIML_INLINE void oiml_compute_backward(oiml_context* ctx, oiml_cgraph* cgraph, int i, const bool* grads_needed) {
	oiml_tensor* tensor = cgraph->nodes[i];
	oiml_tensor* grad	= oiml_graph_get_grad(cgraph, tensor);

	if (!grad) {
		return;
	}

	oiml_tensor* src0			= tensor->src[0];
	oiml_tensor* src1			= tensor->src[1];
	oiml_tensor* src2			= tensor->src[2];
	oiml_hash_set* hash_set		= &cgraph->visited_hash_set;
	const size_t isrc0			= src0 ? oiml_hash_find(hash_set, src0) : ( size_t )-1;
	const size_t isrc1			= src1 ? oiml_hash_find(hash_set, src1) : ( size_t )-1;
	const size_t isrc2			= src2 ? oiml_hash_find(hash_set, src2) : ( size_t )-1;
	const bool src0_needs_grads = src0 && isrc0 != OIML_HASHSET_FULL && oiml_bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
	const bool src1_needs_grads = src1 && isrc1 != OIML_HASHSET_FULL && oiml_bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
	const bool src2_needs_grads = src2 && isrc2 != OIML_HASHSET_FULL && oiml_bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];

	switch (tensor->op) {
		case OIML_OP_DUP: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, grad);
			}
		} break;
		case OIML_OP_ADD: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, grad);
			}
			if (src1_needs_grads) {
				oiml_tensor* tmp = grad;
				if (!oiml_are_same_shape(src0, src1)) {
					tmp = oiml_repeat_back(ctx, tmp, src1);
				}
				oiml_add_or_set(ctx, cgraph, isrc1, tmp);
			}
		} break;
		case OIML_OP_ADD1: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, grad);
			}
			if (src1_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc1, oiml_mean(ctx, grad));// TODO: should probably be sum instead of mean
			}
		} break;
		case OIML_OP_ACC: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, grad);
			}
			if (src1_needs_grads) {
				const size_t nb1	= (( int32_t* )tensor->op_params)[0];
				const size_t nb2	= (( int32_t* )tensor->op_params)[1];
				const size_t nb3	= (( int32_t* )tensor->op_params)[2];
				const size_t offset = (( int32_t* )tensor->op_params)[3];

				oiml_tensor* tensor_grad_view = oiml_view_4d(ctx, grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], nb1, nb2, nb3, offset);

				oiml_add_or_set(ctx, cgraph, isrc1, oiml_reshape(ctx, oiml_cont(ctx, tensor_grad_view), src1));
			}
		} break;
		case OIML_OP_SUB: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, grad);
			}
			if (src1_needs_grads) {
				oiml_sub_or_set(ctx, cgraph, isrc1, grad);
			}
		} break;
		case OIML_OP_MUL: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_mul(ctx, grad, src1));
			}
			if (src1_needs_grads) {
				oiml_tensor* tmp = oiml_mul(ctx, src0, grad);
				if (!oiml_are_same_shape(src0, src1)) {
					tmp = oiml_repeat_back(ctx, tmp, src1);
				}
				oiml_add_or_set(ctx, cgraph, isrc1, tmp);
			}
		} break;
		case OIML_OP_DIV: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_div(ctx, grad, src1));
			}
			if (src1_needs_grads) {
				oiml_sub_or_set(ctx, cgraph, isrc1, oiml_mul(ctx, grad, oiml_div(ctx, tensor, src1)));
			}
		} break;
		case OIML_OP_SQR: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_scale(ctx, oiml_mul(ctx, src0, grad), 2.0f));
			}
		} break;
		case OIML_OP_SQRT: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_scale(ctx, oiml_div(ctx, grad, tensor), 0.5f));
			}
		} break;
		case OIML_OP_LOG: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_div(ctx, grad, src0));
			}
		} break;
		case OIML_OP_SIN: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_mul(ctx, grad, oiml_cos(ctx, src0)));
			}
		} break;
		case OIML_OP_COS: {
			if (src0_needs_grads) {
				oiml_sub_or_set(ctx, cgraph, isrc0, oiml_mul(ctx, grad, oiml_sin(ctx, src0)));
			}
		} break;
		case OIML_OP_SUM: {
			if (src0_needs_grads) {
				oiml_add1_or_set(ctx, cgraph, isrc0, grad);
			}
		} break;
		case OIML_OP_SUM_ROWS: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_repeat(ctx, grad, src0));
			}
		} break;
		case OIML_OP_MEAN: {
			if (src0_needs_grads) {
				oiml_add1_or_set(ctx, cgraph, isrc0, oiml_scale_impl(ctx, grad, 1.0f / src0->ne[0], false));
			}
		} break;
		case OIML_OP_REPEAT: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_repeat_back(ctx, grad, src0));
			}
		} break;
		case OIML_OP_REPEAT_BACK: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_repeat(ctx, grad, src0));
			}
		} break;
		case OIML_OP_RMS_NORM: {
			if (src0_needs_grads) {
				float eps;
				memcpy(&eps, tensor->op_params, sizeof(float));
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_rms_norm_back(ctx, grad, src0, eps));
			}
		} break;
		case OIML_OP_MUL_MAT: {
			// https://cs231n.github.io/optimization-2/#staged
			// # forward pass
			// s0 = np.random.randn(5, 10)
			// s1 = np.random.randn(10, 3)
			// t = s0.dot(s1)

			// # now suppose we had the gradient on t from above in the circuit
			// dt = np.random.randn(*t.shape) # same shape as t
			// ds0 = dt.dot(s1.T) #.T gives the transpose of the matrix
			// ds1 = t.T.dot(dt)

			// tensor.shape [m,p,qq,rr]
			// src0.shape   [n,m,q1,r1]
			// src1.shape   [n,p,qq,rr]

			if (src0_needs_grads) {
				OIML_ASSERT(grad->ne[2] == src1->ne[2]);
				OIML_ASSERT(grad->ne[3] == src1->ne[3]);
				oiml_tensor* tmp = oiml_out_prod(ctx,// [n,m,qq,rr]
					src1,// [n,p,qq,rr]
					grad);// [m,p,qq,rr]
				if (!oiml_are_same_shape(tmp, src0)) {
					OIML_ASSERT(tmp->ne[0] == src0->ne[0]);
					OIML_ASSERT(tmp->ne[1] == src0->ne[1]);
					OIML_ASSERT(tmp->ne[3] == 1);

					const int64_t nr2 = tmp->ne[2] / src0->ne[2];
					const size_t nb2  = tmp->nb[2] * nr2;
					const size_t nb3  = tmp->nb[2];

					tmp = oiml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
					tmp = oiml_repeat_back(ctx, tmp, src0);
				}
				oiml_add_or_set(ctx, cgraph, isrc0, tmp);
			}
			if (src1_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc1,
					// oiml_mul_mat(ctx,                   // [n,p,qq,rr]
					//     oiml_cont(ctx,                  // [m,n,q1,r1]
					//         oiml_transpose(ctx, src0)), // [m,n,q1,r1]
					//     grad),                          // [m,p,qq,rr]

					// when src0 is bigger than tensor->grad (this is mostly the case in llama),
					// avoid transpose of src0, rather transpose smaller tensor->grad
					// and then use oiml_out_prod
					oiml_out_prod(ctx,// [n,p,qq,rr]
						src0,// [n,m,q1,r1]
						oiml_transpose(ctx,// [p,m,qq,rr]
							grad)));// [m,p,qq,rr]
			}
		} break;
		case OIML_OP_SCALE: {
			if (src0_needs_grads) {
				float s;
				memcpy(&s, tensor->op_params, sizeof(float));
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_scale_impl(ctx, grad, s, false));
			}
		} break;
		case OIML_OP_SET: {
			const size_t nb1	= (( const int32_t* )tensor->op_params)[0];
			const size_t nb2	= (( const int32_t* )tensor->op_params)[1];
			const size_t nb3	= (( const int32_t* )tensor->op_params)[2];
			const size_t offset = (( const int32_t* )tensor->op_params)[3];

			oiml_tensor* tensor_grad_view = NULL;

			if (src0_needs_grads || src1_needs_grads) {
				OIML_ASSERT(src0->type == tensor->type);
				OIML_ASSERT(!cgraph->grads[isrc0] || cgraph->grads[isrc0]->type == grad->type);
				OIML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

				tensor_grad_view = oiml_view_4d(ctx, grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], nb1, nb2, nb3, offset);
			}

			if (src0_needs_grads) {
				oiml_tensor* tmp = oiml_neg(ctx, tensor_grad_view);
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
			}

			if (src1_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc1, oiml_reshape(ctx, oiml_cont(ctx, tensor_grad_view), src1));
			}
		} break;
		case OIML_OP_CPY: {
			// cpy overwrites value of src1 by src0 and returns view(src1)
			// the overwriting is mathematically equivalent to:
			// tensor = src0 * 1 + src1 * 0
			if (src0_needs_grads) {
				// dsrc0 = dtensor * 1
				oiml_add_or_set(ctx, cgraph, isrc0, grad);
			}
			if (src1_needs_grads) {
				// dsrc1 = dtensor * 0 -> noop
			}
		} break;
		case OIML_OP_CONT: {
			// same as cpy
			if (src0_needs_grads) {
				OIML_ASSERT(!cgraph->grads[isrc0] || oiml_is_contiguous(cgraph->grads[isrc0]));
				OIML_ASSERT(oiml_is_contiguous(grad));
				OIML_ASSERT(oiml_nelements(tensor) == oiml_nelements(src0));
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_are_same_shape(tensor, src0) ? grad : oiml_reshape(ctx, grad, src0));
			}
		} break;
		case OIML_OP_RESHAPE: {
			if (src0_needs_grads) {
				oiml_tensor* grad_cont = oiml_is_contiguous(grad) ? grad : oiml_cont(ctx, grad);
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_reshape(ctx, grad_cont, src0));
			}
		} break;
		case OIML_OP_VIEW: {
			if (src0_needs_grads) {
				size_t offset;

				memcpy(&offset, tensor->op_params, sizeof(offset));

				size_t nb1 = tensor->nb[1];
				size_t nb2 = tensor->nb[2];
				size_t nb3 = tensor->nb[3];

				if (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
					// gradient is typically F32, but src0 could be other type
					size_t ng = oiml_element_size(cgraph->grads[isrc0]);
					size_t n0 = oiml_element_size(src0);
					OIML_ASSERT(offset % n0 == 0);
					OIML_ASSERT(nb1 % n0 == 0);
					OIML_ASSERT(nb2 % n0 == 0);
					OIML_ASSERT(nb3 % n0 == 0);
					offset = (offset / n0) * ng;
					nb1	   = (nb1 / n0) * ng;
					nb2	   = (nb2 / n0) * ng;
					nb3	   = (nb3 / n0) * ng;
				}

				oiml_acc_or_set(ctx, cgraph, isrc0, grad, nb1, nb2, nb3, offset);
			}
		} break;
		case OIML_OP_PERMUTE: {
			if (src0_needs_grads) {
				const int32_t* axes = ( const int32_t* )tensor->op_params;
				const int axis0		= axes[0] & 0x3;
				const int axis1		= axes[1] & 0x3;
				const int axis2		= axes[2] & 0x3;
				const int axis3		= axes[3] & 0x3;
				int axb[4]			= { 0, 0, 0, 0 };// axes backward
				axb[axis0]			= 0;
				axb[axis1]			= 1;
				axb[axis2]			= 2;
				axb[axis3]			= 3;
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
			}
		} break;
		case OIML_OP_TRANSPOSE: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_transpose(ctx, grad));
			}
		} break;
		case OIML_OP_GET_ROWS: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_get_rows_back(ctx, grad, src1, src0));
			}
			if (src1_needs_grads) {
				// noop
			}
		} break;
		case OIML_OP_DIAG_MASK_INF: {
			if (src0_needs_grads) {
				/* oiml_diag_mask_inf_impl() shouldn't be here */
				/* ref:  https://github.com/ggerganov/llama.cpp/pull/4203#discussion_r1412377992 */
				const int n_past = (( const int32_t* )tensor->op_params)[0];
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_diag_mask_zero_impl(ctx, grad, n_past, false));
			}
		} break;
		case OIML_OP_DIAG_MASK_ZERO: {
			if (src0_needs_grads) {
				const int n_past = (( const int32_t* )tensor->op_params)[0];
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_diag_mask_zero_impl(ctx, grad, n_past, false));
			}
		} break;
		case OIML_OP_SOFT_MAX: {
			if (src0_needs_grads) {
				float scale	   = 1.0f;
				float max_bias = 0.0f;

				memcpy(&scale, ( const float* )tensor->op_params + 0, sizeof(float));
				memcpy(&max_bias, ( const float* )tensor->op_params + 1, sizeof(float));

				oiml_add_or_set(ctx, cgraph, isrc0, oiml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
			}
			OIML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
		} break;
		case OIML_OP_ROPE: {
			if (src0_needs_grads) {
				//const int n_past = ((int32_t *) tensor->op_params)[0];
				const int n_dims = (( const int32_t* )tensor->op_params)[1];
				const int mode	 = (( const int32_t* )tensor->op_params)[2];
				//const int n_ctx      = ((int32_t *) tensor->op_params)[3];
				const int n_ctx_orig = (( const int32_t* )tensor->op_params)[4];
				float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
				int sections[4] = { 0, 0, 0, 0 };

				memcpy(&freq_base, ( const float* )tensor->op_params + 5, sizeof(float));
				memcpy(&freq_scale, ( const float* )tensor->op_params + 6, sizeof(float));
				memcpy(&ext_factor, ( const float* )tensor->op_params + 7, sizeof(float));
				memcpy(&attn_factor, ( const float* )tensor->op_params + 8, sizeof(float));
				memcpy(&beta_fast, ( const float* )tensor->op_params + 9, sizeof(float));
				memcpy(&beta_slow, ( const float* )tensor->op_params + 10, sizeof(float));
				memcpy(&sections, tensor->op_params + 11, sizeof(sections));

				oiml_tensor* rope_back = grad->ne[2] == src1->ne[0]
					? oiml_rope_ext_back(ctx, grad, src1, src2, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
					: oiml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
				oiml_add_or_set(ctx, cgraph, isrc0, rope_back);
			}
			OIML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
		} break;
		case OIML_OP_IM2COL: {
			if (src1_needs_grads) {
				const int32_t s0 = oiml_get_op_params_i32(tensor, 0);
				const int32_t s1 = oiml_get_op_params_i32(tensor, 1);
				const int32_t p0 = oiml_get_op_params_i32(tensor, 2);
				const int32_t p1 = oiml_get_op_params_i32(tensor, 3);
				const int32_t d0 = oiml_get_op_params_i32(tensor, 4);
				const int32_t d1 = oiml_get_op_params_i32(tensor, 5);
				const bool is_2D = oiml_get_op_params_i32(tensor, 6) == 1;

				oiml_add_or_set(ctx, cgraph, isrc1, oiml_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
			}
		} break;
		case OIML_OP_POOL_2D: {
			if (src0_needs_grads) {
				const enum oiml_op_pool op = ( oiml_op_pool )oiml_get_op_params_i32(tensor, 0);
				const int32_t k0		   = oiml_get_op_params_i32(tensor, 1);
				const int32_t k1		   = oiml_get_op_params_i32(tensor, 2);
				const int32_t s0		   = oiml_get_op_params_i32(tensor, 3);
				const int32_t s1		   = oiml_get_op_params_i32(tensor, 4);
				const int32_t p0		   = oiml_get_op_params_i32(tensor, 5);
				const int32_t p1		   = oiml_get_op_params_i32(tensor, 6);

				oiml_add_or_set(ctx, cgraph, isrc0, oiml_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
			}
		} break;
		case OIML_OP_WIN_PART:
		case OIML_OP_WIN_UNPART:
		case OIML_OP_UNARY: {
			switch (oiml_get_unary_op(tensor)) {
				case OIML_UNARY_OP_ABS: {
					if (src0_needs_grads) {
						oiml_add_or_set(ctx, cgraph, isrc0, oiml_mul(ctx, oiml_sgn(ctx, src0), grad));
					}
				} break;
				case OIML_UNARY_OP_SGN: {
					// noop
				} break;
				case OIML_UNARY_OP_NEG: {
					if (src0_needs_grads) {
						oiml_sub_or_set(ctx, cgraph, isrc0, grad);
					}
				} break;
				case OIML_UNARY_OP_STEP: {
					// noop
				} break;
				case OIML_UNARY_OP_RELU: {
					if (src0_needs_grads) {
						oiml_add_or_set(ctx, cgraph, isrc0, oiml_mul(ctx, oiml_step(ctx, src0), grad));
					}
				} break;
				case OIML_UNARY_OP_SILU: {
					if (src0_needs_grads) {
						oiml_add_or_set(ctx, cgraph, isrc0, oiml_silu_back(ctx, grad, src0));
					}
				} break;
				case OIML_UNARY_OP_EXP: {
					if (src0_needs_grads) {
						oiml_add_or_set(ctx, cgraph, isrc0, oiml_mul(ctx, tensor, grad));
					}
				} break;
				default: {
					fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n", __func__, oiml_unary_op_name(oiml_get_unary_op(tensor)));
					OIML_ABORT("fatal error");
				}//break;
			}
		} break;
		case OIML_OP_CROSS_ENTROPY_LOSS: {
			if (src0_needs_grads) {
				oiml_add_or_set(ctx, cgraph, isrc0, oiml_cross_entropy_loss_back(ctx, grad, src0, src1));
			}
			OIML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
		} break;
		case OIML_OP_NONE: {
			// noop
		} break;
		case OIML_OP_COUNT:
		default: {
			fprintf(stderr, "%s: unsupported oiml op for backward pass: %s\n", __func__, oiml_op_name(tensor->op));
			OIML_ABORT("fatal error");
		}//break;
	}

	OIML_ASSERT(!src0_needs_grads || oiml_are_same_shape(src0, cgraph->grads[isrc0]));
	OIML_ASSERT(!src1_needs_grads || oiml_are_same_shape(src1, cgraph->grads[isrc1]));
	OIML_ASSERT(!src2_needs_grads || oiml_are_same_shape(src2, cgraph->grads[isrc2]));
}

OIML_INLINE void oiml_visit_parents(oiml_cgraph* cgraph, oiml_tensor* node) {
	// check if already visited
	if (oiml_hash_insert(&cgraph->visited_hash_set, node) == OIML_HASHSET_ALREADY_EXISTS) {
		return;
	}

	for (int i = 0; i < OIML_MAX_SRC; ++i) {
		const int k = (cgraph->order == OIML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i
			: (cgraph->order == OIML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT)		  ? (OIML_MAX_SRC - 1 - i)
																			  :
																	  /* unknown order, just fall back to using i*/ i;
		if (node->src[k]) {
			oiml_visit_parents(cgraph, node->src[k]);
		}
	}

	if (node->op == OIML_OP_NONE && !(node->flags & OIML_TENSOR_FLAG_PARAM)) {
		// reached a leaf node, not part of the gradient graph (e.g. a constant)
		OIML_ASSERT(cgraph->n_leafs < cgraph->size);

		if (strlen(node->name) == 0) {
			oiml_format_name(node, "leaf_%d", cgraph->n_leafs);
		}

		cgraph->leafs[cgraph->n_leafs] = node;
		cgraph->n_leafs++;
	} else {
		OIML_ASSERT(cgraph->n_nodes < cgraph->size);

		if (strlen(node->name) == 0) {
			oiml_format_name(node, "node_%d", cgraph->n_nodes);
		}

		cgraph->nodes[cgraph->n_nodes] = node;
		cgraph->n_nodes++;
	}
}

OIML_INLINE void oiml_build_forward_impl(oiml_cgraph* cgraph, oiml_tensor* tensor, bool expand) {
	if (!expand) {
		// TODO: this branch isn't accessible anymore, maybe move this to oiml_build_forward_expand
		oiml_graph_clear(cgraph);
	}

	const int n0 = cgraph->n_nodes;

	oiml_visit_parents(cgraph, tensor);

	const int n_new = cgraph->n_nodes - n0;
	OIML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

	if (n_new > 0) {
		// the last added node should always be starting point
		OIML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
	}
}

OIML_INLINE void oiml_build_forward_expand(oiml_cgraph* cgraph, oiml_tensor* tensor) {
	oiml_build_forward_impl(cgraph, tensor, true);
}

OIML_INLINE void oiml_build_backward_expand(oiml_context* ctx_static, oiml_context* ctx_compute, oiml_cgraph* cgraph, bool accumulate) {
	OIML_ASSERT(cgraph->n_nodes > 0);
	OIML_ASSERT(cgraph->grads);
	OIML_ASSERT(cgraph->grad_accs);

	const int n_nodes_f = cgraph->n_nodes;

	memset(cgraph->grads, 0, cgraph->visited_hash_set.size * sizeof(oiml_tensor*));
	memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size * sizeof(oiml_tensor*));
	bool* grads_needed = ( bool* )calloc(cgraph->visited_hash_set.size, sizeof(bool));

	{
		bool any_params = false;
		bool any_loss	= false;
		for (int i = 0; i < n_nodes_f; ++i) {
			oiml_tensor* node = cgraph->nodes[i];
			any_params		  = any_params || (node->flags & OIML_TENSOR_FLAG_PARAM);
			any_loss		  = any_loss || (node->flags & OIML_TENSOR_FLAG_LOSS);
		}
		OIML_ASSERT(any_params && "no trainable parameters found, did you forget to call oiml_set_param?");
		OIML_ASSERT(any_loss && "no training loss found, did you forget to call oiml_set_loss?");
	}

	for (int i = 0; i < n_nodes_f; ++i) {
		oiml_tensor* node = cgraph->nodes[i];

		if (node->type == oiml::oiml_representation_types::int_32) {
			continue;
		}

		bool node_needs_grad		  = (node->flags & OIML_TENSOR_FLAG_PARAM) || (node->flags & OIML_TENSOR_FLAG_LOSS);
		bool ignore_src[OIML_MAX_SRC] = { false };
		switch (node->op) {
			// gradients in node->src[0] for one reason or another have no effect on output gradients
			case OIML_OP_IM2COL:// only used for its shape
			case OIML_OP_IM2COL_BACK:// same as IM2COL
				ignore_src[0] = true;
				break;
			case OIML_OP_UNARY: {
				const enum oiml_unary_op uop = oiml_get_unary_op(node);
				// SGN and STEP unary ops are piecewise constant
				if (uop == OIML_UNARY_OP_SGN || uop == OIML_UNARY_OP_STEP) {
					ignore_src[0] = true;
				}
			} break;

			// gradients in node->src[1] for one reason or another have no effect on output gradients
			case OIML_OP_CPY:// gradients in CPY target are irrelevant
			case OIML_OP_GET_ROWS:// row indices not differentiable
			case OIML_OP_GET_ROWS_BACK:// same as for GET_ROWS
			case OIML_OP_ROPE:// positions not differentiable
				ignore_src[1] = true;
				break;

			default:
				break;
		}
		for (int j = 0; j < OIML_MAX_SRC; ++j) {
			if (!node->src[j] || ignore_src[j] || !grads_needed[oiml_hash_find(&cgraph->visited_hash_set, node->src[j])]) {
				continue;
			}
			OIML_ASSERT(node->src[j]->type == oiml::oiml_representation_types::float_32 || node->src[j]->type == oiml::oiml_representation_types::float_16);
			node_needs_grad = true;
			break;
		}
		if (!node_needs_grad) {
			continue;
		}

		// inplace operations are currently not supported
		OIML_ASSERT(
			!node->view_src || node->op == OIML_OP_CPY || node->op == OIML_OP_VIEW || node->op == OIML_OP_RESHAPE || node->op == OIML_OP_PERMUTE || node->op == OIML_OP_TRANSPOSE);

		const size_t igrad = oiml_hash_find(&cgraph->visited_hash_set, node);
		OIML_ASSERT(igrad != OIML_HASHSET_FULL);
		OIML_ASSERT(oiml_bitset_get(cgraph->visited_hash_set.used, igrad));
		if ((accumulate && (node->flags & OIML_TENSOR_FLAG_PARAM)) || (node->flags & OIML_TENSOR_FLAG_LOSS)) {
			cgraph->grad_accs[igrad] = oiml_dup_tensor(ctx_static, node);
			cgraph->grads[igrad]	 = cgraph->grad_accs[igrad];
			oiml_format_name(cgraph->grad_accs[igrad], "grad acc for %s", node->name);
		}
		grads_needed[igrad] = true;
	}

	for (int i = n_nodes_f - 1; i >= 0; --i) {
		// inplace operations to add gradients are not created by oiml_compute_backward except for gradient accumulation
		// use allocator to automatically make inplace operations
		oiml_compute_backward(ctx_compute, cgraph, i, grads_needed);
	}

	free(grads_needed);
}

OIML_INLINE void* incr_ptr_aligned(void** p, size_t size, size_t align) {
	void* ptr = *p;
	ptr		  = ( void* )OIML_PAD(( uintptr_t )ptr, align);
	*p		  = ( void* )(( char* )ptr + size);
	return ptr;
}

OIML_INLINE size_t oiml_graph_nbytes(size_t size, bool grads) {
	size_t hash_size = oiml_hash_size(size * 2);
	void* p			 = 0;
	incr_ptr_aligned(&p, sizeof(struct oiml_cgraph), 1);
	incr_ptr_aligned(&p, size * sizeof(oiml_tensor*), sizeof(oiml_tensor*));// nodes
	incr_ptr_aligned(&p, size * sizeof(oiml_tensor*), sizeof(oiml_tensor*));// leafs
	incr_ptr_aligned(&p, hash_size * sizeof(oiml_tensor*), sizeof(oiml_tensor*));// hash keys
	if (grads) {
		incr_ptr_aligned(&p, hash_size * sizeof(oiml_tensor*), sizeof(oiml_tensor*));// grads
		incr_ptr_aligned(&p, hash_size * sizeof(oiml_tensor*), sizeof(oiml_tensor*));// grad_accs
	}
	incr_ptr_aligned(&p, oiml_bitset_size(hash_size) * sizeof(oiml_bitset_t), sizeof(oiml_bitset_t));

	size_t nbytes = ( size_t )p;
	return nbytes;
}

OIML_INLINE size_t oiml_graph_overhead_custom(size_t size, bool grads) {
	return OIML_OBJECT_SIZE + OIML_PAD(oiml_graph_nbytes(size, grads), OIML_MEM_ALIGN);
}

OIML_INLINE size_t oiml_graph_overhead() {
	return oiml_graph_overhead_custom(OIML_DEFAULT_GRAPH_SIZE, false);
}

OIML_INLINE oiml_cgraph* oiml_new_graph_custom(oiml_context* ctx, size_t size, bool grads) {
	const size_t obj_size = oiml_graph_nbytes(size, grads);
	oiml_object* obj	  = oiml_new_object(ctx, OIML_OBJECT_TYPE_GRAPH, obj_size);
	oiml_cgraph* cgraph	  = ( oiml_cgraph* )(( char* )ctx->mem_buffer + obj->offs);

	// the size of the hash table is doubled since it needs to hold both nodes and leafs
	size_t hash_size = oiml_hash_size(size * 2);

	void* p = cgraph + 1;

	oiml_tensor** nodes_ptr		= ( oiml_tensor** )incr_ptr_aligned(&p, size * sizeof(oiml_tensor*), sizeof(oiml_tensor*));
	oiml_tensor** leafs_ptr		= ( oiml_tensor** )incr_ptr_aligned(&p, size * sizeof(oiml_tensor*), sizeof(oiml_tensor*));
	oiml_tensor** hash_keys_ptr = ( oiml_tensor** )incr_ptr_aligned(&p, hash_size * sizeof(oiml_tensor*), sizeof(oiml_tensor*));
	oiml_tensor** grads_ptr		= ( oiml_tensor** )(grads ? incr_ptr_aligned(&p, hash_size * sizeof(oiml_tensor*), sizeof(oiml_tensor*)) : NULL);
	oiml_tensor** grad_accs_ptr = ( oiml_tensor** )(grads ? incr_ptr_aligned(&p, hash_size * sizeof(oiml_tensor*), sizeof(oiml_tensor*)) : NULL);

	oiml_bitset_t* hash_used = ( oiml_bitset_t* )incr_ptr_aligned(&p, oiml_bitset_size(hash_size) * sizeof(oiml_bitset_t), sizeof(oiml_bitset_t));

	// check that we allocated the correct amount of memory
	assert(obj_size == ( size_t )(( char* )p - ( char* )cgraph));

	*cgraph = oiml_cgraph{
		/*.size         =*/static_cast<int32_t>(size),
		/*.n_nodes      =*/0,
		/*.n_leafs      =*/0,
		/*.nodes        =*/nodes_ptr,
		/*.grads        =*/grads_ptr,
		/*.grad_accs    =*/grad_accs_ptr,
		/*.leafs        =*/leafs_ptr,
		/*.hash_table   =*/{ hash_size, hash_used, hash_keys_ptr },
		/*.order        =*/OIML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT,
	};

	oiml_hash_set_reset(&cgraph->visited_hash_set);
	if (grads) {
		memset(cgraph->grads, 0, hash_size * sizeof(oiml_tensor*));
		memset(cgraph->grad_accs, 0, hash_size * sizeof(oiml_tensor*));
	}

	return cgraph;
}

OIML_INLINE oiml_cgraph* oiml_new_graph(oiml_context* ctx) {
	return oiml_new_graph_custom(ctx, OIML_DEFAULT_GRAPH_SIZE, false);
}

OIML_INLINE struct oiml_cgraph oiml_graph_view(oiml_cgraph* cgraph0, int i0, int i1) {
	struct oiml_cgraph cgraph = {
		/*.size             =*/0,
		/*.n_nodes          =*/i1 - i0,
		/*.n_leafs          =*/0,
		/*.nodes            =*/cgraph0->nodes + i0,
		/*.grads            =*/NULL,// gradients would need visited_hash_set
		/*.grad_accs        =*/NULL,
		/*.leafs            =*/NULL,
		/*.visited_hash_set =*/{ 0, NULL, NULL },
		/*.order            =*/cgraph0->order,
	};

	return cgraph;
}

OIML_INLINE void oiml_graph_cpy(oiml_cgraph* src, oiml_cgraph* dst) {
	OIML_ASSERT(dst->size >= src->n_leafs);
	OIML_ASSERT(dst->size >= src->n_nodes);
	OIML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

	dst->n_leafs = src->n_leafs;
	dst->n_nodes = src->n_nodes;
	dst->order	 = src->order;

	for (int i = 0; i < src->n_leafs; ++i) {
		dst->leafs[i] = src->leafs[i];
	}

	for (int i = 0; i < src->n_nodes; ++i) {
		dst->nodes[i] = src->nodes[i];
	}

	for (size_t i = 0; i < src->visited_hash_set.size; ++i) {
		// copy all hashset keys (tensors) that are in use
		if (oiml_bitset_get(src->visited_hash_set.used, i)) {
			oiml_hash_insert(&dst->visited_hash_set, src->visited_hash_set.keys[i]);
		}
	}

	if (dst->grads) {
		memset(dst->grads, 0, dst->visited_hash_set.size * sizeof(oiml_tensor*));
		memset(dst->grad_accs, 0, dst->visited_hash_set.size * sizeof(oiml_tensor*));
	}
	if (src->grads) {
		OIML_ASSERT(dst->grads != NULL);
		OIML_ASSERT(dst->grad_accs != NULL);
		for (int i = 0; i < src->n_nodes; ++i) {
			const size_t igrad_src = oiml_hash_find(&src->visited_hash_set, src->nodes[i]);
			const size_t igrad_dst = oiml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

			OIML_ASSERT(igrad_src != OIML_HASHSET_FULL);
			OIML_ASSERT(oiml_bitset_get(src->visited_hash_set.used, igrad_src));
			OIML_ASSERT(igrad_dst != OIML_HASHSET_FULL);
			OIML_ASSERT(oiml_bitset_get(dst->visited_hash_set.used, igrad_dst));

			dst->grads[igrad_dst]	  = src->grads[igrad_src];
			dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
		}
	}
}

OIML_INLINE oiml_cgraph* oiml_graph_dup(oiml_context* ctx, oiml_cgraph* cgraph) {
	oiml_cgraph* result = oiml_new_graph_custom(ctx, cgraph->size, cgraph->grads != NULL);
	oiml_graph_cpy(cgraph, result);
	return result;
}

OIML_INLINE oiml_tensor* oiml_set_zero(oiml_tensor* tensor) {
	if (oiml_is_empty(tensor)) {
		return tensor;
	}
	if (tensor->buffer) {
		oiml_backend_tensor_memset(tensor, 0, 0, oiml_nbytes(tensor));
	} else {
		OIML_ASSERT(tensor->data);
		memset(tensor->data, 0, oiml_nbytes(tensor));
	}
	return tensor;
}

OIML_INLINE void oiml_graph_reset(oiml_cgraph* cgraph) {
	OIML_ASSERT(cgraph->grads != NULL);

	for (int i = 0; i < cgraph->n_nodes; i++) {
		oiml_tensor* node	  = cgraph->nodes[i];
		oiml_tensor* grad_acc = oiml_graph_get_grad_acc(cgraph, node);

		if (node->op == OIML_OP_OPT_STEP_ADAMW) {
			// clear momenta
			oiml_set_zero(node->src[2]);
			oiml_set_zero(node->src[3]);
		}

		// initial gradients of loss should be 1, 0 otherwise
		if (grad_acc) {
			if (node->flags & OIML_TENSOR_FLAG_LOSS) {
				OIML_ASSERT(grad_acc->type == oiml::oiml_representation_types::float_32);
				OIML_ASSERT(oiml_is_scalar(grad_acc));

				const float onef = 1.0f;
				if (grad_acc->buffer) {
					oiml_backend_tensor_set(grad_acc, &onef, 0, sizeof(float));
				} else {
					OIML_ASSERT(grad_acc->data);
					*(( float* )grad_acc->data) = onef;
				}
			} else {
				oiml_set_zero(grad_acc);
			}
		}
	}
}

OIML_INLINE void oiml_graph_clear(oiml_cgraph* cgraph) {
	cgraph->n_leafs = 0;
	cgraph->n_nodes = 0;
	oiml_hash_set_reset(&cgraph->visited_hash_set);
}

OIML_INLINE int oiml_graph_size(oiml_cgraph* cgraph) {
	return cgraph->size;
}

OIML_INLINE oiml_tensor* oiml_graph_node(oiml_cgraph* cgraph, int i) {
	if (i < 0) {
		OIML_ASSERT(cgraph->n_nodes + i >= 0);
		return cgraph->nodes[cgraph->n_nodes + i];
	}

	OIML_ASSERT(i < cgraph->n_nodes);
	return cgraph->nodes[i];
}

OIML_INLINE oiml_tensor** oiml_graph_nodes(oiml_cgraph* cgraph) {
	return cgraph->nodes;
}

OIML_INLINE int oiml_graph_n_nodes(oiml_cgraph* cgraph) {
	return cgraph->n_nodes;
}

OIML_INLINE void oiml_graph_add_node(oiml_cgraph* cgraph, oiml_tensor* tensor) {
	OIML_ASSERT(cgraph->size > cgraph->n_nodes);
	cgraph->nodes[cgraph->n_nodes] = tensor;
	cgraph->n_nodes++;
}

OIML_INLINE oiml_tensor* oiml_graph_get_tensor(const oiml_cgraph* cgraph, const char* name) {
	for (int i = 0; i < cgraph->n_leafs; i++) {
		oiml_tensor* leaf = cgraph->leafs[i];

		if (strcmp(leaf->name, name) == 0) {
			return leaf;
		}
	}

	for (int i = 0; i < cgraph->n_nodes; i++) {
		oiml_tensor* node = cgraph->nodes[i];

		if (strcmp(node->name, name) == 0) {
			return node;
		}
	}

	return NULL;
}

OIML_INLINE oiml_tensor* oiml_graph_get_grad(const oiml_cgraph* cgraph, const oiml_tensor* node) {
	const size_t igrad = oiml_hash_find(&cgraph->visited_hash_set, node);
	return igrad != OIML_HASHSET_FULL && oiml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads ? cgraph->grads[igrad] : NULL;
}

OIML_INLINE oiml_tensor* oiml_graph_get_grad_acc(const oiml_cgraph* cgraph, const oiml_tensor* node) {
	const size_t igrad = oiml_hash_find(&cgraph->visited_hash_set, node);
	return igrad != OIML_HASHSET_FULL && oiml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : NULL;
}

OIML_INLINE void oiml_graph_print(const oiml_cgraph* cgraph) {
	OIML_LOG_INFO("=== GRAPH ===\n");

	OIML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
	for (int i = 0; i < cgraph->n_nodes; i++) {
		oiml_tensor* node = cgraph->nodes[i];

		OIML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n", i, node->ne[0], node->ne[1], node->ne[2], oiml_op_name(node->op),
			(node->flags & OIML_TENSOR_FLAG_PARAM)	? "x"
				: oiml_graph_get_grad(cgraph, node) ? "g"
													: " ");
	}

	OIML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
	for (int i = 0; i < cgraph->n_leafs; i++) {
		oiml_tensor* node = cgraph->leafs[i];

		OIML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n", i, node->ne[0], node->ne[1], oiml_op_name(node->op), oiml_get_name(node));
	}

	OIML_LOG_INFO("========================================\n");
}

// check if node is part of the graph
OIML_INLINE bool oiml_graph_find(const oiml_cgraph* cgraph, const oiml_tensor* node) {
	if (cgraph == NULL) {
		return true;
	}

	for (int i = 0; i < cgraph->n_nodes; i++) {
		if (cgraph->nodes[i] == node) {
			return true;
		}
	}

	return false;
}

OIML_INLINE oiml_tensor* oiml_graph_get_parent(const oiml_cgraph* cgraph, const oiml_tensor* node) {
	for (int i = 0; i < cgraph->n_nodes; i++) {
		oiml_tensor* parent = cgraph->nodes[i];
		oiml_tensor* grad	= oiml_graph_get_grad(cgraph, parent);

		if (grad == node) {
			return parent;
		}
	}

	return NULL;
}

OIML_INLINE void oiml_graph_dump_dot_node_edge(FILE* fp, const oiml_cgraph* gb, oiml_tensor* node, oiml_tensor* parent, const char* label) {
	oiml_tensor* gparent  = oiml_graph_get_parent(gb, node);
	oiml_tensor* gparent0 = oiml_graph_get_parent(gb, parent);
	fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ arrowhead = %s; style = %s; label = \"%s\"; ]\n", gparent0 ? ( void* )gparent0 : ( void* )parent, gparent0 ? "g" : "x",
		gparent ? ( void* )gparent : ( void* )node, gparent ? "g" : "x", gparent ? "empty" : "vee", gparent ? "dashed" : "solid", label);
}

OIML_INLINE void oiml_graph_dump_dot_leaf_edge(FILE* fp, oiml_tensor* node, oiml_tensor* parent, const char* label) {
	fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ label = \"%s\"; ]\n", ( void* )parent, "x", ( void* )node, "x", label);
}

OIML_INLINE void oiml_graph_dump_dot(const oiml_cgraph* gb, const oiml_cgraph* gf, const char* filename) {
	char color[16];

	FILE* fp = oiml_fopen(filename, "w");
	OIML_ASSERT(fp);

	fprintf(fp, "digraph G {\n");
	fprintf(fp, "  newrank = true;\n");
	fprintf(fp, "  rankdir = TB;\n");

	for (int i = 0; i < gb->n_nodes; i++) {
		oiml_tensor* node = gb->nodes[i];
		oiml_tensor* grad = oiml_graph_get_grad(gb, node);

		if (oiml_graph_get_parent(gb, node) != NULL) {
			continue;
		}

		if (node->flags & OIML_TENSOR_FLAG_PARAM) {
			snprintf(color, sizeof(color), "yellow");
		} else if (grad) {
			if (oiml_graph_find(gf, node)) {
				snprintf(color, sizeof(color), "green");
			} else {
				snprintf(color, sizeof(color), "lightblue");
			}
		} else {
			snprintf(color, sizeof(color), "white");
		}

		fprintf(fp,
			"  \"%p\" [ "
			"style = filled; fillcolor = %s; shape = record; "
			"label=\"",
			( void* )node, color);

		if (strlen(node->name) > 0) {
			fprintf(fp, "%s (%s)|", node->name, oiml_type_name(node->type));
		} else {
			fprintf(fp, "(%s)|", oiml_type_name(node->type));
		}

		if (oiml_is_matrix(node)) {
			fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, node->ne[0], node->ne[1], oiml_op_symbol(node->op));
		} else {
			fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, node->ne[0], node->ne[1], node->ne[2], oiml_op_symbol(node->op));
		}

		if (grad) {
			fprintf(fp, " | <g>%s\"; ]\n", oiml_op_symbol(grad->op));
		} else {
			fprintf(fp, "\"; ]\n");
		}
	}

	for (int i = 0; i < gb->n_leafs; i++) {
		oiml_tensor* node = gb->leafs[i];

		snprintf(color, sizeof(color), "pink");

		fprintf(fp,
			"  \"%p\" [ "
			"style = filled; fillcolor = %s; shape = record; "
			"label=\"<x>",
			( void* )node, color);

		if (strlen(node->name) > 0) {
			fprintf(fp, "%s (%s)|", node->name, oiml_type_name(node->type));
		} else {
			fprintf(fp, "(%s)|", oiml_type_name(node->type));
		}

		fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, node->ne[0], node->ne[1]);
		if (oiml_nelements(node) < 5 && node->data != NULL) {
			fprintf(fp, " | (");
			for (int j = 0; j < oiml_nelements(node); j++) {
				// FIXME: use oiml-backend to obtain the tensor data
				//if (node->type == OIML_TYPE_I8 || node->type == OIML_TYPE_I16 || node->type == oiml::oiml_representation_types::int_32) {
				//    fprintf(fp, "%d", oiml_get_i32_1d(node, j));
				//}
				//else if (node->type == oiml::oiml_representation_types::float_32 ||
				//         node->type == oiml::oiml_representation_types::float_16 ||
				//         node->type == oiml::oiml_representation_types::brain_float_16) {
				//    fprintf(fp, "%.1e", (double)oiml_get_f32_1d(node, j));
				//}
				//else
				{
					fprintf(fp, "#");
				}
				if (j < oiml_nelements(node) - 1) {
					fprintf(fp, ", ");
				}
			}
			fprintf(fp, ")");
		}
		fprintf(fp, "\"; ]\n");
	}

	for (int i = 0; i < gb->n_nodes; i++) {
		oiml_tensor* node = gb->nodes[i];

		for (int j = 0; j < OIML_MAX_SRC; j++) {
			if (node->src[j]) {
				char label[16];
				snprintf(label, sizeof(label), "src %d", j);
				oiml_graph_dump_dot_node_edge(fp, gb, node, node->src[j], label);
			}
		}
	}

	for (int i = 0; i < gb->n_leafs; i++) {
		oiml_tensor* node = gb->leafs[i];

		for (int j = 0; j < OIML_MAX_SRC; j++) {
			if (node->src[j]) {
				char label[16];
				snprintf(label, sizeof(label), "src %d", j);
				oiml_graph_dump_dot_leaf_edge(fp, node, node->src[j], label);
			}
		}
	}

	fprintf(fp, "}\n");

	fclose(fp);

	OIML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

OIML_INLINE void oiml_set_input(oiml_tensor* tensor) {
	tensor->flags |= OIML_TENSOR_FLAG_INPUT;
}

OIML_INLINE void oiml_set_output(oiml_tensor* tensor) {
	tensor->flags |= OIML_TENSOR_FLAG_OUTPUT;
}

OIML_INLINE void oiml_set_param(oiml_context* ctx, oiml_tensor* tensor) {
	OIML_UNUSED(ctx);// TODO: remove this parameter
	tensor->flags |= OIML_TENSOR_FLAG_PARAM;
}

OIML_INLINE void oiml_set_loss(oiml_tensor* tensor) {
	OIML_ASSERT(oiml_is_scalar(tensor));
	OIML_ASSERT(tensor->type == oiml::oiml_representation_types::float_32);
	tensor->flags |= OIML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

OIML_INLINE bool oiml_quantize_requires_imatrix(oiml::oiml_representation_types type) {
	OIML_ASSERT(false && "remove this path");
	return false;
	//	return type == OIML_TYPE_IQ2_XXS || type == OIML_TYPE_IQ2_XS || type == OIML_TYPE_IQ1_S;//   ||
	//type == OIML_TYPE_IQ1_M;
}

OIML_INLINE size_t oiml_quantize_chunk(oiml::oiml_representation_types type, const float* src, void* dst, int64_t start, int64_t nrows, int64_t n_per_row, const float* imatrix) {
	const int64_t n = ( int64_t )nrows * n_per_row;

	if (oiml_quantize_requires_imatrix(type)) {
		OIML_ASSERT(imatrix != NULL);
	}

	OIML_ASSERT(start % type_traits.at(type).blck_size == 0);
	OIML_ASSERT(start % n_per_row == 0);

	const size_t start_row = start / n_per_row;
	const size_t row_size  = oiml_row_size(type, n_per_row);

	size_t result = 0;

	switch (type) {
		case oiml::oiml_representation_types::q8_0:
			result = quantize_q8_0(src + start, reinterpret_cast<uint8_t*>(dst) + start_row * row_size, nrows, n_per_row, imatrix);
			break;
		case oiml::oiml_representation_types::float_16: {
			size_t elemsize = sizeof(oiml_fp16_t);
			oiml_fp32_to_fp16_row(src + start, reinterpret_cast<oiml_fp16_t*>(dst) + start, n);
			result = n * elemsize;
		} break;
		case oiml::oiml_representation_types::brain_float_16: {
			size_t elemsize = sizeof(oiml_bf16_t);
			oiml_fp32_to_bf16_row_ref(src + start, reinterpret_cast<oiml_bf16_t*>(dst) + start, n);
			result = n * elemsize;
		} break;
		case oiml::oiml_representation_types::float_32: {
			size_t elemsize = sizeof(float);
			result			= n * elemsize;
			memcpy(( int8_t* )dst + start * elemsize, src + start, result);
		} break;
		default:
			assert(false);
	}

	OIML_ASSERT(result == nrows * row_size);

	return result;
}
////////////////////////////////////////////////////////////////////////////////

OIML_INLINE void oiml_log_set(oiml_log_callback log_callback, void* user_data) {
	g_oiml_logger_state.log_callback		   = log_callback ? log_callback : oiml_log_callback_default;
	g_oiml_logger_state.log_callback_user_data = user_data;
}

OIML_INLINE void oiml_threadpool_params_init(oiml_threadpool_params* p, int n_threads) {
	p->n_threads  = n_threads;
	p->prio		  = ( oiml_sched_priority )0;
	p->poll		  = 50;// hybrid-polling enabled
	p->strict_cpu = false;// no strict placement (all threads share same cpumask)
	p->paused	  = false;// threads are ready to go
	memset(p->cpumask, 0, OIML_MAX_N_THREADS);// all-zero means use the default affinity (usually inherited)
}

OIML_INLINE struct oiml_threadpool_params oiml_threadpool_params_default(int n_threads) {
	struct oiml_threadpool_params p;
	oiml_threadpool_params_init(&p, n_threads);
	return p;
}

OIML_INLINE bool oiml_threadpool_params_match(const oiml_threadpool_params* p0, const oiml_threadpool_params* p1) {
	if (p0->n_threads != p1->n_threads)
		return false;
	if (p0->prio != p1->prio)
		return false;
	if (p0->poll != p1->poll)
		return false;
	if (p0->strict_cpu != p1->strict_cpu)
		return false;
	return memcmp(p0->cpumask, p1->cpumask, OIML_MAX_N_THREADS) == 0;
}
