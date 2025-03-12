#pragma once

// OIML internal header

#include <oiml/legacy/oiml-legacy-common/oiml-final.hpp>
#include <oiml/legacy/oiml-legacy-common/oigguf.hpp>
#include <oiml/common/config.hpp>
#include <oiml/common/util_functions.hpp>
#include <bit>
#include <array>

#include <assert.h>
#include <math.h>
#include <stdlib.h>// load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#ifdef __ARM_FEATURE_SVE
	#include <arm_sve.h>
#endif// __ARM_FEATURE_SVE

#if defined(__ARM_NEON) && !defined(__CUDACC__)
	// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
	//
	//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
	//
	#include <arm_neon.h>
#endif

#if defined(__F16C__)
	#include <immintrin.h>
#endif

#ifndef MIN
	#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
	#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// required for mmap as oigguf only guarantees 32-byte alignment
#define TENSOR_ALIGNMENT 32

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ifndef __cplusplus
	#ifndef static_assert
		#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
			#define static_assert(cond, msg) _Static_assert(cond, msg)
		#else
			#define static_assert(cond, msg) struct global_scope_noop_trick
		#endif
	#endif
#endif

OIML_INLINE static int oiml_up32(int n) {
	return (n + 31) & ~31;
}

//OIML_INLINE static int oiml_up64(int n) {
//    return (n + 63) & ~63;
//}

OIML_INLINE static int oiml_up(int n, int m) {
	// assert m is a power of 2
	OIML_ASSERT((m & (m - 1)) == 0);
	return (n + m - 1) & ~(m - 1);
}

//
// logging
//

OIML_ATTRIBUTE_FORMAT(2, 3)
void oiml_log_internal(enum oiml_log_level level, const char* format, ...);
void oiml_log_callback_default(enum oiml_log_level level, const char* text, void* user_data);

#define OIML_LOG(...) oiml_log_internal(OIML_LOG_LEVEL_NONE, __VA_ARGS__)
#define OIML_LOG_INFO(...) oiml_log_internal(OIML_LOG_LEVEL_INFO, __VA_ARGS__)
#define OIML_LOG_WARN(...) oiml_log_internal(OIML_LOG_LEVEL_WARN, __VA_ARGS__)
#define OIML_LOG_ERROR(...) oiml_log_internal(OIML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define OIML_LOG_DEBUG(...) oiml_log_internal(OIML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define OIML_LOG_CONT(...) oiml_log_internal(OIML_LOG_LEVEL_CONT, __VA_ARGS__)

#define OIML_DEBUG 0

#if (OIML_DEBUG >= 1)
	#define OIML_PRINT_DEBUG(...) OIML_LOG_DEBUG(__VA_ARGS__)
#else
	#define OIML_PRINT_DEBUG(...)
#endif

#if (OIML_DEBUG >= 5)
	#define OIML_PRINT_DEBUG_5(...) OIML_LOG_DEBUG(__VA_ARGS__)
#else
	#define OIML_PRINT_DEBUG_5(...)
#endif

#if (OIML_DEBUG >= 10)
	#define OIML_PRINT_DEBUG_10(...) OIML_LOG_DEBUG(__VA_ARGS__)
#else
	#define OIML_PRINT_DEBUG_10(...)
#endif

// tensor params

OIML_INLINE static void oiml_set_op_params(oiml_tensor* tensor, const void* params, size_t params_size) {
	OIML_ASSERT(tensor != NULL);// silence -Warray-bounds warnings
	assert(params_size <= OIML_MAX_OP_PARAMS);
	memcpy(tensor->op_params, params, params_size);
}

OIML_INLINE static int32_t oiml_get_op_params_i32(const oiml_tensor* tensor, uint32_t i) {
	assert(i < OIML_MAX_OP_PARAMS / sizeof(int32_t));
	return (( const int32_t* )(tensor->op_params))[i];
}

OIML_INLINE static float oiml_get_op_params_f32(const oiml_tensor* tensor, uint32_t i) {
	assert(i < OIML_MAX_OP_PARAMS / sizeof(float));
	return (( const float* )(tensor->op_params))[i];
}

OIML_INLINE static void oiml_set_op_params_i32(oiml_tensor* tensor, uint32_t i, int32_t value) {
	assert(i < OIML_MAX_OP_PARAMS / sizeof(int32_t));
	(( int32_t* )(tensor->op_params))[i] = value;
}

OIML_INLINE static void oiml_set_op_params_f32(oiml_tensor* tensor, uint32_t i, float value) {
	assert(i < OIML_MAX_OP_PARAMS / sizeof(float));
	(( float* )(tensor->op_params))[i] = value;
}

struct oiml_map_custom1_op_params {
	oiml_custom1_op_t fun;
	int n_tasks;
	void* userdata;
};

struct oiml_map_custom2_op_params {
	oiml_custom2_op_t fun;
	int n_tasks;
	void* userdata;
};

struct oiml_map_custom3_op_params {
	oiml_custom3_op_t fun;
	int n_tasks;
	void* userdata;
};

// bitset

typedef uint32_t oiml_bitset_t;

static_assert(sizeof(oiml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5// log2(sizeof(oiml_bitset_t)*8)
#define BITSET_MASK (sizeof(oiml_bitset_t) * 8 - 1)

OIML_INLINE static size_t oiml_bitset_size(size_t n) {
	return (n + BITSET_MASK) >> BITSET_SHR;
}

OIML_INLINE static bool oiml_bitset_get(const oiml_bitset_t* bitset, size_t i) {
	return !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

OIML_INLINE static void oiml_bitset_set(oiml_bitset_t* bitset, size_t i) {
	bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

OIML_INLINE static void oiml_bitset_clear(oiml_bitset_t* bitset, size_t i) {
	bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define OIML_HASHSET_FULL (( size_t )-1)
#define OIML_HASHSET_ALREADY_EXISTS (( size_t )-2)

struct oiml_hash_set {
	size_t size;
	oiml_bitset_t* used;// whether or not the keys are in use i.e. set
	oiml_tensor** keys;// actual tensors in the set, keys[i] is only defined if oiml_bitset_get(used, i)
};

struct oiml_hash_set oiml_hash_set_new(size_t size);
void oiml_hash_set_free(oiml_hash_set* hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t oiml_hash_size(size_t min_sz);

// remove all elements from the hash set
void oiml_hash_set_reset(oiml_hash_set* hash_set);

// returns true if key is in the hash set
static bool oiml_hash_contains(const oiml_hash_set* hash_set, oiml_tensor* key);

// returns OIML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
static size_t oiml_hash_find(const oiml_hash_set* hash_set, const oiml_tensor* key);

// returns OIML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
static size_t oiml_hash_insert(oiml_hash_set* hash_set, oiml_tensor* key);

// return index, asserts if table is full
static size_t oiml_hash_find_or_insert(oiml_hash_set* hash_set, oiml_tensor* key);

// hash function for oiml_tensor
OIML_INLINE static size_t oiml_hash(const oiml_tensor* p) {
	// the last 4 bits are always zero due to alignment
	return ( size_t )( uintptr_t )p >> 4;
}

OIML_INLINE static size_t oiml_hash_find(const oiml_hash_set* hash_set, const oiml_tensor* key) {
	size_t h = oiml_hash(key) % hash_set->size;

	// linear probing
	size_t i = h;
	while (oiml_bitset_get(hash_set->used, i) && hash_set->keys[i] != key) {
		i = (i + 1) % hash_set->size;
		if (i == h) {
			// visited all hash table entries -> not found
			return OIML_HASHSET_FULL;
		}
	}
	return i;
}

OIML_INLINE static bool oiml_hash_contains(const oiml_hash_set* hash_set, oiml_tensor* key) {
	size_t i = oiml_hash_find(hash_set, key);
	return i != OIML_HASHSET_FULL && oiml_bitset_get(hash_set->used, i);
}

OIML_INLINE static size_t oiml_hash_insert(oiml_hash_set* hash_set, oiml_tensor* key) {
	size_t h = oiml_hash(key) % hash_set->size;

	// linear probing
	size_t i = h;
	do {
		if (!oiml_bitset_get(hash_set->used, i)) {
			oiml_bitset_set(hash_set->used, i);
			hash_set->keys[i] = key;
			return i;
		}
		if (hash_set->keys[i] == key) {
			return OIML_HASHSET_ALREADY_EXISTS;
		}
		i = (i + 1) % hash_set->size;
	} while (i != h);

	// visited all hash table entries -> not found
	OIML_ABORT("fatal error");
}

OIML_INLINE static size_t oiml_hash_find_or_insert(oiml_hash_set* hash_set, oiml_tensor* key) {
	size_t h = oiml_hash(key) % hash_set->size;

	// linear probing
	size_t i = h;
	do {
		if (!oiml_bitset_get(hash_set->used, i)) {
			oiml_bitset_set(hash_set->used, i);
			hash_set->keys[i] = key;
			return i;
		}
		if (hash_set->keys[i] == key) {
			return i;
		}
		i = (i + 1) % hash_set->size;
	} while (i != h);

	// visited all hash table entries -> not found
	OIML_ABORT("fatal error");
}

// computation graph

enum oiml_cgraph_eval_order { OIML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0, OIML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT, OIML_CGRAPH_EVAL_ORDER_COUNT };

struct oiml_cgraph {
	int size;// maximum number of nodes/leafs/grads/grad_accs
	int n_nodes;// number of nodes currently in use
	int n_leafs;// number of leafs currently in use

	oiml_tensor** nodes;// tensors with data that can change if the graph is evaluated
	oiml_tensor** grads;// the outputs of these tensors are the gradients of the nodes
	oiml_tensor** grad_accs;// accumulators for node gradients
	oiml_tensor** leafs;// tensors with constant data

	struct oiml_hash_set visited_hash_set;

	enum oiml_cgraph_eval_order order;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
struct oiml_cgraph oiml_graph_view(oiml_cgraph* cgraph, int i0, int i1);

// Memory allocation

void* oiml_aligned_malloc(size_t size);
void oiml_aligned_free(void* ptr, size_t size);

// FP16 to FP32 conversion


/**
 * Converts brain16 to float32.
 *
 * The bfloat16 floating point format has the following structure:
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───┐
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───────────────────┐
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * For comparison, the standard fp16 format has fewer exponent bits.
 *
 *       ┌sign
 *       │
 *       │  ┌exponent
 *       │  │
 *       │  │    ┌mantissa
 *       │  │    │
 *       │┌─┴─┐┌─┴──────┐
 *     0b0000000000000000 IEEE binary16
 *
 * @see IEEE 754-2008
 */


#include <vector>

// expose GGUF internals for test code
size_t oigguf_type_size(enum oigguf_type type);
struct oigguf_context* oigguf_init_from_file_impl(FILE* file, struct oigguf_init_params params);
void oigguf_write_to_buf(const struct oigguf_context* ctx, std::vector<int8_t>& buf, bool only_meta);
