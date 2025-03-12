#pragma once

//
// OIML Tensor Library
//
// This documentation is still a work in progress.
// If you wish some specific topics to be covered, feel free to drop a comment:
//
//   https://github.com/ggerganov/whisper.cpp/issues/40
//
// ## Overview
//
// This library implements:
//
//  - a set of tensor operations
//  - automatic differentiation
//  - basic optimization algorithms
//
// The aim of this library is to provide a minimalistic approach for various machine learning tasks. This includes,
// but is not limited to, the following:
//
//  - linear regression
//  - support vector machines
//  - neural networks
//
// The library allows the user to define a certain function using the available tensor operations. This function
// definition is represented internally via a computation graph. Each tensor operation in the function definition
// corresponds to a node in the graph. Having the computation graph defined, the user can choose to compute the
// function's value and/or its gradient with respect to the input variables. Optionally, the function can be optimized
// using one of the available optimization algorithms.
//
// For example, here we define the function: f(x) = a*x^2 + b
//
//   {
//       struct oiml_init_params params = {
//           .mem_size   = 16*1024*1024,
//           .mem_buffer = NULL,
//       };
//
//       // memory allocation happens here
//       struct oiml_context * ctx = oiml_init(params);
//
//       struct oiml_tensor * x = oiml_new_tensor_1d(ctx, oiml::oiml_representation_types::float_32, 1);
//
//       oiml_set_param(ctx, x); // x is an input variable
//
//       struct oiml_tensor * a  = oiml_new_tensor_1d(ctx, oiml::oiml_representation_types::float_32, 1);
//       struct oiml_tensor * b  = oiml_new_tensor_1d(ctx, oiml::oiml_representation_types::float_32, 1);
//       struct oiml_tensor * x2 = oiml_mul(ctx, x, x);
//       struct oiml_tensor * f  = oiml_add(ctx, oiml_mul(ctx, a, x2), b);
//
//       ...
//   }
//
// Notice that the function definition above does not involve any actual computation. The computation is performed only
// when the user explicitly requests it. For example, to compute the function's value at x = 2.0:
//
//   {
//       ...
//
//       struct oiml_cgraph * gf = oiml_new_graph(ctx);
//       oiml_build_forward_expand(gf, f);
//
//       // set the input variable and parameter values
//       oiml_set_f32(x, 2.0f);
//       oiml_set_f32(a, 3.0f);
//       oiml_set_f32(b, 4.0f);
//
//       oiml_graph_compute_with_ctx(ctx, &gf, n_threads);
//
//       printf("f = %f\n", oiml_get_f32_1d(f, 0));
//
//       ...
//   }
//
// The actual computation is performed in the oiml_graph_compute() function.
//
// The oiml_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
// oiml_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
// in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
// and after defining the computation graph, call the oiml_used_mem() function to find out how much memory was
// actually needed.
//
// The oiml_set_param() function marks a tensor as an input variable. This is used by the automatic
// differentiation and optimization algorithms.
//
// The described approach allows to define the function graph once and then compute its forward or backward graphs
// multiple times. All computations will use the same memory buffer allocated in the oiml_init() function. This way
// the user can avoid the memory allocation overhead at runtime.
//
// The library supports multi-dimensional tensors - up to 4 dimensions. The FP16 and FP32 data types are first class
// citizens, but in theory the library can be extended to support FP8 and integer data types.
//
// Each tensor operation produces a new tensor. Initially the library was envisioned to support only the use of unary
// and binary operations. Most of the available operations fall into one of these two categories. With time, it became
// clear that the library needs to support more complex operations. The way to support these operations is not clear
// yet, but a few examples are demonstrated in the following operations:
//
//   - oiml_permute()
//   - oiml_conv_1d_1s()
//   - oiml_conv_1d_2s()
//
// For each tensor operator, the library implements a forward and backward computation function. The forward function
// computes the output tensor value given the input tensor values. The backward function computes the adjoint of the
// input tensors given the adjoint of the output tensor. For a detailed explanation of what this means, take a
// calculus class, or watch the following video:
//
//   What is Automatic Differentiation?
//   https://www.youtube.com/watch?v=wG_nF1awSSY
//
//
// ## Tensor data (struct oiml_tensor)
//
// The tensors are stored in memory via the oiml_tensor struct. The structure provides information about the size of
// the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
// pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
//
//   {
//       struct oiml_tensor * c = oiml_add(ctx, a, b);
//
//       assert(c->src[0] == a);
//       assert(c->src[1] == b);
//   }
//
// The multi-dimensional tensors are stored in row-major order. The oiml_tensor struct contains fields for the
// number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
// to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
// permutation. All tensor operations have to take the stride into account and not assume that the tensor is
// contiguous in memory.
//
// The data of the tensor is accessed via the "data" pointer. For example:
//
//   {
//       const int nx = 2;
//       const int ny = 3;
//
//       struct oiml_tensor * a = oiml_new_tensor_2d(ctx, oiml::oiml_representation_types::float_32, nx, ny);
//
//       for (int y = 0; y < ny; y++) {
//           for (int x = 0; x < nx; x++) {
//               *(float *) ((char *) a->data + y*a->nb[1] + x*a->nb[0]) = x + y;
//           }
//       }
//
//       ...
//   }
//
// Alternatively, there are helper functions, such as oiml_get_f32_1d() and oiml_set_f32_1d() that can be used.
//
// ## The matrix multiplication operator (oiml_mul_mat)
//
// TODO
//
//
// ## Multi-threading
//
// TODO
//
//
// ## Overview of oiml.c
//
// TODO
//
//
// ## SIMD optimizations
//
// TODO
//
//
// ## Debugging oiml
//
// TODO
//
//

#ifndef __GNUC__
	#define OIML_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__)
	#define OIML_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
	#define OIML_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <oiml/common/util_functions.hpp>
#include <oiml/common/representation_traits.hpp>

#define OIML_FILE_MAGIC 0x67676d6c// "oiml"
#define OIML_FILE_VERSION 2

#define OIML_QNT_VERSION 2// bump this on quantization format changes
#define OIML_QNT_VERSION_FACTOR 1000// do not change this

#define OIML_MAX_DIMS 4
#define OIML_MAX_PARAMS 2048
#define OIML_MAX_SRC 10
#define OIML_MAX_N_THREADS 512
#define OIML_MAX_OP_PARAMS 64

#ifndef OIML_MAX_NAME
	#define OIML_MAX_NAME 64
#endif

#define OIML_DEFAULT_N_THREADS 4
#define OIML_DEFAULT_GRAPH_SIZE 2048

#if UINTPTR_MAX == 0xFFFFFFFF
	#define OIML_MEM_ALIGN 4
#else
	#define OIML_MEM_ALIGN 16
#endif

#define OIML_EXIT_SUCCESS 0
#define OIML_EXIT_ABORTED 1

#define OIML_ROPE_TYPE_NEOX 2
#define OIML_ROPE_TYPE_MROPE 8
#define OIML_ROPE_TYPE_VISION 24

#define OIML_UNUSED(x) ( void )(x)

#define OIML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#ifndef NDEBUG
	#define OIML_UNREACHABLE() \
		do { \
			fprintf(stderr, "statement should be unreachable\n"); \
			abort(); \
		} while (0)
#elif defined(__GNUC__)
	#define OIML_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
	#define OIML_UNREACHABLE() __assume(0)
#else
	#define OIML_UNREACHABLE() (( void )0)
#endif

#ifdef __cplusplus
	#define OIML_NORETURN [[noreturn]]
#elif defined(_MSC_VER)
	#define OIML_NORETURN __declspec(noreturn)
#else
	#define OIML_NORETURN _Noreturn
#endif

#define OIML_ABORT(...) oiml_abort(__FILE__, __LINE__, __VA_ARGS__)
#define OIML_ASSERT(x) \
	if (!(x)) \
	OIML_ABORT("OIML_ASSERT(%s) failed", #x)

// used to copy the number of elements and stride in bytes of tensors into local variables.
// main purpose is to reduce code duplication and improve readability.
//
// example:
//
//    OIML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
//    OIML_TENSOR_LOCALS(size_t,  nb1, src1, nb);
//
#define OIML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
	const type prefix##0 = (pointer)->array[0]; \
	OIML_UNUSED(prefix##0);
#define OIML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
	OIML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
	const type prefix##1 = (pointer)->array[1]; \
	OIML_UNUSED(prefix##1);
#define OIML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
	OIML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
	const type prefix##2 = (pointer)->array[2]; \
	OIML_UNUSED(prefix##2);
#define OIML_TENSOR_LOCALS(type, prefix, pointer, array) \
	OIML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
	const type prefix##3 = (pointer)->array[3]; \
	OIML_UNUSED(prefix##3);

#define OIML_TENSOR_UNARY_OP_LOCALS \
	OIML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
	OIML_TENSOR_LOCALS(size_t, nb0, src0, nb) \
	OIML_TENSOR_LOCALS(int64_t, ne, dst, ne) \
	OIML_TENSOR_LOCALS(size_t, nb, dst, nb)

#define OIML_TENSOR_BINARY_OP_LOCALS \
	OIML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
	OIML_TENSOR_LOCALS(size_t, nb0, src0, nb) \
	OIML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
	OIML_TENSOR_LOCALS(size_t, nb1, src1, nb) \
	OIML_TENSOR_LOCALS(int64_t, ne, dst, ne) \
	OIML_TENSOR_LOCALS(size_t, nb, dst, nb)

#define OIML_TENSOR_BINARY_OP_LOCALS01 \
	OIML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
	OIML_TENSOR_LOCALS(size_t, nb0, src0, nb) \
	OIML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
	OIML_TENSOR_LOCALS(size_t, nb1, src1, nb)



OIML_NORETURN OIML_ATTRIBUTE_FORMAT(3, 4) void oiml_abort(const char* file, int line, const char* fmt, ...);

enum oiml_status {
	OIML_STATUS_ALLOC_FAILED = -2,
	OIML_STATUS_FAILED		 = -1,
	OIML_STATUS_SUCCESS		 = 0,
	OIML_STATUS_ABORTED		 = 1,
};

// get oiml_status name string
const char* oiml_status_to_string(oiml_status status);

struct oiml_tensor_binding;

// ieee 754-2008 half-precision float16
// todo: make this not an integral type
typedef uint16_t oiml_fp16_t;

using oiml_bf16_t = uint16_t;

struct oiml_object;
struct oiml_context;
struct oiml_cgraph;

// precision
enum oiml_prec {
	OIML_PREC_DEFAULT,
	OIML_PREC_F32,
};

// model file types
enum oiml_ftype {
	OIML_FTYPE_UNKNOWN				= -1,
	OIML_FTYPE_ALL_F32				= 0,
	OIML_FTYPE_MOSTLY_F16			= 1,// except 1d tensors
	OIML_FTYPE_MOSTLY_Q4_0			= 2,// except 1d tensors
	OIML_FTYPE_MOSTLY_Q4_1			= 3,// except 1d tensors
	OIML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,// tok_embeddings.weight and output.weight are F16
	OIML_FTYPE_MOSTLY_Q8_0			= 7,// except 1d tensors
	OIML_FTYPE_MOSTLY_Q5_0			= 8,// except 1d tensors
	OIML_FTYPE_MOSTLY_Q5_1			= 9,// except 1d tensors
	OIML_FTYPE_MOSTLY_Q2_K			= 10,// except 1d tensors
	OIML_FTYPE_MOSTLY_Q3_K			= 11,// except 1d tensors
	OIML_FTYPE_MOSTLY_Q4_K			= 12,// except 1d tensors
	OIML_FTYPE_MOSTLY_Q5_K			= 13,// except 1d tensors
	OIML_FTYPE_MOSTLY_Q6_K			= 14,// except 1d tensors
	OIML_FTYPE_MOSTLY_IQ2_XXS		= 15,// except 1d tensors
	OIML_FTYPE_MOSTLY_IQ2_XS		= 16,// except 1d tensors
	OIML_FTYPE_MOSTLY_IQ3_XXS		= 17,// except 1d tensors
	OIML_FTYPE_MOSTLY_IQ1_S			= 18,// except 1d tensors
	OIML_FTYPE_MOSTLY_IQ4_NL		= 19,// except 1d tensors
	OIML_FTYPE_MOSTLY_IQ3_S			= 20,// except 1d tensors
	OIML_FTYPE_MOSTLY_IQ2_S			= 21,// except 1d tensors
	OIML_FTYPE_MOSTLY_IQ4_XS		= 22,// except 1d tensors
	OIML_FTYPE_MOSTLY_IQ1_M			= 23,// except 1d tensors
	OIML_FTYPE_MOSTLY_BF16			= 24,// except 1d tensors
};

// available tensor operations:
enum oiml_op {
	OIML_OP_NONE = 0,

	OIML_OP_DUP,
	OIML_OP_ADD,
	OIML_OP_ADD1,
	OIML_OP_ACC,
	OIML_OP_SUB,
	OIML_OP_MUL,
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
	OIML_OP_RMS_NORM,
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
	OIML_OP_GET_ROWS,
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

	OIML_OP_COUNT,
};

enum oiml_unary_op {
	OIML_UNARY_OP_ABS,
	OIML_UNARY_OP_SGN,
	OIML_UNARY_OP_NEG,
	OIML_UNARY_OP_STEP,
	OIML_UNARY_OP_TANH,
	OIML_UNARY_OP_ELU,
	OIML_UNARY_OP_RELU,
	OIML_UNARY_OP_SIGMOID,
	OIML_UNARY_OP_GELU,
	OIML_UNARY_OP_GELU_QUICK,
	OIML_UNARY_OP_SILU,
	OIML_UNARY_OP_HARDSWISH,
	OIML_UNARY_OP_HARDSIGMOID,
	OIML_UNARY_OP_EXP,

	OIML_UNARY_OP_COUNT,
};

enum oiml_object_type { OIML_OBJECT_TYPE_TENSOR, OIML_OBJECT_TYPE_GRAPH, OIML_OBJECT_TYPE_WORK_BUFFER };

enum oiml_log_level {
	OIML_LOG_LEVEL_NONE	 = 0,
	OIML_LOG_LEVEL_DEBUG = 1,
	OIML_LOG_LEVEL_INFO	 = 2,
	OIML_LOG_LEVEL_WARN	 = 3,
	OIML_LOG_LEVEL_ERROR = 4,
	OIML_LOG_LEVEL_CONT	 = 5,// continue previous log
};

// this tensor...
enum oiml_tensor_flag {
	OIML_TENSOR_FLAG_INPUT	= 1,// ...is an input for the OIML compute graph
	OIML_TENSOR_FLAG_OUTPUT = 2,// ...is an output for the OIML compute graph
	OIML_TENSOR_FLAG_PARAM	= 4,// ...contains trainable parameters
	OIML_TENSOR_FLAG_LOSS	= 8,// ...defines loss for numerical optimization (multiple loss tensors add up)
};

struct oiml_init_params {
	// memory pool
	size_t mem_size;// bytes
	void* mem_buffer;// if NULL, memory will be allocated internally
	bool no_alloc;// don't allocate memory for the tensor data
};

static constexpr size_t OIML_MAX_DATA_CHANNELS = 3; 

enum class oiml_data_channel_type : int32_t {
	value,
	block,
	quant,
	scale
};

struct oiml_data_channel {
	oiml_data_channel_type type	= oiml_data_channel_type::value;	// type of channel
	oiml::oiml_representation_types data_type = oiml::oiml_representation_types::float_32;// data type of elements in the channel
	size_t repeat_count = 1;										// apply a single value from this channel to <repeat_count> elements
	size_t strides[4]{};											// num_bytes between elems, rows, matrices, volume.
	void* data = nullptr;											// backend device specific pointer
	char padding[8]{};
};

// n-dimensional tensor
struct oiml_tensor {
	oiml::oiml_representation_types type;

	struct oiml_backend_buffer* buffer;

	int64_t ne[OIML_MAX_DIMS];// number of elements
	size_t nb[OIML_MAX_DIMS];// stride in bytes:
		// nb[0] = oiml_type_size(type)
		// nb[1] = nb[0]   * (ne[0] / oiml_blck_size(type)) + padding
		// nb[i] = nb[i-1] * ne[i-1]

	// compute data
	enum oiml_op op;

	// op params - allocated as int32_t for alignment
	int32_t op_params[OIML_MAX_OP_PARAMS / sizeof(int32_t)];

	int32_t flags;

	oiml_tensor* src[OIML_MAX_SRC];

	// source tensor and offset for views
	oiml_tensor* view_src;
	size_t view_offs;

	void* data;

	char name[OIML_MAX_NAME];

	void* extra;// extra things e.g. for oiml-cuda.cu

	size_t num_channels;
	oiml_data_channel data_channels[OIML_MAX_DATA_CHANNELS];
};

static constexpr size_t OIML_TENSOR_SIZE = sizeof(struct oiml_tensor);

struct oiml_tensor_binding {
	uint32_t dims[4] = { 1, 1, 1, 1 }; 	// dimensions
	uint32_t axis[4] = { 0, 1, 2, 3 };	// axis mapping. used for permuting without changing data

	oiml::oiml_representation_types type = oiml::oiml_representation_types::float_32;// overall tensor type

	size_t num_channels = 0;	// num data channels actually used
	oiml_data_channel data_channels[OIML_MAX_DATA_CHANNELS];

	static oiml_tensor_binding from_tensor(const oiml_tensor* tensor) {
		oiml_tensor_binding bind = {
			.type = tensor->type,
			.num_channels = tensor->num_channels,
		};
		for (size_t i = 0; i < OIML_MAX_DIMS; ++i) {
			bind.dims[i] = static_cast<uint32_t>(tensor->ne[i]);
		}
		for (size_t i = 0; i < OIML_MAX_DATA_CHANNELS; ++i) {
			bind.data_channels[i] = tensor->data_channels[i];
		}
		return bind;
	}
};

// Abort callback
// If not NULL, called before oiml computation
// If it returns true, the computation is aborted
typedef bool (*oiml_abort_callback)(void* data);


//
// GUID
//

// GUID types
typedef uint8_t oiml_guid[16];
typedef oiml_guid* oiml_guid_t;

bool oiml_guid_matches(oiml_guid_t guid_a, oiml_guid_t guid_b);

// misc

int64_t oiml_time_ms();
int64_t oiml_time_us();
int64_t oiml_cycles();
int64_t oiml_cycles_per_ms();

// accepts a UTF-8 path, even on Windows
FILE* oiml_fopen(const char* fname, const char* mode);

void oiml_print_object(const oiml_object* obj);
void oiml_print_objects(const oiml_context* ctx);

int64_t oiml_nelements(const oiml_tensor* tensor);
int64_t oiml_nrows(const oiml_tensor* tensor);
size_t oiml_nbytes(const oiml_tensor* tensor);
size_t oiml_nbytes_pad(const oiml_tensor* tensor);// same as oiml_nbytes() but padded to OIML_MEM_ALIGN

int64_t oiml_blck_size(oiml::oiml_representation_types type);
size_t oiml_type_size(oiml::oiml_representation_types type);// size in bytes for all elements in a block
size_t oiml_row_size(oiml::oiml_representation_types type, int64_t ne);// size in bytes for all elements in a row

const char* oiml_type_name(oiml::oiml_representation_types type);
const char* oiml_op_name(enum oiml_op op);
const char* oiml_op_symbol(enum oiml_op op);

const char* oiml_unary_op_name(enum oiml_unary_op op);
const char* oiml_op_desc(const oiml_tensor* t);// unary or op name

size_t oiml_element_size(const oiml_tensor* tensor);

bool oiml_is_quantized(oiml::oiml_representation_types type);

// TODO: temporary until model loading of oiml examples is refactored
oiml::oiml_representation_types oiml_ftype_to_oiml_type(enum oiml_ftype ftype);

bool oiml_is_transposed(const oiml_tensor* tensor);
bool oiml_is_permuted(const oiml_tensor* tensor);
bool oiml_is_empty(const oiml_tensor* tensor);
bool oiml_is_scalar(const oiml_tensor* tensor);
bool oiml_is_vector(const oiml_tensor* tensor);
bool oiml_is_matrix(const oiml_tensor* tensor);
bool oiml_is_3d(const oiml_tensor* tensor);
int oiml_n_dims(const oiml_tensor* tensor);// returns 1 for scalars

bool oiml_is_contiguous(const oiml_tensor* tensor);
bool oiml_is_contiguous_0(const oiml_tensor* tensor);// same as oiml_is_contiguous()
bool oiml_is_contiguous_1(const oiml_tensor* tensor);// contiguous for dims >= 1
bool oiml_is_contiguous_2(const oiml_tensor* tensor);// contiguous for dims >= 2

bool oiml_are_same_shape(const oiml_tensor* t0, const oiml_tensor* t1);
bool oiml_are_same_stride(const oiml_tensor* t0, const oiml_tensor* t1);

bool oiml_can_repeat(const oiml_tensor* t0, const oiml_tensor* t1);

// use this to compute the memory overhead of a tensor
size_t oiml_tensor_overhead();

bool oiml_validate_row_data(oiml::oiml_representation_types type, const void* data, size_t nbytes);

// main

oiml_context* oiml_init(struct oiml_init_params params);
void oiml_reset(oiml_context* ctx);
void oiml_free(oiml_context* ctx);

size_t oiml_used_mem(const oiml_context* ctx);

bool oiml_get_no_alloc(oiml_context* ctx);
void oiml_set_no_alloc(oiml_context* ctx, bool no_alloc);

void* oiml_get_mem_buffer(const oiml_context* ctx);
size_t oiml_get_mem_size(const oiml_context* ctx);
size_t oiml_get_max_tensor_size(const oiml_context* ctx);

oiml_tensor* oiml_new_tensor(oiml_context* ctx, oiml::oiml_representation_types type, int n_dims, const int64_t* ne);

oiml_tensor* oiml_new_tensor_1d(oiml_context* ctx, oiml::oiml_representation_types type, int64_t ne0);

oiml_tensor* oiml_new_tensor_2d(oiml_context* ctx, oiml::oiml_representation_types type, int64_t ne0, int64_t ne1);

oiml_tensor* oiml_new_tensor_3d(oiml_context* ctx, oiml::oiml_representation_types type, int64_t ne0, int64_t ne1, int64_t ne2);

oiml_tensor* oiml_new_tensor_4d(oiml_context* ctx, oiml::oiml_representation_types type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

void* oiml_new_buffer(oiml_context* ctx, size_t nbytes);

oiml_tensor* oiml_dup_tensor(oiml_context* ctx, const oiml_tensor* src);
oiml_tensor* oiml_view_tensor(oiml_context* ctx, oiml_tensor* src);

// Context tensor enumeration and lookup
oiml_tensor* oiml_get_first_tensor(const oiml_context* ctx);
oiml_tensor* oiml_get_next_tensor(const oiml_context* ctx, oiml_tensor* tensor);
oiml_tensor* oiml_get_tensor(oiml_context* ctx, const char* name);

// Converts a flat index into coordinates
void oiml_unravel_index(const oiml_tensor* tensor, int64_t i, int64_t* i0, int64_t* i1, int64_t* i2, int64_t* i3);

enum oiml_unary_op oiml_get_unary_op(const oiml_tensor* tensor);

void* oiml_get_data(const oiml_tensor* tensor);
float* oiml_get_data_f32(const oiml_tensor* tensor);

const char* oiml_get_name(const oiml_tensor* tensor);
oiml_tensor* oiml_set_name(oiml_tensor* tensor, const char* name);
OIML_ATTRIBUTE_FORMAT(2, 3)
oiml_tensor* oiml_format_name(oiml_tensor* tensor, const char* fmt, ...);

// Tensor flags
void oiml_set_input(oiml_tensor* tensor);
void oiml_set_output(oiml_tensor* tensor);
void oiml_set_param(oiml_context* ctx, oiml_tensor* tensor);
void oiml_set_loss(oiml_tensor* tensor);

//
// operations on tensors with backpropagation
//

oiml_tensor* oiml_dup(oiml_context* ctx, oiml_tensor* a);

// in-place, returns view(a)
oiml_tensor* oiml_dup_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_add(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

oiml_tensor* oiml_add_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

oiml_tensor* oiml_add_cast(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml::oiml_representation_types type);

oiml_tensor* oiml_add1(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

oiml_tensor* oiml_add1_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

// dst = a
// view(dst, nb1, nb2, nb3, offset) += b
// return dst
oiml_tensor* oiml_acc(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset);

oiml_tensor* oiml_acc_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset);

oiml_tensor* oiml_sub(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

oiml_tensor* oiml_sub_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

oiml_tensor* oiml_mul(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

oiml_tensor* oiml_mul_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

oiml_tensor* oiml_div(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

oiml_tensor* oiml_div_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

oiml_tensor* oiml_sqr(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_sqr_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_sqrt(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_sqrt_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_log(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_log_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_sin(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_sin_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_cos(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_cos_inplace(oiml_context* ctx, oiml_tensor* a);

// return scalar
oiml_tensor* oiml_sum(oiml_context* ctx, oiml_tensor* a);

// sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
oiml_tensor* oiml_sum_rows(oiml_context* ctx, oiml_tensor* a);

// mean along rows
oiml_tensor* oiml_mean(oiml_context* ctx, oiml_tensor* a);

// argmax along rows
oiml_tensor* oiml_argmax(oiml_context* ctx, oiml_tensor* a);

// count number of equal elements in a and b
oiml_tensor* oiml_count_equal(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

// if a is the same shape as b, and a is not parameter, return a
// otherwise, return a new tensor: repeat(a) to fit in b
oiml_tensor* oiml_repeat(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

// sums repetitions in a into shape of b
oiml_tensor* oiml_repeat_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

// concat a and b along dim
// used in stable-diffusion
oiml_tensor* oiml_concat(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int dim);

oiml_tensor* oiml_abs(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_abs_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_sgn(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_sgn_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_neg(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_neg_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_step(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_step_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_tanh(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_tanh_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_elu(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_elu_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_relu(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_leaky_relu(oiml_context* ctx, oiml_tensor* a, float negative_slope, bool inplace);

oiml_tensor* oiml_relu_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_sigmoid(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_sigmoid_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_gelu(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_gelu_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_gelu_quick(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_gelu_quick_inplace(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_silu(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_silu_inplace(oiml_context* ctx, oiml_tensor* a);

// a - x
// b - dy
oiml_tensor* oiml_silu_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

// hardswish(x) = x * relu6(x + 3) / 6
oiml_tensor* oiml_hardswish(oiml_context* ctx, oiml_tensor* a);

// hardsigmoid(x) = relu6(x + 3) / 6
oiml_tensor* oiml_hardsigmoid(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_exp(oiml_context* ctx, oiml_tensor* a);

oiml_tensor* oiml_exp_inplace(oiml_context* ctx, oiml_tensor* a);

// normalize along rows
oiml_tensor* oiml_norm(oiml_context* ctx, oiml_tensor* a, float eps);

oiml_tensor* oiml_norm_inplace(oiml_context* ctx, oiml_tensor* a, float eps);

oiml_tensor* oiml_rms_norm(oiml_context* ctx, oiml_tensor* a, float eps);

oiml_tensor* oiml_rms_norm_inplace(oiml_context* ctx, oiml_tensor* a, float eps);

// group normalize along ne0*ne1*n_groups
// used in stable-diffusion
oiml_tensor* oiml_group_norm(oiml_context* ctx, oiml_tensor* a, int n_groups, float eps);

oiml_tensor* oiml_group_norm_inplace(oiml_context* ctx, oiml_tensor* a, int n_groups, float eps);

// a - x
// b - dy
oiml_tensor* oiml_rms_norm_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, float eps);

// A: k columns, n rows => [ne03, ne02, n, k]
// B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
// result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
oiml_tensor* oiml_mul_mat(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

// change the precision of a matrix multiplication
// set to OIML_PREC_F32 for higher precision (useful for phi-2)
void oiml_mul_mat_set_prec(oiml_tensor* a, enum oiml_prec prec);

// indirect matrix multiplication
oiml_tensor* oiml_mul_mat_id(oiml_context* ctx, oiml_tensor* as, oiml_tensor* b, oiml_tensor* ids);

// A: m columns, n rows,
// B: p columns, n rows,
// result is m columns, p rows
oiml_tensor* oiml_out_prod(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

//
// operations on tensors without backpropagation
//

oiml_tensor* oiml_scale(oiml_context* ctx, oiml_tensor* a, float s);

// in-place, returns view(a)
oiml_tensor* oiml_scale_inplace(oiml_context* ctx, oiml_tensor* a, float s);

// b -> view(a,offset,nb1,nb2,3), return modified a
oiml_tensor* oiml_set(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1, size_t nb2, size_t nb3,
	size_t offset);// in bytes

// b -> view(a,offset,nb1,nb2,3), return view(a)
oiml_tensor* oiml_set_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1, size_t nb2, size_t nb3,
	size_t offset);// in bytes

oiml_tensor* oiml_set_1d(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b,
	size_t offset);// in bytes

oiml_tensor* oiml_set_1d_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b,
	size_t offset);// in bytes

// b -> view(a,offset,nb1,nb2,3), return modified a
oiml_tensor* oiml_set_2d(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1,
	size_t offset);// in bytes

// b -> view(a,offset,nb1,nb2,3), return view(a)
oiml_tensor* oiml_set_2d_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, size_t nb1,
	size_t offset);// in bytes

// a -> b, return view(b)
oiml_tensor* oiml_cpy(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

oiml_tensor* oiml_cast(oiml_context* ctx, oiml_tensor* a, oiml::oiml_representation_types type);

// make contiguous
oiml_tensor* oiml_cont(oiml_context* ctx, oiml_tensor* a);

// make contiguous, with new shape
oiml_tensor* oiml_cont_1d(oiml_context* ctx, oiml_tensor* a, int64_t ne0);

oiml_tensor* oiml_cont_2d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1);

oiml_tensor* oiml_cont_3d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2);

oiml_tensor* oiml_cont_4d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

// return view(a), b specifies the new shape
// TODO: when we start computing gradient, make a copy instead of view
oiml_tensor* oiml_reshape(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
oiml_tensor* oiml_reshape_1d(oiml_context* ctx, oiml_tensor* a, int64_t ne0);

oiml_tensor* oiml_reshape_2d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
oiml_tensor* oiml_reshape_3d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2);

oiml_tensor* oiml_reshape_4d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

// offset in bytes
oiml_tensor* oiml_view_1d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, size_t offset);

oiml_tensor* oiml_view_2d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1,
	size_t nb1,// row stride in bytes
	size_t offset);

oiml_tensor* oiml_view_3d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2,
	size_t nb1,// row   stride in bytes
	size_t nb2,// slice stride in bytes
	size_t offset);

oiml_tensor* oiml_view_4d(oiml_context* ctx, oiml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
	size_t nb1,// row   stride in bytes
	size_t nb2,// slice stride in bytes
	size_t nb3, size_t offset);

oiml_tensor* oiml_permute(oiml_context* ctx, oiml_tensor* a, int axis0, int axis1, int axis2, int axis3);

// alias for oiml_permute(ctx, a, 1, 0, 2, 3)
oiml_tensor* oiml_transpose(oiml_context* ctx, oiml_tensor* a);

// supports 3D: a->ne[2] == b->ne[1]
oiml_tensor* oiml_get_rows(oiml_context* ctx,
	oiml_tensor* a,// data
	oiml_tensor* b);// row indices

oiml_tensor* oiml_get_rows_back(oiml_context* ctx,
	oiml_tensor* a,// gradients of oiml_get_rows result
	oiml_tensor* b,// row indices
	oiml_tensor* c);// data for oiml_get_rows, only used for its shape

oiml_tensor* oiml_diag(oiml_context* ctx, oiml_tensor* a);

// set elements above the diagonal to -INF
oiml_tensor* oiml_diag_mask_inf(oiml_context* ctx, oiml_tensor* a, int n_past);

// in-place, returns view(a)
oiml_tensor* oiml_diag_mask_inf_inplace(oiml_context* ctx, oiml_tensor* a, int n_past);

// set elements above the diagonal to 0
oiml_tensor* oiml_diag_mask_zero(oiml_context* ctx, oiml_tensor* a, int n_past);

// in-place, returns view(a)
oiml_tensor* oiml_diag_mask_zero_inplace(oiml_context* ctx, oiml_tensor* a, int n_past);

oiml_tensor* oiml_soft_max(oiml_context* ctx, oiml_tensor* a);

// in-place, returns view(a)
oiml_tensor* oiml_soft_max_inplace(oiml_context* ctx, oiml_tensor* a);

// fused soft_max(a*scale + mask*(ALiBi slope))
// mask is optional
// max_bias = 0.0f for no ALiBi
oiml_tensor* oiml_soft_max_ext(oiml_context* ctx, oiml_tensor* a, oiml_tensor* mask, float scale, float max_bias);

oiml_tensor* oiml_soft_max_ext_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, float scale, float max_bias);

// in-place, returns view(a)
oiml_tensor* oiml_soft_max_ext_back_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, float scale, float max_bias);

// rotary position embedding
// if (mode & 1) - skip n_past elements (NOT SUPPORTED)
// if (mode & OIML_ROPE_TYPE_NEOX) - GPT-NeoX style
//
// b is an int32 vector with size a->ne[2], it contains the positions
oiml_tensor* oiml_rope(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int n_dims, int mode);

// in-place, returns view(a)
oiml_tensor* oiml_rope_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int n_dims, int mode);

// custom RoPE
// c is freq factors (e.g. phi3-128k), (optional)
oiml_tensor* oiml_rope_ext(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale,
	float ext_factor, float attn_factor, float beta_fast, float beta_slow);

oiml_tensor* oiml_rope_multi(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, int n_dims, int sections[4], int mode, int n_ctx_orig, float freq_base,
	float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);

// in-place, returns view(a)
oiml_tensor* oiml_rope_ext_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale,
	float ext_factor, float attn_factor, float beta_fast, float beta_slow);

// compute correction dims for YaRN RoPE scaling
void oiml_rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]);

// rotary position embedding backward, i.e compute dx from dy
// a - dy
oiml_tensor* oiml_rope_ext_back(oiml_context* ctx,
	oiml_tensor* a,// gradients of oiml_rope result
	oiml_tensor* b,// positions
	oiml_tensor* c,// freq factors
	int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);

oiml_tensor* oiml_rope_multi_back(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, int n_dims, int sections[4], int mode, int n_ctx_orig, float freq_base,
	float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);


// clamp
// in-place, returns view(a)
oiml_tensor* oiml_clamp(oiml_context* ctx, oiml_tensor* a, float min, float max);

// im2col
// converts data into a format that effectively results in a convolution when combined with matrix multiplication
oiml_tensor* oiml_im2col(oiml_context* ctx,
	oiml_tensor* a,// convolution kernel
	oiml_tensor* b,// data
	int s0,// stride dimension 0
	int s1,// stride dimension 1
	int p0,// padding dimension 0
	int p1,// padding dimension 1
	int d0,// dilation dimension 0
	int d1,// dilation dimension 1
	bool is_2D, oiml::oiml_representation_types dst_type);

oiml_tensor* oiml_im2col_back(oiml_context* ctx,
	oiml_tensor* a,// convolution kernel
	oiml_tensor* b,// gradient of im2col output
	int64_t* ne,// shape of im2col input
	int s0,// stride dimension 0
	int s1,// stride dimension 1
	int p0,// padding dimension 0
	int p1,// padding dimension 1
	int d0,// dilation dimension 0
	int d1,// dilation dimension 1
	bool is_2D);

oiml_tensor* oiml_conv_1d(oiml_context* ctx,
	oiml_tensor* a,// convolution kernel
	oiml_tensor* b,// data
	int s0,// stride
	int p0,// padding
	int d0);// dilation

// conv_1d with padding = half
// alias for oiml_conv_1d(a, b, s, a->ne[0]/2, d)
oiml_tensor* oiml_conv_1d_ph(oiml_context* ctx,
	oiml_tensor* a,// convolution kernel
	oiml_tensor* b,// data
	int s,// stride
	int d);// dilation

// depthwise
// TODO: this is very likely wrong for some cases! - needs more testing
oiml_tensor* oiml_conv_1d_dw(oiml_context* ctx,
	oiml_tensor* a,// convolution kernel
	oiml_tensor* b,// data
	int s0,// stride
	int p0,// padding
	int d0);// dilation

oiml_tensor* oiml_conv_1d_dw_ph(oiml_context* ctx,
	oiml_tensor* a,// convolution kernel
	oiml_tensor* b,// data
	int s0,// stride
	int d0);// dilation

oiml_tensor* oiml_conv_transpose_1d(oiml_context* ctx,
	oiml_tensor* a,// convolution kernel
	oiml_tensor* b,// data
	int s0,// stride
	int p0,// padding
	int d0);// dilation

oiml_tensor* oiml_conv_2d(oiml_context* ctx,
	oiml_tensor* a,// convolution kernel
	oiml_tensor* b,// data
	int s0,// stride dimension 0
	int s1,// stride dimension 1
	int p0,// padding dimension 0
	int p1,// padding dimension 1
	int d0,// dilation dimension 0
	int d1);// dilation dimension 1

// kernel size is a->ne[0] x a->ne[1]
// stride is equal to kernel size
// padding is zero
// example:
// a:     16   16    3  768
// b:   1024 1024    3    1
// res:   64   64  768    1
// used in sam
oiml_tensor* oiml_conv_2d_sk_p0(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

// kernel size is a->ne[0] x a->ne[1]
// stride is 1
// padding is half
// example:
// a:      3    3    256  256
// b:     64   64    256    1
// res:   64   64    256    1
// used in sam
oiml_tensor* oiml_conv_2d_s1_ph(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b);

// depthwise
oiml_tensor* oiml_conv_2d_dw(oiml_context* ctx,
	oiml_tensor* a,// convolution kernel
	oiml_tensor* b,// data
	int s0,// stride dimension 0
	int s1,// stride dimension 1
	int p0,// padding dimension 0
	int p1,// padding dimension 1
	int d0,// dilation dimension 0
	int d1);// dilation dimension 1

oiml_tensor* oiml_conv_transpose_2d_p0(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, int stride);

enum oiml_op_pool {
	OIML_OP_POOL_MAX,
	OIML_OP_POOL_AVG,
	OIML_OP_POOL_COUNT,
};

oiml_tensor* oiml_pool_1d(oiml_context* ctx, oiml_tensor* a, enum oiml_op_pool op,
	int k0,// kernel size
	int s0,// stride
	int p0);// padding

// the result will have 2*p0 padding for the first dimension
// and 2*p1 padding for the second dimension
oiml_tensor* oiml_pool_2d(oiml_context* ctx, oiml_tensor* a, enum oiml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1);

oiml_tensor* oiml_pool_2d_back(oiml_context* ctx, oiml_tensor* a,
	oiml_tensor* af,// "a"/input used in forward pass
	enum oiml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1);

// nearest interpolate
// multiplies ne0 and ne1 by scale factor
// used in stable-diffusion
oiml_tensor* oiml_upscale(oiml_context* ctx, oiml_tensor* a, int scale_factor);

// nearest interpolate
// nearest interpolate to specified dimensions
// used in tortoise.cpp
oiml_tensor* oiml_upscale_ext(oiml_context* ctx, oiml_tensor* a, int ne0, int ne1, int ne2, int ne3);

// pad each dimension with zeros: [x, ..., x] -> [x, ..., x, 0, ..., 0]
oiml_tensor* oiml_pad(oiml_context* ctx, oiml_tensor* a, int p0, int p1, int p2, int p3);

// pad each dimension with reflection: [a, b, c, d] -> [b, a, b, c, d, c]
oiml_tensor* oiml_pad_reflect_1d(oiml_context* ctx, oiml_tensor* a, int p0, int p1);

// Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
// timesteps: [N,]
// return: [N, dim]
oiml_tensor* oiml_timestep_embedding(oiml_context* ctx, oiml_tensor* timesteps, int dim, int max_period);

// sort rows
enum oiml_sort_order {
	OIML_SORT_ORDER_ASC,
	OIML_SORT_ORDER_DESC,
};

oiml_tensor* oiml_argsort(oiml_context* ctx, oiml_tensor* a, enum oiml_sort_order order);

oiml_tensor* oiml_arange(oiml_context* ctx, float start, float stop, float step);

// top k elements per row
oiml_tensor* oiml_top_k(oiml_context* ctx, oiml_tensor* a, int k);

#define OIML_KQ_MASK_PAD 32

// q:    [n_embd, n_batch,     n_head,    1]
// k:    [n_embd, n_kv,        n_head_kv, 1]
// v:    [n_embd, n_kv,        n_head_kv, 1] !! not transposed !!
// mask: [n_kv,   n_batch_pad, 1,         1] !! n_batch_pad = OIML_PAD(n_batch, OIML_KQ_MASK_PAD) !!
// res:  [n_embd, n_head,      n_batch,   1] !! permuted !!
oiml_tensor* oiml_flash_attn_ext(oiml_context* ctx, oiml_tensor* q, oiml_tensor* k, oiml_tensor* v, oiml_tensor* mask, float scale, float max_bias, float logit_softcap);

void oiml_flash_attn_ext_set_prec(oiml_tensor* a, enum oiml_prec prec);

enum oiml_prec oiml_flash_attn_ext_get_prec(const oiml_tensor* a);

// TODO: needs to be adapted to oiml_flash_attn_ext
oiml_tensor* oiml_flash_attn_back(oiml_context* ctx, oiml_tensor* q, oiml_tensor* k, oiml_tensor* v, oiml_tensor* d, bool masked);

oiml_tensor* oiml_ssm_conv(oiml_context* ctx, oiml_tensor* sx, oiml_tensor* c);

oiml_tensor* oiml_ssm_scan(oiml_context* ctx, oiml_tensor* s, oiml_tensor* x, oiml_tensor* dt, oiml_tensor* A, oiml_tensor* B, oiml_tensor* C);

// partition into non-overlapping windows with padding if needed
// example:
// a:   768   64   64    1
// w:    14
// res: 768   14   14    25
// used in sam
oiml_tensor* oiml_win_part(oiml_context* ctx, oiml_tensor* a, int w);

// reverse of oiml_win_part
// used in sam
oiml_tensor* oiml_win_unpart(oiml_context* ctx, oiml_tensor* a, int w0, int h0, int w);

oiml_tensor* oiml_unary(oiml_context* ctx, oiml_tensor* a, enum oiml_unary_op op);

oiml_tensor* oiml_unary_inplace(oiml_context* ctx, oiml_tensor* a, enum oiml_unary_op op);

// used in sam
oiml_tensor* oiml_get_rel_pos(oiml_context* ctx, oiml_tensor* a, int qh, int kh);

// used in sam
oiml_tensor* oiml_add_rel_pos(oiml_context* ctx, oiml_tensor* a, oiml_tensor* pw, oiml_tensor* ph);

oiml_tensor* oiml_add_rel_pos_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* pw, oiml_tensor* ph);

oiml_tensor* oiml_rwkv_wkv6(oiml_context* ctx, oiml_tensor* k, oiml_tensor* v, oiml_tensor* r, oiml_tensor* tf, oiml_tensor* td, oiml_tensor* state);

oiml_tensor* oiml_gated_linear_attn(oiml_context* ctx, oiml_tensor* k, oiml_tensor* v, oiml_tensor* q, oiml_tensor* g, oiml_tensor* state, float scale);

// custom operators

typedef void (*oiml_unary_op_f32_t)(const int, float*, const float*);
typedef void (*oiml_binary_op_f32_t)(const int, float*, const float*, const float*);

typedef void (*oiml_custom1_op_f32_t)(oiml_tensor*, const oiml_tensor*);
typedef void (*oiml_custom2_op_f32_t)(oiml_tensor*, const oiml_tensor*, const oiml_tensor*);
typedef void (*oiml_custom3_op_f32_t)(oiml_tensor*, const oiml_tensor*, const oiml_tensor*, const oiml_tensor*);

// custom operators v2

typedef void (*oiml_custom1_op_t)(oiml_tensor* dst, const oiml_tensor* a, int ith, int nth, void* userdata);
typedef void (*oiml_custom2_op_t)(oiml_tensor* dst, const oiml_tensor* a, const oiml_tensor* b, int ith, int nth, void* userdata);
typedef void (*oiml_custom3_op_t)(oiml_tensor* dst, const oiml_tensor* a, const oiml_tensor* b, const oiml_tensor* c, int ith, int nth, void* userdata);

#define OIML_N_TASKS_MAX (-1)
// n_tasks == OIML_N_TASKS_MAX means to use max number of tasks

oiml_tensor* oiml_map_custom1(oiml_context* ctx, oiml_tensor* a, oiml_custom1_op_t fun, int n_tasks, void* userdata);

oiml_tensor* oiml_map_custom1_inplace(oiml_context* ctx, oiml_tensor* a, oiml_custom1_op_t fun, int n_tasks, void* userdata);

oiml_tensor* oiml_map_custom2(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_custom2_op_t fun, int n_tasks, void* userdata);

oiml_tensor* oiml_map_custom2_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_custom2_op_t fun, int n_tasks, void* userdata);

oiml_tensor* oiml_map_custom3(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, oiml_custom3_op_t fun, int n_tasks, void* userdata);

oiml_tensor* oiml_map_custom3_inplace(oiml_context* ctx, oiml_tensor* a, oiml_tensor* b, oiml_tensor* c, oiml_custom3_op_t fun, int n_tasks, void* userdata);

// loss function

oiml_tensor* oiml_cross_entropy_loss(oiml_context* ctx,
	oiml_tensor* a,// logits
	oiml_tensor* b);// labels

oiml_tensor* oiml_cross_entropy_loss_back(oiml_context* ctx,
	oiml_tensor* a,// logits
	oiml_tensor* b,// labels
	oiml_tensor* c);// gradients of cross_entropy_loss result

// AdamW optimizer step
// Paper: https://arxiv.org/pdf/1711.05101v3.pdf
// PyTorch: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
oiml_tensor* oiml_opt_step_adamw(oiml_context* ctx, oiml_tensor* a, oiml_tensor* grad, oiml_tensor* m, oiml_tensor* v,
	oiml_tensor* adamw_params);// parameters such a the learning rate

//
// automatic differentiation
//

void oiml_build_forward_expand(oiml_cgraph* cgraph, oiml_tensor* tensor);
void oiml_build_backward_expand(oiml_context* ctx_static,// context for static gradients (loss + gradient accumulation)
	oiml_context* ctx_compute,// context for gradient computation
	oiml_cgraph* cgraph,
	bool accumulate);// whether or not gradients should be accumulated, requires static allocation of tensors in ctx_static

// graph allocation in a context
oiml_cgraph* oiml_new_graph(oiml_context* ctx);// size = OIML_DEFAULT_GRAPH_SIZE, grads = false
oiml_cgraph* oiml_new_graph_custom(oiml_context* ctx, size_t size, bool grads);
oiml_cgraph* oiml_graph_dup(oiml_context* ctx, oiml_cgraph* cgraph);
void oiml_graph_cpy(oiml_cgraph* src, oiml_cgraph* dst);
void oiml_graph_reset(oiml_cgraph* cgraph);// set regular grads + optimizer momenta to 0, set loss grad to 1
void oiml_graph_clear(oiml_cgraph* cgraph);

int oiml_graph_size(oiml_cgraph* cgraph);
oiml_tensor* oiml_graph_node(oiml_cgraph* cgraph, int i);// if i < 0, returns nodes[n_nodes + i]
oiml_tensor** oiml_graph_nodes(oiml_cgraph* cgraph);
int oiml_graph_n_nodes(oiml_cgraph* cgraph);

void oiml_graph_add_node(oiml_cgraph* cgraph, oiml_tensor* tensor);

size_t oiml_graph_overhead();
size_t oiml_graph_overhead_custom(size_t size, bool grads);

oiml_tensor* oiml_graph_get_tensor(const oiml_cgraph* cgraph, const char* name);
oiml_tensor* oiml_graph_get_grad(const oiml_cgraph* cgraph, const oiml_tensor* node);
oiml_tensor* oiml_graph_get_grad_acc(const oiml_cgraph* cgraph, const oiml_tensor* node);

void oiml_graph_export(const oiml_cgraph* cgraph, const char* fname);
oiml_cgraph* oiml_graph_import(const char* fname, oiml_context** ctx_data, oiml_context** ctx_eval);

// print info and performance information for the graph
void oiml_graph_print(const oiml_cgraph* cgraph);

// dump the graph into a file using the dot format
void oiml_graph_dump_dot(const oiml_cgraph* gb, const oiml_cgraph* gf, const char* filename);

// TODO these functions were sandwiched in the old optimization interface, is there a better place for them?
typedef void (*oiml_log_callback)(enum oiml_log_level level, const char* text, void* user_data);

// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
void oiml_log_set(oiml_log_callback log_callback, void* user_data);

oiml_tensor* oiml_set_zero(oiml_tensor* tensor);

//
// quantization
//

// some quantization type cannot be used without an importance matrix
bool oiml_quantize_requires_imatrix(oiml::oiml_representation_types type);

// calls oiml_quantize_init internally (i.e. can allocate memory)
size_t oiml_quantize_chunk(oiml::oiml_representation_types type, const float* src, oiml_tensor_binding* dst, int64_t offset, int64_t start, int64_t nrows, int64_t n_per_row, const float* imatrix);

typedef void (*oiml_to_float_t)(const oiml_tensor_binding* __restrict x, float* __restrict y, int64_t offset, int64_t k);
typedef void (*oiml_from_float_t)(const float* __restrict x, oiml_tensor_binding* __restrict y, int64_t offset, int64_t k);
typedef void (*oiml_vec_dot_t)(int n, float* __restrict s, size_t bs, const oiml_tensor_binding* __restrict x, size_t x_off, size_t bx, const oiml_tensor_binding* __restrict y,
	size_t y_off, size_t by, int nrc);

struct oiml_type_traits {
	oiml_from_float_t from_float;
	oiml_vec_dot_t vec_dot;
	oiml::oiml_representation_types vec_dot_type;
	int64_t nrows;// number of rows to process simultaneously
	const char* type_name;
	int64_t blck_size;
	int64_t blck_size_interleave;// interleave elements in blocks
	size_t type_size;
	bool is_quantized;
	oiml_to_float_t to_float;
	oiml_from_float_t from_float_ref;
};

const oiml_type_traits* oiml_get_type_traits(oiml::oiml_representation_types type);

// oiml threadpool
// TODO: currently, only a few functions are in the base oiml API, while the rest are in the CPU backend
// the goal should be to create an API that other backends can use move everything to the oiml base

// scheduling priorities
enum oiml_sched_priority { OIML_SCHED_PRIO_NORMAL, OIML_SCHED_PRIO_MEDIUM, OIML_SCHED_PRIO_HIGH, OIML_SCHED_PRIO_REALTIME };

// threadpool params
// Use oiml_threadpool_params_default() or oiml_threadpool_params_init() to populate the defaults
struct oiml_threadpool_params {
	bool cpumask[OIML_MAX_N_THREADS];// mask of cpu cores (all-zeros means use default affinity settings)
	int n_threads;// number of threads
	enum oiml_sched_priority prio;// thread priority
	uint32_t poll;// polling level (0 - no polling, 100 - aggressive polling)
	bool strict_cpu;// strict cpu placement
	bool paused;// start in paused state
};

struct oiml_threadpool;// forward declaration, see oiml.c

typedef oiml_threadpool* oiml_threadpool_t;

struct oiml_threadpool_params oiml_threadpool_params_default(int n_threads);
void oiml_threadpool_params_init(oiml_threadpool_params* p, int n_threads);
bool oiml_threadpool_params_match(const oiml_threadpool_params* p0, const oiml_threadpool_params* p1);
