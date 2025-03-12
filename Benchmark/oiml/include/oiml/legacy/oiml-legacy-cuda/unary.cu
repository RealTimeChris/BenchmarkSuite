#include "unary.cuh"

static __device__ __forceinline__ float op_abs(float x) {
	return fabsf(x);
}

static __device__ __forceinline__ float op_sgn(float x) {
	return (x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f)));
}

static __device__ __forceinline__ float op_neg(float x) {
	return -x;
}

static __device__ __forceinline__ float op_step(float x) {
	return x > 0.0f;
}

static __device__ __forceinline__ float op_gelu(float x) {
	const float GELU_COEF_A	   = 0.044715f;
	const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

	return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

static __device__ __forceinline__ float op_gelu_quick(float x) {
	const float GELU_QUICK_COEF = -1.702f;

	return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

static __device__ __forceinline__ float op_silu(float x) {
	return x / (1.0f + expf(-x));
}

static __device__ __forceinline__ float op_tanh(float x) {
	return tanhf(x);
}

static __device__ __forceinline__ float op_relu(float x) {
	return fmaxf(x, 0);
}

static __device__ __forceinline__ float op_sigmoid(float x) {
	return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float op_hardsigmoid(float x) {
	return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_hardswish(float x) {
	return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_exp(float x) {
	return expf(x);
}

static __device__ __forceinline__ float op_sqr(float x) {
	return x * x;
}

static __device__ __forceinline__ float op_sqrt(float x) {
	return sqrtf(x);
}

static __device__ __forceinline__ float op_sin(float x) {
	return sinf(x);
}

static __device__ __forceinline__ float op_cos(float x) {
	return cosf(x);
}

static __device__ __forceinline__ float op_log(float x) {
	return logf(x);
}

template<float (*op)(float), typename T> static __global__ void unary_op_kernel(const T* x, T* dst, const int k) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= k) {
		return;
	}

	dst[i] = ( T )op(( float )x[i]);
}

template<float (*op)(float), typename T> static void unary_cuda(const T* x, T* dst, const int k, cudaStream_t stream) {
	const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
	unary_op_kernel<op><<<num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template<float (*op)(float)> void oiml_cuda_op_unary(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* src0 = dst->src[0];
	const void* src0_d		= src0->data;
	void* dst_d				= dst->data;
	cudaStream_t stream		= ctx.stream();

	OIML_ASSERT(oiml_is_contiguous(src0));

	OIML_ASSERT(src0->type == oiml::oiml_representation_types::float_32 || src0->type == oiml::oiml_representation_types::float_16);
	OIML_ASSERT(dst->type == oiml::oiml_representation_types::float_32 || dst->type == oiml::oiml_representation_types::float_16);
	OIML_ASSERT(src0->type == dst->type);

	if (src0->type == oiml::oiml_representation_types::float_16) {
		unary_cuda<op>(( const half* )src0_d, ( half* )dst_d, oiml_nelements(src0), stream);
	} else {
		unary_cuda<op>(( const float* )src0_d, ( float* )dst_d, oiml_nelements(src0), stream);
	}
}

void oiml_cuda_op_abs(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_abs>(ctx, dst);
}

void oiml_cuda_op_sgn(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_sgn>(ctx, dst);
}

void oiml_cuda_op_neg(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_neg>(ctx, dst);
}

void oiml_cuda_op_step(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_step>(ctx, dst);
}

void oiml_cuda_op_gelu(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_gelu>(ctx, dst);
}

void oiml_cuda_op_gelu_quick(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_gelu_quick>(ctx, dst);
}

void oiml_cuda_op_silu(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_silu>(ctx, dst);
}

void oiml_cuda_op_tanh(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_tanh>(ctx, dst);
}

void oiml_cuda_op_relu(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_relu>(ctx, dst);
}

void oiml_cuda_op_sigmoid(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_sigmoid>(ctx, dst);
}

void oiml_cuda_op_hardsigmoid(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_hardsigmoid>(ctx, dst);
}

void oiml_cuda_op_hardswish(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_hardswish>(ctx, dst);
}

void oiml_cuda_op_exp(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_exp>(ctx, dst);
}

void oiml_cuda_op_sqr(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_sqr>(ctx, dst);
}

void oiml_cuda_op_sqrt(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_sqrt>(ctx, dst);
}

void oiml_cuda_op_sin(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_sin>(ctx, dst);
}

void oiml_cuda_op_cos(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_cos>(ctx, dst);
}

void oiml_cuda_op_log(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_cuda_op_unary<op_log>(ctx, dst);
}

/* silu_back */

static __device__ __forceinline__ float op_silu_back(float grad, float x) {
	const float s = 1.0f / (1.0f + expf(-x));
	return grad * s * (1.0f + x * (1.0f - s));
}

template<class T> static __global__ void silu_back_kernel(const T* grad, const T* xf, T* dst, const int k) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= k) {
		return;
	}

	dst[i] = ( T )op_silu_back(( float )grad[i], ( float )xf[i]);
}

template<class T> static void silu_back_cuda(const T* grad, const T* x, T* dst, const int k, cudaStream_t stream) {
	const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
	silu_back_kernel<<<num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream>>>(grad, x, dst, k);
}

void oiml_cuda_op_silu_back(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* src0 = dst->src[0];// input from forward pass
	const oiml_tensor* src1 = dst->src[1];// grads of forward pass output

	const float* src0_d = ( const float* )src0->data;
	const float* src1_d = ( const float* )src1->data;
	float* dst_d		= ( float* )dst->data;

	cudaStream_t stream = ctx.stream();

	OIML_ASSERT(oiml_is_contiguous(src0));

	OIML_ASSERT(src0->type == oiml::oiml_representation_types::float_32 || src0->type == oiml::oiml_representation_types::float_16);
	OIML_ASSERT(dst->type == oiml::oiml_representation_types::float_32 || dst->type == oiml::oiml_representation_types::float_16);
	OIML_ASSERT(src0->type == dst->type);

	if (src0->type == oiml::oiml_representation_types::float_16) {
		silu_back_cuda(( const half* )src0_d, ( const half* )src1_d, ( half* )dst_d, oiml_nelements(src0), stream);
	} else {
		silu_back_cuda(( const float* )src0_d, ( const float* )src1_d, ( float* )dst_d, oiml_nelements(src0), stream);
	}
}

/* leaky relu */

static __device__ __forceinline__ float op_leaky_relu(float x, const float negative_slope) {
	return fmaxf(x, 0) + fminf(x, 0.0f) * negative_slope;
}

template<class T> static __global__ void leaky_relu_kernel(const T* x, T* dst, const int k, const float negative_slope) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= k) {
		return;
	}

	dst[i] = ( T )op_leaky_relu(( float )x[i], negative_slope);
}

template<class T> static void leaky_relu_cuda(const T* x, T* dst, const int k, const float negative_slope, cudaStream_t stream) {
	const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
	leaky_relu_kernel<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

void oiml_cuda_op_leaky_relu(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* src0 = dst->src[0];
	const void* src0_d		= src0->data;
	void* dst_d				= dst->data;
	cudaStream_t stream		= ctx.stream();

	OIML_ASSERT(oiml_is_contiguous(src0));

	OIML_ASSERT(src0->type == oiml::oiml_representation_types::float_32 || src0->type == oiml::oiml_representation_types::float_16);
	OIML_ASSERT(dst->type == oiml::oiml_representation_types::float_32 || dst->type == oiml::oiml_representation_types::float_16);
	OIML_ASSERT(src0->type == dst->type);

	float negative_slope;
	memcpy(&negative_slope, dst->op_params, sizeof(float));

	if (src0->type == oiml::oiml_representation_types::float_16) {
		leaky_relu_cuda(( const half* )src0_d, ( half* )dst_d, oiml_nelements(src0), negative_slope, stream);
	} else {
		leaky_relu_cuda(( const float* )src0_d, ( float* )dst_d, oiml_nelements(src0), negative_slope, stream);
	}
}
