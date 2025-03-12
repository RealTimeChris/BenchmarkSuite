#include "common.cuh"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256

void oiml_cuda_op_abs(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_sgn(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_neg(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_step(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_gelu(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_silu(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_silu_back(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_gelu_quick(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_tanh(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_relu(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_sigmoid(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_hardsigmoid(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_exp(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_hardswish(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_leaky_relu(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_sqr(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_sqrt(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_sin(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_cos(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_log(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
