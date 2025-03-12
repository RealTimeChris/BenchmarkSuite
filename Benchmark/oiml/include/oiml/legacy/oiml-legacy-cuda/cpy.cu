#include "cpy.cuh"
#include "dequantize.cuh"

typedef void (*cpy_kernel_t)(const char* cx, char* cdst);

static __device__ void cpy_1_f32_f32(const char* cxi, char* cdsti) {
	const float* xi = ( const float* )cxi;
	float* dsti		= ( float* )cdsti;

	*dsti = *xi;
}

static __device__ void cpy_1_f32_f16(const char* cxi, char* cdsti) {
	const float* xi = ( const float* )cxi;
	half* dsti		= ( half* )cdsti;

	*dsti = __float2half(*xi);
}

static __device__ void cpy_1_f16_f16(const char* cxi, char* cdsti) {
	const half* xi = ( const half* )cxi;
	half* dsti	   = ( half* )cdsti;

	*dsti = *xi;
}

static __device__ void cpy_1_f16_f32(const char* cxi, char* cdsti) {
	const half* xi = ( const half* )cxi;
	float* dsti	   = ( float* )cdsti;

	*dsti = *xi;
}

template<cpy_kernel_t cpy_1> static __global__ void cpy_f32_f16(const char* cx, char* cdst, const int ne, const int ne00, const int ne01, const int ne02, const int nb00,
	const int nb01, const int nb02, const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13) {
	const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= ne) {
		return;
	}

	// determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
	// then combine those indices with the corresponding byte offsets to get the total offsets
	const int64_t i03	   = i / (ne00 * ne01 * ne02);
	const int64_t i02	   = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
	const int64_t i01	   = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
	const int64_t i00	   = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
	const int64_t x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

	const int64_t i13		 = i / (ne10 * ne11 * ne12);
	const int64_t i12		 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
	const int64_t i11		 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
	const int64_t i10		 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
	const int64_t dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

	cpy_1(cx + x_offset, cdst + dst_offset);
}

static __device__ void cpy_blck_f32_q8_0(const char* cxi, char* cdsti) {
	const float* xi						   = ( const float* )cxi;
	oiml::block_q8_0<oiml_half_cuda>* dsti = ( oiml::block_q8_0<oiml_half_cuda>* )cdsti;

	float amax = 0.0f;// absolute max

	for (int j = 0; j < oiml::Q_SIZE; j++) {
		const float v = xi[j];
		amax		  = fmaxf(amax, fabsf(v));
	}

	const float d  = amax / ((1 << 7) - 1);
	const float id = d ? 1.0f / d : 0.0f;

	dsti->d = d;

	for (int j = 0; j < oiml::Q_SIZE; ++j) {
		const float x0 = xi[j] * id;

		dsti->qs[j] = roundf(x0);
	}
}

static __device__ void cpy_blck_q8_0_f32(const char* cxi, char* cdsti) {
	float* cdstf = ( float* )(cdsti);

#pragma unroll
	for (int j = 0; j < oiml::Q_SIZE; j += 2) {
		dfloat2 dq;
		dequantize_q8_0(cxi, 0, j, dq);
		*(cdstf + j)	 = dq.x;
		*(cdstf + j + 1) = dq.y;
	}
}

template<dequantize_kernel_t dequant, int qk> static __device__ void cpy_blck_q_f32(const char* cxi, char* cdsti) {
	float* cdstf = ( float* )(cdsti);

#pragma unroll
	for (int j = 0; j < qk / 2; j++) {
		dfloat2 dq;
		dequant(cxi, 0, j, dq);
		*(cdstf + j)		  = dq.x;
		*(cdstf + j + qk / 2) = dq.y;
	}
}

static __device__ __forceinline__ int best_index_int8_cuda(int n, const int8_t* val, float x) {
	if (x <= val[0])
		return 0;
	if (x >= val[n - 1])
		return n - 1;
	int ml = 0, mu = n - 1;
	while (mu - ml > 1) {
		int mav = (ml + mu) / 2;
		if (x < val[mav])
			mu = mav;
		else
			ml = mav;
	}
	return x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

template<cpy_kernel_t cpy_blck, int qk> static __global__ void cpy_f32_q(const char* cx, char* cdst, const int ne, const int ne00, const int ne01, const int ne02, const int nb00,
	const int nb01, const int nb02, const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13) {
	const int i = (blockDim.x * blockIdx.x + threadIdx.x) * qk;

	if (i >= ne) {
		return;
	}

	const int i03	   = i / (ne00 * ne01 * ne02);
	const int i02	   = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
	const int i01	   = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
	const int i00	   = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
	const int x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

	const int i13		 = i / (ne10 * ne11 * ne12);
	const int i12		 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
	const int i11		 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
	const int i10		 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
	const int dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

	cpy_blck(cx + x_offset, cdst + dst_offset);
}

template<cpy_kernel_t cpy_blck, int qk> static __global__ void cpy_q_f32(const char* cx, char* cdst, const int ne, const int ne00, const int ne01, const int ne02, const int nb00,
	const int nb01, const int nb02, const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13) {
	const int i = (blockDim.x * blockIdx.x + threadIdx.x) * qk;

	if (i >= ne) {
		return;
	}

	const int i03	   = i / (ne00 * ne01 * ne02);
	const int i02	   = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
	const int i01	   = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
	const int i00	   = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
	const int x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

	const int i13		 = i / (ne10 * ne11 * ne12);
	const int i12		 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
	const int i11		 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
	const int i10		 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
	const int dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

	cpy_blck(cx + x_offset, cdst + dst_offset);
}

static void oiml_cpy_f16_f32_cuda(const char* cx, char* cdst, const int ne, const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
	const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {
	const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
	cpy_f32_f16<cpy_1_f16_f32><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void oiml_cpy_f32_f32_cuda(const char* cx, char* cdst, const int ne, const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
	const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {
	const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
	cpy_f32_f16<cpy_1_f32_f32><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void oiml_cpy_f32_f16_cuda(const char* cx, char* cdst, const int ne, const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
	const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {
	const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
	cpy_f32_f16<cpy_1_f32_f16><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void oiml_cpy_f32_q8_0_cuda(const char* cx, char* cdst, const int ne, const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
	const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {
	OIML_ASSERT(ne % oiml::Q_SIZE == 0);
	const int num_blocks = ne / oiml::Q_SIZE;
	cpy_f32_q<cpy_blck_f32_q8_0, oiml::Q_SIZE><<<num_blocks, 1, 0, stream>>>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void oiml_cpy_q8_0_f32_cuda(const char* cx, char* cdst, const int ne, const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
	const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {
	const int num_blocks = ne;
	cpy_q_f32<cpy_blck_q8_0_f32, oiml::Q_SIZE><<<num_blocks, 1, 0, stream>>>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

void oiml_cuda_cpy(oiml_backend_cuda_context& ctx, const oiml_tensor* src0, oiml_tensor* src1) {
	const int64_t ne = oiml_nelements(src0);
	OIML_ASSERT(ne == oiml_nelements(src1));

	OIML_ASSERT(oiml_nbytes(src0) <= INT_MAX);
	OIML_ASSERT(oiml_nbytes(src1) <= INT_MAX);

	const int64_t ne00 = src0->ne[0];
	const int64_t ne01 = src0->ne[1];
	const int64_t ne02 = src0->ne[2];

	//OIML_ASSERT(src0->ne[3] == 1);

	const int64_t nb00 = src0->nb[0];
	const int64_t nb01 = src0->nb[1];
	const int64_t nb02 = src0->nb[2];
	const int64_t nb03 = src0->nb[3];

	const int64_t ne10 = src1->ne[0];
	const int64_t ne11 = src1->ne[1];
	const int64_t ne12 = src1->ne[2];

	//OIML_ASSERT(src1->ne[3] == 1);

	const int64_t nb10 = src1->nb[0];
	const int64_t nb11 = src1->nb[1];
	const int64_t nb12 = src1->nb[2];
	const int64_t nb13 = src1->nb[3];

	cudaStream_t main_stream = ctx.stream();

	char* src0_ddc = ( char* )src0->data;
	char* src1_ddc = ( char* )src1->data;

	if (src0->type == src1->type && oiml_is_contiguous(src0) && oiml_is_contiguous(src1)) {
		OIML_ASSERT(oiml_nbytes(src0) == oiml_nbytes(src1));
		CUDA_CHECK(cudaMemcpyAsync(src1_ddc, src0_ddc, oiml_nbytes(src0), cudaMemcpyDeviceToDevice, main_stream));
	} else if (src0->type == oiml::oiml_representation_types::float_32 && src1->type == oiml::oiml_representation_types::float_32) {
		oiml_cpy_f32_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
	} else if (src0->type == oiml::oiml_representation_types::float_32 && src1->type == oiml::oiml_representation_types::float_16) {
		oiml_cpy_f32_f16_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
	} else if (src0->type == oiml::oiml_representation_types::float_32 && src1->type == oiml::oiml_representation_types::q8_0) {
		oiml_cpy_f32_q8_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
	} else if (src0->type == oiml::oiml_representation_types::q8_0 && src1->type == oiml::oiml_representation_types::float_32) {
		oiml_cpy_q8_0_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
	} else if (src0->type == oiml::oiml_representation_types::float_16 && src1->type == oiml::oiml_representation_types::float_32) {
		oiml_cpy_f16_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
	} else {
		OIML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__, oiml_type_name(src0->type), oiml_type_name(src1->type));
	}
}

void oiml_cuda_dup(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* src0 = dst->src[0];
	oiml_cuda_cpy(ctx, src0, dst);
}

void* oiml_cuda_cpy_fn(const oiml_tensor* src0, oiml_tensor* src1) {
	if (src0->type == src1->type && oiml_is_contiguous(src0) && oiml_is_contiguous(src1)) {
		return nullptr;
	} else if (src0->type == oiml::oiml_representation_types::float_32 && src1->type == oiml::oiml_representation_types::float_32) {
		return ( void* )cpy_f32_f16<cpy_1_f32_f32>;
	} else if (src0->type == oiml::oiml_representation_types::float_32 && src1->type == oiml::oiml_representation_types::float_16) {
		return ( void* )cpy_f32_f16<cpy_1_f32_f16>;
	} else if (src0->type == oiml::oiml_representation_types::float_32 && src1->type == oiml::oiml_representation_types::q8_0) {
		return ( void* )cpy_f32_q<cpy_blck_f32_q8_0, oiml::Q_SIZE>;
	} else if (src0->type == oiml::oiml_representation_types::q8_0 && src1->type == oiml::oiml_representation_types::float_32) {
		return ( void* )cpy_q_f32<cpy_blck_q8_0_f32, oiml::Q_SIZE>;
	} else if (src0->type == oiml::oiml_representation_types::float_16 && src1->type == oiml::oiml_representation_types::float_16) {
		return ( void* )cpy_f32_f16<cpy_1_f32_f16>;
	} else if (src0->type == oiml::oiml_representation_types::float_16 && src1->type == oiml::oiml_representation_types::float_32) {
		return ( void* )cpy_f32_f16<cpy_1_f16_f32>;
	} else {
		OIML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__, oiml_type_name(src0->type), oiml_type_name(src1->type));
	}
}
