#include "getrows.cuh"
#include "dequantize.cuh"

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t> static __global__ void k_get_rows(const void* __restrict__ src0, const int32_t* __restrict__ src1,
	dst_t* __restrict__ dst, const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
	/*const int64_t ne10, const int64_t ne11,*/ const int64_t ne12, /*const int64_t ne13,*/
	/*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
	/*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03, const size_t s10, const size_t s11, const size_t s12 /*, const size_t s13*/) {
	const int i00 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	const int i10 = blockDim.y * blockIdx.y + threadIdx.y;
	const int i11 = (blockIdx.z * blockDim.z + threadIdx.z) / ne12;
	const int i12 = (blockIdx.z * blockDim.z + threadIdx.z) % ne12;

	if (i00 >= ne00) {
		return;
	}

	const int i01 = src1[i10 * s10 + i11 * s11 + i12 * s12];

	dst_t* dst_row		 = dst + i10 * s1 + i11 * s2 + i12 * s3;
	const void* src0_row = ( const char* )src0 + i01 * nb01 + i11 * nb02 + i12 * nb03;

	const int ib	   = i00 / qk;// block index
	const int iqs	   = (i00 % qk) / qr;// quant index
	const int iybs	   = i00 - i00 % qk;// dst block start index
	const int y_offset = qr == 1 ? 1 : qk / 2;

	// dequantize
	dfloat2 v;
	dequantize_kernel(src0_row, ib, iqs, v);

	dst_row[iybs + iqs + 0]		   = v.x;
	dst_row[iybs + iqs + y_offset] = v.y;
}

template<typename src0_t, typename dst_t> static __global__ void k_get_rows_float(const src0_t* __restrict__ src0, const int32_t* __restrict__ src1, dst_t* __restrict__ dst,
	const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
	/*const int64_t ne10, const int64_t ne11,*/ const int64_t ne12, /*const int64_t ne13,*/
	/*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
	/*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03, const size_t s10, const size_t s11, const size_t s12 /*, const size_t s13*/) {
	const int i00 = blockIdx.x * blockDim.x + threadIdx.x;
	const int i10 = blockDim.y * blockIdx.y + threadIdx.y;
	const int i11 = (blockIdx.z * blockDim.z + threadIdx.z) / ne12;
	const int i12 = (blockIdx.z * blockDim.z + threadIdx.z) % ne12;

	if (i00 >= ne00) {
		return;
	}

	const int i01 = src1[i10 * s10 + i11 * s11 + i12 * s12];

	dst_t* dst_row		   = dst + i10 * s1 + i11 * s2 + i12 * s3;
	const src0_t* src0_row = ( const src0_t* )(( const char* )src0 + i01 * nb01 + i11 * nb02 + i12 * nb03);

	dst_row[i00] = src0_row[i00];
}

template<typename grad_t, typename dst_t> static __global__ void k_get_rows_back_float(const grad_t* __restrict__ grad, const int32_t* __restrict__ rows, dst_t* __restrict__ dst,
	const int64_t ncols, const int64_t nrows_grad) {
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col >= ncols) {
		return;
	}

	const int dst_row = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0.0f;

	for (int64_t i = 0; i < nrows_grad; ++i) {
		if (rows[i] != dst_row) {
			continue;
		}
		sum += grad[i * ncols + col];
	}

	dst[dst_row * ncols + col] = sum;
}

template<int qk, int qr, dequantize_kernel_t dq>
static void get_rows_cuda(const oiml_tensor* src0, const oiml_tensor* src1, oiml_tensor* dst, const void* src0_dd, const int32_t* src1_dd, float* dst_dd, cudaStream_t stream) {
	OIML_TENSOR_BINARY_OP_LOCALS

	const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
	const int block_num_x = (ne00 + 2 * CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2 * CUDA_GET_ROWS_BLOCK_SIZE);
	const dim3 block_nums(block_num_x, ne10, ne11 * ne12);

	// strides in elements
	//const size_t s0 = nb0 / oiml_element_size(dst);
	const size_t s1 = nb1 / oiml_element_size(dst);
	const size_t s2 = nb2 / oiml_element_size(dst);
	const size_t s3 = nb3 / oiml_element_size(dst);

	const size_t s10 = nb10 / oiml_element_size(src1);
	const size_t s11 = nb11 / oiml_element_size(src1);
	const size_t s12 = nb12 / oiml_element_size(src1);
	//const size_t s13 = nb13 / oiml_element_size(src1);

	OIML_ASSERT(ne00 % 2 == 0);

	k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(src0_dd, src1_dd, dst_dd, ne00, /*ne01, ne02, ne03,*/
		/*ne10, ne11,*/ ne12, /*ne13,*/
		/* s0,*/ s1, s2, s3,
		/* nb00,*/ nb01, nb02, nb03, s10, s11, s12 /*, s13*/);

	OIML_UNUSED(dst);
}

template<typename src0_t> static void get_rows_cuda_float(const oiml_tensor* src0, const oiml_tensor* src1, oiml_tensor* dst, const src0_t* src0_dd, const int32_t* src1_dd,
	float* dst_dd, cudaStream_t stream) {
	OIML_TENSOR_BINARY_OP_LOCALS

	OIML_ASSERT(ne13 == 1);

	const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
	const int block_num_x = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
	const dim3 block_nums(block_num_x, ne10, ne11 * ne12);

	// strides in elements
	//const size_t s0 = nb0 / oiml_element_size(dst);
	const size_t s1 = nb1 / oiml_element_size(dst);
	const size_t s2 = nb2 / oiml_element_size(dst);
	const size_t s3 = nb3 / oiml_element_size(dst);

	const size_t s10 = nb10 / oiml_element_size(src1);
	const size_t s11 = nb11 / oiml_element_size(src1);
	const size_t s12 = nb12 / oiml_element_size(src1);
	//const size_t s13 = nb13 / oiml_element_size(src1);

	k_get_rows_float<<<block_nums, block_dims, 0, stream>>>(src0_dd, src1_dd, dst_dd, ne00, /*ne01, ne02, ne03,*/
		/*ne10, ne11,*/ ne12, /*ne13,*/
		/* s0,*/ s1, s2, s3,
		/* nb00,*/ nb01, nb02, nb03, s10, s11, s12 /*, s13*/);

	OIML_UNUSED(dst);
}

void oiml_cuda_op_get_rows(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* src0 = dst->src[0];
	const oiml_tensor* src1 = dst->src[1];

	const void* src0_d	  = ( const void* )src0->data;
	const int32_t* src1_d = ( const int32_t* )src1->data;
	float* dst_d		  = ( float* )dst->data;

	cudaStream_t stream = ctx.stream();

	OIML_ASSERT(src1->type == oiml::oiml_representation_types::int_32);
	OIML_ASSERT(dst->type == oiml::oiml_representation_types::float_32);

	OIML_ASSERT(src0->nb[0] == oiml_type_size(src0->type));
	OIML_ASSERT(src1->nb[0] == oiml_type_size(src1->type));
	OIML_ASSERT(dst->nb[0] == oiml_type_size(dst->type));

	switch (src0->type) {
		case oiml::oiml_representation_types::float_16:
			get_rows_cuda_float(src0, src1, dst, ( const half* )src0_d, src1_d, dst_d, stream);
			break;
		case oiml::oiml_representation_types::float_32:
			get_rows_cuda_float(src0, src1, dst, ( const float* )src0_d, src1_d, dst_d, stream);
			break;
		case oiml::oiml_representation_types::q8_0:
			get_rows_cuda<oiml::Q_SIZE, oiml::QR8_0, dequantize_q8_0>(src0, src1, dst, src0_d, src1_d, dst_d, stream);
			break;
		default:
			// TODO: k-quants
			OIML_ABORT("%s: unsupported type: %s\n", __func__, oiml_type_name(src0->type));
			break;
	}
}

void oiml_cuda_op_get_rows_back(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* src0 = dst->src[0];// gradients of forward pass output
	const oiml_tensor* src1 = dst->src[1];// src1 in forward pass

	OIML_TENSOR_BINARY_OP_LOCALS

	const float* src0_d	  = ( const float* )src0->data;
	const int32_t* src1_d = ( const int32_t* )src1->data;
	float* dst_d		  = ( float* )dst->data;

	cudaStream_t stream = ctx.stream();

	OIML_ASSERT(src0->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(src1->type == oiml::oiml_representation_types::int_32);
	OIML_ASSERT(dst->type == oiml::oiml_representation_types::float_32);

	OIML_ASSERT(oiml_is_contiguous(src0));
	OIML_ASSERT(oiml_is_contiguous(src1));
	OIML_ASSERT(oiml_is_contiguous(dst));

	OIML_ASSERT(ne02 * ne03 == 1);
	OIML_ASSERT(ne12 * ne13 == 1);
	OIML_ASSERT(ne2 * ne3 == 1);

	const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
	const int block_num_x = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
	const dim3 block_nums(block_num_x, ne1, 1);

	k_get_rows_back_float<<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne10);
}
