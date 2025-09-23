#include <BnchSwt/BenchmarkSuite.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

static constexpr uint64_t total_iterations{ 2 };
static constexpr uint64_t measured_iterations{ 2 };

template<auto multiple, typename value_01_type = decltype(multiple)> BNCH_SWT_INLINE constexpr value_01_type round_up_to_multiple(value_01_type value) noexcept {
	if constexpr ((multiple & (multiple - 1)) == 0) {
		constexpr value_01_type mulSub1{ multiple - 1 };
		constexpr value_01_type notMulSub1{ static_cast<value_01_type>(~mulSub1) };
		return (value + (mulSub1)) & notMulSub1;
	} else {
		const value_01_type remainder = value % multiple;
		return remainder == 0 ? value : value + (multiple - remainder);
	}
}

struct cuda_buffer {
	using value_type													= uint8_t;
	using pointer														= value_type*;
	using size_type														= uint64_t;
	BNCH_SWT_INLINE cuda_buffer() noexcept								= default;
	BNCH_SWT_INLINE cuda_buffer& operator=(const cuda_buffer&) noexcept = delete;
	BNCH_SWT_INLINE cuda_buffer(const cuda_buffer&) noexcept			= delete;

	BNCH_SWT_INLINE cuda_buffer& operator=(cuda_buffer&& other) noexcept {
		if (this != &other) {
			std::swap(data_val, other.data_val);
			std::swap(size_val, other.size_val);
		}
		return *this;
	}

	BNCH_SWT_INLINE cuda_buffer(cuda_buffer&& other) noexcept {
		*this = std::move(other);
	}

	BNCH_SWT_INLINE void init(uint64_t size) {
		if (data_val) {
			clear();
		}

		cudaError_t result = cudaMalloc(&data_val, size);
		if (result != cudaSuccess) {
			data_val = nullptr;
			throw std::runtime_error{ "cuda_buffer - failed to allocate GPU memory" };
		}

		size_val = size;
	}

	BNCH_SWT_INLINE void deinit() noexcept {
		clear();
	}

	BNCH_SWT_INLINE uint64_t size() noexcept {
		return size_val;
	}

	BNCH_SWT_INLINE void* data() noexcept {
		return data_val;
	}

	BNCH_SWT_INLINE void* claim_memory(uint64_t offset_to_claim) {
		uint64_t aligned_amount = round_up_to_multiple<64>(offset_to_claim);
		if (aligned_amount > size_val) {
			throw std::runtime_error{ "cuda_buffer - not enough memory allocated!" };
		}
		pointer return_value = data_val + aligned_amount;
		return return_value;
	}

	BNCH_SWT_INLINE ~cuda_buffer() noexcept {
		clear();
	}

  protected:
	value_type* data_val{};
	uint64_t size_val{};

	BNCH_SWT_INLINE void clear() noexcept {
		if (data_val) {
			cudaError_t result = cudaFree(data_val);
			data_val		   = nullptr;
			size_val		   = 0;
		}
	}
};

using q8_quant = int8_t;

inline static uint16_t fp32_to_fp16(float f) {
	return static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0));
}

struct block_q8_0 {
	static constexpr uint64_t block_count{ 32 };
	int16_t scale;
	int8_t quants[block_count];
};

inline block_q8_0 generate_block(const float* x) {
	block_q8_0 return_values{};

	float amax = 0.0f;

	for (int32_t j = 0; j < 32; j++) {
		const float v = x[j];
		amax		  = std::max(amax, fabsf(v));
	}

	const float d  = amax / ((1 << 7) - 1);
	const float id = d ? 1.0f / d : 0.0f;

	return_values.scale = fp32_to_fp16(d);

	for (int32_t j = 0; j < 32; ++j) {
		const float x0 = x[j] * id;

		return_values.quants[j] = roundf(x0);
	}
	return return_values;
}

inline std::vector<std::vector<block_q8_0>> generate_blocks(const std::vector<std::vector<float>>& floats) {
	std::vector<std::vector<block_q8_0>> result;
	result.reserve(floats.size());

	for (const auto& row: floats) {
		const uint64_t row_elements	 = row.size();
		const uint64_t blocks_needed = (row_elements + 31) / 32;

		std::vector<block_q8_0> row_blocks;
		row_blocks.reserve(blocks_needed);
		for (uint64_t x = 0; x < row_elements / 32; ++x) {
			row_blocks.emplace_back(generate_block(row.data() + x * 32));
		}

		result.emplace_back(std::move(row_blocks));
	}

	return result;
}

inline std::vector<std::vector<std::vector<block_q8_0>>> generate_blocks_final(const std::vector<std::vector<std::vector<float>>>& floats) {
	std::vector<std::vector<std::vector<block_q8_0>>> result;
	result.reserve(floats.size());

	for (const auto& values: floats) {
		result.emplace_back(generate_blocks(values));
	}

	return result;
}

inline float generate_llm_float() {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::normal_distribution<float> dist(0.0f, 0.02f);
	float value = dist(gen);
	return std::clamp(value, -0.5f, 0.5f);
}

template<uint64_t dim_00, uint64_t dim_01> inline std::vector<std::vector<float>> generate_floats() {
	std::vector<std::vector<float>> result;
	result.resize(dim_00);
	for (uint64_t x = 0; x < dim_00; ++x) {
		result[x].reserve(dim_01);
	}
	for (uint64_t i = 0; i < dim_00; ++i) {
		for (uint64_t j = 0; j < dim_01; ++j) {
			result[i].emplace_back(generate_llm_float());
		}
	}
	return result;
}

template<uint64_t iteration_count, uint64_t dim_00, uint64_t dim_01> inline std::vector<std::vector<std::vector<float>>> generate_floats_final() {
	std::vector<std::vector<std::vector<float>>> result;
	result.reserve(iteration_count);
	for (uint64_t x = 0; x < iteration_count; ++x) {
		result.emplace_back(generate_floats<dim_00, dim_01>());
	}
	return result;
}

template<typename value_type> inline std::vector<value_type> linearize_values(const std::vector<std::vector<value_type>>& values) {
	std::vector<value_type> return_values{};
	return_values.reserve(values.size() * values[0].size());
	for (uint64_t x = 0; x < values.size(); ++x) {
		for (uint64_t y = 0; y < values[x].size(); ++y) {
			return_values.emplace_back(values[x][y]);
		}
	}
	return return_values;
}

template<typename value_type> inline std::vector<std::vector<value_type>> transpose_values(const std::vector<std::vector<value_type>>& floats) {
	const uint64_t rows = floats.size();
	const uint64_t cols = floats.empty() ? 0 : floats[0].size();

	std::vector<std::vector<value_type>> result;
	result.resize(cols);
	for (uint64_t x = 0; x < cols; ++x) {
		result[x].reserve(rows);
	}

	for (uint64_t i = 0; i < rows; ++i) {
		for (uint64_t j = 0; j < cols; ++j) {
			result[j].emplace_back(floats[i][j]);
		}
	}
	return result;
}

template<typename value_type> inline std::vector<std::vector<std::vector<value_type>>> transpose_values_final(const std::vector<std::vector<std::vector<value_type>>>& floats) {
	std::vector<std::vector<std::vector<value_type>>> result;
	result.reserve(floats.size());
	for (uint64_t x = 0; x < floats.size(); ++x) {
		result.emplace_back(transpose_values(floats[x]));
	}
	return result;
}

template<typename value_type> inline std::vector<std::vector<value_type>> generate_values_final(const std::vector<std::vector<std::vector<value_type>>>& values) {
	std::vector<std::vector<value_type>> return_values{};
	for (uint64_t x = 0; x < values.size(); ++x) {
		return_values.emplace_back(linearize_values(values[x]));
	}
	return return_values;
}

BNCH_SWT_INLINE static constexpr float fp32_from_bits(uint32_t w) noexcept {
	return std::bit_cast<float>(w);
}

BNCH_SWT_INLINE static constexpr uint32_t fp32_to_bits(float f) noexcept {
	return std::bit_cast<uint32_t>(f);
}

BNCH_SWT_INLINE static float compute_fp16_to_fp32(half h) noexcept {
	const uint32_t w	 = static_cast<uint32_t>(h) << 16;
	const uint32_t sign	 = w & 0x80000000u;
	const uint32_t two_w = w + w;

	constexpr uint32_t exp_offset = 0xE0u << 23;
	constexpr float exp_scale	  = fp32_from_bits(0x7800000u);
	const float normalized_value  = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

	constexpr uint32_t magic_mask  = 126u << 23;
	constexpr float magic_bias	   = 0.5f;
	const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

	constexpr uint32_t denormalized_cutoff = 1u << 27;
	const uint32_t result				   = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
	return fp32_from_bits(result);
}

alignas(64) static float* __restrict fp16_to_fp32_array{ []() {
	alignas(64) static std::array<float, (1 << 16)> return_values_new{};
	for (uint64_t i = 0; i < (1 << 16); ++i) {
		return_values_new[i] = float{ compute_fp16_to_fp32(static_cast<uint16_t>(i)) };
	}
	return return_values_new.data();
}() };

BNCH_SWT_INLINE static float fp16_to_fp32(uint16_t f) {
	return fp16_to_fp32_array[f];
}

template<uint64_t M, uint64_t K> struct reference_mul_mat {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats, std::vector<std::vector<block_q8_0>>& blocks,
		std::vector<std::vector<float>>& outputs, uint64_t N) {
		const auto& current_blocks = blocks[current_index];
		const auto& current_floats = floats[current_index];
		auto& current_outputs	   = outputs[current_index];

		for (uint64_t row = 0; row < M; ++row) {
			for (uint64_t col = 0; col < N; ++col) {
				float sum = 0.0f;

				for (uint64_t k = 0; k < K; ++k) {
					const uint64_t block_idx	 = (row * K + k) / 32;
					const uint64_t elem_in_block = (row * K + k) % 32;

					const auto& block  = current_blocks[block_idx];
					const float scale  = __half2float(*reinterpret_cast<const __half*>(&block.scale));
					const float a_elem = scale * static_cast<float>(block.quants[elem_in_block]);

					const float b_elem = current_floats[k * N + col];

					sum += a_elem * b_elem;
				}

				current_outputs[row * N + col] = sum;
			}
		}
		++current_index;
		return current_outputs.size() * sizeof(float);
	}
};

template<uint64_t M, uint64_t K> struct cuda_mul_mat_01_prep {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats_A, std::vector<std::vector<float>>& floats_B,
		std::vector<std::vector<float>>& outputs, uint64_t N) {
		const uint64_t floats_A_size  = (M * K) * sizeof(float);
		const uint64_t floats_B_size  = (K * N) * sizeof(float);
		const uint64_t outputs_C_size = (M * N) * sizeof(float);

		float* A_ptr	= reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()));
		uint64_t offset = round_up_to_multiple<64>(floats_A_size);

		float* B_ptr = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset		 = round_up_to_multiple<64>(offset + floats_B_size);

		float* C_ptr = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);

		const auto& current_floats_A = floats_A[current_index];
		const auto& current_floats_B = floats_B[current_index];

		cudaError_t err = cudaMemcpy(A_ptr, current_floats_A.data(), floats_A_size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "Failed to copy Matrix A floats to device: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		err = cudaMemcpy(B_ptr, current_floats_B.data(), floats_B_size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "Failed to copy Matrix B floats to device: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		return 0;
	}
};

template<uint64_t M, uint64_t K> struct cuda_mul_mat_01_post {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats_A, std::vector<std::vector<float>>& floats_B,
		std::vector<std::vector<float>>& outputs, uint64_t N) {
		const uint64_t floats_A_size  = (M * K) * sizeof(float);
		const uint64_t floats_B_size  = (K * N) * sizeof(float);
		const uint64_t outputs_C_size = (M * N) * sizeof(float);

		uint64_t offset = round_up_to_multiple<64>(floats_A_size);
		offset			= round_up_to_multiple<64>(offset + floats_B_size);

		float* C_ptr = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);

		auto& previous_outputs = outputs[current_index];
		cudaError_t err		   = cudaMemcpy(previous_outputs.data(), C_ptr, outputs_C_size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::cerr << "Failed to copy previous outputs from device: " + std::string(cudaGetErrorString(err)) << std::endl;
		}
		err = cudaMemset(C_ptr, 0, outputs_C_size);
		if (err != cudaSuccess) {
			std::cerr << "Failed to zero output buffer: " + std::string(cudaGetErrorString(err)) << std::endl;
		}
		++current_index;
		return 0;
	}
};

template<uint64_t M, uint64_t K> __global__ void ggml_cuda_mul_mat_float_kernel(const float* input_A, const float* input_B, float* output, uint64_t N) {
	const uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= M || col >= N)
		return;

	float sum = 0.0f;

	const uint64_t k_end = K & ~3;
	uint64_t k			 = 0;

	for (; k < k_end; k += 4) {
#pragma unroll
		for (uint64_t i = 0; i < 4; ++i) {
			const uint64_t k_idx = k + i;
			const float a_elem	 = input_A[row * K + k_idx];
			const float b_elem	 = input_B[k_idx * N + col];
			sum += a_elem * b_elem;
		}
	}

	for (; k < K; ++k) {
		const float a_elem = input_A[row * K + k];
		const float b_elem = input_B[k * N + col];
		sum += a_elem * b_elem;
	}

	output[row * N + col] = sum;
}

template<uint64_t M, uint64_t K> struct ggml_cuda_mul_mat {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats_A, std::vector<std::vector<float>>& floats_B,
		std::vector<std::vector<float>>& outputs, uint64_t N) {
		auto& current_outputs = outputs[current_index];

		const uint64_t floats_A_size  = (M * K) * sizeof(float);
		const uint64_t floats_B_size  = (K * N) * sizeof(float);
		const uint64_t outputs_C_size = (M * N) * sizeof(float);

		const float* A_ptr = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()));
		uint64_t offset	   = round_up_to_multiple<64>(floats_A_size);

		const float* B_ptr = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset			   = round_up_to_multiple<64>(offset + floats_B_size);

		float* C_ptr = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);

		uint64_t block_dim_x, block_dim_y;
		if (N <= 4) {
			block_dim_x = N;
			block_dim_y = 256 / block_dim_x;
		} else if (M <= 16) {
			block_dim_x = 32;
			block_dim_y = 16;
		} else {
			block_dim_x = 16;
			block_dim_y = 32;
		}

		block_dim_x = std::min(block_dim_x, N);
		block_dim_y = std::min(block_dim_y, M);

		const uint64_t grid_dim_x = (N + block_dim_x - 1) / block_dim_x;
		const uint64_t grid_dim_y = (M + block_dim_y - 1) / block_dim_y;

		dim3 blockDim(static_cast<uint64_t>(block_dim_x), static_cast<uint64_t>(block_dim_y));
		dim3 gridDim(static_cast<uint64_t>(grid_dim_x), static_cast<uint64_t>(grid_dim_y));

		ggml_cuda_mul_mat_float_kernel<M, K><<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, N);

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "GGML CUDA float kernel launch failed: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "GGML CUDA float kernel execution failed: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		return M * N * sizeof(float);
	}
};

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<uint64_t M, uint64_t K> struct reference_mul_mat_float {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats_A, std::vector<std::vector<float>>& floats_B,
		std::vector<std::vector<float>>& outputs, uint64_t N) {
		auto& A = floats_A[current_index];
		auto& B = floats_B[current_index];
		auto& C = outputs[current_index];

		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				float sum = 0.0f;
				for (int k = 0; k < K; ++k) {
					sum += A[i * K + k] * B[k * N + j];
				}
				C[i * N + j] = sum;
			}
		}
		++current_index;
		return M * N * sizeof(float);
	}
};

template<uint64_t M, uint64_t K> __global__ void rt_tm_gemm_float_kernel(const float* input_A, const float* input_B, float* output, uint64_t N) {
	const uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= M || col >= N)
		return;

	float sum = 0.0f;

	const uint64_t k_end = K & ~3;
	uint64_t k			 = 0;

	for (; k < k_end; k += 4) {
#pragma unroll
		for (uint64_t i = 0; i < 4; ++i) {
			const uint64_t k_idx = k + i;
			const float a_elem	 = input_A[row * K + k_idx];
			const float b_elem	 = input_B[k_idx * N + col];
			sum += a_elem * b_elem;
		}
	}

	for (; k < K; ++k) {
		const float a_elem = input_A[row * K + k];
		const float b_elem = input_B[k * N + col];
		sum += a_elem * b_elem;
	}

	output[row * N + col] = sum;
}

#include <nihilus_gemm/gemm/device/gemm.h>

using element_a = float;
using element_b = float;
using element_c = float;
using layout_a	= nihilus_gemm::layout::RowMajor;
using layout_b	= nihilus_gemm::layout::RowMajor;
using layout_c	= nihilus_gemm::layout::RowMajor;

__global__ void dequantize_a_matrix_kernel(const block_q8_0* input_blocks, float* output, uint64_t total_elements) {
	const uint64_t idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t stride = blockDim.x * gridDim.x;

	for (uint64_t i = idx; i < total_elements; i += stride) {
		const uint64_t block_idx	 = i >> 5;
		const uint64_t elem_in_block = i & 31;

		const block_q8_0& block = input_blocks[block_idx];
		const float scale		= __half2float(*reinterpret_cast<const __half*>(&block.scale));
		output[i]				= scale * static_cast<float>(block.quants[elem_in_block]);
	}
}

template<uint64_t M, uint64_t K> __global__ void nihilus_custom_cuda_kernel(const float* input_A, const float* input_B, float* output, uint64_t N) {
	const uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= M || col >= N)
		return;
	float sum = 0.0f;

	const uint64_t k_end = K & ~3;
	uint64_t k			 = 0;

	for (; k < k_end; k += 4) {
#pragma unroll
		for (uint64_t i = 0; i < 4; ++i) {
			const uint64_t k_idx = k + i;
			const float a_elem	 = input_A[row * K + k_idx];
			const float b_elem	 = input_B[k_idx * N + col];
			sum += a_elem * b_elem;
		}
	}

	output[row * N + col] = sum;
}

template<uint64_t M, uint64_t K> struct nihilus_mul_mat {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats_A, std::vector<std::vector<float>>& floats_B,
		std::vector<std::vector<float>>& outputs, uint64_t N) {
		const uint64_t floats_A_size  = (M * K) * sizeof(float);
		const uint64_t floats_B_size  = (K * N) * sizeof(float);
		const uint64_t outputs_C_size = (M * N) * sizeof(float);

		uint64_t offset = 0;

		const float* A_ptr = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset			   = round_up_to_multiple<64>(offset + floats_A_size);

		const float* B_ptr = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset			   = round_up_to_multiple<64>(offset + floats_B_size);

		float* C_ptr = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		if constexpr (M <= 4096) {
			if (N <= 32) {
				uint64_t block_dim_x, block_dim_y;
				if (N <= 4) {
					block_dim_x = N;
					block_dim_y = 256 / block_dim_x;
				} else if (M <= 16) {
					block_dim_x = 32;
					block_dim_y = 16;
				} else {
					block_dim_x = 16;
					block_dim_y = 32;
				}

				block_dim_x = std::min(block_dim_x, N);
				block_dim_y = std::min(block_dim_y, M);

				const uint64_t grid_dim_x = (N + block_dim_x - 1) / block_dim_x;
				const uint64_t grid_dim_y = (M + block_dim_y - 1) / block_dim_y;
				dim3 blockDim(static_cast<uint32_t>(block_dim_x), static_cast<uint32_t>(block_dim_y));

				dim3 gridDim((N + block_dim_x - 1) / block_dim_x, (M + block_dim_y - 1) / block_dim_y);

				nihilus_custom_cuda_kernel<M, K><<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, N);
			} else {
				using index_type		= nihilus_gemm::gemm::GemmCoord::Index;
				using nihilus_gemm_type = nihilus_gemm::gemm::device::Gemm<M, K, element_a, layout_a, element_b, layout_b, element_c, layout_c, element_c>;
				nihilus_gemm_type gemm_op;
				nihilus_gemm::Status status =
					gemm_op({ { static_cast<index_type>(M), static_cast<index_type>(N), static_cast<index_type>(K) }, { A_ptr, static_cast<index_type>(K) },
						{ B_ptr, static_cast<index_type>(N) }, { C_ptr, static_cast<index_type>(N) }, { C_ptr, static_cast<index_type>(N) }, { 1.0f, 0.0f } });

				if (status != nihilus_gemm::Status::kSuccess) {
					std::cerr << "Nihilus Gemm failed: " << nihilus_gemm::cutlassGetStatusString(status) << std::endl;
				}
			}

		} else {
			using index_type		= nihilus_gemm::gemm::GemmCoord::Index;
			using nihilus_gemm_type = nihilus_gemm::gemm::device::Gemm<M, K, element_a, layout_a, element_b, layout_b, element_c, layout_c, element_c>;
			nihilus_gemm_type gemm_op;
			nihilus_gemm::Status status = gemm_op({ { static_cast<index_type>(M), static_cast<index_type>(N), static_cast<index_type>(K) }, { A_ptr, static_cast<index_type>(K) },
				{ B_ptr, static_cast<index_type>(N) }, { C_ptr, static_cast<index_type>(N) }, { C_ptr, static_cast<index_type>(N) }, { 1.0f, 0.0f } });

			if (status != nihilus_gemm::Status::kSuccess) {
				std::cerr << "Nihilus Gemm failed: " << nihilus_gemm::cutlassGetStatusString(status) << std::endl;
			}
		}

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "❌ CUDA error after Nihilus: " << cudaGetErrorString(err) << std::endl;
		}

		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "❌ CUDA synchronization failed: " << cudaGetErrorString(err) << std::endl;
		}

		return M * N * sizeof(float);
	}
};

#include <cutlass/gemm/device/gemm.h>

template<uint64_t M, uint64_t K> struct cutlass_mul_mat {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats_A, std::vector<std::vector<float>>& floats_B,
		std::vector<std::vector<float>>& outputs, uint64_t N) {
		const uint64_t floats_A_size  = (M * K) * sizeof(float);
		const uint64_t floats_B_size  = (K * N) * sizeof(float);
		const uint64_t outputs_C_size = (M * N) * sizeof(float);

		using element_a = float;
		using element_b = float;
		using element_c = float;
		using layout_a	= cutlass::layout::RowMajor;
		using layout_b	= cutlass::layout::RowMajor;
		using layout_c	= cutlass::layout::RowMajor;
		uint64_t offset = 0;

		const float* A_ptr = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset			   = round_up_to_multiple<64>(offset + floats_A_size);

		const float* B_ptr = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset			   = round_up_to_multiple<64>(offset + floats_B_size);

		float* C_ptr			= reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		using index_type		= cutlass::gemm::GemmCoord::Index;
		using cutlass_gemm_type = cutlass::gemm::device::Gemm<element_a, layout_a, element_b, layout_b, element_c, layout_c, element_c>;
		cutlass_gemm_type gemm_op;
		cutlass::Status status = gemm_op({ { static_cast<index_type>(M), static_cast<index_type>(N), static_cast<index_type>(K) }, { A_ptr, static_cast<index_type>(K) },
			{ B_ptr, static_cast<index_type>(N) }, { C_ptr, static_cast<index_type>(N) }, { C_ptr, static_cast<index_type>(N) }, { 1.0f, 0.0f } });

		if (status != cutlass::Status::kSuccess) {
			std::cerr << "Cutlass Gemm failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
		}

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "❌ CUDA error after Nihilus: " << cudaGetErrorString(err) << std::endl;
		}

		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "❌ CUDA synchronization failed: " << cudaGetErrorString(err) << std::endl;
		}

		return M * N * sizeof(float);
	}
};

template<bnch_swt::string_literal rhs> inline void compare_outputs(const std::vector<std::vector<float>>& outputs01, const std::vector<std::vector<float>>& outputs02) {
	static constexpr float relative_tolerance = 1e-1f;
	static constexpr float absolute_tolerance = 1e-30f;
	if (outputs01.size() != outputs02.size()) {
		std::cerr << "Unequal output sizes!" << std::endl;
		return;
	}
	for (uint64_t x = 0; x < outputs02.size(); ++x) {
		if (outputs01[x].size() != outputs02[x].size()) {
			std::cerr << "Unequal output sizes!" << std::endl;
			return;
		}
		for (uint64_t y = 0; y < outputs01[x].size(); ++y) {
			const float val1 = outputs01[x][y];
			const float val2 = outputs02[x][y];

			const float abs_diff = std::abs(val1 - val2);
			const float max_val	 = std::max(std::abs(val1), std::abs(val2));

			if (std::isinf(val1) || std::isinf(val2) || std::isnan(val1) || std::isnan(val2) || !((abs_diff <= absolute_tolerance) || (abs_diff <= relative_tolerance * max_val))) {
				std::cerr << rhs.operator std::string_view() << ": Mismatch at [" << x << "," << y << "]: Ref Val: " << val1 << " vs Incorrect Val: " << val2 << std::endl;
				std::cerr << "Relative difference: " << (abs_diff / max_val) * 100.0f << "%" << std::endl;
				return;
			}
		}
	}
}

template<uint64_t M, uint64_t K, uint64_t matB_dim_00, uint64_t N> BNCH_SWT_INLINE void test_function_floats() {
	static constexpr uint64_t total_elements_C{ M * N };
	std::vector<std::vector<float>> floats_a{ generate_values_final(generate_floats_final<total_iterations, M, K>()) };
	std::vector<std::vector<float>> floats_b{ generate_values_final(generate_floats_final<total_iterations, K, N>()) };
	std::vector<std::vector<float>> outputs01{};
	std::vector<std::vector<float>> outputs02{};
	std::vector<std::vector<float>> outputs03{};
	std::vector<std::vector<float>> outputs04{};
	std::vector<std::vector<float>> outputs05{};
	outputs01.resize(total_iterations);
	outputs02.resize(total_iterations);
	outputs03.resize(total_iterations);
	outputs04.resize(total_iterations);
	outputs05.resize(total_iterations);
	for (uint64_t x = 0; x < total_iterations; ++x) {
		outputs01[x].resize(total_elements_C);
		outputs02[x].resize(total_elements_C);
		outputs03[x].resize(total_elements_C);
		outputs04[x].resize(total_elements_C);
		outputs05[x].resize(total_elements_C);
	}

	static constexpr bnch_swt::string_literal stage_name{ "(F32 * F32) mul_mat: [" + bnch_swt::internal::toStringLiteral<M>() + "x" + bnch_swt::internal::toStringLiteral<K>() +
		" * " + bnch_swt::internal::toStringLiteral<matB_dim_00>() + "x" + bnch_swt::internal::toStringLiteral<N>() + "]" };
	constexpr uint64_t total_elements_A = M * K;
	constexpr uint64_t total_elements_B = K * N;
	constexpr uint64_t floats_A_size	= total_elements_A * sizeof(float);
	constexpr uint64_t floats_B_size	= total_elements_B * sizeof(float);
	constexpr uint64_t floats_C_size	= total_elements_C * sizeof(float);

	uint64_t total_buffer_size = 0;
	total_buffer_size += round_up_to_multiple<64>(floats_A_size);
	total_buffer_size += round_up_to_multiple<64>(floats_B_size);
	total_buffer_size += round_up_to_multiple<64>(floats_C_size);

	cuda_buffer buffer{};
	buffer.init(total_buffer_size);

	uint64_t current_index{};
	//bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmark<"reference_mul_mat", reference_mul_mat_float<M, K>>(buffer,
	//current_index, floats_a, floats_b, outputs01, N);
	//current_index = 0;

	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrepAndPost<"ggml_cuda_mul_mat", cuda_mul_mat_01_prep<M, K>,
		ggml_cuda_mul_mat<M, K>, cuda_mul_mat_01_post<M, K>>(buffer, current_index, floats_a, floats_b, outputs01, N);
	current_index = 0;

	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrepAndPost<"nihilus_mul_mat", cuda_mul_mat_01_prep<M, K>,
		nihilus_mul_mat<M, K>, cuda_mul_mat_01_post<M, K>>(buffer, current_index, floats_a, floats_b, outputs02, N);
	current_index = 0;

	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrepAndPost<"cutlass_mul_mat", cuda_mul_mat_01_prep<M, K>,
	cutlass_mul_mat<M, K>, cuda_mul_mat_01_post<M, K>>(buffer, current_index, floats_a, floats_b, outputs03, N);
	current_index = 0;

	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::printResults();
	compare_outputs<"nihilus_mul_mat Incorrect Value">(outputs01, outputs02);
	compare_outputs<"cutlass_mul_mat Incorrect Value">(outputs01, outputs03);
};

int32_t main() {
	test_function_floats<4096, 4096, 4096, 1>();
	test_function_floats<4096, 4096, 4096, 2>();
	test_function_floats<4096, 4096, 4096, 4>();
	test_function_floats<4096, 4096, 4096, 8>();
	test_function_floats<4096, 4096, 4096, 16>();
	test_function_floats<4096, 4096, 4096, 32>();
	test_function_floats<4096, 4096, 4096, 64>();
	test_function_floats<4096, 4096, 4096, 128>();
	test_function_floats<4096, 4096, 4096, 256>();
	test_function_floats<4096, 4096, 4096, 512>();
	test_function_floats<4096, 4096, 4096, 1024>();
	test_function_floats<4096, 4096, 4096, 2048>();
	test_function_floats<4096, 4096, 4096, 4096>();
	test_function_floats<4096, 4096, 4096, 8192>();
	test_function_floats<4096, 4096, 4096, 16384>();
	test_function_floats<14336, 4096, 4096, 1>();
	test_function_floats<14336, 4096, 4096, 2>();
	test_function_floats<14336, 4096, 4096, 4>();
	test_function_floats<14336, 4096, 4096, 8>();
	test_function_floats<14336, 4096, 4096, 16>();
	test_function_floats<14336, 4096, 4096, 32>();
	test_function_floats<14336, 4096, 4096, 64>();
	test_function_floats<14336, 4096, 4096, 128>();
	test_function_floats<14336, 4096, 4096, 256>();
	test_function_floats<14336, 4096, 4096, 512>();
	test_function_floats<14336, 4096, 4096, 1024>();
	test_function_floats<14336, 4096, 4096, 2048>();
	test_function_floats<14336, 4096, 4096, 4096>();
	test_function_floats<14336, 4096, 4096, 8192>();
	test_function_floats<14336, 4096, 4096, 16384>();
	return 0;
}