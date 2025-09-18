#include <BnchSwt/BenchmarkSuite.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

static constexpr uint64_t total_iterations{ 8 };
static constexpr uint64_t measured_iterations{ 1 };

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

	BNCH_SWT_INLINE void init(uint64_t size) noexcept {
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

	BNCH_SWT_INLINE void* claim_memory(uint64_t offset_to_claim) noexcept {
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

struct block_q8_0 {
	q8_quant qs[32]{};
	uint16_t d{};
};

inline static uint16_t fp32_to_fp16(float f) {
	return static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0));
}

template<std::size_t count> inline std::vector<block_q8_0> generate_blocks() {
	std::vector<block_q8_0> result;
	result.reserve(count);

	for (std::size_t i = 0; i < count; ++i) {
		block_q8_0 block{};

		for (auto& q: block.qs) {
			q = static_cast<int8_t>((bnch_swt::random_generator::generateValue<uint8_t>() % 13) - 6);
		}

		float scale_float = (bnch_swt::random_generator::generateValue<float>() / std::numeric_limits<float>::max()) * 0.1f;

		block.d = static_cast<uint16_t>(fp32_to_fp16(scale_float));

		result.emplace_back(block);
	}
	return result;
}

template<uint64_t iteration_count, std::size_t count> inline std::vector<std::vector<block_q8_0>> generate_blocks_final() {
	std::vector<std::vector<block_q8_0>> return_values{};
	for (uint64_t x = 0; x < iteration_count; ++x) {
		return_values.emplace_back(generate_blocks<count>());
	}
	return return_values;
}

template<std::size_t count> inline std::vector<float> generate_floats() {
	std::vector<float> result;
	result.reserve(count);

	for (std::size_t i = 0; i < count; ++i) {
		result.emplace_back(bnch_swt::random_generator::generateValue<float>());
	}
	return result;
}

template<uint64_t iteration_count, std::size_t count> inline std::vector<std::vector<float>> generate_floats_final() {
	std::vector<std::vector<float>> return_values{};
	for (uint64_t x = 0; x < iteration_count; ++x) {
		return_values.emplace_back(generate_floats<count>());
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

template<uint64_t mat_a_dim_00, uint64_t mat_a_dim_01, uint64_t mat_b_dim_01> struct reference_mul_mat {
	BNCH_SWT_INLINE static uint64_t impl(uint64_t& current_index, std::vector<std::vector<float>>& floats, std::vector<std::vector<block_q8_0>>& blocks,
		std::vector<std::vector<float>>& outputs) {
		const auto& current_blocks = blocks[current_index];
		const auto& current_floats = floats[current_index];
		auto& current_outputs	   = outputs[current_index];

		for (uint64_t row = 0; row < mat_a_dim_00; ++row) {
			for (uint64_t col = 0; col < mat_b_dim_01; ++col) {
				float sum = 0.0f;

				for (uint64_t k = 0; k < mat_a_dim_01; ++k) {
					const uint64_t block_idx	 = (row * mat_a_dim_01 + k) / 32;
					const uint64_t elem_in_block = (row * mat_a_dim_01 + k) % 32;

					const auto& block  = current_blocks[block_idx];
					const float d	   = __half2float(*reinterpret_cast<const __half*>(&block.d));
					const float a_elem = d * static_cast<float>(block.qs[elem_in_block]);

					const float b_elem = current_floats[k * mat_b_dim_01 + col];

					sum += a_elem * b_elem;
				}

				current_outputs[row * mat_b_dim_01 + col] = sum;
			}
		}

		return current_outputs.size() * sizeof(float);
	}
};

template<uint64_t mat_a_dim_00, uint64_t mat_a_dim_01, uint64_t mat_b_dim_00> struct cuda_mul_mat_01_prep {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats, std::vector<std::vector<block_q8_0>>& blocks,
		std::vector<std::vector<float>>& outputs, uint64_t mat_b_dim_01) {
		constexpr uint64_t total_blocks_A = ((mat_a_dim_00 * mat_a_dim_01) + 32 - 1) / 32;
		constexpr uint64_t blocks_size	  = total_blocks_A * sizeof(block_q8_0);
		const uint64_t floats_B_size  = (mat_a_dim_01 * mat_b_dim_01) * sizeof(float);
		const uint64_t outputs_C_size	  = (mat_a_dim_00 * mat_b_dim_01) * sizeof(float);
		uint64_t offset					  = 0;
		block_q8_0* d_blocks			  = reinterpret_cast<block_q8_0*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset							  = round_up_to_multiple<64>(offset + blocks_size);

		float* d_floats = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset			= round_up_to_multiple<64>(offset + floats_B_size);

		float* d_outputs = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);

		if (current_index > 0) {
			auto& previous_outputs = outputs[current_index - 1];
			cudaError_t err		   = cudaMemcpy(previous_outputs.data(), d_outputs, outputs_C_size, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				std::cerr << "Failed to copy previous outputs from device: " + std::string(cudaGetErrorString(err)) << std::endl;
			}
		}

		const auto& current_blocks = blocks[current_index];
		const auto& current_floats = floats[current_index];

		cudaError_t err = cudaMemcpy(d_blocks, current_blocks.data(), blocks_size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "Failed to copy blocks to device: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		err = cudaMemcpy(d_floats, current_floats.data(), floats_B_size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "Failed to copy floats to device: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		err = cudaMemset(d_outputs, 0, outputs_C_size);
		if (err != cudaSuccess) {
			std::cerr << "Failed to zero output buffer: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		return 0;
	}
};

template<uint64_t mat_a_dim_00, uint64_t mat_a_dim_01, uint64_t matB_dim_01>
__global__ void ggml_cuda_mul_mat_kernel(const float* input01, const block_q8_0* input02, float* output, uint64_t mat_b_dim_01) {
	const uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= mat_a_dim_00 || col >= mat_b_dim_01)
		return;

	float sum = 0.0f;

	const uint64_t k_end = mat_a_dim_01 & ~3;

	uint64_t k = 0;
	for (; k < k_end; k += 4) {
#pragma unroll
		for (int i = 0; i < 4; ++i) {
			const uint64_t k_idx		 = k + i;
			const uint64_t linear_idx	 = row * mat_a_dim_01 + k_idx;
			const uint64_t block_idx	 = linear_idx / 32;
			const uint64_t elem_in_block = linear_idx % 32;

			const block_q8_0& block = input02[block_idx];
			const float d			= __half2float(*reinterpret_cast<const __half*>(&block.d));
			const float a_elem		= d * static_cast<float>(block.qs[elem_in_block]);
			const float b_elem		= input01[k_idx * mat_b_dim_01 + col];

			sum += a_elem * b_elem;
		}
	}

	for (; k < mat_a_dim_01; ++k) {
		const uint64_t linear_idx	 = row * mat_a_dim_01 + k;
		const uint64_t block_idx	 = linear_idx / 32;
		const uint64_t elem_in_block = linear_idx % 32;

		const block_q8_0& block = input02[block_idx];
		const float d			= __half2float(*reinterpret_cast<const __half*>(&block.d));
		const float a_elem		= d * static_cast<float>(block.qs[elem_in_block]);
		const float b_elem		= input01[k * mat_b_dim_01 + col];

		sum += a_elem * b_elem;
	}

	output[row * mat_b_dim_01 + col] = sum;
}

template<uint64_t mat_a_dim_00, uint64_t mat_a_dim_01, uint64_t mat_b_dim_00> struct ggml_cuda_mul_mat {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats, std::vector<std::vector<block_q8_0>>& blocks,
		std::vector<std::vector<float>>& outputs, uint64_t mat_b_dim_01) {
		auto& current_outputs = outputs[current_index];

		static constexpr uint64_t total_blocks_A = ((mat_a_dim_00 * mat_a_dim_01) + 32 - 1) / 32;
		static constexpr uint64_t blocks_size	 = total_blocks_A * sizeof(block_q8_0);
		const uint64_t floats_B_size			 = (mat_a_dim_01 * mat_b_dim_01) * sizeof(float);
		const uint64_t outputs_C_size			 = (mat_a_dim_00 * mat_b_dim_01) * sizeof(float);

		uint64_t offset			   = 0;
		const block_q8_0* d_blocks = reinterpret_cast<const block_q8_0*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset					   = round_up_to_multiple<64>(offset + blocks_size);

		const float* d_floats = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset				  = round_up_to_multiple<64>(offset + floats_B_size);

		float* d_outputs = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);

		uint64_t block_dim_x, block_dim_y;
		if (mat_b_dim_01 <= 4) {
			block_dim_x = mat_b_dim_01;
			block_dim_y = 256 / block_dim_x;
		} else if (mat_a_dim_00 <= 16) {
			block_dim_x = 32;
			block_dim_y = 16;
		} else {
			block_dim_x = 16;
			block_dim_y = 32;
		}

		block_dim_x = std::min(block_dim_x, mat_b_dim_01);
		block_dim_y = std::min(block_dim_y, mat_a_dim_00);

		const uint64_t grid_dim_x = (mat_b_dim_01 + block_dim_x - 1) / block_dim_x;
		const uint64_t grid_dim_y = (mat_a_dim_00 + block_dim_y - 1) / block_dim_y;

		dim3 blockDim(static_cast<unsigned int>(block_dim_x), static_cast<unsigned int>(block_dim_y));
		dim3 gridDim(static_cast<unsigned int>(grid_dim_x), static_cast<unsigned int>(grid_dim_y));

		ggml_cuda_mul_mat_kernel<mat_a_dim_00, mat_a_dim_01, mat_b_dim_00><<<gridDim, blockDim>>>(d_floats, d_blocks, d_outputs, mat_b_dim_01);

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "GGML CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "GGML CUDA kernel execution failed: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		++current_index;
		return current_outputs.size() * sizeof(float);
	}
};

enum class kernel_type_profiles : uint8_t {
	fp16_mha,
	fp16_moe,
	bf16_mha,
	bf16_gqa,
	q4_mha,
	q4_gqa,
	q4_moe,
	q8_mha,
	q8_gqa,
	q8_moe,
	mixed_fp16_fp32,
	mixed_bf16_fp32,
	count,
};

enum class mul_mat_types {
	q,
	k,
	v,
	kq,
	kqv,
	kqv_out,
	ffn_gate,
	ffn_up,
	ffn_out,
};

template<mul_mat_types core_type, kernel_type_profiles kernel_type_profile> struct mul_mat_params;

template<> struct mul_mat_params<mul_mat_types::q, kernel_type_profiles::q8_gqa> {
	static constexpr uint64_t block_size{ 32 };
	static constexpr uint64_t tile_size{ 32 };
};

template<> struct mul_mat_params<mul_mat_types::ffn_up, kernel_type_profiles::q8_gqa> {
	static constexpr uint64_t block_size{ 32 };
	static constexpr uint64_t tile_size{ 16 };
};

struct gpu_properties {
	static constexpr uint64_t sm_count{ 70ull };
	static constexpr uint64_t max_threads_per_sm{ 1536ull };
	static constexpr uint64_t max_threads_per_block{ 1024ull };
	static constexpr uint64_t warp_size{ 32ull };
	static constexpr uint64_t l2_cache_size{ 50331648ull };
	static constexpr uint64_t shared_mem_per_block{ 49152ull };
	static constexpr uint64_t memory_bus_width{ 256ull };
	static constexpr uint64_t memory_clock_rate{ 14001000ull };
	static constexpr uint64_t major_compute_capability{ 12ull };
	static constexpr uint64_t minor_compute_capability{ 0ull };
	static constexpr uint64_t max_grid_size_x{ 2147483647ull };
	static constexpr uint64_t gpu_arch_index{ 4ull };
	static constexpr uint64_t total_threads{ 107520ull };
	static constexpr uint64_t optimal_block_size{ 512ull };
};

struct nihilus_dim3 {
	uint64_t x{};
	uint64_t y{};
	uint64_t z{};
	BNCH_SWT_INLINE operator dim3() const {
		return { static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z) };
	}
};

struct cuda_launch_params {
	uint64_t shared_memory_element_count{};
	nihilus_dim3 block_dim{};
	nihilus_dim3 grid_dim{};
};

template<uint64_t mat_a_dim_00, uint64_t mat_a_dim_01, typename matB_type> BNCH_SWT_INLINE static consteval cuda_launch_params calculate_gpu_launch_params_stage_01() {
	cuda_launch_params params;
	constexpr uint64_t total_memory_per_col{ (sizeof(matB_type) * mat_a_dim_01) };
	constexpr uint64_t memory_multiplier{ gpu_properties::shared_mem_per_block / total_memory_per_col };
	if constexpr (memory_multiplier > 0) {
		params.shared_memory_element_count = (total_memory_per_col * memory_multiplier) / sizeof(matB_type);
		constexpr uint64_t cols_per_block  = memory_multiplier;
		return params;
	}

	return params;
}

template<uint64_t mat_a_dim_00, uint64_t mat_a_dim_01, typename matB_type>
BNCH_SWT_INLINE static cuda_launch_params calculate_gpu_launch_params_stage_02(cuda_launch_params params, uint64_t sequence_length) {
	constexpr uint64_t total_memory_per_col{ (sizeof(matB_type) * mat_a_dim_01) };
	constexpr uint64_t memory_multiplier{ gpu_properties::shared_mem_per_block / total_memory_per_col };
	if constexpr (memory_multiplier > 0) {
		params.block_dim.x = mat_a_dim_00 / gpu_properties::optimal_block_size;
		params.block_dim.y = 1;
		params.block_dim.z = 1;
		params.grid_dim.x  = mat_a_dim_00 / params.block_dim.x;
		params.grid_dim.y  = sequence_length;
		params.grid_dim.z  = 1;
		return params;
	}
	return params;
}

template<uint64_t mat_a_dim_00, uint64_t mat_a_dim_01, uint64_t element_count>
__global__ void cuda_mul_mat_kernel(const block_q8_0* input01, const float* input02, float* output, uint64_t mat_b_dim_01) {
	__shared__ float matrix_b_shared[mat_a_dim_01];

	const uint64_t col = blockIdx.y;

	for (uint64_t k = threadIdx.x; k < mat_a_dim_01; k += blockDim.x) {
		matrix_b_shared[k] = input02[k * mat_b_dim_01 + col];
	}
	__syncthreads();

	const uint64_t row	  = blockIdx.x * blockDim.x + threadIdx.x;

	float sum			  = 0.0f;
	constexpr uint64_t QK = 32;
	const uint64_t nb	  = (mat_a_dim_01 + QK - 1) / QK;

	for (uint64_t i = 0; i < nb; i++) {
		const block_q8_0* a_block = &input01[row * nb + i];
		const float scale		  = __half2float(*reinterpret_cast<const __half*>(&a_block->d));

		for (int j = 0; j < QK && i * QK + j < mat_a_dim_01; j++) {
			const float a_val = scale * static_cast<float>(a_block->qs[j]);
			const float b_val = matrix_b_shared[i * QK + j];
			sum += a_val * b_val;
		}
	}

	output[row * mat_b_dim_01 + col] = sum;
}

template<uint64_t mat_a_dim_00, uint64_t mat_a_dim_01, uint64_t mat_b_dim_00> struct cuda_mul_mat {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats, std::vector<std::vector<block_q8_0>>& blocks,
		std::vector<std::vector<float>>& outputs, uint64_t mat_b_dim_01) {
		auto& current_outputs = outputs[current_index];

		static constexpr uint64_t total_blocks_A = ((mat_a_dim_00 * mat_a_dim_01) + 32 - 1) / 32;
		static constexpr uint64_t blocks_size	 = total_blocks_A * sizeof(block_q8_0);
		const uint64_t floats_B_size			 = (mat_a_dim_01 * mat_b_dim_01) * sizeof(float);
		const uint64_t outputs_C_size			 = (mat_a_dim_00 * mat_b_dim_01) * sizeof(float);

		uint64_t offset			   = 0;
		const block_q8_0* d_blocks = reinterpret_cast<const block_q8_0*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset					   = round_up_to_multiple<64>(offset + blocks_size);

		const float* d_floats = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset				  = round_up_to_multiple<64>(offset + floats_B_size);

		float* d_outputs = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);

		static constexpr auto launch_params_01 = calculate_gpu_launch_params_stage_01<mat_a_dim_00, mat_a_dim_01, float>();
		const auto launch_params			   = calculate_gpu_launch_params_stage_02<mat_a_dim_00, mat_a_dim_01, float>(launch_params_01, mat_b_dim_01);
		cuda_mul_mat_kernel<mat_a_dim_00, mat_a_dim_01, launch_params_01.shared_memory_element_count>
			<<<launch_params.grid_dim, launch_params.block_dim>>>(d_blocks, d_floats, d_outputs, mat_b_dim_01);

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "CUDA kernel execution failed: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		return current_outputs.size() * sizeof(float);
	}
};

template<uint64_t matA_rows, uint64_t matA_cols, mul_mat_types mul_mat_type>
__global__ void cuda_mul_mat_kernel_optimized(const block_q8_0* input02, const float* input01, float* output, uint64_t matB_cols) {
	__shared__ float tile_A[mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size][mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size];
	__shared__ float tile_B[mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size][mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size];

	const uint64_t row = blockIdx.y * mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size + threadIdx.y;
	const uint64_t col = blockIdx.x * mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size + threadIdx.x;

	float sum = 0.0f;

	for (uint64_t tile = 0;
		tile < (matA_cols + mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size - 1) / mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size;
		++tile) {
		const uint64_t a_col = tile * mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size + threadIdx.x;
		const uint64_t b_row = tile * mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size + threadIdx.y;

		const uint64_t linear_idx	 = row * matA_cols + a_col;
		const uint64_t block_idx	 = linear_idx / 32;
		const uint64_t elem_in_block = linear_idx % 32;

		const block_q8_0& block			 = input02[block_idx];
		const float scale				 = __half2float(*reinterpret_cast<const __half*>(&block.d));
		tile_A[threadIdx.y][threadIdx.x] = scale * static_cast<float>(block.qs[elem_in_block]);

		tile_B[threadIdx.y][threadIdx.x] = input01[b_row * matB_cols + col];

		__syncthreads();

#pragma unroll
		for (uint64_t k = 0; k < mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size; ++k) {
			sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < matA_rows && col < matB_cols) {
		output[row * matB_cols + col] = sum;
	}
}

template<uint64_t mat_a_dim_00, uint64_t mat_a_dim_01, uint64_t mat_b_dim_00, mul_mat_types mul_mat_type> struct nihilus_cuda_mul_mat {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats, std::vector<std::vector<block_q8_0>>& blocks,
		std::vector<std::vector<float>>& outputs, uint64_t mat_b_dim_01) {
		auto& current_outputs = outputs[current_index];

		static constexpr uint64_t total_blocks_A = ((mat_a_dim_00 * mat_a_dim_01) + 32 - 1) / 32;
		static constexpr uint64_t blocks_size	 = total_blocks_A * sizeof(block_q8_0);
		const uint64_t floats_B_size			 = (mat_a_dim_01 * mat_b_dim_01) * sizeof(float);
		const uint64_t outputs_C_size			 = (mat_a_dim_00 * mat_b_dim_01) * sizeof(float);

		uint64_t offset			   = 0;
		const block_q8_0* d_blocks = reinterpret_cast<const block_q8_0*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset					   = round_up_to_multiple<64>(offset + blocks_size);

		const float* d_floats = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset				  = round_up_to_multiple<64>(offset + floats_B_size);

		float* d_outputs = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		if (mat_b_dim_01 <= 4) {
			cuda_mul_mat<mat_a_dim_00, mat_a_dim_01, mat_b_dim_00>::impl(buffer, current_index, floats, blocks, outputs, mat_b_dim_01);
		} else {
			const dim3 blockDim(mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size, mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size);
			const dim3 gridDim((mat_b_dim_01 + mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size - 1) /
					mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size,
				(mat_a_dim_00 + mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size - 1) / mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size);
			cuda_mul_mat_kernel_optimized<mat_a_dim_00, mat_a_dim_01, mul_mat_type><<<gridDim, blockDim>>>(d_blocks, d_floats, d_outputs, mat_b_dim_01);
		}

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "CUDA kernel execution failed: " + std::string(cudaGetErrorString(err)) << std::endl;
		}

		++current_index;
		return current_outputs.size() * sizeof(float);
	}
};

template<bnch_swt::string_literal rhs> inline void compare_outputs(const std::vector<std::vector<float>>& outputs01, const std::vector<std::vector<float>>& outputs02,
	uint64_t total_iterations, uint64_t matC_total_elems) {
	static constexpr float relative_tolerance = 3e-2f;
	static constexpr float absolute_tolerance = 1e-30f;
	for (uint64_t x = 0; x < total_iterations; ++x) {
		for (uint64_t y = 0; y < matC_total_elems; ++y) {
			const float val1 = outputs01[x][y];
			const float val2 = outputs02[x][y];

			if (std::isnan(val1) && std::isnan(val2)) {
				continue;
			}
			if (std::isnan(val1) || std::isnan(val2)) {
				std::cerr << rhs.operator std::string_view() << ": NaN mismatch at [" << x << "," << y << "]: " << val1 << " vs " << val2 << std::endl;
				break;
			}
			if (std::isinf(val1) && std::isinf(val2)) {
				if ((val1 > 0) == (val2 > 0)) {
					continue;
				} else {
					std::cerr << rhs.operator std::string_view() << ": Infinity sign mismatch at [" << x << "," << y << "]: " << val1 << " vs " << val2 << std::endl;
					break;
				}
			}
			if (std::isinf(val1) || std::isinf(val2)) {
				std::cerr << rhs.operator std::string_view() << ": Infinity mismatch at [" << x << "," << y << "]: " << val1 << " vs " << val2 << std::endl;
				break;
			}

			const float abs_diff = std::abs(val1 - val2);
			const float max_val	 = std::max(std::abs(val1), std::abs(val2));

			const bool values_match = (abs_diff <= absolute_tolerance) || (abs_diff <= relative_tolerance * max_val);

			if (!values_match) {
				std::cerr << rhs.operator std::string_view() << ": Value mismatch at [" << x << "," << y << "]: " << val1 << " vs " << val2 << std::endl;
				std::cerr << rhs.operator std::string_view() << ": Relative difference: " << (abs_diff / max_val) * 100.0f << "%" << std::endl;
				break;
			}
		}
	}
}

template<uint64_t matA_dim_00, uint64_t matA_dim_01, uint64_t matB_dim_00, uint64_t matB_dim_01, mul_mat_types mul_mat_type> BNCH_SWT_INLINE void test_function() {
	static constexpr uint64_t matC_dim_00{ matA_dim_00 };
	static constexpr uint64_t matC_dim_01{ matB_dim_01 };
	static constexpr uint64_t matC_total_elems{ matC_dim_00 * matC_dim_01 };
	static constexpr uint64_t total_blocks_a{ static_cast<uint64_t>(static_cast<float>(matA_dim_00 * matA_dim_01) * static_cast<float>(sizeof(block_q8_0)) /
												  static_cast<float>(std::size(block_q8_0{}.qs))) /
		sizeof(block_q8_0) };
	static constexpr uint64_t total_floats_b{ matB_dim_00 * matB_dim_01 };
	auto blocks = generate_blocks_final<total_iterations, total_blocks_a>();
	auto floats = generate_floats_final<total_iterations, total_floats_b>();
	std::vector<std::vector<float>> outputs01{};
	std::vector<std::vector<float>> outputs02{};
	outputs01.resize(total_iterations);
	outputs02.resize(total_iterations);
	for (uint64_t x = 0; x < total_iterations; ++x) {
		outputs01[x].resize(matC_total_elems);
		outputs02[x].resize(matC_total_elems);
	}

	static constexpr bnch_swt::string_literal stage_name{ "mul_mat: [" + bnch_swt::internal::toStringLiteral<matA_dim_00>() + "x" +
		bnch_swt::internal::toStringLiteral<matA_dim_01>() + " * " + bnch_swt::internal::toStringLiteral<matB_dim_00>() + "x" + bnch_swt::internal::toStringLiteral<matB_dim_01>() +
		"]" };
	static constexpr uint64_t total_elements_A = matA_dim_00 * matA_dim_01;
	static constexpr uint64_t total_blocks_A   = (total_elements_A + 32 - 1) / 32;
	static constexpr uint64_t blocks_size	   = total_blocks_A * sizeof(block_q8_0);
	static constexpr uint64_t floats_B_count   = matB_dim_00 * matB_dim_01;
	static constexpr uint64_t floats_B_size	   = floats_B_count * sizeof(float);
	static constexpr uint64_t outputs_C_count  = matA_dim_00 * matB_dim_01;
	static constexpr uint64_t outputs_C_size   = outputs_C_count * sizeof(float);

	uint64_t total_buffer_size = 0;
	total_buffer_size += round_up_to_multiple<64>(blocks_size);
	total_buffer_size += round_up_to_multiple<64>(floats_B_size);
	total_buffer_size += round_up_to_multiple<64>(outputs_C_size);

	cuda_buffer buffer{};
	buffer.init(total_buffer_size);

	uint64_t current_index{};
	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrep<"ggml_cuda_mul_mat",
		cuda_mul_mat_01_prep<matA_dim_00, matA_dim_01, matB_dim_00>, ggml_cuda_mul_mat<matA_dim_00, matA_dim_01, matB_dim_00>>(buffer, current_index, floats, blocks, outputs01,
		matB_dim_01);
	current_index = 0;
	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrep<"cuda_mul_mat",
		cuda_mul_mat_01_prep<matA_dim_00, matA_dim_01, matB_dim_00>, nihilus_cuda_mul_mat<matA_dim_00, matA_dim_01, matB_dim_00, mul_mat_type>>(buffer, current_index, floats,
		blocks, outputs02, matB_dim_01);

	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::printResults();
	compare_outputs<"outputs02">(outputs01, outputs02, total_iterations, matC_total_elems);
};

int main() {
	test_function<14336, 4096, 4096, 1, mul_mat_types::ffn_up>();
	test_function<14336, 4096, 4096, 2, mul_mat_types::ffn_up>();
	test_function<14336, 4096, 4096, 4, mul_mat_types::ffn_up>();
	test_function<14336, 4096, 4096, 8, mul_mat_types::ffn_up>();
	test_function<14336, 4096, 4096, 16, mul_mat_types::ffn_up>();
	test_function<14336, 4096, 4096, 32, mul_mat_types::ffn_up>();
	test_function<14336, 4096, 4096, 64, mul_mat_types::ffn_up>();
	test_function<14336, 4096, 4096, 128, mul_mat_types::ffn_up>();
	test_function<14336, 4096, 4096, 256, mul_mat_types::ffn_up>();
	test_function<14336, 4096, 4096, 512, mul_mat_types::ffn_up>();
	test_function<14336, 4096, 4096, 1024, mul_mat_types::ffn_up>();
	test_function<14336, 4096, 4096, 2048, mul_mat_types::ffn_up>();
	test_function<14336, 4096, 4096, 4096, mul_mat_types::ffn_up>();
	test_function<4096, 4096, 4096, 1, mul_mat_types::q>();
	test_function<4096, 4096, 4096, 2, mul_mat_types::q>();
	test_function<4096, 4096, 4096, 4, mul_mat_types::q>();
	test_function<4096, 4096, 4096, 8, mul_mat_types::q>();
	test_function<4096, 4096, 4096, 16, mul_mat_types::q>();
	test_function<4096, 4096, 4096, 32, mul_mat_types::q>();
	test_function<4096, 4096, 4096, 64, mul_mat_types::q>();
	test_function<4096, 4096, 4096, 128, mul_mat_types::q>();
	test_function<4096, 4096, 4096, 256, mul_mat_types::q>();
	test_function<4096, 4096, 4096, 512, mul_mat_types::q>();
	test_function<4096, 4096, 4096, 1024, mul_mat_types::q>();
	test_function<4096, 4096, 4096, 2048, mul_mat_types::q>();
	test_function<4096, 4096, 4096, 4096, mul_mat_types::q>();
	return 0;
}
