#include <BnchSwt/BenchmarkSuite.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

static constexpr uint64_t total_iterations{ 1 };
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
	q8_quant quants[32]{};
	uint16_t scale{};
};

inline static uint16_t fp32_to_fp16(float f) {
	return static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0));
}

template<std::size_t count> inline std::vector<block_q8_0> generate_blocks() {
	std::vector<block_q8_0> result;
	result.reserve(count);

	for (std::size_t i = 0; i < count; ++i) {
		block_q8_0 block{};

		for (auto& q: block.quants) {
			q = static_cast<int8_t>((bnch_swt::random_generator::generateValue<uint8_t>() % 13) - 6);
		}

		float scale_float = (bnch_swt::random_generator::generateValue<float>() / std::numeric_limits<float>::max()) * 0.1f;

		block.scale = static_cast<uint16_t>(fp32_to_fp16(scale_float));

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
		float raw_val	 = bnch_swt::random_generator::generateValue<float>();
		float normalized = (raw_val / std::numeric_limits<float>::max());
		result.emplace_back(normalized);
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

template<uint64_t matA_dim_00, uint64_t matA_dim_01, uint64_t matB_dim_01, uint64_t block_size> struct reference_mul_mat {
	BNCH_SWT_INLINE static uint64_t impl(uint64_t& current_index, std::vector<std::vector<float>>& floats, std::vector<std::vector<block_q8_0>>& blocks,
		std::vector<std::vector<float>>& outputs) {
		const auto& current_blocks = blocks[current_index];
		const auto& current_floats = floats[current_index];
		auto& current_outputs	   = outputs[current_index];

		for (uint64_t row = 0; row < matA_dim_00; ++row) {
			for (uint64_t col = 0; col < matB_dim_01; ++col) {
				float sum = 0.0f;

				for (uint64_t k = 0; k < matA_dim_01; ++k) {
					const uint64_t block_idx	 = (row * matA_dim_01 + k) / block_size;
					const uint64_t elem_in_block = (row * matA_dim_01 + k) % block_size;

					const auto& block  = current_blocks[block_idx];
					const float scale  = __half2float(*reinterpret_cast<const __half*>(&block.scale));
					const float a_elem = scale * static_cast<float>(block.quants[elem_in_block]);

					const float b_elem = current_floats[k * matB_dim_01 + col];

					sum += a_elem * b_elem;
				}

				current_outputs[row * matB_dim_01 + col] = sum;
			}
		}

		return current_outputs.size() * sizeof(float);
	}
};

template<uint64_t matA_dim_00, uint64_t matA_dim_01, uint64_t block_size> struct cuda_mul_mat_01_prep {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats, std::vector<std::vector<block_q8_0>>& blocks,
		std::vector<std::vector<float>>& outputs, uint64_t matB_dim_01) {
		constexpr uint64_t total_blocks_A = ((matA_dim_00 * matA_dim_01) + block_size - 1) / block_size;
		constexpr uint64_t blocks_size	  = total_blocks_A * sizeof(block_q8_0);
		const uint64_t floats_B_size	  = (matA_dim_01 * matB_dim_01) * sizeof(float);
		const uint64_t outputs_C_size	  = (matA_dim_00 * matB_dim_01) * sizeof(float);
		uint64_t offset					  = 0;
		block_q8_0* d_blocks			  = reinterpret_cast<block_q8_0*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset							  = round_up_to_multiple<64>(offset + blocks_size);

		float* d_floats = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset			= round_up_to_multiple<64>(offset + floats_B_size);

		float* d_outputs = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);

		cudaMemset(d_floats, 0, floats_B_size);
		cudaMemset(d_outputs, 0, outputs_C_size);
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

template<uint64_t matA_dim_00, uint64_t matA_dim_01, uint64_t block_size>
__global__ void ggml_cuda_mul_mat_kernel(const float* input01, const block_q8_0* input02, float* output, uint64_t matB_dim_01) {
	const uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
	const uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= matA_dim_00 || col >= matB_dim_01)
		return;

	float sum = 0.0f;

	const uint64_t k_end = matA_dim_01 & ~3;

	uint64_t k = 0;
	for (; k < k_end; k += 4) {
#pragma unroll
		for (uint64_t i = 0; i < 4; ++i) {
			const uint64_t k_idx		 = k + i;
			const uint64_t linear_idx	 = row * matA_dim_01 + k_idx;
			const uint64_t block_idx	 = linear_idx / block_size;
			const uint64_t elem_in_block = linear_idx % block_size;

			const block_q8_0& block = input02[block_idx];
			const float scale		= __half2float(*reinterpret_cast<const __half*>(&block.scale));
			const float a_elem		= scale * static_cast<float>(block.quants[elem_in_block]);
			const float b_elem		= input01[k_idx * matB_dim_01 + col];

			sum += a_elem * b_elem;
		}
	}

	for (; k < matA_dim_01; ++k) {
		const uint64_t linear_idx	 = row * matA_dim_01 + k;
		const uint64_t block_idx	 = linear_idx / block_size;
		const uint64_t elem_in_block = linear_idx % block_size;

		const block_q8_0& block = input02[block_idx];
		const float scale		= __half2float(*reinterpret_cast<const __half*>(&block.scale));
		const float a_elem		= scale * static_cast<float>(block.quants[elem_in_block]);
		const float b_elem		= input01[k * matB_dim_01 + col];

		sum += a_elem * b_elem;
	}

	output[row * matB_dim_01 + col] = sum;
}

template<uint64_t matA_dim_00, uint64_t matA_dim_01, uint64_t block_size> struct ggml_cuda_mul_mat {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats, std::vector<std::vector<block_q8_0>>& blocks,
		std::vector<std::vector<float>>& outputs, uint64_t matB_dim_01) {
		auto& current_outputs = outputs[current_index];

		static constexpr uint64_t total_blocks_A = ((matA_dim_00 * matA_dim_01) + block_size - 1) / block_size;
		static constexpr uint64_t blocks_size	 = total_blocks_A * sizeof(block_q8_0);
		const uint64_t floats_B_size			 = (matA_dim_01 * matB_dim_01) * sizeof(float);
		const uint64_t outputs_C_size			 = (matA_dim_00 * matB_dim_01) * sizeof(float);

		uint64_t offset			   = 0;
		const block_q8_0* d_blocks = reinterpret_cast<const block_q8_0*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset					   = round_up_to_multiple<64>(offset + blocks_size);

		const float* d_floats = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset				  = round_up_to_multiple<64>(offset + floats_B_size);

		float* d_outputs = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);

		uint64_t block_dim_x, block_dim_y;
		if (matB_dim_01 <= 4) {
			block_dim_x = matB_dim_01;
			block_dim_y = 256 / block_dim_x;
		} else if (matA_dim_00 <= 16) {
			block_dim_x = 32;
			block_dim_y = 16;
		} else {
			block_dim_x = 16;
			block_dim_y = 32;
		}

		block_dim_x = std::min(block_dim_x, matB_dim_01);
		block_dim_y = std::min(block_dim_y, matA_dim_00);

		const uint64_t grid_dim_x = (matB_dim_01 + block_dim_x - 1) / block_dim_x;
		const uint64_t grid_dim_y = (matA_dim_00 + block_dim_y - 1) / block_dim_y;

		dim3 blockDim(static_cast<uint64_t>(block_dim_x), static_cast<uint64_t>(block_dim_y));
		dim3 gridDim(static_cast<uint64_t>(grid_dim_x), static_cast<uint64_t>(grid_dim_y));

		ggml_cuda_mul_mat_kernel<matA_dim_00, matA_dim_01, block_size><<<gridDim, blockDim>>>(d_floats, d_blocks, d_outputs, matB_dim_01);

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
	static constexpr uint64_t tile_size{ 16 };
};

template<> struct mul_mat_params<mul_mat_types::ffn_up, kernel_type_profiles::q8_gqa> {
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
	static constexpr uint64_t optimal_grid_size{ 210ull };
};

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename value_type> using x_type = decltype(std::remove_cvref_t<value_type>::x);

template<typename value_type>
concept uint_cuda_types = std::is_unsigned_v<x_type<value_type>> && std::is_integral_v<x_type<value_type>>;

template<typename value_type>
concept int_cuda_types = std::is_signed_v<x_type<value_type>> && std::is_integral_v<x_type<value_type>> && !uint_cuda_types<value_type>;

template<typename value_type>
concept int8_cuda_types = int_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

template<typename value_type>
concept int16_cuda_types = int_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

template<typename value_type>
concept int32_cuda_types = int_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept int64_cuda_types = int_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept uint8_cuda_types = uint_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

template<typename value_type>
concept uint16_cuda_types = uint_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

template<typename value_type>
concept uint32_cuda_types = uint_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept uint64_cuda_types = uint_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept float_cuda_types = std::floating_point<x_type<value_type>>;

template<typename value_type>
concept float32_cuda_types = float_cuda_types<value_type> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept float64_cuda_types = float_cuda_types<value_type> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept r_value_reference_types = std::is_rvalue_reference_v<value_type>;

template<typename value_type> BNCH_SWT_INLINE __device__ constexpr value_type&& forward_device(value_type& arg) noexcept {
	return static_cast<value_type&&>(arg);
}

template<r_value_reference_types value_type> __device__ BNCH_SWT_INLINE constexpr value_type forward_device(value_type arg) noexcept {
	return arg;
}

enum class get_value_type_errors {
	invalid_type,
};

template<typename value_type>
concept dim04_types = requires() { std::remove_cvref_t<value_type>::w; };

template<typename value_type>
concept dim03_types = requires() { std::remove_cvref_t<value_type>::z; } && !dim04_types<value_type>;

template<typename value_type>
concept dim02_types = requires() { std::remove_cvref_t<value_type>::y; } && !dim03_types<value_type> && !dim04_types<value_type>;

template<typename value_type>
concept dim01_types = requires() { std::remove_cvref_t<value_type>::x; } && !dim02_types<value_type> && !dim03_types<value_type> && !dim04_types<value_type>;

template<typename value_type>
concept dim_types = requires() { std::remove_cvref_t<value_type>::x; };

template<typename value_type> struct get_value_type {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr auto impl(value_types&&... args) {
		static_assert(false, "Failed to specialize this class!");
	}
};

template<int8_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr auto impl(value_types&&... args) {
		if constexpr (dim01_types<value_type>) {
			return make_char1(forward_device<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_char2(forward_device<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_char3(forward_device<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_char4(forward_device<value_types>(args)...);
		}
	}
};

template<int16_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr auto impl(value_types&&... args) {
		if constexpr (dim01_types<value_type>) {
			return make_short1(forward_device<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_short2(forward_device<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_short3(forward_device<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_short4(forward_device<value_types>(args)...);
		}
	}
};

template<int32_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr auto impl(value_types&&... args) {
		if constexpr (dim01_types<value_type>) {
			return make_int1(forward_device<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_int2(forward_device<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_int3(forward_device<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_int4(forward_device<value_types>(args)...);
		}
	}
};

template<int64_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr auto impl(value_types&&... args) {
		if constexpr (dim01_types<value_type>) {
			return make_long1(forward_device<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_long2(forward_device<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_long3(forward_device<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_long4(forward_device<value_types>(args)...);
		}
	}
};

template<uint8_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr auto impl(value_types&&... args) {
		if constexpr (dim01_types<value_type>) {
			return make_uchar1(forward_device<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_uchar2(forward_device<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_uchar3(forward_device<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_uchar4(forward_device<value_types>(args)...);
		}
	}
};

template<uint16_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr auto impl(value_types&&... args) {
		if constexpr (dim01_types<value_type>) {
			return make_ushort1(forward_device<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_ushort2(forward_device<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_ushort3(forward_device<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_ushort4(forward_device<value_types>(args)...);
		}
	}
};

template<uint32_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr auto impl(value_types&&... args) {
		if constexpr (dim01_types<value_type>) {
			return make_uint1(forward_device<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_uint2(forward_device<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_uint3(forward_device<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_uint4(forward_device<value_types>(args)...);
		}
	}
};

template<uint64_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr auto impl(value_types&&... args) {
		if constexpr (dim01_types<value_type>) {
			return make_ulong1(forward_device<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_ulong2(forward_device<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_ulong3(forward_device<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_ulong4(forward_device<value_types>(args)...);
		}
	}
};

template<float32_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr auto impl(value_types&&... args) {
		if constexpr (dim01_types<value_type>) {
			return make_float1(forward_device<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_float2(forward_device<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_float3(forward_device<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_float4(forward_device<value_types>(args)...);
		}
	}
};

template<float64_cuda_types value_type> struct get_value_type<value_type> {
	template<typename... value_types> BNCH_SWT_INLINE __device__ static constexpr auto impl(value_types&&... args) {
		if constexpr (dim01_types<value_type>) {
			return make_double1(forward_device<value_types>(args)...);
		} else if constexpr (dim02_types<value_type>) {
			return make_double2(forward_device<value_types>(args)...);
		} else if constexpr (dim03_types<value_type>) {
			return make_double3(forward_device<value_types>(args)...);
		} else if constexpr (dim04_types<value_type>) {
			return make_double4(forward_device<value_types>(args)...);
		}
	}
};

enum class binary_op_types {
	add,
	sub,
	mul,
	div,
};

template<binary_op_types> struct binary_op_core;

template<> struct binary_op_core<binary_op_types::add> {
	template<typename value_type01, std::convertible_to<value_type01> value_type02>
	BNCH_SWT_INLINE static __device__ value_type01 impl(const value_type01& val01, value_type02&& val02) {
		return val01 + static_cast<value_type01>(val02);
	}
};

template<> struct binary_op_core<binary_op_types::sub> {
	template<typename value_type01, std::convertible_to<value_type01> value_type02>
	BNCH_SWT_INLINE static __device__ value_type01 impl(const value_type01& val01, value_type02&& val02) {
		return val01 - static_cast<value_type01>(val02);
	}
};

template<> struct binary_op_core<binary_op_types::mul> {
	template<typename value_type01, std::convertible_to<value_type01> value_type02>
	BNCH_SWT_INLINE static __device__ value_type01 impl(const value_type01& val01, value_type02&& val02) {
		return static_cast<value_type01>(val02);
	}
};

template<> struct binary_op_core<binary_op_types::div> {
	template<typename value_type01, std::convertible_to<value_type01> value_type02>
	BNCH_SWT_INLINE static __device__ value_type01 impl(const value_type01& val01, value_type02&& val02) {
		return val01 / static_cast<value_type01>(val02);
	}
};

template<binary_op_types binary_op_type> struct binary_op_base {
	using op_core_type = binary_op_core<binary_op_type>;
	template<typename value_type01, std::convertible_to<value_type01> value_type02>
	BNCH_SWT_INLINE static __device__ value_type01 impl_one(const value_type01& val01, value_type02&& val02) {
		return get_value_type<value_type01>::impl(op_core_type::impl(val01.x, forward_device<value_type02>(val02).x));
	}

	template<typename value_type01, std::convertible_to<value_type01> value_type02>
	BNCH_SWT_INLINE static __device__ value_type01 impl_two(const value_type01& val01, value_type02&& val02) {
		return get_value_type<value_type01>::impl(op_core_type::impl(val01.x, forward_device<value_type02>(val02).x),
			op_core_type::impl(val01.y, forward_device<value_type02>(val02).y));
	}

	template<typename value_type01, std::convertible_to<value_type01> value_type02>
	BNCH_SWT_INLINE static __device__ value_type01 impl_three(const value_type01& val01, value_type02&& val02) {
		return get_value_type<value_type01>::impl(op_core_type::impl(val01.x, forward_device<value_type02>(val02).x),
			op_core_type::impl(val01.y, forward_device<value_type02>(val02).y), op_core_type::impl(val01.z, forward_device<value_type02>(val02).z));
	}

	template<typename value_type01, std::convertible_to<value_type01> value_type02>
	BNCH_SWT_INLINE static __device__ value_type01 impl_four(const value_type01& val01, value_type02&& val02) {
		return get_value_type<value_type01>::impl(op_core_type::impl(val01.x, forward_device<value_type02>(val02).x),
			op_core_type::impl(val01.y, forward_device<value_type02>(val02).y), op_core_type::impl(val01.z, forward_device<value_type02>(val02).z),
			op_core_type::impl(val01.w, forward_device<value_type02>(val02).w));
	}
};

template<binary_op_types binary_op_type> struct binary_op {
	template<typename value_type01, std::convertible_to<value_type01> value_type02>
	BNCH_SWT_INLINE static __device__ value_type01 impl(const value_type01& val01, value_type02&& val02) {
		if constexpr (dim04_types<value_type01>) {
			return binary_op_base<binary_op_type>::impl_four(val01, forward_device<value_type02>(val02));
		} else if constexpr (dim03_types<value_type01>) {
			return binary_op_base<binary_op_type>::impl_three(val01, forward_device<value_type02>(val02));
		} else if constexpr (dim02_types<value_type01>) {
			return binary_op_base<binary_op_type>::impl_two(val01, forward_device<value_type02>(val02));
		} else {
			return binary_op_base<binary_op_type>::impl_one(val01, forward_device<value_type02>(val02));
		}
	}
};

template<dim_types value_type01, std::convertible_to<value_type01> value_type02>
BNCH_SWT_INLINE __device__ value_type01 operator+=(const value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::add>::impl(val01, val02);
}

template<dim_types value_type01, std::convertible_to<value_type01> value_type02>
BNCH_SWT_INLINE __device__ value_type01 operator+(const value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::add>::impl(val01, val02);
}

template<dim_types value_type01, std::convertible_to<value_type01> value_type02>
BNCH_SWT_INLINE __device__ value_type01 operator*=(const value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::mul>::impl(val01, val02);
}

template<dim_types value_type01, std::convertible_to<value_type01> value_type02>
BNCH_SWT_INLINE __device__ value_type01 operator*(const value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::mul>::impl(val01, val02);
}

template<dim_types value_type01, std::convertible_to<value_type01> value_type02>
BNCH_SWT_INLINE __device__ value_type01 operator-=(const value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::sub>::impl(val01, val02);
}

template<dim_types value_type01, std::convertible_to<value_type01> value_type02>
BNCH_SWT_INLINE __device__ value_type01 operator-(const value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::sub>::impl(val01, val02);
}

template<dim_types value_type01, std::convertible_to<value_type01> value_type02>
BNCH_SWT_INLINE __device__ value_type01 operator/=(const value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::div>::impl(val01, val02);
}

template<dim_types value_type01, std::convertible_to<value_type01> value_type02>
BNCH_SWT_INLINE __device__ value_type01 operator/(const value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::div>::impl(val01, val02);
}

template<uint64_t block_m, uint64_t block_n, uint64_t block_k, uint64_t warp_m_new, uint64_t warp_n_new, uint64_t thread_m_new, uint64_t thread_n_new> struct cuda_kernel_traits {
	static constexpr uint64_t block_tile_m	= block_m;
	static constexpr uint64_t block_tile_n	= block_n;
	static constexpr uint64_t block_tile_k	= block_k;
	static constexpr uint64_t warp_tile_m	= warp_m_new;
	static constexpr uint64_t warp_tile_n	= warp_n_new;
	static constexpr uint64_t thread_tile_m = thread_m_new;
	static constexpr uint64_t thread_tile_n = thread_n_new;

	static constexpr uint64_t warps_m			= block_m / warp_m_new;
	static constexpr uint64_t warps_n			= block_n / warp_n_new;
	static constexpr uint64_t threads_per_warp	= 32;
	static constexpr uint64_t threads_per_block = warps_m * warps_n * threads_per_warp;
};

using mul_mat_4097_to_8192 = cuda_kernel_traits<256, 128, 16, gpu_properties::warp_size * 2, 64, 4, 4>;
using mul_mat_2049_to_4096 = cuda_kernel_traits<256, 128, 16, gpu_properties::warp_size * 2, 64, 4, 4>;
using mul_mat_1025_to_2048 = cuda_kernel_traits<256, 128, 16, gpu_properties::warp_size * 2, 64, 4, 4>;
using mul_mat_1_to_1024	   = cuda_kernel_traits<32, 64, 16, gpu_properties::warp_size, 32, 4, 4>;

template<uint64_t M, typename policy>
__device__ __forceinline__ void load_smem_tile_A(float* smem_A, const block_q8_0* global_A, uint64_t K, uint64_t block_row, uint64_t k_offset) {
	constexpr uint64_t block_m = policy::block_tile_m;
	constexpr uint64_t block_k = policy::block_tile_k;

	const uint64_t tid					 = threadIdx.x;
	constexpr uint64_t threads_per_block = policy::threads_per_block;

	constexpr uint64_t vec_size			= 4;
	constexpr uint64_t total_elements	= block_m * block_k;
	constexpr uint64_t loads_per_thread = (total_elements + threads_per_block * vec_size - 1) / (threads_per_block * vec_size);

#pragma unroll
	for (uint64_t i = 0; i < loads_per_thread; i++) {
		const uint64_t elem_id = tid * vec_size + i * threads_per_block * vec_size;
		if (elem_id < total_elements) {
			const uint64_t row		  = elem_id / block_k;
			const uint64_t col		  = elem_id % block_k;
			const uint64_t global_row = block_row + row;
			const uint64_t global_col = k_offset + col;

			if (global_row < M && global_col < K) {
				const uint64_t block_idx	 = (global_row * K + global_col) / 32;
				const uint64_t elem_in_block = (global_row * K + global_col) % 32;

				const block_q8_0& block = global_A[block_idx];
				float scale				= __half2float(*reinterpret_cast<const __half*>(&block.scale));
				float4 quantized_vals = make_float4(block.quants[elem_in_block], block.quants[elem_in_block + 1], block.quants[elem_in_block + 2], block.quants[elem_in_block + 3]);
				float4 dequantized	  = binary_op<binary_op_types::mul>::impl(make_float4(scale, scale, scale, scale), quantized_vals);
			}
		}
	}
}

template<uint64_t N, typename policy> __device__ __forceinline__ void load_smem_tile_B(float* smem_B, const float* global_B, uint64_t K, uint64_t k_offset, uint64_t block_col) {
	constexpr uint64_t block_k = policy::block_tile_k;
	constexpr uint64_t block_n = policy::block_tile_n;

	const uint64_t tid					 = threadIdx.x;
	constexpr uint64_t threads_per_block = policy::threads_per_block;
	constexpr uint64_t vec_size			 = 4;
	constexpr uint64_t total_elements	 = block_k * block_n;
	constexpr uint64_t loads_per_thread	 = (total_elements + threads_per_block * vec_size - 1) / (threads_per_block * vec_size);

#pragma unroll
	for (uint64_t i = 0; i < loads_per_thread; i++) {
		const uint64_t elem_id = tid * vec_size + i * threads_per_block * vec_size;
		if (elem_id < total_elements) {
			const uint64_t row		  = elem_id / block_n;
			const uint64_t col		  = elem_id % block_n;
			const uint64_t global_row = k_offset + row;
			const uint64_t global_col = block_col + col;

			if (global_row < K && global_col < N) {
				const float4* global_B_vec = reinterpret_cast<const float4*>(global_B + global_row * N + global_col);
				float4* smem_B_vec		   = reinterpret_cast<float4*>(smem_B + row * block_n + col);
				smem_B_vec[0]			   = global_B_vec[0];
			}
		}
	}
}

template<typename policy> __device__ __forceinline__ void compute_warp_tile(float* smem_A, float* smem_B, float* accum, uint64_t warp_m, uint64_t warp_n, uint64_t k_step) {
	constexpr uint64_t warp_m_new	= policy::warp_tile_m;
	constexpr uint64_t warp_n_new	= policy::warp_tile_n;
	constexpr uint64_t thread_m_new = policy::thread_tile_m;
	constexpr uint64_t thread_n_new = policy::thread_tile_n;
	constexpr uint64_t block_k		= policy::block_tile_k;
	constexpr uint64_t block_n		= policy::block_tile_n;

	const uint64_t lane_id	= threadIdx.x % 32;
	const uint64_t thread_m = lane_id / (warp_n_new / thread_n_new);
	const uint64_t thread_n = lane_id % (warp_n_new / thread_n_new);

	float frag_A[thread_m_new];
	float frag_B[thread_n_new];

#pragma unroll
	for (uint64_t tm = 0; tm < (thread_m_new - 4); tm += 4) {
		const uint64_t smem_row					= warp_m + thread_m * thread_m_new + tm;
		*reinterpret_cast<float4*>(&frag_A[tm]) = make_float4(smem_A[smem_row * block_k + k_step], smem_A[(smem_row * block_k + k_step) + 1],
			smem_A[(smem_row * block_k + k_step) + 2], smem_A[(smem_row * block_k + k_step) + 3]);
	}

#pragma unroll
	for (uint64_t tn = 0; tn < (thread_n_new - 4); tn += 4) {
		const uint64_t smem_col					= warp_n + thread_n * thread_n_new + tn;
		*reinterpret_cast<float4*>(&frag_B[tn]) = make_float4(smem_B[k_step * block_n + smem_col], smem_B[(k_step * block_n + smem_col) + 1],
			smem_B[(k_step * block_n + smem_col) + 2], smem_B[(k_step * block_n + smem_col) + 3]);
	}

#pragma unroll
	for (uint64_t tm = 0; tm < thread_m_new; tm++) {
		float4* accum_vec = reinterpret_cast<float4*>(accum + tm * thread_n_new);

#pragma unroll
		for (uint64_t tn = 0; tn < thread_n_new; tn += 4) {
			float4 current_accum = accum_vec[tn / 4];
			float4 frag_A_vec	 = make_float4(frag_A[tm], frag_A[tm], frag_A[tm], frag_A[tm]);
			float4 frag_B_vec	 = make_float4(frag_B[tn], frag_B[tn + 1], frag_B[tn + 2], frag_B[tn + 3]);
			float4 result		 = binary_op<binary_op_types::add>::impl(current_accum, binary_op<binary_op_types::mul>::impl(frag_A_vec, frag_B_vec));
			accum_vec[tn / 4]	 = result;
		}
	}
}

template<uint64_t M, uint64_t N, typename policy> __device__ __forceinline__ void store_output_tile(float* global_C, float* accum, uint64_t block_row, uint64_t block_col) {
	constexpr uint64_t warp_m_new	= policy::warp_tile_m;
	constexpr uint64_t warp_n_new	= policy::warp_tile_n;
	constexpr uint64_t thread_m_new = policy::thread_tile_m;
	constexpr uint64_t thread_n_new = policy::thread_tile_n;
	constexpr uint64_t warps_m		= policy::warps_m;
	constexpr uint64_t warps_n		= policy::warps_n;

	const uint64_t warp_id = threadIdx.x / 32;
	const uint64_t lane_id = threadIdx.x % 32;

	const uint64_t warp_m = (warp_id / warps_n) * warp_m_new;
	const uint64_t warp_n = (warp_id % warps_n) * warp_n_new;

	const uint64_t thread_m = lane_id / (warp_n_new / thread_n_new);
	const uint64_t thread_n = lane_id % (warp_n_new / thread_n_new);

#pragma unroll
	for (uint64_t tm = 0; tm < thread_m_new; tm++) {
		const uint64_t global_row = block_row + warp_m + thread_m * thread_m_new + tm;

		if (global_row < (M - 4)) {
			float4* global_C_vec = reinterpret_cast<float4*>(global_C + global_row * N);

#pragma unroll
			for (uint64_t tn = 0; tn < thread_n_new; tn += 4) {
				const uint64_t global_col = block_col + warp_n + thread_n * thread_n_new + tn;

				if (global_col < (N - 4)) {
					global_C_vec[global_col / 4] =
						make_float4(accum[tm * thread_n_new + tn], accum[tm * thread_n_new + tn + 1], accum[tm * thread_n_new + tn + 2], accum[tm * thread_n_new + tn + 3]);
				}
			}
		}
	}
}

template<uint64_t M, uint64_t N, typename policy> __global__ void rt_tm_gemm_kernel(const block_q8_0* A, const float* B, float* C, uint64_t K) {
	constexpr uint64_t block_m		= policy::block_tile_m;
	constexpr uint64_t block_n		= policy::block_tile_n;
	constexpr uint64_t block_k		= policy::block_tile_k;
	constexpr uint64_t warp_m_new	= policy::warp_tile_m;
	constexpr uint64_t warp_n_new	= policy::warp_tile_n;
	constexpr uint64_t thread_m_new = policy::thread_tile_m;
	constexpr uint64_t thread_n_new = policy::thread_tile_n;

	__shared__ float smem_A[2][block_m * block_k];
	__shared__ float smem_B[2][block_k * block_n];

	const uint64_t block_row = blockIdx.y * block_m;
	const uint64_t block_col = blockIdx.x * block_n;

	constexpr uint64_t accum_size = thread_m_new * thread_n_new;
	float accum[accum_size];

#pragma unroll
	for (uint64_t i = 0; i < accum_size; i++) {
		accum[i] = 0.0f;
	}

	uint64_t smem_read_stage  = 0;
	uint64_t smem_write_stage = 1;

	load_smem_tile_A<M, policy>(smem_A[0], A, K, block_row, 0);
	load_smem_tile_B<N, policy>(smem_B[0], B, K, 0, block_col);
	__syncthreads();

	for (uint64_t k_tile = 0; k_tile < K; k_tile += block_k) {
		if (k_tile + block_k < K) {
			load_smem_tile_A<M, policy>(smem_A[smem_write_stage], A, K, block_row, k_tile + block_k);
			load_smem_tile_B<N, policy>(smem_B[smem_write_stage], B, K, k_tile + block_k, block_col);
		}

		const uint64_t warp_id = threadIdx.x / 32;
		const uint64_t warp_m  = (warp_id / policy::warps_n) * warp_m_new;
		const uint64_t warp_n  = (warp_id % policy::warps_n) * warp_n_new;

#pragma unroll
		for (uint64_t k_step = 0; k_step < block_k; k_step++) {
			compute_warp_tile<policy>(smem_A[smem_read_stage], smem_B[smem_read_stage], accum, warp_m, warp_n, k_step);
		}

		__syncthreads();
		smem_read_stage ^= 1;
		smem_write_stage ^= 1;
	}

	store_output_tile<M, N, policy>(C, accum, block_row, block_col);
}

template<uint64_t matA_dim_00, uint64_t matA_dim_01, uint64_t block_size, mul_mat_types mul_mat_type> struct rt_tm_mul_mat {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats, std::vector<std::vector<block_q8_0>>& blocks,
		std::vector<std::vector<float>>& outputs, uint64_t matB_dim_01) {
		auto& current_outputs = outputs[current_index];

		static constexpr uint64_t total_blocks_A = ((matA_dim_00 * matA_dim_01) + block_size - 1) / block_size;
		static constexpr uint64_t blocks_size	 = total_blocks_A * sizeof(block_q8_0);
		const uint64_t floats_B_size			 = (matA_dim_01 * matB_dim_01) * sizeof(float);
		const uint64_t outputs_C_size			 = (matA_dim_00 * matB_dim_01) * sizeof(float);

		uint64_t offset			   = 0;
		const block_q8_0* d_blocks = reinterpret_cast<const block_q8_0*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset					   = round_up_to_multiple<64>(offset + blocks_size);

		const float* d_floats = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset				  = round_up_to_multiple<64>(offset + floats_B_size);

		float* d_outputs = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);

		if (matB_dim_01 >= 4096) {
			using policy = mul_mat_4097_to_8192;
			dim3 block(policy::threads_per_block);
			dim3 grid((matB_dim_01 + policy::block_tile_n - 1) / policy::block_tile_n, (matA_dim_00 + policy::block_tile_m - 1) / policy::block_tile_m);
			rt_tm_gemm_kernel<matA_dim_00, matA_dim_01, policy><<<grid, block>>>(d_blocks, d_floats, d_outputs, matB_dim_01);
		} else if (matB_dim_01 >= 2048) {
			using policy = mul_mat_2049_to_4096;
			dim3 block(policy::threads_per_block);
			dim3 grid((matB_dim_01 + policy::block_tile_n - 1) / policy::block_tile_n, (matA_dim_00 + policy::block_tile_m - 1) / policy::block_tile_m);
			rt_tm_gemm_kernel<matA_dim_00, matA_dim_01, policy><<<grid, block>>>(d_blocks, d_floats, d_outputs, matB_dim_01);
		} else if (matB_dim_01 >= 1024) {
			using policy = mul_mat_1025_to_2048;
			dim3 block(policy::threads_per_block);
			dim3 grid((matB_dim_01 + policy::block_tile_n - 1) / policy::block_tile_n, (matA_dim_00 + policy::block_tile_m - 1) / policy::block_tile_m);
			rt_tm_gemm_kernel<matA_dim_00, matA_dim_01, policy><<<grid, block>>>(d_blocks, d_floats, d_outputs, matB_dim_01);
		} else {
			using policy = mul_mat_1_to_1024;
			dim3 block(policy::threads_per_block);
			dim3 grid((matB_dim_01 + policy::block_tile_n - 1) / policy::block_tile_n, (matA_dim_00 + policy::block_tile_m - 1) / policy::block_tile_m);
			rt_tm_gemm_kernel<matA_dim_00, matA_dim_01, policy><<<grid, block>>>(d_blocks, d_floats, d_outputs, matB_dim_01);
		}

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
		}

		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
		}

		++current_index;
		return current_outputs.size() * sizeof(float);
	}
};

template<uint64_t matA_dim_00, uint64_t matA_dim_01, uint64_t tile_size, mul_mat_types mul_mat_type>
__global__ void cuda_mul_mat_kernel_optimized(const float* input01, const block_q8_0* input02, float* output, uint64_t matB_dim_01) {
	__shared__ float tile_A[tile_size][tile_size];
	__shared__ float tile_B[tile_size][tile_size];

	const uint64_t row = blockIdx.y * tile_size + threadIdx.y;
	const uint64_t col = blockIdx.x * tile_size + threadIdx.x;

	float sum = 0.0f;

	for (uint64_t tile = 0; tile < (matA_dim_01 + tile_size - 1) / tile_size; ++tile) {
		const uint64_t a_col = tile * tile_size + threadIdx.x;
		const uint64_t b_row = tile * tile_size + threadIdx.y;

		const uint64_t linear_idx	 = row * matA_dim_01 + a_col;
		const uint64_t block_idx	 = linear_idx / 32;
		const uint64_t elem_in_block = linear_idx % 32;

		const block_q8_0& block			 = input02[block_idx];
		const float scale				 = __half2float(*reinterpret_cast<const __half*>(&block.scale));
		tile_A[threadIdx.y][threadIdx.x] = scale * static_cast<float>(block.quants[elem_in_block]);

		tile_B[threadIdx.y][threadIdx.x] = input01[b_row * matB_dim_01 + col];

		__syncthreads();

#pragma unroll
		for (uint64_t k = 0; k < tile_size; k += gpu_properties::warp_size) {
#pragma unroll
			for (uint64_t i = 0; i < gpu_properties::warp_size; ++i) {
				if (k + i < tile_size) {
					sum += tile_A[threadIdx.y][k + i] * tile_B[k + i][threadIdx.x];
				}
			}
		}

		__syncthreads();
	}

	if (row < matA_dim_00 && col < matB_dim_01) {
		output[row * matB_dim_01 + col] = sum;
	}
}

template<uint64_t matA_dim_00, uint64_t matA_dim_01, uint64_t block_size, mul_mat_types mul_mat_type> struct rt_tm_cuda_mul_mat {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats, std::vector<std::vector<block_q8_0>>& blocks,
		std::vector<std::vector<float>>& outputs, uint64_t matB_dim_01) {
		auto& current_outputs = outputs[current_index];

		static constexpr uint64_t total_blocks_A = ((matA_dim_00 * matA_dim_01) + block_size - 1) / block_size;
		static constexpr uint64_t blocks_size	 = total_blocks_A * sizeof(block_q8_0);
		static constexpr uint64_t floats_B_size	 = (matA_dim_01 * matB_dim_01) * sizeof(float);

		uint64_t offset			   = 0;
		const block_q8_0* d_blocks = reinterpret_cast<const block_q8_0*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset					   = round_up_to_multiple<64>(offset + blocks_size);

		const float* d_floats = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset				  = round_up_to_multiple<64>(offset + floats_B_size);

		float* d_outputs = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		if constexpr (matB_dim_01 <= 4) {
			static constexpr uint64_t tile_size{ mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size / 2 };
			const dim3 blockDim(tile_size, tile_size);
			const dim3 gridDim((matB_dim_01 + tile_size - 1) / tile_size, (matA_dim_00 + tile_size - 1) / tile_size);
			cuda_mul_mat_kernel_optimized<matA_dim_00, matA_dim_01, matB_dim_01, tile_size, mul_mat_type><<<gridDim, blockDim>>>(d_floats, d_blocks, d_outputs);
		} else {
			static constexpr uint64_t tile_size{ mul_mat_params<mul_mat_type, kernel_type_profiles::q8_gqa>::tile_size };
			const dim3 blockDim(tile_size, tile_size);
			const dim3 gridDim((matB_dim_01 + tile_size - 1) / tile_size, (matA_dim_00 + tile_size - 1) / tile_size);
			cuda_mul_mat_kernel_optimized<matA_dim_00, matA_dim_01, matB_dim_01, tile_size, mul_mat_type><<<gridDim, blockDim>>>(d_floats, d_blocks, d_outputs);
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

__global__ void dequantize_q8_vectorized_kernel(const block_q8_0* input_blocks, float* output, uint64_t total_elements) {
	const uint64_t idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t stride = blockDim.x * gridDim.x;

	for (uint64_t i = idx; i < total_elements; i += stride) {
		const uint64_t block_idx	 = i / 32;
		const uint64_t elem_in_block = i % 32;

		const block_q8_0& block = input_blocks[block_idx];
		const float scale		= __half2float(*reinterpret_cast<const __half*>(&block.scale));
		output[i]				= scale * static_cast<float>(block.quants[elem_in_block]);
	}
}

template<uint64_t matA_dim_00, uint64_t matA_dim_01, uint64_t block_size, mul_mat_types mul_mat_type> struct nvcuda_cublas_mul_mat_simple {
	inline static cublasHandle_t cublas_handle;
	inline static bool handle_initialized;

	static void initialize_handle() {
		if (!handle_initialized) {
			cublasCreate(&cublas_handle);
			cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
			handle_initialized = true;
		}
	}

	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats, std::vector<std::vector<block_q8_0>>& blocks,
		std::vector<std::vector<float>>& outputs, uint64_t matB_dim_01) {
		initialize_handle();
		auto& current_outputs = outputs[current_index];

		static constexpr uint64_t total_blocks_A = ((matA_dim_00 * matA_dim_01) + block_size - 1) / block_size;
		static constexpr uint64_t blocks_size	 = total_blocks_A * sizeof(block_q8_0);
		const uint64_t floats_B_size			 = (matA_dim_01 * matB_dim_01) * sizeof(float);
		const uint64_t outputs_C_size			 = (matA_dim_00 * matB_dim_01) * sizeof(float);
		static constexpr uint64_t dequant_A_size = (matA_dim_00 * matA_dim_01) * sizeof(float);
		uint64_t offset							 = 0;
		const block_q8_0* d_blocks				 = reinterpret_cast<const block_q8_0*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset									 = round_up_to_multiple<64>(offset + blocks_size);

		const float* d_floats_B = reinterpret_cast<const float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset					= round_up_to_multiple<64>(offset + floats_B_size);

		float* d_outputs = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);
		offset			 = round_up_to_multiple<64>(offset + outputs_C_size);

		float* d_dequant_A = reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.data()) + offset);

		const uint64_t required_size = offset + dequant_A_size;
		if (required_size > buffer.size()) {
			std::cerr << "❌ BUFFER OVERFLOW! Required: " << required_size << " Available: " << buffer.size() << std::endl;
			return 0;
		}

		if (( uint8_t* )d_dequant_A < ( uint8_t* )buffer.data() || ( uint8_t* )d_dequant_A >= ( uint8_t* )buffer.data() + buffer.size()) {
			std::cerr << "❌ INVALID POINTER! d_dequant_A is outside buffer bounds!" << std::endl;
			std::cerr << "   Buffer start: " << ( void* )buffer.data() << std::endl;
			std::cerr << "   Buffer end: " << ( void* )(( uint8_t* )buffer.data() + buffer.size()) << std::endl;
			std::cerr << "   d_dequant_A: " << ( void* )d_dequant_A << std::endl;
			return 0;
		}

		const uint64_t total_elements_A = matA_dim_00 * matA_dim_01;
		const dim3 dequant_grid((total_elements_A + 1023) / 1024);
		const dim3 dequant_block(1024);

		dequantize_q8_vectorized_kernel<<<dequant_grid, dequant_block>>>(d_blocks, d_dequant_A, total_elements_A);

		cudaError_t dequant_err = cudaDeviceSynchronize();
		if (dequant_err != cudaSuccess) {
			std::cerr << "❌ Dequantization kernel failed: " << cudaGetErrorString(dequant_err) << std::endl;
			return 0;
		}


		const float alpha = 1.0f;
		const float beta  = 0.0f;

		cublasStatus_t status = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, matB_dim_01, matA_dim_00, matA_dim_01, &alpha, d_floats_B, matB_dim_01, d_dequant_A,
			matA_dim_01, &beta, d_outputs, matB_dim_01);

		if (status != CUBLAS_STATUS_SUCCESS) {
			std::cerr << "❌ cuBLAS failed with status: " << status << std::endl;
		}

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "❌ CUDA error after cuBLAS: " << cudaGetErrorString(err) << std::endl;
		}

		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cerr << "❌ CUDA synchronization failed: " << cudaGetErrorString(err) << std::endl;
		}


		++current_index;
		return current_outputs.size() * sizeof(float);
	}
};

__global__ void float_to_half_kernel(const float* input, __half* output, uint64_t total_elements) {
	const uint64_t idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t stride = blockDim.x * gridDim.x;

	for (uint64_t i = idx; i < total_elements; i += stride) {
		output[i] = __float2half(input[i]);
	}
}

template<bnch_swt::string_literal rhs> inline void compare_outputs(const std::vector<std::vector<float>>& outputs01, const std::vector<std::vector<float>>& outputs02,
	uint64_t total_iterations, uint64_t matC_total_elems) {
	static constexpr float relative_tolerance = 5e-2f;
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
				std::cerr << rhs.operator std::string_view() << ": Value mismatch at [" << x << "," << y << "]: Reference Value: " << val1 << " vs Incorrect Value: " << val2
						  << std::endl;
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
												  static_cast<float>(std::size(block_q8_0{}.quants))) /
		sizeof(block_q8_0) };
	static constexpr uint64_t total_floats_b{ matB_dim_00 * matB_dim_01 };
	auto blocks = generate_blocks_final<total_iterations, total_blocks_a>();
	auto floats = generate_floats_final<total_iterations, total_floats_b>();
	std::vector<std::vector<float>> outputs01{};
	std::vector<std::vector<float>> outputs02{};
	std::vector<std::vector<float>> outputs03{};
	outputs01.resize(total_iterations);
	outputs02.resize(total_iterations);
	outputs03.resize(total_iterations);
	for (uint64_t x = 0; x < total_iterations; ++x) {
		outputs01[x].resize(matC_total_elems);
		outputs02[x].resize(matC_total_elems);
		outputs03[x].resize(matC_total_elems);
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
	total_buffer_size += round_up_to_multiple<64>((matA_dim_00 * matA_dim_01) * sizeof(float));
	total_buffer_size += round_up_to_multiple<64>(outputs_C_size);

	cuda_buffer buffer{};
	buffer.init(total_buffer_size);

	uint64_t current_index{};
	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrep<"ggml_cuda_mul_mat",
		cuda_mul_mat_01_prep<matA_dim_00, matA_dim_01, 32>, ggml_cuda_mul_mat<matA_dim_00, matA_dim_01, 32>>(buffer, current_index, floats, blocks, outputs01, matB_dim_01);
	current_index = 0;

	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrep<"nvcuda_cublas_mul_mat",
		cuda_mul_mat_01_prep<matA_dim_00, matA_dim_01, 32>, nvcuda_cublas_mul_mat_simple<matA_dim_00, matA_dim_01, 32, mul_mat_type>>(buffer, current_index, floats, blocks,
		outputs03, matB_dim_01);
	current_index = 0;
	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrep<"rt_tm_cuda_mul_mat",
		cuda_mul_mat_01_prep<matA_dim_00, matA_dim_01, 32>, rt_tm_mul_mat<matA_dim_00, matA_dim_01, 32, mul_mat_type>>(buffer, current_index, floats, blocks, outputs02,
		matB_dim_01);

	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::printResults();
	compare_outputs<"outputs02">(outputs01, outputs02, total_iterations, matC_total_elems);
	compare_outputs<"nvcuda_cublas_mul_mat_simple producing incorrect values">(outputs01, outputs03, total_iterations, matC_total_elems);
};

int main() {
	test_function<14336, 4096, 4096, 16384, mul_mat_types::ffn_up>();
	test_function<4096, 4096, 4096, 16384, mul_mat_types::q>();
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
	test_function<14336, 4096, 4096, 8192, mul_mat_types::ffn_up>();
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
	test_function<4096, 4096, 4096, 8192, mul_mat_types::q>();
	return 0;
}