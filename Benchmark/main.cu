#include <BnchSwt/BenchmarkSuite.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

static constexpr uint64_t total_iterations{ 2 };
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

template<uint64_t matA_dim_00, uint64_t matA_dim_01, uint64_t block_size> struct reference_mul_mat {
	BNCH_SWT_INLINE static uint64_t impl(cuda_buffer& buffer, uint64_t& current_index, std::vector<std::vector<float>>& floats, std::vector<std::vector<block_q8_0>>& blocks,
		std::vector<std::vector<float>>& outputs, uint64_t matB_dim_01) {
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

		//cudaMemset(d_floats, 0, floats_B_size);
		//cudaMemset(d_outputs, 0, outputs_C_size);
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

		//err = cudaMemset(d_outputs, 0, outputs_C_size);
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
		return val01 * static_cast<value_type01>(val02);
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

template<uint64_t block_m, uint64_t block_n, uint64_t block_k, uint64_t warp_m_new, uint64_t warp_n_new, uint64_t thread_m_new, uint64_t thread_n_new> struct cuda_kernel_traits {
	static constexpr uint64_t block_tile_m		= block_m;
	static constexpr uint64_t block_tile_n		= block_n;
	static constexpr uint64_t block_tile_k		= block_k;
	static constexpr uint64_t warp_tile_m		= warp_m_new;
	static constexpr uint64_t warp_tile_n		= warp_n_new;
	static constexpr uint64_t thread_tile_m		= thread_m_new;
	static constexpr uint64_t thread_tile_n		= thread_n_new;
	static constexpr uint64_t warps_m			= block_m / warp_m_new;
	static constexpr uint64_t warps_n			= block_n / warp_n_new;
	static constexpr uint64_t threads_per_warp	= gpu_properties::warp_size;
	static constexpr uint64_t threads_per_block = warps_m * warps_n * threads_per_warp;

	static_assert(block_m > 0, "block_m must be greater than 0");
	static_assert(block_n > 0, "block_n must be greater than 0");
	static_assert(block_k > 0, "block_k must be greater than 0");
	static_assert(warp_m_new > 0, "warp_m must be greater than 0");
	static_assert(warp_n_new > 0, "warp_n must be greater than 0");
	static_assert(thread_m_new > 0, "thread_m must be greater than 0");
	static_assert(thread_n_new > 0, "thread_n must be greater than 0");

	static_assert(block_m % warp_m_new == 0, "block_m must be evenly divisible by warp_m");
	static_assert(block_n % warp_n_new == 0, "block_n must be evenly divisible by warp_n");

	static_assert(warp_m_new % thread_m_new == 0, "warp_m must be evenly divisible by thread_m");
	static_assert(warp_n_new % thread_n_new == 0, "warp_n must be evenly divisible by thread_n");

	static_assert((warp_m_new / thread_m_new) * (warp_n_new / thread_n_new) == gpu_properties::warp_size, "Warp configuration must result in exactly warp_size threads per warp");

	static_assert(threads_per_block <= gpu_properties::max_threads_per_block, "threads_per_block cannot exceed max_threads_per_block");
	static_assert(threads_per_block >= gpu_properties::warp_size, "threads_per_block must be at least warp_size");

	static_assert(block_m <= 512, "block_m should not exceed 512 for reasonable shared memory usage");
	static_assert(block_n <= 512, "block_n should not exceed 512 for reasonable shared memory usage");
	static_assert(block_k <= 64, "block_k should not exceed 64 for reasonable register usage");

	static_assert(block_k % 4 == 0, "block_k should be a multiple of 4 for vectorized loads");

	static_assert(thread_m_new <= 8, "thread_m should not exceed 8 for reasonable register usage");
	static_assert(thread_n_new <= 8, "thread_n should not exceed 8 for reasonable register usage");

	static_assert(warps_m > 0 && warps_n > 0, "Must have at least one warp in each dimension");
	static_assert(warps_m * warps_n <= 32, "Total warps per block should not exceed 32");

	static constexpr uint64_t shared_mem_usage = 2 * (block_m * block_k + block_k * block_n) * sizeof(float);
	static_assert(shared_mem_usage <= gpu_properties::shared_mem_per_block, "Estimated shared memory usage exceeds shared_mem_per_block limit");

	static_assert(threads_per_block % gpu_properties::warp_size == 0, "threads_per_block must be a multiple of warp_size");
};

using mul_mat_1_to_1024 = cuda_kernel_traits<32, 64, 16, 16, 32, 4, 4>;

template<uint64_t M, uint64_t K, typename traits>
__device__ __forceinline__ void load_smem_tile_A(float* smem_A, const block_q8_0* A_global, uint64_t N, uint64_t k_offset, uint64_t block_row) {
	constexpr uint64_t block_m			 = traits::block_tile_m;
	constexpr uint64_t block_k			 = traits::block_tile_k;
	constexpr uint64_t threads_per_block = traits::threads_per_block;
	const uint64_t tid					 = threadIdx.x;
	const uint64_t k_blocks				 = (K + 31) / 32;
	const uint64_t elements_per_block	 = block_m * block_k;
	const uint64_t vec4_elements		 = elements_per_block / 4;
	const uint64_t vec4_per_thread		 = (vec4_elements + threads_per_block - 1) / threads_per_block;
	for (uint64_t i = 0; i < vec4_per_thread; i++) {
		const uint64_t vec4_idx	  = tid + i * threads_per_block;
		const uint64_t linear_idx = vec4_idx * 4;
		const uint64_t row		  = linear_idx / block_k;
		const uint64_t col		  = linear_idx % block_k;
		const uint64_t global_row	= block_row + row;
		const uint64_t global_col	= k_offset + col;
		const uint64_t q8_block_row = global_row;
		const uint64_t q8_block_col = global_col / 32;
		const uint64_t q8_elem_idx	= global_col % 32;
		const uint64_t q8_block_idx = q8_block_row * k_blocks + q8_block_col;
		const block_q8_0& q8_block	= A_global[q8_block_idx];
		const float scale_raw		= __half2float(*reinterpret_cast<const __half*>(&q8_block.scale));
		const uint64_t smem_offset						 = row * block_k + col;
		*reinterpret_cast<float4*>(&smem_A[smem_offset]) = make_float4(static_cast<float>(q8_block.quants[q8_elem_idx]), static_cast<float>(q8_block.quants[q8_elem_idx + 1]),
															   static_cast<float>(q8_block.quants[q8_elem_idx + 2]), static_cast<float>(q8_block.quants[q8_elem_idx + 3])) *
			make_float4(scale_raw, scale_raw, scale_raw, scale_raw);
	}
}

template<uint64_t M, uint64_t K, typename traits>
__device__ __forceinline__ void load_smem_tile_B(float* smem_B, const float* B_global, uint64_t N, uint64_t k_offset, uint64_t block_col) {
	constexpr uint64_t block_n			 = traits::block_tile_n;
	constexpr uint64_t block_k			 = traits::block_tile_k;
	constexpr uint64_t threads_per_block = traits::threads_per_block;

	const uint64_t tid					 = threadIdx.x;
	const uint64_t vec4_cols_per_row	 = block_n / 4;
	const uint64_t total_vec4_loads		 = block_k * vec4_cols_per_row;
	const uint64_t vec4_loads_per_thread = (total_vec4_loads + threads_per_block - 1) / threads_per_block;

	for (uint64_t i = 0; i < vec4_loads_per_thread; i++) {
		const uint64_t vec4_idx = tid + i * threads_per_block;
		if (vec4_idx < total_vec4_loads) {
			const uint64_t row		= vec4_idx / vec4_cols_per_row;
			const uint64_t vec4_col = vec4_idx % vec4_cols_per_row;
			const uint64_t col		= vec4_col * 4;

			const uint64_t global_row = k_offset + row;
			const uint64_t global_col = block_col + col;

			if (global_row < K && global_col + 3 < N) {
				const uint64_t global_offset					 = global_row * N + global_col;
				const uint64_t smem_offset						 = row * block_n + col;
				*reinterpret_cast<float4*>(&smem_B[smem_offset]) = *reinterpret_cast<const float4*>(&B_global[global_offset]);
			} else {
				for (uint64_t elem = 0; elem < 4; elem++) {
					const uint64_t elem_global_col = global_col + elem;
					const uint64_t elem_col		   = col + elem;
					if (global_row < K && elem_global_col < N && elem_col < block_n) {
						smem_B[row * block_n + elem_col] = B_global[global_row * N + elem_global_col];
					}
				}
			}
		}
	}
}

template<typename traits> __device__ __forceinline__ void compute_warp_tile(float* smem_A, float* smem_B, float accumulator[traits::thread_tile_m][traits::thread_tile_n],
	uint64_t warp_row, uint64_t warp_col) {
	constexpr uint64_t warp_m	= traits::warp_tile_m;
	constexpr uint64_t warp_n	= traits::warp_tile_n;
	constexpr uint64_t thread_m = traits::thread_tile_m;
	constexpr uint64_t thread_n = traits::thread_tile_n;
	constexpr uint64_t block_k	= traits::block_tile_k;
	constexpr uint64_t block_n	= traits::block_tile_n;
	constexpr uint64_t block_m	= traits::block_tile_m;

	uint64_t lane_id		 = threadIdx.x % 32;
	uint64_t threads_per_row = warp_n / thread_n;
	uint64_t thread_row		 = lane_id / threads_per_row;
	uint64_t thread_col		 = lane_id % threads_per_row;

	float frag_A[thread_m];
	float frag_B[thread_n];

	for (uint64_t k = 0; k < block_k; k++) {
#pragma unroll
		for (uint64_t tm = 0; tm < thread_m; tm++) {
			uint64_t smem_row = warp_row + thread_row * thread_m + tm;
			frag_A[tm]		  = smem_A[smem_row * block_k + k];
		}

#pragma unroll
		for (uint64_t tn = 0; tn < thread_n; tn++) {
			uint64_t smem_col = warp_col + thread_col * thread_n + tn;
			frag_B[tn]		  = smem_B[k * block_n + smem_col];
		}

#pragma unroll
		for (uint64_t tm = 0; tm < thread_m; tm++) {
#pragma unroll
			for (uint64_t tn = 0; tn < thread_n; tn++) {
				accumulator[tm][tn] += frag_A[tm] * frag_B[tn];
			}
		}
	}
}

template<typename traits> __device__ __forceinline__ void store_output_tile(float* C_global, float accumulator[traits::thread_tile_m][traits::thread_tile_n], uint64_t M,
	uint64_t N, uint64_t block_row, uint64_t block_col, uint64_t warp_row, uint64_t warp_col) {
	constexpr uint64_t thread_m = traits::thread_tile_m;
	constexpr uint64_t thread_n = traits::thread_tile_n;
	constexpr uint64_t warp_n	= traits::warp_tile_n;

	uint64_t lane_id		 = threadIdx.x % 32;
	uint64_t threads_per_row = warp_n / thread_n;
	uint64_t thread_row		 = lane_id / threads_per_row;
	uint64_t thread_col		 = lane_id % threads_per_row;

#pragma unroll
	for (uint64_t tm = 0; tm < thread_m; tm++) {
#pragma unroll
		for (uint64_t tn = 0; tn < thread_n; tn++) {
			uint64_t global_row = block_row + warp_row + thread_row * thread_m + tm;
			uint64_t global_col = block_col + warp_col + thread_col * thread_n + tn;

			if (global_row < M && global_col < N) {
				C_global[global_row * N + global_col] = accumulator[tm][tn];
			}
		}
	}
}

template<uint64_t M, uint64_t K> __global__ void rt_tm_gemm_kernel(const block_q8_0* A, const float* B, float* C, uint64_t N) {
	using traits = mul_mat_1_to_1024;

	constexpr uint64_t block_m	= traits::block_tile_m;
	constexpr uint64_t block_n	= traits::block_tile_n;
	constexpr uint64_t block_k	= traits::block_tile_k;
	constexpr uint64_t warp_m	= traits::warp_tile_m;
	constexpr uint64_t warp_n	= traits::warp_tile_n;
	constexpr uint64_t thread_m = traits::thread_tile_m;
	constexpr uint64_t thread_n = traits::thread_tile_n;
	constexpr uint64_t warps_m	= traits::warps_m;
	constexpr uint64_t warps_n	= traits::warps_n;

	__shared__ float smem_A[2][block_m * block_k];
	__shared__ float smem_B[2][block_k * block_n];

	uint64_t block_row = blockIdx.y * block_m;
	uint64_t block_col = blockIdx.x * block_n;

	uint64_t warp_id  = threadIdx.x / 32;
	uint64_t warp_row = (warp_id / warps_n) * warp_m;
	uint64_t warp_col = (warp_id % warps_n) * warp_n;

	float accumulator[thread_m][thread_n];
#pragma unroll
	for (uint64_t tm = 0; tm < thread_m; tm++) {
#pragma unroll
		for (uint64_t tn = 0; tn < thread_n; tn++) {
			accumulator[tm][tn] = 0.0f;
		}
	}

	uint64_t smem_write_stage = 0;
	uint64_t smem_read_stage  = 0;

	load_smem_tile_A<M, K, traits>(smem_A[smem_write_stage], A, N, 0, block_row);
	load_smem_tile_B<M, K, traits>(smem_B[smem_write_stage], B, N, 0, block_col);
	__syncthreads();

	for (uint64_t k_tile = 0; k_tile < K; k_tile += block_k) {
		smem_read_stage	 = smem_write_stage;
		smem_write_stage = 1 - smem_write_stage;

		if (k_tile + block_k < K) {
			load_smem_tile_A<M, K, traits>(smem_A[smem_write_stage], A, N, k_tile + block_k, block_row);
			load_smem_tile_B<M, K, traits>(smem_B[smem_write_stage], B, N, k_tile + block_k, block_col);
		}

		compute_warp_tile<traits>(smem_A[smem_read_stage], smem_B[smem_read_stage], accumulator, warp_row, warp_col);

		__syncthreads();
	}

	store_output_tile<traits>(C, accumulator, M, N, block_row, block_col, warp_row, warp_col);
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

		using traits = mul_mat_1_to_1024;

		dim3 grid{ static_cast<uint32_t>((matB_dim_01 + traits::block_tile_n - 1) / traits::block_tile_n),
			static_cast<uint32_t>((matA_dim_00 + traits::block_tile_m - 1) / traits::block_tile_m) };
		dim3 block{ traits::threads_per_block };
		rt_tm_gemm_kernel<matA_dim_00, matA_dim_01><<<grid, block>>>(d_blocks, d_floats, d_outputs, matB_dim_01);

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
				const float abs_diff = std::abs(val1 - val2);
				const float max_val	 = std::max(std::abs(val1), std::abs(val2));

				const bool values_match = (abs_diff <= absolute_tolerance) || (abs_diff <= relative_tolerance * max_val);
				std::cerr << rhs.operator std::string_view() << ": Value mismatch at [" << x << "," << y << "]: Reference Value: " << val1 << " vs Incorrect Value: " << val2
						  << std::endl;
				std::cerr << rhs.operator std::string_view() << ": Relative difference: " << (abs_diff / max_val) * 100.0f << "%" << std::endl;
				break;
			}
			if (std::isinf(val1) && std::isinf(val2)) {
				if ((val1 > 0) == (val2 > 0)) {
					continue;
				} else {
					const float abs_diff = std::abs(val1 - val2);
					const float max_val	 = std::max(std::abs(val1), std::abs(val2));

					const bool values_match = (abs_diff <= absolute_tolerance) || (abs_diff <= relative_tolerance * max_val);
					std::cerr << rhs.operator std::string_view() << ": Value mismatch at [" << x << "," << y << "]: Reference Value: " << val1 << " vs Incorrect Value: " << val2
							  << std::endl;
					std::cerr << rhs.operator std::string_view() << ": Relative difference: " << (abs_diff / max_val) * 100.0f << "%" << std::endl;
					break;
				}
			}
			if (std::isinf(val1) || std::isinf(val2)) {
				const float abs_diff = std::abs(val1 - val2);
				const float max_val	 = std::max(std::abs(val1), std::abs(val2));

				const bool values_match = (abs_diff <= absolute_tolerance) || (abs_diff <= relative_tolerance * max_val);
				std::cerr << rhs.operator std::string_view() << ": Value mismatch at [" << x << "," << y << "]: Reference Value: " << val1 << " vs Incorrect Value: " << val2
						  << std::endl;
				std::cerr << rhs.operator std::string_view() << ": Relative difference: " << (abs_diff / max_val) * 100.0f << "%" << std::endl;
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

	//bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrep<"reference_mul_mat",
	//cuda_mul_mat_01_prep<matA_dim_00, matA_dim_01, 32>, reference_mul_mat<matA_dim_00, matA_dim_01, 32>>(buffer, current_index, floats, blocks, outputs01, matB_dim_01);
	//current_index = 0;

	//bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrep<"nvcuda_cublas_mul_mat",
	//		cuda_mul_mat_01_prep<matA_dim_00, matA_dim_01, 32>, nvcuda_cublas_mul_mat_simple<matA_dim_00, matA_dim_01, 32, mul_mat_type>>(buffer, current_index, floats, blocks,
	//		outputs03, matB_dim_01);
	current_index = 0;
	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::template runBenchmarkWithPrep<"rt_tm_cuda_mul_mat",
		cuda_mul_mat_01_prep<matA_dim_00, matA_dim_01, 32>, rt_tm_mul_mat<matA_dim_00, matA_dim_01, 32, mul_mat_type>>(buffer, current_index, floats, blocks, outputs02,
		matB_dim_01);

	bnch_swt::benchmark_stage<stage_name, total_iterations, measured_iterations>::printResults();
	compare_outputs<"outputs02">(outputs01, outputs02, total_iterations, matC_total_elems);
	//compare_outputs<"nvcuda_cublas_mul_mat_simple producing incorrect values">(outputs01, outputs03, total_iterations, matC_total_elems);
};

int main() {
	//test_function<14336, 4096, 4096, 16384, mul_mat_types::ffn_up>();
	//test_function<4096, 4096, 4096, 16384, mul_mat_types::q>();
	//test_function<14336, 4096, 4096, 1, mul_mat_types::ffn_up>();
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
	test_function<14336, 4096, 4096, 8192, mul_mat_types::ffn_up>(); /*
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
	test_function<4096, 4096, 4096, 8192, mul_mat_types::q>();*/
	return 0;
}