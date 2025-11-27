/*
	MIT License

	Copyright (c) 2024 RealTimeChris

	Permission is hereby granted, free of charge, to any person obtaining a copy of this
	software and associated documentation files (the "Software"), to deal in the Software
	without restriction, including without limitation the rights to use, copy, modify, merge,
	publish, distribute, sublicense, and/or sell copies of the Software, and to permit
	persons to whom the Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all copies or
	substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	DEALINGS IN THE SOFTWARE.
*/
/// https://github.com/RealTimeChris/BenchmarkSuite
/*
    MIT License
    Copyright (c) 2024 RealTimeChris
    
    Compute vs Memory Bound Workload Analysis
    Using __device__ globals for kernel access!
*/
#include <BnchSwt/index,hpp>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>

static constexpr uint64_t total_iterations{ 100 };
static constexpr uint64_t measured_iterations{ 10 };
static constexpr size_t ARRAY_SIZE = 64 * 1024 * 1024;// 64M floats = 256 MB

// ============================================================================
// DEVICE-SIDE GLOBALS (accessible from both host and device!)
// ============================================================================
__device__ float* g_d_input	 = nullptr;
__device__ float* g_d_output = nullptr;

// Host-side copies for management
float* h_d_input  = nullptr;
float* h_d_output = nullptr;

// ============================================================================
// FRAMEWORK-COMPATIBLE KERNELS
// ============================================================================

// 1. Pure Memory Bound (Intensity = 0)
struct kernel_memory_only {
	BNCH_SWT_DEVICE static void impl() {
		size_t idx	  = blockIdx.x * blockDim.x + threadIdx.x;
		size_t stride = blockDim.x * gridDim.x;

		for (size_t i = idx; i < ARRAY_SIZE; i += stride) {
			g_d_output[i] = g_d_input[i];// ✅ Now accessible!
		}
	}
};

// 2. SAXPY: y = a*x + y (Intensity ≈ 0.17)
struct kernel_saxpy {
	BNCH_SWT_DEVICE static void impl() {
		size_t idx	  = blockIdx.x * blockDim.x + threadIdx.x;
		size_t stride = blockDim.x * gridDim.x;

		for (size_t i = idx; i < ARRAY_SIZE; i += stride) {
			g_d_output[i] = 2.5f * g_d_input[i] + g_d_output[i];
		}
	}
};

// 3. Polynomial evaluation (Intensity ≈ 1.25)
struct kernel_polynomial {
	BNCH_SWT_DEVICE static void impl() {
		size_t idx	  = blockIdx.x * blockDim.x + threadIdx.x;
		size_t stride = blockDim.x * gridDim.x;

		for (size_t i = idx; i < ARRAY_SIZE; i += stride) {
			float x		  = g_d_input[i];
			float result  = 1.0f + x * (2.0f + x * (3.0f + x * (4.0f + x * (5.0f + x * 6.0f))));
			g_d_output[i] = result;
		}
	}
};

// 4. Transcendental functions (Intensity ≈ 12.5)
struct kernel_transcendental {
	BNCH_SWT_DEVICE static void impl() {
		size_t idx	  = blockIdx.x * blockDim.x + threadIdx.x;
		size_t stride = blockDim.x * gridDim.x;

		for (size_t i = idx; i < ARRAY_SIZE; i += stride) {
			float x		  = g_d_input[i];
			float result  = sinf(x) * cosf(x) + expf(x * 0.1f) + sqrtf(fabsf(x));
			g_d_output[i] = result;
		}
	}
};

// 5. FMA Heavy (Intensity = 16)
struct kernel_fma_heavy {
	BNCH_SWT_DEVICE static void impl() {
		size_t idx	  = blockIdx.x * blockDim.x + threadIdx.x;
		size_t stride = blockDim.x * gridDim.x;

		for (size_t i = idx; i < ARRAY_SIZE; i += stride) {
			float x	  = g_d_input[i];
			float acc = 0.0f;

#pragma unroll 64
			for (int j = 0; j < 64; ++j) {
				acc = fmaf(x, 1.01f, acc);
			}

			g_d_output[i] = acc;
		}
	}
};

// 6. Extreme Compute (Intensity = 128)
struct kernel_compute_extreme {
	BNCH_SWT_DEVICE static void impl() {
		size_t idx	  = blockIdx.x * blockDim.x + threadIdx.x;
		size_t stride = blockDim.x * gridDim.x;

		for (size_t i = idx; i < ARRAY_SIZE; i += stride) {
			float x	  = g_d_input[i];
			float acc = 0.0f;

#pragma unroll 8
			for (int outer = 0; outer < 8; ++outer) {
#pragma unroll 64
				for (int inner = 0; inner < 64; ++inner) {
					acc = fmaf(x, 1.001f, acc);
				}
			}

			g_d_output[i] = acc;
		}
	}
};

// 7. Shared Memory - Light Compute
struct compute_light_shared {
	BNCH_SWT_DEVICE static void impl() {
		__shared__ float data[1024];
		int tid = threadIdx.x;

		if (tid < 1024) {
			data[tid] = tid * 1.5f;
		}
		__syncthreads();

		for (int iter = 0; iter < 100; ++iter) {
			if (tid < 1024) {
				data[tid] = data[tid] * 1.01f + 0.5f;
			}
			__syncthreads();
		}
	}
};

// 8. Shared Memory - Heavy Compute
struct compute_heavy_shared {
	BNCH_SWT_DEVICE static void impl() {
		__shared__ float data[1024];
		int tid = threadIdx.x;

		if (tid < 1024) {
			data[tid] = tid * 1.5f;
		}
		__syncthreads();

		for (int iter = 0; iter < 100; ++iter) {
			if (tid < 1024) {
				float acc = data[tid];
#pragma unroll 32
				for (int j = 0; j < 32; ++j) {
					acc = fmaf(acc, 1.001f, 0.5f);
				}
				data[tid] = acc;
			}
			__syncthreads();
		}
	}
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================
void allocate_device_memory() {
	// Allocate device memory
	cudaMalloc(&h_d_input, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&h_d_output, ARRAY_SIZE * sizeof(float));

	// Copy pointers to device-side globals
	cudaMemcpyToSymbol(g_d_input, &h_d_input, sizeof(float*));
	cudaMemcpyToSymbol(g_d_output, &h_d_output, sizeof(float*));

	// Initialize input data
	std::vector<float> h_data(ARRAY_SIZE);
	for (size_t i = 0; i < ARRAY_SIZE; ++i) {
		h_data[i] = static_cast<float>(i) / 1000.0f;
	}
	cudaMemcpy(h_d_input, h_data.data(), ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	// Initialize output to non-zero for SAXPY
	cudaMemcpy(h_d_output, h_data.data(), ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
}

void cleanup_device_memory() {
	if (h_d_input)
		cudaFree(h_d_input);
	if (h_d_output)
		cudaFree(h_d_output);
}

void print_roofline_header() {
	std::cout << std::string(120, '=') << "\n";
	std::cout << std::left << std::setw(30) << "Kernel Name" << std::setw(12) << "Time (ms)" << std::setw(15) << "Bandwidth" << std::setw(15) << "GFLOPS" << std::setw(15)
			  << "Intensity" << std::setw(20) << "Bottleneck"
			  << "\n";
	std::cout << std::string(120, '=') << "\n";
}

void print_roofline_result(const char* name, double time_ms, uint64_t bytes, uint64_t flops) {
	double bandwidth_gb = (bytes / (time_ms * 1e-3)) / 1e9;
	double gflops		= flops / (time_ms * 1e-3) / 1e9;
	double intensity	= static_cast<double>(flops) / static_cast<double>(bytes);

	const char* bottleneck = intensity < 10.0 ? "MEMORY BOUND" : "COMPUTE BOUND";

	std::cout << std::left << std::setw(30) << name << std::setw(12) << std::fixed << std::setprecision(2) << time_ms << std::setw(15) << std::fixed << std::setprecision(2)
			  << bandwidth_gb << std::setw(15) << std::fixed << std::setprecision(2) << gflops << std::setw(15) << std::fixed << std::setprecision(2) << intensity << std::setw(20)
			  << bottleneck << "\n";
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
	std::cout << "\n" << std::string(120, '=') << "\n";
	std::cout << "  COMPUTE vs MEMORY BOUND WORKLOAD ANALYSIS (RTX 5070 Ti)\n";
	std::cout << "  Using BenchmarkSuite Framework Result Collection\n";
	std::cout << std::string(120, '=') << "\n\n";

	// GPU info
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout << "GPU: " << prop.name << "\n";
	std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
	double theoretical_bw = (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	std::cout << "Peak Memory Bandwidth: " << theoretical_bw << " GB/s\n";
	std::cout << "Test array size: " << (ARRAY_SIZE * sizeof(float) / (1024 * 1024)) << " MB\n\n";

	// Allocate memory
	allocate_device_memory();

	// Grid/block config
	dim3 block(256);
	dim3 grid(2048);

	// Bytes transferred per kernel execution
	size_t bytes_transferred = ARRAY_SIZE * sizeof(float) * 2;// Read + Write

	// ========================================================================
	// RUN ALL BENCHMARKS
	// ========================================================================
	using benchmark = bnch_swt::benchmark_stage<"Roofline Analysis", total_iterations, measured_iterations, bnch_swt::benchmark_types::cuda>;

	std::cout << "Running benchmarks (this will take ~30 seconds)...\n\n";

	// Memory bound tests
	benchmark::run_benchmark<"01_Memory_Copy", kernel_memory_only>(grid, block, 0, bytes_transferred);

	benchmark::run_benchmark<"02_SAXPY", kernel_saxpy>(grid, block, 0, bytes_transferred);

	benchmark::run_benchmark<"03_Polynomial", kernel_polynomial>(grid, block, 0, bytes_transferred);

	benchmark::run_benchmark<"04_Transcendental", kernel_transcendental>(grid, block, 0, bytes_transferred);

	benchmark::run_benchmark<"05_FMA_Heavy", kernel_fma_heavy>(grid, block, 0, bytes_transferred);

	benchmark::run_benchmark<"06_Extreme_Compute", kernel_compute_extreme>(grid, block, 0, bytes_transferred);

	// Shared memory tests
	benchmark::run_benchmark<"07_Shared_Light", compute_light_shared>(dim3(32), dim3(256), 4096, 1024 * 100 * 2 * 4);

	benchmark::run_benchmark<"08_Shared_Heavy", compute_heavy_shared>(dim3(32), dim3(256), 4096, 1024 * 100 * 64 * 4);

	// ========================================================================
	// PRINT FRAMEWORK RESULTS
	// ========================================================================
	std::cout << "\n";
	benchmark::print_results();

	// ========================================================================
	// EXTRACT AND ANALYZE RESULTS
	// ========================================================================
	std::cout << "\n\n" << std::string(120, '=') << "\n";
	std::cout << "  ROOFLINE MODEL ANALYSIS\n";
	std::cout << std::string(120, '=') << "\n\n";

	auto results = benchmark::get_results();

	print_roofline_header();

	// Define FLOPS per element for each kernel
	struct KernelInfo {
		const char* name_prefix;
		uint64_t flops_per_element;
		const char* display_name;
	};

	std::vector<KernelInfo> kernel_info = { { "01_Memory_Copy", 0, "Memory Copy (0 FLOPS)" }, { "02_SAXPY", 2, "SAXPY (2 FLOPS/elem)" },
		{ "03_Polynomial", 10, "Polynomial (10 FLOPS/elem)" }, { "04_Transcendental", 100, "Transcendental (100 FLOPS)" }, { "05_FMA_Heavy", 128, "FMA Heavy (128 FLOPS/elem)" },
		{ "06_Extreme_Compute", 1024, "Extreme (1024 FLOPS/elem)" }, { "07_Shared_Light", 0, "Shared Light (200 FLOPS)" }, { "08_Shared_Heavy", 0, "Shared Heavy (6400 FLOPS)" } };

	for (auto& result: results) {
		KernelInfo* info = nullptr;
		for (auto& ki: kernel_info) {
			if (result.name.find(ki.name_prefix) != std::string::npos) {
				info = &ki;
				break;
			}
		}

		if (!info)
			continue;

		double time_ms = result.time_in_ns / 1e6;
		uint64_t bytes = result.bytes_processed;
		uint64_t flops = 0;

		if (info->flops_per_element > 0) {
			flops = ARRAY_SIZE * info->flops_per_element;
		} else if (result.name.find("07_Shared_Light") != std::string::npos) {
			flops = 32 * 1024 * 100 * 2;
		} else if (result.name.find("08_Shared_Heavy") != std::string::npos) {
			flops = 32 * 1024 * 100 * 64;
		}

		print_roofline_result(info->display_name, time_ms, bytes, flops);
	}

	std::cout << std::string(120, '=') << "\n\n";

	// ========================================================================
	// PERFORMANCE SUMMARY
	// ========================================================================
	std::cout << "PERFORMANCE SUMMARY:\n";
	std::cout << std::string(120, '-') << "\n";

	if (results.size() > 0) {
		auto& memory_bound = results[0];
		double peak_bw_gb  = (memory_bound.bytes_processed / (memory_bound.time_in_ns * 1e-9)) / 1e9;
		std::cout << "Peak Measured Bandwidth:  " << std::fixed << std::setprecision(2) << peak_bw_gb << " GB/s (" << (peak_bw_gb / theoretical_bw * 100.0)
				  << "% of theoretical)\n";

		std::cout << "\nTHROUGHPUT STABILITY:\n";
		for (auto& result: results) {
			std::cout << "  " << std::left << std::setw(30) << result.name << ": ±" << std::fixed << std::setprecision(2) << result.throughput_percentage_deviation << "%\n";
		}
	}

	std::cout << std::string(120, '=') << "\n\n";

	std::cout << "INTERPRETATION:\n";
	std::cout << "  • Memory Bound (Intensity < 10):  Performance limited by bandwidth\n";
	std::cout << "  • Compute Bound (Intensity > 10): Performance limited by FLOPS\n";
	std::cout << "  • Transition Point: Where bottleneck changes from memory → compute\n\n";

	std::cout << "OPTIMIZATION HINTS:\n";
	std::cout << "  • Memory bound kernels: Focus on coalescing, caching, shared memory\n";
	std::cout << "  • Compute bound kernels: Focus on ILP, occupancy, instruction mix\n";
	std::cout << "  • Your RTX 5070 Ti theoretical peak: ~" << theoretical_bw << " GB/s\n";
	std::cout << std::string(120, '=') << "\n";

	// Cleanup
	cleanup_device_memory();

	return 0;
}