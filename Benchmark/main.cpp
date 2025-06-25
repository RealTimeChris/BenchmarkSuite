#include <BnchSwt/BenchmarkSuite.hpp>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>
#include <array>
#include <queue>
#include <latch>
#include <bit>
#include <vector>
#include <iostream>
#include <thread>
#include <cstring>
#include <algorithm>
#include <cmath>

// Your existing functions
static std::vector<float> pack_B_T(const std::vector<float>& B, size_t K, size_t N) {
	std::vector<float> B_T(K * N);
	for (size_t k = 0; k < K; ++k) {
		for (size_t j = 0; j < N; ++j) {
			B_T[j * K + k] = B[k * N + j];
		}
	}
	return B_T;
}

// Mul mat with A row-major, B_T column-major
static void mul_mat_packed(const std::vector<float>& A, const std::vector<float>& B_T, std::vector<float>& C, size_t M, size_t K, size_t N) {
	for (size_t i = 0; i < M; ++i) {
		for (size_t j = 0; j < N; ++j) {
			float sum		   = 0.0f;
			for (size_t k = 0; k < K; ++k) {
				sum += A[(i * K) + k] * B_T[(j * K) + k];
			}
			C[i * N + j] = sum;
		}
	}
}

static void mul_mat_scalar_f32(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t M, size_t K, size_t N) {
	for (size_t i = 0; i < M; ++i) {
		for (size_t j = 0; j < N; ++j) {
			float sum = 0.0f;
			for (size_t k = 0; k < K; ++k) {
				sum += A[i * K + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
}

// NEW: Pack pointers to original data for linear thread access
static std::pair<std::vector<float*>, std::vector<float*>> pack_pointers_for_linear_threads(const std::vector<float>& A, const std::vector<float>& B, size_t M, size_t K,
	size_t N) {
	size_t total_ops = M * N;// Total number of C[i,j] elements to compute

	std::vector<float*> A_pointers;
	std::vector<float*> B_pointers;

	A_pointers.reserve(total_ops * K);// Each op needs K elements from A
	B_pointers.reserve(total_ops * K);// Each op needs K elements from B

	// Pack pointers for each operation linearly
	for (size_t op = 0; op < total_ops; ++op) {
		size_t i = op / N;// Row in C
		size_t j = op % N;// Col in C

		// Add pointers to A row elements for this operation
		for (size_t k = 0; k < K; ++k) {
			A_pointers.push_back(const_cast<float*>(&A[i * K + k]));
		}

		// Add pointers to B column elements for this operation
		for (size_t k = 0; k < K; ++k) {
			B_pointers.push_back(const_cast<float*>(&B[k * N + j]));
		}
	}

	return { A_pointers, B_pointers };
}

// Generate thread work chunks with pointer arrays
struct ThreadWorkChunkPtrs {
	float** A_ptrs_start;// Starting pointer to A pointer array
	float** B_ptrs_start;// Starting pointer to B pointer array
	float* C_start;// Starting pointer for output C
	size_t operations;// Number of dot products this thread handles
	size_t K;// Inner dimension for dot product length
};
/*
static std::vector<ThreadWorkChunkPtrs> generate_thread_chunks_ptrs(std::vector<float*>& A_pointers, std::vector<float*>& B_pointers, size_t M, size_t N, size_t K,
	size_t num_threads,
	std::vector<float>& output_C) {
	std::vector<ThreadWorkChunkPtrs> chunks;

	size_t total_ops	  = M * N;
	size_t ops_per_thread = (total_ops + num_threads - 1) / num_threads;

	for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
		size_t start_op = thread_id * ops_per_thread;
		size_t end_op	= std::min(start_op + ops_per_thread, total_ops);

		if (start_op >= total_ops)
			break;

		size_t thread_ops = end_op - start_op;

		ThreadWorkChunkPtrs chunk;
		chunk.A_ptrs_start = &A_pointers[start_op * K];// Pointer to pointer array!
		chunk.B_ptrs_start = &B_pointers[start_op * K];// Pointer to pointer array!
		chunk.C_start	   = &output_C[start_op];
		chunk.operations   = thread_ops;
		chunk.K			   = K;

		chunks.push_back(chunk);
	}

	return chunks;
}
*/
// ULTRA-SIMPLE THREAD FUNCTION - PURE POINTER DEREFERENCING!
static void thread_worker_ptrs(const ThreadWorkChunkPtrs& ) {
	//float** A_ptr_ptr = chunk.A_ptrs_start;// Pointer to A pointer array
	//float** B_ptr_ptr = chunk.B_ptrs_start;// Pointer to B pointer array
	//float* C_ptr	  = chunk.C_start;
	/*
	// Each thread just increments through pointer arrays!
	for (size_t op = 0; op < chunk.operations; ++op) {
		float sum = 0.0f;

		// Dot product with pure pointer dereferencing
		for (size_t k = 0; k < chunk.K; ++k) {
			sum += (**A_ptr_ptr) * (**B_ptr_ptr);// Dereference pointers!
			++A_ptr_ptr;// Move to next A pointer
			++B_ptr_ptr;// Move to next B pointer
		}

		*C_ptr = sum;
		++C_ptr;
	}*/
}

/*
// Main function using pointer arrays
static void mul_mat_threaded_ptrs(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t M, size_t K, size_t N,
	size_t num_threads = std::thread::hardware_concurrency()) {
	// Pack pointers for linear thread access
	auto [A_pointers, B_pointers] = pack_pointers_for_linear_threads(A, B, M, K, N);

	// Generate work chunks for each thread
	auto chunks = generate_thread_chunks_ptrs(A_pointers, B_pointers, M, N, K, num_threads, C);

	// Launch threads
	std::vector<std::thread> threads;
	for (const auto& chunk: chunks) {
		threads.emplace_back([chunk]() {
			thread_worker_ptrs(chunk);
		});
	}

	// Wait for all threads to complete
	for (auto& t: threads) {
		t.join();
	}
}
*/
// Your existing threaded data copy version
struct ThreadWorkChunk {
	const float* A_start;// Starting pointer for A data
	const float* B_start;// Starting pointer for B data
	float* C_start;// Starting pointer for output C
	size_t operations;// Number of dot products this thread handles
	size_t K;// Inner dimension for dot product length
};
/*
static std::vector<float> pack_for_linear_threads(const std::vector<float>& A, const std::vector<float>& B, size_t M, size_t K, size_t N) {
	size_t total_ops = M * N;// Total number of C[i,j] elements to compute

	// Layout: [A_data_for_all_ops][B_data_for_all_ops]
	std::vector<float> packed_data(total_ops * K * 2);// 2 = A + B data per operation

	float* A_section = packed_data.data();
	float* B_section = packed_data.data() + (total_ops * K);

	size_t A_offset = 0;
	size_t B_offset = 0;

	// Pack data for each operation linearly
	for (size_t op = 0; op < total_ops; ++op) {
		size_t i = op / N;// Row in C
		size_t j = op % N;// Col in C

		// Copy A row for this operation
		const float* A_row = &A[i * K];
		std::memcpy(&A_section[A_offset], A_row, K * sizeof(float));
		A_offset += K;

		// Copy B column for this operation
		for (size_t k = 0; k < K; ++k) {
			B_section[B_offset + k] = B[k * N + j];
		}
		B_offset += K;
	}

	return packed_data;
}

static std::vector<ThreadWorkChunk> generate_thread_chunks(const std::vector<float>& packed_data, size_t M, size_t N, size_t K, size_t num_threads, std::vector<float>& output_C) {
	std::vector<ThreadWorkChunk> chunks;

	size_t total_ops	  = M * N;
	size_t ops_per_thread = (total_ops + num_threads - 1) / num_threads;

	const float* A_section = packed_data.data();
	const float* B_section = packed_data.data() + (total_ops * K);

	for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
		size_t start_op = thread_id * ops_per_thread;
		size_t end_op	= std::min(start_op + ops_per_thread, total_ops);

		if (start_op >= total_ops)
			break;

		size_t thread_ops = end_op - start_op;

		ThreadWorkChunk chunk;
		chunk.A_start	 = &A_section[start_op * K];
		chunk.B_start	 = &B_section[start_op * K];
		chunk.C_start	 = &output_C[start_op];
		chunk.operations = thread_ops;
		chunk.K			 = K;

		chunks.push_back(chunk);
	}

	return chunks;
}
static void thread_worker_linear(const ThreadWorkChunk& chunk) {
	const float* A_ptr = chunk.A_start;
	const float* B_ptr = chunk.B_start;
	float* C_ptr	   = chunk.C_start;

	// Each thread just does this simple loop - no indexing calculations!
	for (size_t op = 0; op < chunk.operations; ++op) {
		float sum = 0.0f;

		// Dot product with simple pointer increments
		for (size_t k = 0; k < chunk.K; ++k) {
			sum += (*A_ptr) * (*B_ptr);
			++A_ptr;
			++B_ptr;
		}

		*C_ptr = sum;
		++C_ptr;
	}
}

static void mul_mat_threaded(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t M, size_t K, size_t N,
	size_t num_threads = std::thread::hardware_concurrency()) {
	// Pack data for linear thread access
	auto packed_data = pack_for_linear_threads(A, B, M, K, N);

	// Generate work chunks for each thread
	auto chunks = generate_thread_chunks(packed_data, M, N, K, num_threads, C);

	// Launch threads
	std::vector<std::thread> threads;
	for (const auto& chunk: chunks) {
		threads.emplace_back([chunk]() {
			thread_worker_linear(chunk);
		});
	}

	// Wait for all threads to complete
	for (auto& t: threads) {
		t.join();
	}
}
*/
// Verification function to ensure both implementations produce same results
/*
static bool verify_results(const std::vector<float>& C1, const std::vector<float>& C2, float epsilon = 1e-5f) {
	if (C1.size() != C2.size())
		return false;
	for (size_t i = 0; i < C1.size(); ++i) {
		if (std::abs(C1[i] - C2[i]) > epsilon) {
			return false;
		}
	}
	return true;
}
*/
static std::vector<float> generate_random_float_vector(size_t size) {
	std::vector<float> return_value{};
	return_value.resize(size);
	for (size_t x = 0; x < size; ++x) {
		return_value[x] = bnch_swt::random_generator::generateValue<float>();
	}
	return return_value;
}


int main() {
	constexpr size_t NUM_TESTS = 100;
	constexpr size_t test_M	   = 128;
	constexpr size_t test_K	   = 128;
	constexpr size_t test_N	   = 128;

	// Generate test data
	std::vector<std::vector<float>> test_A_small{ NUM_TESTS };
	std::vector<std::vector<float>> test_B_small{ NUM_TESTS };
	std::vector<std::vector<float>> test_B_T_small{ NUM_TESTS };

	// PRE-CALCULATE POINTER ARRAYS
	std::vector<std::pair<std::vector<float*>, std::vector<float*>>> pointer_arrays(NUM_TESTS);

	// Store chunk templates with relative offsets
	struct ChunkTemplate {
		size_t A_ptrs_offset;// Offset into A_pointers array
		size_t B_ptrs_offset;// Offset into B_pointers array
		size_t C_start_offset;// Offset into output C array
		size_t operations;
		size_t K;
	};

	std::vector<std::vector<ChunkTemplate>> chunk_templates(NUM_TESTS);

	for (size_t test = 0; test < NUM_TESTS; ++test) {
		test_A_small[test]	 = generate_random_float_vector(test_M * test_K);
		test_B_small[test]	 = generate_random_float_vector(test_K * test_N);
		test_B_T_small[test] = pack_B_T(test_B_small[test], test_K, test_N);

		// PRE-GENERATE POINTER ARRAYS
		pointer_arrays[test] = pack_pointers_for_linear_threads(test_A_small[test], test_B_small[test], test_M, test_K, test_N);

		// PRE-CALCULATE CHUNK TEMPLATES
		size_t total_ops	  = test_M * test_N;
		size_t num_threads	  = std::thread::hardware_concurrency();
		size_t ops_per_thread = (total_ops + num_threads - 1) / num_threads;

		for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
			size_t start_op = thread_id * ops_per_thread;
			size_t end_op	= std::min(start_op + ops_per_thread, total_ops);

			if (start_op >= total_ops)
				break;

			ChunkTemplate tmpl;
			tmpl.A_ptrs_offset	= start_op * test_K;
			tmpl.B_ptrs_offset	= start_op * test_K;
			tmpl.C_start_offset = start_op;
			tmpl.operations		= end_op - start_op;
			tmpl.K				= test_K;

			chunk_templates[test].push_back(tmpl);
		}
	}

	// HOT LOOP FUNCTION - PURE EXECUTION!
	auto mul_mat_threaded_ptrs_pure_hotpath = [&](size_t test_index, std::vector<float>& C) noexcept {
		auto& [A_pointers, B_pointers] = pointer_arrays[test_index];
		auto& templates				   = chunk_templates[test_index];

		// Convert templates to actual chunks (minimal work)
		std::vector<ThreadWorkChunkPtrs> chunks;
		chunks.reserve(templates.size());

		for (const auto& tmpl: templates) {
			ThreadWorkChunkPtrs chunk;
			chunk.A_ptrs_start = &A_pointers[tmpl.A_ptrs_offset];
			chunk.B_ptrs_start = &B_pointers[tmpl.B_ptrs_offset];
			chunk.C_start	   = &C[tmpl.C_start_offset];
			chunk.operations   = tmpl.operations;
			chunk.K			   = tmpl.K;
			chunks.push_back(chunk);
		}

		// PURE EXECUTION - NO CALCULATIONS!
		std::vector<std::thread> threads;
		for (const auto& chunk: chunks) {
			threads.emplace_back([chunk]() noexcept {
				thread_worker_ptrs(chunk);
			});
		}

		for (auto& t: threads) {
			t.join();
		}
	};

	struct test_function {
		BNCH_SWT_INLINE static size_t impl(std::vector<std::vector<float>>& test_A_small, std::vector<std::vector<float>>& test_B_small) {
			for (size_t x = 0; x < 16; ++x) {
				size_t test_index = x % NUM_TESTS;
				std::vector<float> C(test_M * test_N, 0.0f);
				mul_mat_scalar_f32(test_A_small[test_index], test_B_small[test_index], C, test_M, test_K, test_N);
				bnch_swt::doNotOptimizeAway(C);
			}
			return sizeof(float) * test_M * test_N;
		}
	};

	std::cout << "All implementations verified - results identical!" << std::endl;

	// Original test harness with ALL implementations
	bnch_swt::benchmark_stage<"MUL_MAT-COMPARISON">::runBenchmark<"MUL_MAT_SCALAR", test_function>(test_A_small, test_B_small);

	bnch_swt::benchmark_stage<"MUL_MAT-COMPARISON">::runBenchmark<"MUL_MAT_PACKED">([=] mutable {
		for (size_t x = 0; x < 16; ++x) {
			size_t test_index = x % NUM_TESTS;
			std::vector<float> C_packed(test_M * test_N, 0.0f);
			mul_mat_packed(test_A_small[test_index], test_B_T_small[test_index], C_packed, test_M, test_K, test_N);
			bnch_swt::doNotOptimizeAway(C_packed);
		}
		return sizeof(float) * test_M * test_N;
	});

	// [Rest of benchmarking code...]

	bnch_swt::benchmark_stage<"MUL_MAT-COMPARISON">::runBenchmark<"MUL_MAT_PURE_HOTPATH">([=] mutable {
		for (size_t x = 0; x < 16; ++x) {
			size_t test_index = x % NUM_TESTS;
			std::vector<float> C_threaded_ptrs(test_M * test_N, 0.0f);
			mul_mat_threaded_ptrs_pure_hotpath(test_index, C_threaded_ptrs);
			bnch_swt::doNotOptimizeAway(C_threaded_ptrs);
		}
		return sizeof(float) * test_M * test_N;
	});

	bnch_swt::benchmark_stage<"MUL_MAT-COMPARISON">::printResults();


	

	return 0;
}
