#include <BnchSwt/BenchmarkSuite.hpp>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>
#include <array>
#include <queue>
#include <latch>
#include <bit>#include <vector>
#include <iostream>
#include <thread>
#include <cstring>
#include <algorithm>
#include <cmath>

// Your existing functions
std::vector<float> pack_B_T(const std::vector<float>& B, size_t K, size_t N) {
	std::vector<float> B_T(K * N);
	for (size_t k = 0; k < K; ++k) {
		for (size_t j = 0; j < N; ++j) {
			B_T[j * K + k] = B[k * N + j];
		}
	}
	return B_T;
}

// Mul mat with A row-major, B_T column-major
void mul_mat_packed(const std::vector<float>& A, const std::vector<float>& B_T, std::vector<float>& C, size_t M, size_t K, size_t N) {
	for (size_t i = 0; i < M; ++i) {
		const float* A_row = &A[i * K];
		for (size_t j = 0; j < N; ++j) {
			const float* B_col = &B_T[j * K];
			float sum		   = 0.0f;
			for (size_t k = 0; k < K; ++k) {
				sum += A_row[k] * B_col[k];
			}
			C[i * N + j] = sum;
		}
	}
}

void mul_mat_scalar_f32(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t M, size_t K, size_t N) {
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

// Verification function to ensure both implementations produce same results
bool verify_results(const std::vector<float>& C1, const std::vector<float>& C2, float epsilon = 1e-5f) {
	if (C1.size() != C2.size())
		return false;
	for (size_t i = 0; i < C1.size(); ++i) {
		if (std::abs(C1[i] - C2[i]) > epsilon) {
			return false;
		}
	}
	return true;
}

std::vector<float> generate_random_float_vector(size_t size) {
	std::vector<float> return_value{};
	return_value.resize(size);
	for (size_t x = 0; x < size; ++x) {
		return_value[x] = bnch_swt::random_generator::generateValue<float>();
	}
	return return_value;
}

// NEW: Thread-friendly linear packing structures
struct ThreadWorkChunk {
	const float* A_start;// Starting pointer for A data
	const float* B_start;// Starting pointer for B data
	float* C_start;// Starting pointer for output C
	size_t operations;// Number of dot products this thread handles
	size_t K;// Inner dimension for dot product length
};

// Pack matrices for linear thread access - separate A and B sections
std::vector<float> pack_for_linear_threads(const std::vector<float>& A, const std::vector<float>& B, size_t M, size_t K, size_t N) {
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

// Generate thread work chunks with simple linear pointers
std::vector<ThreadWorkChunk> generate_thread_chunks(const std::vector<float>& packed_data, size_t M, size_t N, size_t K, size_t num_threads, std::vector<float>& output_C) {
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

// ULTRA-SIMPLE THREAD FUNCTION - JUST INCREMENT POINTERS!
void thread_worker_linear(const ThreadWorkChunk& chunk) {
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

// Main threaded matrix multiplication function
void mul_mat_threaded(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t M, size_t K, size_t N,
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

int main() {
	constexpr size_t NUM_TESTS = 100;

	// Use original dimensions for consistency with your harness
	constexpr size_t test_M = 128;
	constexpr size_t test_K = 128;
	constexpr size_t test_N = 128;
	std::cout << "\nRunning benchmarks with original test harness.\nDimensions: [" << test_M << ", " << test_K << ", " << test_N << "]" << std::endl;

	// Generate 100 unique test vectors for the original dimensions
	std::vector<std::vector<float>> test_A_small(NUM_TESTS);
	std::vector<std::vector<float>> test_B_small(NUM_TESTS);
	std::vector<std::vector<float>> test_B_T_small(NUM_TESTS);

	for (size_t test = 0; test < NUM_TESTS; ++test) {
		test_A_small[test]	 = generate_random_float_vector(test_M * test_K);
		test_B_small[test]	 = generate_random_float_vector(test_K * test_N);
		test_B_T_small[test] = pack_B_T(test_B_small[test], test_K, test_N);
	}

	// Verify small matrices too
	for (size_t test = 0; test < NUM_TESTS; ++test) {
		std::vector<float> C_scalar(test_M * test_N, 0.0f);
		std::vector<float> C_packed(test_M * test_N, 0.0f);
		std::vector<float> C_threaded(test_M * test_N, 0.0f);

		mul_mat_scalar_f32(test_A_small[test], test_B_small[test], C_scalar, test_M, test_K, test_N);
		mul_mat_packed(test_A_small[test], test_B_T_small[test], C_packed, test_M, test_K, test_N);
		mul_mat_threaded(test_A_small[test], test_B_small[test], C_threaded, test_M, test_K, test_N);

		if (!verify_results(C_scalar, C_packed) || !verify_results(C_scalar, C_threaded)) {
			std::cout << "VERIFICATION FAILED for small matrix test " << test << std::endl;
			return 1;
		}
	}

	// Original test harness adapted for 100 unique test vectors
	bnch_swt::benchmark_stage<"MUL_MAT-COMPARISON">::runBenchmark<"MUL_MAT_SCALAR", "CYAN">([=] mutable {
		for (size_t x = 0; x < 16; ++x) {
			// Use index to cycle through the 100 unique test vectors

			size_t test_index = x % NUM_TESTS;
			std::vector<float> C(test_M * test_N, 0.0f);
			mul_mat_scalar_f32(test_A_small[test_index], test_B_small[test_index], C, test_M, test_K, test_N);
			bnch_swt::doNotOptimizeAway(C);
		}
		return sizeof(float) * test_M * test_N;
	});

	bnch_swt::benchmark_stage<"MUL_MAT-COMPARISON">::runBenchmark<"MUL_MAT_PACKED", "GREEN">([=] mutable {
		for (size_t x = 0; x < 16; ++x) {
			// Use index to cycle through the 100 unique test vectors
			size_t test_index = x % NUM_TESTS;
			std::vector<float> C_packed(test_M * test_N, 0.0f);
			mul_mat_packed(test_A_small[test_index], test_B_T_small[test_index], C_packed, test_M, test_K, test_N);
			bnch_swt::doNotOptimizeAway(C_packed);
		}
		return sizeof(float) * test_M * test_N;
	});

	bnch_swt::benchmark_stage<"MUL_MAT-COMPARISON">::runBenchmark<"MUL_MAT_THREADED", "YELLOW">([=] mutable {
		for (size_t x = 0; x < 16; ++x) {
			// Use index to cycle through the 100 unique test vectors
			size_t test_index = x % NUM_TESTS;
			std::vector<float> C_threaded(test_M * test_N, 0.0f);
			mul_mat_threaded(test_A_small[test_index], test_B_small[test_index], C_threaded, test_M, test_K, test_N);
			bnch_swt::doNotOptimizeAway(C_threaded);
		}
		return sizeof(float) * test_M * test_N;
	});

	bnch_swt::benchmark_stage<"MUL_MAT-COMPARISON">::printResults();

	return 0;
}