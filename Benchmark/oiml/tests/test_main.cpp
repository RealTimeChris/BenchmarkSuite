#include <BnchSwt/BenchmarkSuite.hpp>
#include <BnchSwt/RandomGenerators.hpp>
#include <oiml/index.hpp>

inline static constexpr oiml::oiml_array<size_t, 12> sizes{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };

template<typename value_type, size_t size> OIML_FORCE_INLINE bool operator!=(const std::vector<value_type, oiml::alloc_wrapper<value_type>>& lhs,
	const std::vector<value_type, oiml::alloc_wrapper<value_type>>& rhs) {
	if (lhs.size() != rhs.size()) {
		return true;
	}

	constexpr float epsilon = 1e-6f;
	for (size_t x = 0; x < lhs.size(); ++x) {
		float abs_a = std::abs(lhs[x]);
		float abs_b = std::abs(rhs[x]);
		float diff	= std::abs(lhs[x] - rhs[x]);

		if (diff > epsilon * std::max({ abs_a, abs_b, 1.0f })) {
			std::cout << "INCORRECT @ INDEX : " << x << std::endl;
			std::cout << "LHS VALUE: " << lhs[x] << std::endl;
			std::cout << "RHS VALUE: " << rhs[x] << std::endl;
			return true;
		}
	}
	return false;
}

template<size_t dim_a, size_t dim_b, size_t dim_c> struct test {
	inline static constexpr uint64_t num_tensors{ 2 };
	inline static constexpr uint64_t matrix_a_dim_a{ dim_a };
	inline static constexpr uint64_t matrix_a_dim_b{ dim_b };
	inline static constexpr uint64_t matrix_b_dim_a{ dim_a };
	inline static constexpr uint64_t matrix_b_dim_b{ dim_c };
	inline static constexpr uint64_t matrix_c_dim_a{ matrix_a_dim_b };
	inline static constexpr uint64_t matrix_c_dim_b{ matrix_b_dim_b };
#if defined(NDEBUG)
	inline static constexpr uint64_t max_iteration_count{ 20 };
	inline static constexpr uint64_t measured_iteration_count{ 4 };
#else
	inline static constexpr uint64_t max_iteration_count{ 2 };
	inline static constexpr uint64_t measured_iteration_count{ 1 };
#endif
	inline static constexpr uint64_t matrix_a_dims{ matrix_a_dim_a * matrix_a_dim_b };
	inline static constexpr uint64_t matrix_b_dims{ matrix_b_dim_a * matrix_b_dim_b };
	inline static constexpr uint64_t matrix_c_dims{ matrix_c_dim_a * matrix_c_dim_b };

	struct simple_model {
		struct oiml_tensor* matrix_a{};
		struct oiml_tensor* matrix_b{};

		oiml_backend_t backend = nullptr;
		oiml_backend_buffer_t buffer{};
		struct oiml_context* ctx{};
		oiml_init_params params{};

		oiml_gallocr_t allocr{};
		size_t buf_size{};
		std::vector<uint8_t> buf{};
		oiml_context* ctx0{};

		oiml_cgraph* gf{};
		oiml_tensor* result02{};
	};

	struct simple_model_oiml {
		using device_type	= oiml::oiml_backend_device<oiml_backend_device_types::cpu>;
		using matrix_type_a = oiml::oiml_get_tensor_type_t<device_type, oiml::oiml_representation_types::q8_0, oiml::oiml_op_type::oiml_op_mul_mat, matrix_a_dim_a, matrix_a_dim_b>;
		using matrix_type_b =
			oiml::oiml_get_tensor_type_t<device_type, oiml::oiml_representation_types::float_32, oiml::oiml_op_type::oiml_op_mul_mat, matrix_b_dim_a, matrix_b_dim_b>;
		using matrix_type_result =
			oiml::oiml_get_tensor_view_type_t<device_type, oiml::oiml_representation_types::float_32, oiml::oiml_op_type::oiml_op_store, matrix_c_dim_a, matrix_c_dim_b>;
		matrix_type_a matrix_a;
		matrix_type_b matrix_b{};

		simple_model_oiml(const void* data_new_01, const void* data_new_02, float* data_new_03) : matrix_a{}, matrix_b{}, result_matrix{ data_new_03 } {
			simple_model_oiml::device_type::execute_unary_op<oiml::oiml_op_type::oiml_op_load>(matrix_a, ( typename matrix_type_a::value_type*& )(data_new_01));
			simple_model_oiml::device_type::execute_unary_op<oiml::oiml_op_type::oiml_op_load>(matrix_b, ( typename matrix_type_b::value_type*& )(data_new_02));
		};
		matrix_type_result result_matrix{};
	};

	struct simple_model_oiml_dynamic {
		using device_type = oiml::oiml_backend_device<oiml_backend_device_types::cpu>;
		oiml::oiml_dynamic_tensor matrix_a{};
		oiml::oiml_dynamic_tensor matrix_b{};

		simple_model_oiml_dynamic(const void* data_new_01, const void* data_new_02)
			: matrix_a{ oiml::oiml_representation_types::q8_0, matrix_a_dim_a, matrix_a_dim_b },
			  matrix_b{ oiml::oiml_representation_types::float_32, matrix_b_dim_a, matrix_b_dim_b } {
			simple_model_oiml::device_type::execute_unary_op<oiml::oiml_op_type::oiml_op_load>(matrix_a, ( oiml::block_q8_0<oiml_half>*& )(data_new_01));
			simple_model_oiml::device_type::execute_unary_op<oiml::oiml_op_type::oiml_op_load>(matrix_b, ( float*& )(data_new_02));
		};
		oiml::oiml_dynamic_tensor result_matrix{ oiml::oiml_representation_types::float_32, matrix_c_dim_a, matrix_c_dim_b };
	};

	template<size_t rows, size_t cols, typename value_type> static void print_values(value_type* values, const char* string) {
		printf("Tensor - %s (%zu x %zu):\n[", string, rows, cols);

		for (size_t i = 0; i < rows; ++i) {
			if (i > 0) {
				printf("\n");
			}
			for (size_t j = 0; j < cols; ++j) {
				printf(" %.2f", values[i * cols + j]);
			}
		}

		printf(" ]\n");
	}

	OIML_FORCE_INLINE static void compute_graph_oiml(simple_model_oiml& model) {
		simple_model_oiml::device_type::template execute_binary_op<oiml::oiml_op_type::oiml_op_mul_mat>(model.matrix_a, model.matrix_b, model.result_matrix);
	}

	OIML_FORCE_INLINE static void compute_graph_oiml_dynamic(simple_model_oiml_dynamic& model, float* out_data) {
		simple_model_oiml_dynamic::device_type::template execute_binary_op<oiml::oiml_op_type::oiml_op_mul_mat>(model.matrix_a, model.matrix_b, model.result_matrix);
		simple_model_oiml_dynamic::device_type::execute_unary_op<oiml::oiml_op_type::oiml_op_store>(model.result_matrix, out_data);
	}

	OIML_FORCE_INLINE static void compute_graph_prep(simple_model& model, void* matrix_a, void* matrix_b) {
		model.backend = oiml_backend_cpu_init();
		model.params = { oiml_tensor_overhead() * num_tensors, nullptr, true };
		model.ctx = oiml_init(model.params);
		model.matrix_a = oiml_new_tensor_2d(model.ctx, oiml::oiml_representation_types::q8_0, matrix_a_dim_a, matrix_a_dim_b);
		model.matrix_b = oiml_new_tensor_2d(model.ctx, oiml::oiml_representation_types::float_32, matrix_b_dim_a, matrix_b_dim_b);
		model.buffer = oiml_backend_alloc_ctx_tensors(model.ctx, model.backend);
		oiml_backend_tensor_set(model.matrix_a, matrix_a, 0, oiml_nbytes(model.matrix_a));
		oiml_backend_tensor_set(model.matrix_b, matrix_b, 0, oiml_nbytes(model.matrix_b));
		model.allocr = oiml_gallocr_new(oiml_backend_get_default_buffer_type(model.backend));
		model.buf_size = oiml_tensor_overhead() * OIML_DEFAULT_GRAPH_SIZE + oiml_graph_overhead();
		model.buf.resize(model.buf_size);
		model.params = { model.buf_size, model.buf.data(), true };
		model.ctx0	 = oiml_init(model.params);
		model.gf = oiml_new_graph(model.ctx0);
		model.result02 = oiml_mul_mat(model.ctx0, model.matrix_a, model.matrix_b);
		oiml_build_forward_expand(model.gf, model.result02);
		oiml_free(model.ctx0);
		oiml_gallocr_reserve(model.allocr, model.gf);
		oiml_gallocr_alloc_graph(model.allocr, model.gf);
		if (oiml_backend_is_cpu(model.backend)) {
			oiml_backend_cpu_set_n_threads(model.backend, 6);
		}
	}

	OIML_FORCE_INLINE static void compute_graph(simple_model& model, float* out_data) {
		oiml_backend_graph_compute(model.backend, model.gf);
		oiml_backend_tensor_get(model.result02, out_data, 0, oiml_nbytes(model.result02));
	}

	// Function to clean up resources
	OIML_FORCE_INLINE static void compute_graph_cleanup(simple_model& model) {
		oiml_gallocr_free(model.allocr);
		oiml_free(model.ctx);
		oiml_backend_buffer_free(model.buffer);
		oiml_backend_free(model.backend);
	}

	static oiml::block_q8_0<oiml_half>* advance_pointer(int8_t* ptr) {
		return reinterpret_cast<oiml::block_q8_0<oiml_half>*>(ptr) + 1;
	}

	static float* advance_pointer_float(int8_t* ptr) {
		return reinterpret_cast<float*>(ptr + sizeof(oiml::block_q8_0<oiml_half>));
	}

	static int8_t* advance_pointer_uint(int8_t* ptr) {
		return ptr + sizeof(oiml::block_q8_0<oiml_half>);
	}

	struct test_data {
		std::vector<int8_t, oiml::alloc_wrapper<int8_t>> matrix_a{};
		std::vector<int8_t, oiml::alloc_wrapper<int8_t>> matrix_b{};
		std::vector<float, oiml::alloc_wrapper<float>> out_data01{};
		std::vector<float, oiml::alloc_wrapper<float>> out_data02{};
		std::vector<float, oiml::alloc_wrapper<float>> out_data03{};

		OIML_FORCE_INLINE test_data() {
			matrix_a.resize(simple_model_oiml::matrix_type_a::get_total_bytes());
			matrix_b.resize(simple_model_oiml::matrix_type_b::get_total_bytes());
			out_data01.resize(simple_model_oiml::matrix_type_result::get_total_bytes() / 4);
			out_data02.resize(simple_model_oiml::matrix_type_result::get_total_bytes() / 4);
			out_data03.resize(simple_model_oiml::matrix_type_result::get_total_bytes() / 4);
			for (size_t x = 0; x < simple_model_oiml::matrix_type_a::get_total_bytes(); ++x) {
				matrix_a[x] = bnch_swt::random_generator::generateValue<uint8_t>() / 4;
			}

			for (size_t x = 0; x < simple_model_oiml::matrix_type_b::get_total_bytes(); ++x) {
				matrix_b[x] = bnch_swt::random_generator::generateValue<uint8_t>() / 4;
			}
			auto new_ptr = matrix_a.data();
			oiml::block_q8_0<oiml_half>* ptr_block{ reinterpret_cast<oiml::block_q8_0<oiml_half>*>(new_ptr) };
			oiml_half* ptr_float = reinterpret_cast<oiml_half*>(new_ptr);
			int8_t* ptr_uint{ reinterpret_cast<int8_t*>(new_ptr + 2) };
			//std::cout << "NEW-VALUES!" << std::endl;
			for (size_t x = 0; x < 2; ++x) {
				for (size_t y = 0; y < 32; ++y) {
					//std::cout << "Y-01: " << +ptr_block->qs[y] << std::endl;
					//std::cout << "Y-02: " << +ptr_uint[y] << std::endl;
				}
				//std::cout << "F-01: " << +ptr_block->d << std::endl;
				//std::cout << "F-02: " << *ptr_float << std::endl;
				new_ptr += sizeof(oiml::block_q8_0<oiml_half>);
				ptr_block = reinterpret_cast<oiml::block_q8_0<oiml_half>*>(new_ptr);
				ptr_float = reinterpret_cast<oiml_half*>(new_ptr);
				ptr_uint  = new_ptr + 2;
			}
		}
	};

	inline static constexpr bnch_swt::string_literal sl01{ "threadpool-testing-mat_mul-(" };
	inline static constexpr bnch_swt::string_literal sl02{ "x" };
	inline static constexpr bnch_swt::string_literal sl03{ ")" };
	inline static constexpr bnch_swt::string_literal sl04{ "(" };
	inline static constexpr bnch_swt::string_literal sl05{ "," };
	inline static constexpr bnch_swt::string_literal a_dim_a{ bnch_swt::internal::toStringLiteral<matrix_a_dim_a>() };
	inline static constexpr bnch_swt::string_literal a_dim_b{ bnch_swt::internal::toStringLiteral<matrix_a_dim_b>() };
	inline static constexpr bnch_swt::string_literal b_dim_a{ bnch_swt::internal::toStringLiteral<matrix_b_dim_a>() };
	inline static constexpr bnch_swt::string_literal b_dim_b{ bnch_swt::internal::toStringLiteral<matrix_b_dim_b>() };
	inline static constexpr bnch_swt::string_literal name{ sl01 + a_dim_a + sl05 + a_dim_b + sl03 + sl02 + sl04 + b_dim_a + sl05 + b_dim_b + sl03 };

	OIML_INLINE static void run() {
		std::vector<test_data> testData{ max_iteration_count };
		size_t currentIndex{};
		{
			std::vector<simple_model> testModels{ max_iteration_count };
			for (size_t x = 0; x < max_iteration_count; ++x) {
				compute_graph_prep(testModels[x], testData[x].matrix_a.data(), testData[x].matrix_b.data());
			}
			bnch_swt::benchmark_stage<name, max_iteration_count, measured_iteration_count>::template runBenchmark<"ggml-threadpool", "cyan">([&] {
				compute_graph(testModels[currentIndex], testData[currentIndex].out_data01.data());
				auto new_size = testData[currentIndex].matrix_a.size() * 4;
				bnch_swt::doNotOptimizeAway(testData[currentIndex].out_data01.data());
				++currentIndex;
				return new_size;
			});
			for (size_t x = 0; x < max_iteration_count; ++x) {
				compute_graph_cleanup(testModels[x]);
			}
		}
		{
			std::vector<simple_model_oiml> testModelsOiml{};
			for (size_t x = 0; x < max_iteration_count; ++x) {
				testModelsOiml.emplace_back(testData[x].matrix_a.data(), testData[x].matrix_b.data(), testData[x].out_data02.data());
			}
			currentIndex = 0;
			bnch_swt::benchmark_stage<name, max_iteration_count, measured_iteration_count>::template runBenchmark<"oiml-threadpool", "cyan">([&] {
				compute_graph_oiml(testModelsOiml[currentIndex]);
				auto new_size = testData[currentIndex].matrix_a.size() * 4;
				bnch_swt::doNotOptimizeAway(testData[currentIndex].out_data02.data());
				++currentIndex;
				return new_size;
			});
		}
		{ /*
			std::vector<simple_model_oiml_dynamic> testModelsOimlDynamic{};
			for (size_t x = 0; x < max_iteration_count; ++x) {
				testModelsOimlDynamic.emplace_back(testData[x].matrix_a.data(), testData[x].matrix_b.data());
			}
			currentIndex = 0;
			bnch_swt::benchmark_stage<name, max_iteration_count, measured_iteration_count>::template runBenchmark<"oiml-dynamic-threadpool", "cyan">([&] {
				compute_graph_oiml_dynamic(testModelsOimlDynamic[currentIndex], testData[currentIndex].out_data03.data());
				auto new_size = testData[currentIndex].matrix_a.size() * 4;
				++currentIndex;
				bnch_swt::doNotOptimizeAway(testData[currentIndex].out_data03.data());
				return new_size;
			});*/
		}
		for (size_t x = 0; x < max_iteration_count; ++x) {
			if (std::this_thread::get_id() == std::thread::id{}) {
				print_values<matrix_c_dim_a, matrix_c_dim_b>(testData[x].out_data02.data(), "ggml-tensor");
				print_values<matrix_c_dim_a, matrix_c_dim_b>(testData[x].out_data01.data(), "oiml-tensor");
			}
			if (testData[x].out_data01 != testData[x].out_data02) {
				std::cout << "unequal results (for static) at index: " << x << std::endl;
			}
		}
		for (size_t x = 0; x < max_iteration_count; ++x) {
			//if (testData[x].out_data01 != testData[x].out_data03) {
				//std::cout << "unequal results (for dynamic) at index: " << x << std::endl;
			//}
		}
		bnch_swt::benchmark_stage<name, max_iteration_count, measured_iteration_count>::printResults(true, true);
	}
};

int32_t main() {
	test<32, 32, 32>::run();
	test<sizes[6], sizes[6], sizes[6]>::run();
	test<sizes[7], sizes[7], sizes[7]>::run();
	test<sizes[8], sizes[8], sizes[8]>::run();
	test<sizes[9], sizes[9], sizes[9]>::run();
	test<sizes[10], sizes[10], sizes[10]>::run();
	test<sizes[11], sizes[11], sizes[11]>::run();
	test<8192, 1024, 1>::run();
	return 0;
}