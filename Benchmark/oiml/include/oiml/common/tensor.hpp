#pragma once

#include <iostream>
#include <iomanip>
#include <oiml/common/common.hpp>
#include <oiml/common/op_traits.hpp>
#include <oiml/common/ring_buffer.hpp>
#include <iostream>
#include <iomanip>

namespace oiml {

	template<oiml_op_type op_type_new, oiml_representation_types rep_type_new, size_t... dims_new> struct oiml_tensor_creation_data {
		static_assert(sizeof...(dims_new) <= 4, "Sorry, but the maximum supported number of dimensions for a tensor is 4.");
		using rep_traits = oiml_representation_traits<rep_type_new>;
		using value_type = typename rep_traits::value_type;
		inline static constexpr oiml_array<size_t, sizeof...(dims_new)> dims{ dims_new... };
	};

	template<oiml_representation_types rep_type_new, oiml_op_type op_type_new, auto dims_new> struct tensor_traits {
		using rep_traits = oiml_representation_traits<rep_type_new>;
		using value_type = typename rep_traits::value_type;
		using size_type	 = size_t;
		inline static constexpr oiml_op_type op_type{ op_type_new };
		inline static constexpr oiml_array<size_t, 4> dims{ [] {
			oiml_array<size_t, 4> return_values{ 1, 1, 1, 1 };
			for (int32_t i = 0; i < dims_new.size(); i++) {
				return_values[i] = dims_new[i];
			}
			return return_values;
		}() };
		inline static constexpr oiml_array<size_t, 4> byte_strides{ [] {
			oiml_array<size_t, 4> return_values{ 0, 0, 0, 0 };
			return_values[0] = sizeof(value_type);
			return_values[1] = return_values[0] * (dims[0] / rep_traits::block_size);
			for (int32_t i = 2; i < 4; i++) {
				return_values[i] = return_values[i - 1] * dims[i - 1];
			}
			return return_values;
		}() };

		inline static constexpr uint64_t total_dimensions{ [] {
			size_t total = 1;
			for (size_t i = 0; i < dims.size(); ++i) {
				total *= dims[i];
			}
			return total;
		}() };

		inline static constexpr uint64_t total_bytes{ [] {
			size_t nbytes{};
			const size_t blck_size = rep_traits::block_size;
			if (blck_size == 1) {
				nbytes = rep_traits::type_size;
				for (int i = 0; i < OIML_MAX_DIMS; ++i) {
					nbytes += (dims[i] - 1) * byte_strides[i];
				}
			} else {
				nbytes = dims[0] * byte_strides[0] / blck_size;
				for (int i = 1; i < OIML_MAX_DIMS; ++i) {
					nbytes += (dims[i] - 1) * byte_strides[i];
				}
			}

			return nbytes;
		}() };

		OIML_FORCE_INLINE static constexpr size_type get_dimensions() {
			return dims_new.size();
		}

		inline static constexpr bool is_contiguous{ [] {
			size_t next_nb = rep_traits::type_size;
			if (dims[0] != rep_traits::block_size && byte_strides[0] != next_nb) {
				return false;
			}
			next_nb *= dims[0] / rep_traits::block_size;
			for (size_t i = 1; i < OIML_MAX_DIMS; i++) {
				if (dims[i] != 1) {
					if (i > get_dimensions()) {
						if (byte_strides[i] != next_nb) {
							return false;
						}
						next_nb *= dims[i];
					} else {
						next_nb = dims[i] * byte_strides[i];
					}
				}
			}
			return true;
		}() };

		inline static constexpr size_type size_val{ [] {
			size_type size_new{ 1 };
			for (size_t i = 0; i < OIML_MAX_DIMS; i++) {
				size_new *= dims[i];
			}
			return size_new;
		}() };

		OIML_FORCE_INLINE static constexpr size_type get_dimension(size_t axis) {
			if (dims.size() > axis) {
				return dims[axis];
			} else {
				return 0;
			}
		}

		OIML_FORCE_INLINE static constexpr size_type size() {
			return size_val;
		}

		OIML_FORCE_INLINE static constexpr size_type get_row_size(size_type size) {
			return rep_traits::type_size * size / rep_traits::block_size;
		}

		OIML_FORCE_INLINE static constexpr bool oiml_is_contiguous() {
			return is_contiguous;
		}

		OIML_FORCE_INLINE static constexpr size_t get_total_bytes() {
			return total_bytes;
		}
	};

	struct oiml_dynamic_tensor {
		using size_type = size_t;

		OIML_FORCE_INLINE oiml_dynamic_tensor() noexcept = default;

		OIML_FORCE_INLINE oiml_dynamic_tensor(oiml_representation_types type_new, size_t w, size_t x, size_t y = 1, size_t z = 1)
			: dims{ w, x, y, z }, rep_traits{ get_rep_traits(type_new) }, type{ type_new } {
			byte_strides[0] = rep_traits.type_size;
			byte_strides[1] = byte_strides[0] * (dims[0] / rep_traits.block_size);
			for (int i = 2; i < 4; i++) {
				byte_strides[i] = byte_strides[i - 1] * dims[i - 1];
			}
			total_bytes = [&] {
				size_t nbytes{};
				const size_t blck_size = rep_traits.block_size;
				if (blck_size == 1) {
					nbytes = rep_traits.type_size;
					for (int i = 0; i < OIML_MAX_DIMS; ++i) {
						nbytes += (dims[i] - 1) * byte_strides[i];
					}
				} else {
					nbytes = dims[0] * byte_strides[0] / blck_size;
					for (int i = 1; i < OIML_MAX_DIMS; ++i) {
						nbytes += (dims[i] - 1) * byte_strides[i];
					}
				}

				return nbytes;
			}();
			is_contiguous = [&] {
				size_t next_nb = rep_traits.type_size;
				if (dims[0] != rep_traits.block_size && byte_strides[0] != next_nb) {
					return false;
				}
				next_nb *= dims[0] / rep_traits.block_size;
				for (size_t i = 1; i < OIML_MAX_DIMS; i++) {
					if (dims[i] != 1) {
						if (i > get_dimensions()) {
							if (byte_strides[i] != next_nb) {
								return false;
							}
							next_nb *= dims[i];
						} else {
							next_nb = dims[i] * byte_strides[i];
						}
					}
				}
				return true;
			}();
			data_val.resize(total_bytes);
		};

		OIML_FORCE_INLINE void update_tensor_properties(oiml_representation_types type_new, size_t w, size_t x, size_t y = 1, size_t z = 1) {
			dims[0]			= w;
			dims[1]			= x;
			dims[2]			= y;
			dims[3]			= z;
			type			= type_new;
			rep_traits		= get_rep_traits(type);
			byte_strides[0] = rep_traits.type_size;
			byte_strides[1] = byte_strides[0] * (dims[0] / rep_traits.block_size);
			for (int i = 2; i < 4; i++) {
				byte_strides[i] = byte_strides[i - 1] * dims[i - 1];
			}
			total_bytes = [&] {
				size_t nbytes{};
				const size_t blck_size = rep_traits.block_size;
				if (blck_size == 1) {
					nbytes = rep_traits.type_size;
					for (int i = 0; i < OIML_MAX_DIMS; ++i) {
						nbytes += (dims[i] - 1) * byte_strides[i];
					}
				} else {
					nbytes = dims[0] * byte_strides[0] / blck_size;
					for (int i = 1; i < OIML_MAX_DIMS; ++i) {
						nbytes += (dims[i] - 1) * byte_strides[i];
					}
				}

				return nbytes;
			}();
			is_contiguous = [&] {
				size_t next_nb = rep_traits.type_size;
				if (dims[0] != rep_traits.block_size && byte_strides[0] != next_nb) {
					return false;
				}
				next_nb *= dims[0] / rep_traits.block_size;
				for (size_t i = 1; i < OIML_MAX_DIMS; i++) {
					if (dims[i] != 1) {
						if (i > get_dimensions()) {
							if (byte_strides[i] != next_nb) {
								return false;
							}
							next_nb *= dims[i];
						} else {
							next_nb = dims[i] * byte_strides[i];
						}
					}
				}
				return true;
			}();
			if (data_val.size() < total_bytes) {
				data_val.resize(total_bytes);
			}
		};

		OIML_FORCE_INLINE constexpr size_t get_total_bytes() const {
			return total_bytes;
		}

		OIML_FORCE_INLINE constexpr size_type get_dimensions() const {
			return dims.size();
		}

		OIML_FORCE_INLINE constexpr size_type get_row_size(size_type size) const {
			return rep_traits.type_size * size / rep_traits.block_size;
		}

		OIML_FORCE_INLINE constexpr bool oiml_is_contiguous() const {
			return is_contiguous;
		}

		OIML_FORCE_INLINE const void* data() const {
			return data_val.data();
		}

		OIML_FORCE_INLINE void* data() {
			return data_val.data();
		}

		OIML_FORCE_INLINE ~oiml_dynamic_tensor() {
			//buffer.release_space(total_bytes);
		}

		inline static thread_local oiml_ring_buffer<64> buffer{};
		oiml_representation_traits_dynamic rep_traits{};
		oiml_array<size_t, 4> byte_strides{};
		oiml_representation_types type{};
		vector<uint8_t> data_val{};
		oiml_array<size_t, 4> dims{};
		oiml_op_type op_type{};
		size_t total_bytes{};
		bool is_contiguous{};
		bool in_use{};
	};

	template<typename value_type>
	concept is_dynamic_tensor_base = std::is_same_v<std::remove_cvref_t<value_type>, oiml_dynamic_tensor>;

	template<typename value_type>
	concept not_dynamic_tensor_base = !is_dynamic_tensor_base<value_type>;

	template<typename oiml_backend_dev_type, typename tensor_traits_new> class oiml_tensor_view : public tensor_traits_new, public tensor_traits_new::rep_traits {
	  public:
		using device_type	= oiml_backend_dev_type;
		using value_type	= tensor_traits_new::value_type;
		using pointer		= value_type*;
		using const_pointer = const pointer;
		using size_type		= size_t;

		OIML_FORCE_INLINE oiml_tensor_view(pointer data_val_new) noexcept : data_val{ data_val_new } {};

		OIML_FORCE_INLINE const_pointer data() const {
			return data_val;
		}

		OIML_FORCE_INLINE pointer data() {
			return data_val;
		}

		OIML_FORCE_INLINE void print_values() const {
			static constexpr size_t total_dimensions = tensor_traits_new::get_dimensions();
			if (total_dimensions == 2) {
				static constexpr size_t rows = tensor_traits_new::get_dimension(0);
				static constexpr size_t cols = tensor_traits_new::get_dimension(1);

				printf("Tensor (%zu x %zu):\n[", rows, cols);

				for (size_t i = 0; i < rows; ++i) {
					if (i > 0) {
						printf("\n");
					}
					for (size_t j = 0; j < cols; ++j) {
						std::cout << std::setprecision(3) << data_val[i * cols + j];
						if (j < cols - 1) {
							std::cout << ",";
						}
					}
				}
				printf(" ]\n");
			} else {
				printf("Tensor is not a 2D tensor or needs additional handling\n");
			}
		}

	  protected:
		pointer data_val{};
	};

	template<typename oiml_backend_dev_type, typename tensor_traits_new> class oiml_tensor : public tensor_traits_new, public tensor_traits_new::rep_traits {
	  public:
		using device_type	= oiml_backend_dev_type;
		using value_type	= tensor_traits_new::value_type;
		using pointer		= value_type*;
		using const_pointer = const value_type*;
		using size_type		= size_t;

		OIML_FORCE_INLINE oiml_tensor() noexcept : data_val{} {
			data_val.resize(tensor_traits_new::get_total_bytes());
		}

		OIML_FORCE_INLINE const_pointer data() const {
			return reinterpret_cast<const_pointer>(data_val.data());
		}

		OIML_FORCE_INLINE pointer data() {
			return reinterpret_cast<pointer>(data_val.data());
		}

		OIML_FORCE_INLINE void print_values() {
			size_t total_dimensions = tensor_traits_new::get_dimensions();
			if (total_dimensions == 2) {
				size_t rows = tensor_traits_new::get_dimension(0);
				size_t cols = tensor_traits_new::get_dimension(1);

				printf("Tensor (%zu x %zu):\n[", rows, cols);

				for (size_t i = 0; i < rows; ++i) {
					if (i > 0) {
						printf("\n");
					}
					for (size_t j = 0; j < cols; ++j) {
						printf(" %.2f", data_val[i * cols + j]);
					}
				}

				printf(" ]\n");
			} else {
				printf("Tensor is not a 2D tensor or needs additional handling\n");
			}
		}

	  protected:
		vector<uint8_t> data_val{};
	};

	template<typename backend_device_type, oiml_representation_types rep_type, oiml_op_type op_type, size_t... dims> struct oiml_get_tensor_type {
		using oiml_tensor_creation_data_new = oiml_tensor_creation_data<op_type, rep_type, dims...>;
		using tensor_traits					= tensor_traits<oiml_tensor_creation_data_new::rep_traits::type, op_type, oiml_tensor_creation_data_new::dims>;

		using type = oiml_tensor<backend_device_type, tensor_traits>;
	};

	template<typename backend_device_type, oiml_representation_types rep_type, oiml_op_type op_type, size_t... dims> using oiml_get_tensor_type_t =
		oiml_get_tensor_type<backend_device_type, rep_type, op_type, dims...>::type;

	template<typename backend_device_type, oiml_representation_types rep_type, oiml_op_type op_type, size_t... dims> struct oiml_get_tensor_view_type {
		using oiml_tensor_creation_data_new = oiml_tensor_creation_data<op_type, rep_type, dims...>;
		using tensor_traits					= tensor_traits<oiml_tensor_creation_data_new::rep_traits::type, op_type, oiml_tensor_creation_data_new::dims>;

		using type = oiml_tensor_view<backend_device_type, tensor_traits>;
	};

	template<typename backend_device_type, oiml_representation_types rep_type, oiml_op_type op_type, size_t... dims> using oiml_get_tensor_view_type_t =
		oiml_get_tensor_view_type<backend_device_type, rep_type, op_type, dims...>::type;

}// namespace oiml