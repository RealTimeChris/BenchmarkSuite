#pragma once

#include <oiml/common/config.hpp>
#include <oiml/common/allocator.hpp>

namespace oiml {

	template<typename value_type> using vector = std::vector<value_type, alloc_wrapper<value_type>>;

	template<size_t alignment> class oiml_ring_buffer : public alloc_wrapper<uint8_t> {
	  public:
		using allocator	 = alloc_wrapper<uint8_t>;
		using value_type = uint8_t;
		using pointer	 = value_type*;
		using size_type	 = size_t;

		OIML_FORCE_INLINE oiml_ring_buffer() noexcept = default;

		OIML_FORCE_INLINE oiml_ring_buffer(size_type size_new) {
			resize(size_new);
		}

		OIML_FORCE_INLINE void* claim_space(size_type size_new) {
			if (this->get_free_space() < size_new) {
				resize((data_val.size() + size_new) * 2);
			}
			auto return_values = this->get_current_head();
			head += size_new;
			return return_values;
		}

		OIML_FORCE_INLINE void release_space(size_type size_new) {
			size_new = get_alignment_offset(size_new);
			if (head < size_new) {
				throw std::runtime_error("Not enough data to release");
			}
			if (data_val.size() > 0) {
				head += size_new % data_val.size();
			} else {
				head += size_new;
			}
		}

		OIML_FORCE_INLINE void clear() {
			head = 0;
		}

		OIML_FORCE_INLINE void resize(size_type newSize) noexcept {
			data_val.resize(newSize);
		}

		OIML_FORCE_INLINE pointer get_current_head() {
			return data_val.data() + head;
		}

		OIML_FORCE_INLINE ~oiml_ring_buffer() {
			data_val.clear();
		}

	  protected:
		vector<value_type> data_val{};
		size_type head{};

		OIML_FORCE_INLINE size_type get_free_space() const {
			return data_val.size() - head;
		}

		OIML_FORCE_INLINE size_type get_alignment_offset(size_type size_new) const {
			size_type alignment_units = alignment / sizeof(value_type);
			size_type remainder		  = size_new % alignment_units;
			return remainder == 0 ? size_new : alignment - sizeof(value_type) * remainder;
		}
	};

}