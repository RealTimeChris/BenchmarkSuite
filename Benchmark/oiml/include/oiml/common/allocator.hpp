#pragma once

#include <oiml-cpu/cpu_traits.hpp>
#include <oiml/common/config.hpp>
#include <memory_resource>
#include <stdlib.h>

namespace oiml {

	template<typename value_type> OIML_FORCE_INLINE constexpr value_type roundUpToMultiple(value_type value, value_type multiple) noexcept {
		if ((multiple & (multiple - 1)) == 0) {
			auto mulSub1{ multiple - 1 };
			auto notMulSub1{ ~mulSub1 };
			return (value + (mulSub1)) & notMulSub1;
		} else {
			const auto remainder = value % multiple;
			return remainder == 0 ? value : value + (multiple - remainder);
		}
	}

	template<typename value_type_new> class alloc_wrapper {
	  public:
		using value_type	   = value_type_new;
		using pointer		   = value_type_new*;
		using const_pointer	   = const value_type_new*;
		using reference		   = value_type_new&;
		using const_reference  = const value_type_new&;
		using size_type		   = std::size_t;
		using difference_type  = std::ptrdiff_t;
		using allocator_traits = std::allocator_traits<alloc_wrapper<value_type>>;

		template<typename U> struct rebind {
			using other = alloc_wrapper<U>;
		};

		OIML_FORCE_INLINE alloc_wrapper() noexcept = default;

		template<typename U> alloc_wrapper(const alloc_wrapper<U>&) noexcept {
		}

		OIML_FORCE_INLINE static pointer allocate(size_type count) noexcept {
			size_t alignment{ aligments[cpu_arch_index] };
			if OIML_UNLIKELY (count == 0) {
				return nullptr;
			}
#if defined(OIML_WIN) || defined(OIML_LINUX)
			return static_cast<value_type*>(_mm_malloc(roundUpToMultiple(count * sizeof(value_type), alignment), alignment));
#else
			return static_cast<value_type*>(aligned_alloc(alignment, roundUpToMultiple(count * sizeof(value_type), alignment)));
#endif
		}

		OIML_FORCE_INLINE void deallocate(pointer ptr, size_t = 0) noexcept {
			if OIML_LIKELY (ptr) {
#if defined(OIML_WIN) || defined(OIML_LINUX)
				_mm_free(ptr);
#else
				free(ptr);
#endif
			}
		}

		template<typename... arg_types> OIML_FORCE_INLINE static void construct(pointer ptr, arg_types&&... args) noexcept {
			if (ptr) {
				new (ptr) value_type(std::forward<arg_types>(args)...);
			}
		}

		OIML_FORCE_INLINE static size_type maxSize() noexcept {
			return allocator_traits::max_size(alloc_wrapper{});
		}

		OIML_FORCE_INLINE static void destroy(pointer ptr) noexcept {
			ptr->~value_type();
		}
	};

}