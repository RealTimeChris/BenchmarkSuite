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

struct model_config {
	bool use_gradient_checkpointing{};
	bool use_rotary_embeddings{};
	uint64_t kv_cache_block_size{};
	bool use_flash_attention{};
	float norm_epsilon{};
	bool exceptions{};
	bool benchmark{};

	constexpr model_config() {}

  protected:
};

enum class behavioral_axes {
	collect_required_bytes,
	tensor_debug,
	count,
};

template<model_config config, typename base_type_new> struct weight_mapper {
	BNCH_SWT_INLINE weight_mapper() noexcept								 = default;
	BNCH_SWT_INLINE weight_mapper& operator=(const weight_mapper&) noexcept = delete;
	BNCH_SWT_INLINE weight_mapper(const weight_mapper&) noexcept			 = delete;
	BNCH_SWT_INLINE weight_mapper& operator=(weight_mapper&&) noexcept		 = delete;
	BNCH_SWT_INLINE weight_mapper(weight_mapper&&) noexcept				 = delete;
	using base_type																 = base_type_new;
	template<typename core_type> BNCH_SWT_INLINE static constexpr bool filter() {
		return base_type::type % 2 == 0;
	}
	BNCH_SWT_INLINE static void impl(base_type_new& , int64_t& data, std::array<std::array<void*, 32>, 32>& ) {
		data += rand();
	}
};

template<model_config config, typename derived_type> struct behavioral_axis;

template<model_config config, typename derived_type> struct behavioral_axis {
	template<typename core_type> BNCH_SWT_INLINE static constexpr bool filter() {
		return derived_type::type % 2 == 0;
	}

	template<typename core_type> BNCH_SWT_INLINE static void impl(core_type& core) {
		std::cout << "CURRENT OP_TYPE: " << core.type << std::endl;
	}
};

template<model_config config, typename... bases> struct core_bases : public bases... {
	template<template<model_config, typename> typename mixin_type, typename... arg_types> BNCH_SWT_INLINE constexpr void impl(arg_types&&... args) {
		(impl_internal_filtered<mixin_type, bases>(args...), ...);
	}

  protected:
	template<template<model_config, typename> typename mixin_type, typename base_type, typename... arg_types>
	BNCH_SWT_INLINE constexpr void impl_internal_filtered(arg_types&&... args) {
		if constexpr (mixin_type<config, base_type>::template filter<base_type>()) {
			mixin_type<config, base_type>::impl(*static_cast<base_type*>(this), std::forward<arg_types>(args)...);
		}
	}
};

template<model_config config_new, auto op_type> struct core_traits {
	static constexpr size_t type{ op_type };
};

template<model_config config, typename index_sequence> struct get_core_bases;

template<model_config config, uint64_t... index> struct get_core_bases<config, std::index_sequence<index...>> {
	using type = core_bases<config, core_traits<config, index>...>;
};

template<model_config config> using get_core_bases_t = typename get_core_bases<config, std::make_index_sequence<32>>::type;

template<model_config config, typename model_type> struct thread_pool : public get_core_bases_t<config> {
	BNCH_SWT_INLINE thread_pool() noexcept								 = default;
	BNCH_SWT_INLINE thread_pool& operator=(const thread_pool&) noexcept = delete;
	BNCH_SWT_INLINE thread_pool(const thread_pool&) noexcept			 = delete;

	BNCH_SWT_INLINE thread_pool(uint64_t) {}
};

template<model_config config> struct model : public thread_pool<config, model<config>> {
	using thread_pool_t		= thread_pool<config, model<config>>;
	template<auto op_type> auto& get_core_type() {
		return *static_cast<core_traits<config, op_type>*>(static_cast<get_core_bases_t<config>*>(this));
	}
	BNCH_SWT_INLINE model& operator=(const model&) = delete;
	BNCH_SWT_INLINE model(const model&) = delete;
	BNCH_SWT_INLINE model() : thread_pool<config, model>{ } {
	}

	BNCH_SWT_INLINE void init() {
		//memory.init(total_required_bytes);
		//weight_memory = memory_mapped_file{ params.model_file };
		//memory.init(total_required_bytes);
		//std::cout << "Total required bytes: " << total_required_bytes << std::endl;
		//weight_memory = memory_mapped_file{ params.model_file };
		std::array<std::array<void*, 32>, 32> data{};
		//core_bases_type::template impl<memory_mapper>(memory);
		this->template impl<weight_mapper>(32, data);
		//model_graph_data<config> model_construction_data = model_parser<config>::parse_model(data, &weight_memory);
		//static_cast<core_depth_bases_type*>(this)->template impl<weight_mapper, thread_strategy_type::thread_prep>(params.thread_count, data);
		//		static_cast<core_depth_bases_type*>(this)->template impl<weight_mapper, thread_strategy_type::weight_mapping>(params.thread_count, data);
		//stop_watch_val_nihilus.reset();
		//model_graph_data<config> model_construction_data = model_parser<config>::parse_model(data, &weight_memory);
		//std::cout << "Nihilus model Load time: " << stop_watch_val_nihilus.total_time_elapsed() << std::endl;
		//core_bases_type::template impl<tensor_debugger_impl>();
	}
};

int main() {

	static constexpr model_config config{};
	int64_t value{};
	model<config> model{};
	std::array<std::array<void*, 32>, 32> data{};
	model.impl<weight_mapper>(value, data);
	bnch_swt::doNotOptimizeAway(value);
	return 0;
}
