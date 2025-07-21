#include <cstdint>
#include <chrono>
#include <vector>
#include <iostream>
#if defined(__clang__) || (defined(__GNUC__) && defined(__llvm__)) || (defined(__APPLE__) && defined(__clang__))
	#define BNCH_SWT_CLANG 1
#elif defined(_MSC_VER)
	#define BNCH_SWT_MSVC 1
	#pragma warning(disable : 4820)
	#pragma warning(disable : 4371)
	#pragma warning(disable : 4710)
	#pragma warning(disable : 4711)
#elif defined(__GNUC__) && !defined(__clang__)
	#define BNCH_SWT_GNUCXX 1
#endif

#if (defined(__x86_64__) || defined(_M_AMD64)) && !defined(_M_ARM64EC)
	#define BNCH_SWT_IS_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
	#define BNCH_SWT_IS_ARM64 1
#endif

#if defined(macintosh) || defined(Macintosh) || (defined(__APPLE__) && defined(__MACH__))
	#define BNCH_SWT_MAC 1
#elif defined(WIN32) || defined(_WIN32) || defined(_WIN64)
	#define BNCH_SWT_WIN 1
#elif defined(__ANDROID__)
	#define BNCH_SWT_ANDROID 1
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__gnu_linux__)
	#define BNCH_SWT_LINUX 1
#endif

#if defined(BNCH_SWT_FORCE_INLINE)
	#if defined(BNCH_SWT_MSVC)
		#define BNCH_SWT_INLINE [[msvc::forceinline]] inline
	#elif defined(BNCH_SWT_CLANG)
		#define BNCH_SWT_INLINE inline __attribute__((always_inline))
	#elif defined(BNCH_SWT_GNUCXX)
		#define BNCH_SWT_INLINE inline __attribute__((always_inline))
	#else
		#define BNCH_SWT_INLINE inline
	#endif
#else
	#if defined(BNCH_SWT_MSVC)
		#define BNCH_SWT_INLINE inline
	#elif defined(BNCH_SWT_CLANG)
		#define BNCH_SWT_INLINE inline
	#elif defined(BNCH_SWT_GNUCXX)
		#define BNCH_SWT_INLINE inline
	#else
		#define BNCH_SWT_INLINE inline
	#endif
#endif

enum class data_types : uint64_t {
	f32		= 0,
	f16		= 1,
	q4_0	= 2,
	q4_1	= 3,
	q5_0	= 6,
	q5_1	= 7,
	q8_0	= 8,
	q8_1	= 9,
	q2_k	= 10,
	q3_k	= 11,
	q4_k	= 12,
	q5_k	= 13,
	q6_k	= 14,
	q8_k	= 15,
	iq2_xxs = 16,
	iq2_xs	= 17,
	iq3_xxs = 18,
	iq1_s	= 19,
	iq4_nl	= 20,
	iq3_s	= 21,
	iq2_s	= 22,
	iq4_xs	= 23,
	i8		= 24,
	i16		= 25,
	i32		= 26,
	i64		= 27,
	f64		= 28,
	iq1_m	= 29,
	bf16	= 30,
	tq1_0	= 34,
	tq2_0	= 35,
	count	= 39,
};

enum class core_types : uint8_t {
	weights,
	global_inputs,
	token_embeddings,
	mega_qkv_prep_and_cache_publish,
	mega_attention_apply,
	mega_ffn,
	final_norm_and_sampling,
	count,
};

enum class kernel_types : uint8_t {
	weights,
	get_rows,
	rms_norm,
	mul,
	mul_mat,
	reshape,
	transpose,
	permute,
	view,
	rope,
	softmax,
	silu,
	copy,
	cont,
	add,
	sub,
	top_k_filter,
	top_p_filter,
	repetition_penalty,
	presence_penalty,
	temperature_scale,
	frequency_penalty,
	vocab_mask,
	sample_logits,
	count,
};

enum class composite_kernel_types : uint8_t {
	none,
	view,
	get_rows,
	mega_qkv_prep_and_cache,
	mega_attention_apply,
	mega_ffn,
	final_norm_and_sampling,
	count,
};

enum class user_input_types {
	direct_string,
	cin,
	managed,
};

enum class weight_types : uint8_t {
	attn_q,
	attn_k,
	attn_v,
	attn_output,
	attn_norm,
	ffn_gate,
	ffn_up,
	ffn_down,
	ffn_norm,
	token_embd,
	rope_freqs,
	output_norm,
	output,
	count,
};

enum class global_input_types : uint8_t {
	inp_tokens,
	inp_pos,
	cache_k,
	cache_v,
	kq_mask,
	inp_out_ids,
	temperature,
	top_k,
	top_p,
	repetition_penalty,
	presence_penalty,
	frequency_penalty,
	rep_window,
	token_history,
	rng_state,
	logits_bias,
	allowed_vocab_mask,
	count,
};

enum class token_embeddings_types : uint8_t {
	get_rows,
	count,
};

enum class mega_qkv_prep_and_cache_publish_types : uint8_t {
	q_out,
	count,
};

enum class mega_attention_apply_types {
	ffn_inp,
	count,
};

enum class mega_ffn_types {
	l_out,
	count,
};

enum class final_norm_and_sampling_types {
	result_token_id,
	count,
};

enum class global_output_types : uint8_t {
	result_output_composite,
	count,
};

enum class rope_and_cache_types : uint8_t {
	rope_q_permute_type,
	rope_k_copy_type,
	k_rope_view_type,
	v_rope_view_type,
	count,
};

enum class attention_scores_types : uint8_t {
	kq_scores_type,
	count,
};

enum class attention_weighted_values_types : uint8_t {
	attention_output_type,
	count,
};

enum class attention_output_projection_types : uint8_t {
	attn_output_type,
	count,
};

enum class ffn_parallel_projection_types : uint8_t {
	ffn_gate_type,
	ffn_up_type,
	count,
};

enum class ffn_down_projection_types : uint8_t {
	ffn_down_type,
	count,
};

enum class device_types : uint8_t {
	cpu,
	gpu,
	numa,
};

enum class model_arches : uint8_t {
	llama,
	deci,
	falcon,
	baichuan,
	grok,
	gpt2,
	gptj,
	gptneox,
	mpt,
	starcoder,
	refact,
	bert,
	nomic_bert,
	jina_bert_v2,
	bloom,
	stablelm,
	qwen,
	qwen2,
	qwen2moe,
	qwen2vl,
	phi2,
	phi3,
	phimoe,
	plamo,
	codeshell,
	orion,
	internlm2,
	minicpm,
	minicpm3,
	gemma,
	gemma2,
	starcoder2,
	mamba,
	xverse,
	command_r,
	cohere2,
	dbrx,
	olmo,
	olmo2,
	olmoe,
	openelm,
	arctic,
	deepseek,
	deepseek2,
	chatglm,
	bitnet,
	t5,
	t5encoder,
	jais,
	nemotron,
	exaone,
	rwkv6,
	rwkv6qwen2,
	granite,
	granite_moe,
	chameleon,
	wavtokenizer_dec,
	unknown,
	count,
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

enum class rms_norm_types : uint8_t {
	rms_standard,
	rms_parallel,
	rms_grouped,
	layer_norm_standard,
	layer_norm_no_bias,
	rms_norm_welford,
	adaptive_norm,
	count,
};

enum class kv_cache_strategies : uint8_t {
	contiguous,
	paged,
	compressed,
	streaming,
	hierarchical,
	count,
};

enum class rope_scaling_types : uint8_t {
	none,
	linear,
	dynamic,
	yarn,
	longrope,
	count,
};

enum class model_generations : uint8_t {
	v1_v2,
	v3,
	v3_1,
	v3_2,
	count,
};

enum class model_sizes : uint8_t {
	llm_unknown,
	llm_14M,
	llm_17M,
	llm_22M,
	llm_33M,
	llm_60M,
	llm_70M,
	llm_80M,
	llm_109M,
	llm_137M,
	llm_160M,
	llm_220M,
	llm_250M,
	llm_270M,
	llm_335M,
	llm_410M,
	llm_450M,
	llm_770M,
	llm_780M,
	llm_0_5B,
	llm_1B,
	llm_1_3B,
	llm_1_4B,
	llm_1_5B,
	llm_1_6B,
	llm_2B,
	llm_2_8B,
	llm_3B,
	llm_4B,
	llm_6B,
	llm_6_9B,
	llm_7B,
	llm_8B,
	llm_9B,
	llm_11B,
	llm_12B,
	llm_13B,
	llm_14B,
	llm_15B,
	llm_16B,
	llm_20B,
	llm_30B,
	llm_32B,
	llm_34B,
	llm_35B,
	llm_40B,
	llm_65B,
	llm_70B,
	llm_405B,
	llm_SMALL,
	llm_MEDIUM,
	llm_LARGE,
	llm_XL,
	llm_A1_7B,
	llm_A2_7B,
	llm_8x7B,
	llm_8x22B,
	llm_16x12B,
	llm_16x3_8B,
	llm_10B_128x3_66B,
	llm_57B_A14B,
	llm_27B,
	count,
};

enum class tokenizer_types : uint8_t {
	none,
	spm,
	bpe,
	wpm,
	ugm,
	rwkv,
	count,
};

enum class tokenizer_pre_types : uint8_t {
	default_pre,
	llama3,
	deepseek_llm,
	deepseek_coder,
	falcon,
	mpt,
	starcoder,
	gpt2,
	refact,
	command_r,
	stablelm2,
	qwen2,
	olmo,
	dbrx,
	smaug,
	poro,
	chatglm3,
	chatglm4,
	viking,
	jais,
	tekken,
	smollm,
	codeshell,
	bloom,
	gpt3_finnish,
	exaone,
	chameleon,
	minerva,
	deepseek3_llm,
	count,
};

enum class rope_types : int8_t {
	none_rope = -1,
	norm,
	neox,
	mrope,
	vision,
	count,
};

enum class token_types : uint8_t {
	undefined_token,
	normal,
	unknown,
	control,
	user_defined,
	unused,
	byte,
	count,
};

enum class tokens : uint16_t {
	undefined	 = 0,
	unknown		 = 1 << 0,
	unused		 = 1 << 1,
	normal		 = 1 << 2,
	control		 = 1 << 3,
	user_defined = 1 << 4,
	byte		 = 1 << 5,
	normalized	 = 1 << 6,
	lstrip		 = 1 << 7,
	rstrip		 = 1 << 8,
	single_word	 = 1 << 9,
	count,
};

enum class model_formats {
	nh_void,
	gguf,
	count,
};

enum class model_config_types {
	model_generation,
	model_size,
	kernel_type_profile,
	model_arch,
	exceptions,
	default_max_sequence_length,
	default_batch_size,
	kv_cache_strategy,
	user_input_type,
	rope_scaling_type,
	tokenizer_pre_type,
	kv_cache_block_size,
	use_rotary_embeddings,
	rms_norm_type,
	tokenizer_type,
	device_type,
	model_format,
	norm_epsilon,
	benchmark,
	dev
};

struct model_config {
	model_generations model_generation{};
	model_sizes model_size{};
	kernel_type_profiles kernel_type_profile{};
	model_arches model_arch{};
	bool exceptions{};
	uint64_t default_max_sequence_length{};
	uint64_t default_batch_size{};
	kv_cache_strategies kv_cache_strategy{};
	user_input_types user_input_type{};
	rope_scaling_types rope_scaling_type{};
	tokenizer_pre_types tokenizer_pre_type{};
	uint64_t kv_cache_block_size{};
	bool use_rotary_embeddings{};
	rms_norm_types rms_norm_type{};
	tokenizer_types tokenizer_type{};
	device_types device_type{};
	model_formats model_format{};
	float norm_epsilon{};
	bool benchmark{};
	bool dev{};

	BNCH_SWT_INLINE consteval auto update_model_generation(model_generations value) const {
		model_config return_value{ *this };
		return_value.model_generation = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_model_size(model_sizes value) const {
		model_config return_value{ *this };
		return_value.model_size = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_kernel_type_profile(kernel_type_profiles value) const {
		model_config return_value{ *this };
		return_value.kernel_type_profile = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_model_arch(model_arches value) const {
		model_config return_value{ *this };
		return_value.model_arch = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_exceptions(bool value) const {
		model_config return_value{ *this };
		return_value.exceptions = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_default_max_sequence_length(uint64_t value) const {
		model_config return_value{ *this };
		return_value.default_max_sequence_length = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_default_batch_size(uint64_t value) const {
		model_config return_value{ *this };
		return_value.default_batch_size = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_kv_cache_strategy(kv_cache_strategies value) const {
		model_config return_value{ *this };
		return_value.kv_cache_strategy = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_user_input_type(user_input_types value) const {
		model_config return_value{ *this };
		return_value.user_input_type = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_rope_scaling_type(rope_scaling_types value) const {
		model_config return_value{ *this };
		return_value.rope_scaling_type = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_tokenizer_pre_type(tokenizer_pre_types value) const {
		model_config return_value{ *this };
		return_value.tokenizer_pre_type = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_kv_cache_block_size(uint64_t value) const {
		model_config return_value{ *this };
		return_value.kv_cache_block_size = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_use_rotary_embeddings(bool value) const {
		model_config return_value{ *this };
		return_value.use_rotary_embeddings = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_rms_norm_type(rms_norm_types value) const {
		model_config return_value{ *this };
		return_value.rms_norm_type = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_tokenizer_type(tokenizer_types value) const {
		model_config return_value{ *this };
		return_value.tokenizer_type = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_device_type(device_types value) const {
		model_config return_value{ *this };
		return_value.device_type = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_model_format(model_formats value) const {
		model_config return_value{ *this };
		return_value.model_format = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_norm_epsilon(float value) const {
		model_config return_value{ *this };
		return_value.norm_epsilon = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_benchmark(bool value) const {
		model_config return_value{ *this };
		return_value.benchmark = value;
		return return_value;
	}

	BNCH_SWT_INLINE consteval auto update_dev(bool value) const {
		model_config return_value{ *this };
		return_value.dev = value;
		return return_value;
	}
};

BNCH_SWT_INLINE static consteval auto generate_model_config(model_generations model_generation = {}, model_sizes model_size = {}, kernel_type_profiles kernel_type_profile = {},
	model_arches model_arch = {}, device_types device_type = {}, bool exceptions = {}, uint64_t default_max_sequence_length = 1024, uint64_t default_batch_size = {},
	kv_cache_strategies kv_cache_strategy = {}, user_input_types user_input_type = {}, rope_scaling_types rope_scaling_type = {},
	tokenizer_pre_types tokenizer_pre_type = tokenizer_pre_types::llama3, uint64_t kv_cache_block_size = {}, bool use_rotary_embeddings = {}, rms_norm_types rms_norm_type = {},
	tokenizer_types tokenizer_type = tokenizer_types::bpe, model_formats model_format = model_formats::gguf, float norm_epsilon = 0.0f, bool benchmark = {}, bool dev = {}) {
	return model_config{ .model_generation = model_generation,
		.model_size						   = model_size,
		.kernel_type_profile			   = kernel_type_profile,
		.model_arch						   = model_arch,
		.exceptions						   = exceptions,
		.default_max_sequence_length	   = default_max_sequence_length,
		.default_batch_size				   = default_batch_size,
		.kv_cache_strategy				   = kv_cache_strategy,
		.user_input_type				   = user_input_type,
		.rope_scaling_type				   = rope_scaling_type,
		.tokenizer_pre_type				   = tokenizer_pre_type,
		.kv_cache_block_size			   = kv_cache_block_size,
		.use_rotary_embeddings			   = use_rotary_embeddings,
		.rms_norm_type					   = rms_norm_type,
		.tokenizer_type					   = tokenizer_type,
		.device_type					   = device_type,
		.model_format					   = model_format,
		.norm_epsilon					   = norm_epsilon,
		.benchmark						   = benchmark,
		.dev							   = dev };
}

template<typename model_config> static constexpr model_config global_config{ model_config::get_config() };

struct base_struct {
	BNCH_SWT_INLINE virtual void test_function() = 0;
};

template<model_config config> struct test_struct : public base_struct {
	BNCH_SWT_INLINE void test_function() override {
		std::cout << "CURRENTLY HERE!" << std::endl;
	};
};

static constexpr auto model_config_00 = generate_model_config(model_generations::v3_1, model_sizes::llm_8B, kernel_type_profiles::q8_gqa,
	model_arches::llama, device_types::gpu, false, 8192);
static constexpr auto model_config_01 = model_config_00.update_benchmark(true);
static constexpr auto model_config_02 = generate_model_config(model_generations::v3_1, model_sizes::llm_405B, kernel_type_profiles::q8_gqa,
	model_arches::llama, device_types::gpu, false, 8192);

int main() {

	std::vector<std::unique_ptr<base_struct>> test_val{};
	test_val.emplace_back(std::make_unique<test_struct<model_config_00>>());
	for (auto& value: test_val) {
		value->test_function();
	}
	return 0;
}
