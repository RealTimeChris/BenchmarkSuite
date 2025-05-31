#include <BnchSwt/BenchmarkSuite.hpp>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>
#include <array>
#include <queue>
#include <latch>
#include <bit>

struct oiml_tensor_schedule;

// n-dimensional tensor
struct oiml_tensor {

	oiml_tensor_schedule* schedule;
	size_t padding;
};

struct oiml_tensor_schedule {
	oiml_tensor_schedule(size_t num_threads_processing)
		: num_threads_processing(num_threads_processing)
#if OIML_TRACE_OP_EXECUTION
		  ,
		  thread_processing_slice(num_threads_processing)
#endif
	{
	}

	std::vector<oiml_tensor*> dependents;
	std::atomic<size_t> threads_finished   = { 0 };
	std::atomic<size_t> dependencies_ready = { 0 };
	size_t num_threads_processing;
	size_t num_dependencies_to_await = 0;
	// As a thread hits an op, it increments this using fetch_add. If the retrieved value is >= num_threads_processing,
	// the op is skipped for that thread. Otherwise we call compute_forward on it. This way even if a given thread is busy
	// with an OP, other threads may pick up that work instead. The retrieved value becomes the ith parameter.
	// This is done in a loop so that if 1 thread is taking a *very* long time on an OP, we don't need for it to finish
	// for other OPs to retire.
	std::atomic<int> processing_ticket = { 0 };
	std::once_flag op_scratch_init;
#if OIML_TRACE_OP_EXECUTION
	std::vector<size_t> thread_processing_slice;
	std::chrono::high_resolution_clock::time_point available_at;
	std::chrono::high_resolution_clock::time_point first_scheduled_at;
	std::chrono::high_resolution_clock::time_point completed_at;
#endif
};

class per_thread_processing_queue {
  public:
	per_thread_processing_queue() = default;

	std::pair<int, oiml_tensor*> get_next_available_task(size_t thread) {
		std::unique_lock lock{ mutex_ };
		while (true) {
			tasks_available_.wait(lock, [&] {
				return ops_finished_ || !tasks_.empty();
			});

			if (ops_finished_ && tasks_.empty()) {
				return std::make_pair(-1, nullptr);
			}

			oiml_tensor* next = tasks_.front();

			int slice_id = next->schedule->processing_ticket.fetch_add(1, std::memory_order::acq_rel);
			if (slice_id < next->schedule->num_threads_processing) {
#if OIML_TRACE_OP_EXECUTION
				if (slice_id == 0) {
					next->schedule->first_scheduled_at = std::chrono::high_resolution_clock::now();
				}
				next->schedule->thread_processing_slice[slice_id] = thread;
#endif
				return std::make_pair(slice_id, next);
			} else {
				tasks_.pop();

				if (ops_finished_ && tasks_.empty()) {
					return std::make_pair(-1, nullptr);
				}
			}
		}
	}

	void notify_all_ops_finished() {
		{
			std::unique_lock lock{ mutex_ };
			ops_finished_ = true;
		}
		tasks_available_.notify_one();
	}

	void tasks_made_ready(const std::vector<oiml_tensor*>& tensors) {

		{
			std::lock_guard lock{ mutex_ };
			for (oiml_tensor* tensor: tensors) {
				tasks_.emplace(tensor);
			}
		}

		// Only 1 thread could be blocked waiting for this signal.
		tasks_available_.notify_one();
	}

	void ready_immediately(oiml_tensor* tensor) {
		// No locking because this happens before any threads are accessing this data structure.
#if OIML_TRACE_OP_EXECUTION
		tensor->schedule->available_at = std::chrono::high_resolution_clock::now();
#endif
		tasks_.emplace(tensor);
	}

	void reset() {
		ops_finished_ = false;
	}

  private:
	std::mutex mutex_;
	std::condition_variable tasks_available_;
	std::queue<oiml_tensor*> tasks_;
	bool ops_finished_ = false;
};

const char* oiml_op_name(enum oiml_op op);

class processing_queue {
  public:
	processing_queue(size_t threads) : per_thread_(threads), all_ops_complete_(std::make_unique<std::latch>(static_cast<ptrdiff_t>(threads))) {
	}

	void ready_immediately(oiml_tensor* tensor) {
		for (auto& thread: per_thread_) {
			thread.ready_immediately(tensor);
		}
	}

	std::pair<int, oiml_tensor*> get_next_available_task_for_thread(size_t thread) {
		return per_thread_.at(thread).get_next_available_task(thread);
	}

	void thread_finished(oiml_tensor* tensor) {
		const size_t num_threads_processing_op = tensor->schedule->num_threads_processing;

		const size_t threads_finished = 1 + tensor->schedule->threads_finished.fetch_add(1, std::memory_order::acq_rel);
		// fprintf(stderr, "%zu/%zu threads finished %s\n", threads_finished, num_threads_processing_op, tensor->name);

		if (threads_finished != num_threads_processing_op) {
			return;
		}

#if OIML_TRACE_OP_EXECUTION
		tensor->schedule->completed_at = std::chrono::high_resolution_clock::now();
		std::string threads			   = [tensor] {
			   std::stringstream ss;
			   ss << "[";
			   for (size_t thread_id: tensor->schedule->thread_processing_slice) {
				   if (ss.tellp() > 1) {
					   ss << ", ";
				   }
				   ss << thread_id;
			   }
			   ss << "]";
			   return ss.str();
		}();
		std::string op_desc = [tensor] {
			std::stringstream ss;
			ss << oiml_op_name(tensor->op) << "[" << tensor->ne[0] << "x" << tensor->ne[1] << "x" << tensor->ne[2] << "x" << tensor->ne[3] << "](";
			for (size_t i = 0; i < OIML_MAX_SRC; i++) {
				auto src = tensor->src[i];
				if (src == nullptr) {
					break;
				}
				if (i > 0) {
					ss << ", ";
				}
				ss << "\"" << src->name << "\"" << " [" << src->ne[0] << "x" << src->ne[1] << "x" << src->ne[2] << "x" << src->ne[3] << "]";
			}
			ss << ")";
			return ss.str();
		}();
		fprintf(stderr, "Tensor %s %s took %zu microseconds to schedule and %zu microseconds to process using threads %s\n", tensor->name, op_desc.c_str(),
			std::chrono::duration_cast<std::chrono::microseconds>(tensor->schedule->first_scheduled_at - tensor->schedule->available_at).count(),
			std::chrono::duration_cast<std::chrono::microseconds>(tensor->schedule->completed_at - tensor->schedule->first_scheduled_at).count(), threads.c_str());
#endif

		// All threads have completed this OP. We look for dependents that were made ready by this OP completed.
		// It's fine if multiple threads process a dependent concurrently because the threads will in a wait-free
		// way cooperatively determine the set of tasks made ready & broadcast it out (the broadcast is not lock-free).
		std::vector<oiml_tensor*> made_ready;
		for (oiml_tensor* dependent: tensor->schedule->dependents) {
			// The dependent must need at least 1 thing to finish because it depends on us.
			size_t now_ready = 1 + dependent->schedule->dependencies_ready.fetch_add(1, std::memory_order::acq_rel);
			if (now_ready == dependent->schedule->num_dependencies_to_await) {
				// This dependent has had all it's dependencies satisfied & can be added to the queue for
				// all threads. NOTE: This is wait-free in that if multiple threads are marking this off as
				// complete, only 1 will notice it's the last one this dependent was waiting on & claim
				// it into it's made_ready collection.
#if OIML_TRACE_OP_EXECUTION
				dependent->schedule->available_at = std::chrono::high_resolution_clock::now();
#endif
				made_ready.emplace_back(dependent);
			}
		}

		bool notify_completion = false;
		{
			std::lock_guard lock{ ops_mutex_ };
			num_retired_++;
			notify_completion = num_retired_ == num_ops_;
		}

		if (!notify_completion && made_ready.empty()) {
			// No tensors were made ready / not finished processing so there's nothing to nofiy.
			return;
		}

		// This thread noticed 1 or more tensors have been made ready for processing XOR we have completed processing
		// all OPs in the graph. Notify all threads of the update.
		for (per_thread_processing_queue& per_thread: per_thread_) {
			if (!made_ready.empty()) {
				per_thread.tasks_made_ready(made_ready);
			} else {
				per_thread.notify_all_ops_finished();
			}
		}
	}

	void set_num_ops(size_t n) {
		num_ops_		  = n;
		num_retired_	  = 0;
		all_ops_complete_ = std::make_unique<std::latch>(per_thread_.size());
		for (auto& thread: per_thread_) {
			thread.reset();
		}
	}

	void wait_completed() {
		all_ops_complete_->arrive_and_wait();
	}

  private:
	std::vector<per_thread_processing_queue> per_thread_;
	std::mutex ops_mutex_;
	std::unique_ptr<std::latch> all_ops_complete_;
	size_t num_retired_ = 0;
	size_t num_ops_		= 0;
};

inline static constexpr float fp32_from_bits(uint32_t w) {
	return std::bit_cast<float>(w);
}

inline static constexpr uint32_t fp32_to_bits(float f) {
	return std::bit_cast<uint32_t>(f);
}

inline static constexpr float oiml_compute_fp16_to_fp32(uint16_t h) {
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

alignas(64)
inline const std::array<float, (1 << 16)> fp16_to_fp32_table{ [] {
	std::array<float, (1 << 16)> returnValues{};
	for (uint32_t x = 0; x < (1 << 16); ++x) {
		returnValues[x] = oiml_compute_fp16_to_fp32(static_cast<uint16_t>(x));
	}
	return returnValues;
}() };

int main() {
	bnch_swt::benchmark_stage<"test_stage">::runBenchmark<"test01", "Cyan">([] {
		std::this_thread::sleep_for(std::chrono::milliseconds{ 1000 });
		return 20ull;
	});
	bnch_swt::benchmark_stage<"test_stage", 5, 1, false, "TEST">::printResults();
	auto new_value = fp16_to_fp32_table[0];
	std::cout << "CURRENT VALUE: " << new_value << std::endl;
	return 0;
}
