#pragma once

#include <oiml/common/common.hpp>
#include <oiml/common/op_traits.hpp>
#include <oiml/common/unique_ptr.hpp>
#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <type_traits>

namespace oiml {

	class oiml_threadpool {
	  public:
		OIML_FORCE_INLINE oiml_threadpool(size_t num_threads = std::thread::hardware_concurrency()) : stop(false) {
			for (size_t i = 0; i < num_threads; ++i) {
				workers.emplace_back([this] {
					while (true) {
						std::function<void()> task;

						{
							std::unique_lock<std::mutex> lock(this->queue_mutex);
							this->condition.wait(lock, [this] {
								return this->stop.load(std::memory_order_acquire) || !this->tasks.empty();
							});

							if (this->stop.load(std::memory_order_acquire) && this->tasks.empty()) {
								return;
							}

							task = std::move(this->tasks.front());
							this->tasks.pop();
						}

						task();
					}
				});
			}
		}

		template<typename F, typename... Args> OIML_FORCE_INLINE auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
			using return_type = typename std::invoke_result_t<F, Args...>;

			auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

			std::future<return_type> res = task->get_future();

			{
				std::unique_lock<std::mutex> lock(queue_mutex);
				if (stop.load(std::memory_order_acquire)) {
					throw std::runtime_error("oiml_threadpool is stopped");
				}
				tasks.emplace([task_new = std::move(task)]() mutable {
					(*task_new)();
				});
			}

			condition.notify_one();
			return res;
		}

		OIML_FORCE_INLINE ~oiml_threadpool() {
			{
				std::unique_lock<std::mutex> lock(queue_mutex);
				stop.store(true, std::memory_order_release);
			}
			condition.notify_all();
			for (std::thread& worker: workers) {
				worker.join();
			}
		}

	  private:
		std::queue<std::function<void()>> tasks{};
		std::condition_variable condition{};
		std::vector<std::thread> workers{};
		std::mutex queue_mutex{};
		std::atomic_bool stop{};
	};

	inline static oiml_threadpool thread_pool_val{};

}