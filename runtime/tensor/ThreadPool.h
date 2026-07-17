#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace titan {

// A fixed-size pool of worker threads with a shared task queue.
//
// parallel_for splits an index range across the workers plus the calling
// thread and blocks until every chunk finishes — the workhorse for data-
// parallel kernels like matmul. See docs/architecture/tensor.md (Milestone 2).
class ThreadPool {
 public:
  explicit ThreadPool(unsigned num_workers);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  unsigned size() const { return static_cast<unsigned>(workers_.size()); }

  // Split [0, n) into contiguous chunks and run fn(begin, end) on each across
  // the pool + the caller. Blocks until all chunks complete.
  void parallel_for(std::size_t n,
                    const std::function<void(std::size_t, std::size_t)>& fn);

 private:
  std::future<void> enqueue(std::function<void()> task);

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
};

// Process-wide pool sized to the hardware concurrency.
ThreadPool& global_pool();

}  // namespace titan
