#include "ThreadPool.h"

#include <algorithm>

namespace titan {

ThreadPool::ThreadPool(unsigned num_workers) {
  for (unsigned i = 0; i < num_workers; ++i) {
    workers_.emplace_back([this] {
      for (;;) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
          if (stop_ && tasks_.empty()) return;
          task = std::move(tasks_.front());
          tasks_.pop();
        }
        task();
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  for (std::thread& w : workers_) {
    if (w.joinable()) w.join();
  }
}

std::future<void> ThreadPool::enqueue(std::function<void()> task) {
  auto packaged = std::make_shared<std::packaged_task<void()>>(std::move(task));
  std::future<void> fut = packaged->get_future();
  {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.emplace([packaged] { (*packaged)(); });
  }
  cv_.notify_one();
  return fut;
}

void ThreadPool::parallel_for(std::size_t n,
                              const std::function<void(std::size_t, std::size_t)>& fn) {
  if (n == 0) return;
  const unsigned parts = std::max(1u, size() + 1u);  // workers + caller
  const std::size_t chunk = (n + parts - 1) / parts;

  std::vector<std::future<void>> futures;
  for (std::size_t begin = chunk; begin < n; begin += chunk) {
    const std::size_t end = std::min(begin + chunk, n);
    futures.push_back(enqueue([&fn, begin, end] { fn(begin, end); }));
  }
  fn(0, std::min(chunk, n));  // caller runs the first chunk
  for (std::future<void>& f : futures) f.get();
}

ThreadPool& global_pool() {
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  static ThreadPool pool(hw - 1);  // + the calling thread == hw
  return pool;
}

}  // namespace titan
