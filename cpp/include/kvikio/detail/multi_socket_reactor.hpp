/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <cstddef>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <curl/curl.h>

#include <kvikio/detail/concurrent_request_limiter.hpp>
#include <kvikio/detail/remote_multi_transfer.hpp>
#include <kvikio/remote_handle.hpp>

namespace kvikio::detail {

class MultiSocketReactorPool;  // Forward declaration, because reactors need a back-pointer to the
                               // pool.

/**
 * @brief One reactor has one `CURLM*`, one epoll set, one I/O thread, one submit queue, one
 * in-flight map.
 *
 * This is the multi socket-action sibling of `MultiPollReactor`. It shares the same execution model
 * (a leaked pool of reactor threads multiplexing many easy handles, host and device pipelines,
 * concurrency and bounce-buffer gating, pool-wide death propagation) and the same per-transfer and
 * aggregate types (`RemoteMultiTransfer`, `RemoteMultiAggregateContext`). The only difference is
 * how transfers are driven: instead of `curl_multi_perform()` + `curl_multi_poll()`, libcurl pushes
 * socket-interest changes through `CURLMOPT_SOCKETFUNCTION` and timeout hints through
 * `CURLMOPT_TIMERFUNCTION`, and the reactor waits on a persistent `epoll` set and calls
 * `curl_multi_socket_action()` for the exact sockets that fired.
 *
 * `CURLM*` and the epoll fds are not thread-safe. All libcurl-multi calls, all socket/timer
 * callback work, and all `epoll_ctl`/`epoll_wait` calls happen on `_io_thread`. The only
 * cross-thread operation is writing one byte to `_wakefd` (an eventfd registered in the epoll set),
 * used by `submit()` and pool death to break the reactor out of `epoll_wait`. This replaces
 * `curl_multi_wakeup()`, which only works with `curl_multi_poll()`.
 *
 * @note Instances are intentionally never destroyed. They are owned by the leaked
 * `MultiSocketReactorPool` singleton, so their dtor body is empty. Reactor threads run until the
 * process exits.
 */
class MultiSocketReactor {
 public:
  /**
   * @brief Construct a reactor owned by the given pool.
   *
   * @param pool Non-owning back-pointer to the pool that owns this reactor. Used to observe and
   * propagate pool-wide death state. The pool must outlive the reactor, which is guaranteed because
   * the pool is a leaked singleton that owns this reactor by `unique_ptr`.
   * @param max_concurrent_requests This reactor's private share of the total concurrent-request
   * budget (the global cap divided across reactors). `std::nullopt` means unlimited.
   */
  MultiSocketReactor(MultiSocketReactorPool* pool,
                     std::optional<std::size_t> max_concurrent_requests);
  ~MultiSocketReactor() noexcept;
  MultiSocketReactor(MultiSocketReactor const&)            = delete;
  MultiSocketReactor& operator=(MultiSocketReactor const&) = delete;
  MultiSocketReactor(MultiSocketReactor&&)                 = delete;
  MultiSocketReactor& operator=(MultiSocketReactor&&)      = delete;

  /**
   * @brief Hand off a batch of prepared transfers to this reactor. Thread-safe.
   *
   * The reactor picks the transfers up on its next loop iteration. The caller must have already
   * obtained the aggregate future via `aggregate->get_future()` before calling this. If the pool
   * has already declared death, every transfer in the batch is failed immediately with the recorded
   * death reason and never enters the inbox.
   *
   * @param transfers Per-transfer state, ownership transferred to the reactor.
   */
  void submit(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers);

  /**
   * @brief Wake up the reactor out of its `epoll_wait()`. Thread-safe.
   *
   * Writes one byte to the reactor's eventfd. Used by `MultiSocketReactorPool::signal_death` to
   * make every reactor notice pool death promptly rather than waiting for the epoll timeout.
   */
  void wakeup() noexcept;

 private:
  void io_thread_main();

  /**
   * @brief Splice the inbox into `_pending`, then admit as many pending transfers to libcurl as the
   * concurrency and bounce-buffer gates allow. Mirrors `MultiPollReactor` admission stage.
   *
   * @return True if at least one transfer was added to the multi handle this call, so the caller
   * should kick libcurl with a `CURL_SOCKET_TIMEOUT` socket action to start it.
   */
  bool admit_pending();

  /**
   * @brief Drain finished transfers via `curl_multi_info_read` and resolve their aggregates.
   * Mirrors `MultiPollReactor` completion stage, including the device H2D scheduling and the
   * overflow/error-message handling.
   *
   * @return True if at least one transfer completed, which frees a limiter slot and may unblock a
   * deferred transfer.
   */
  bool drain_completions();

  /**
   * @brief Run one `curl_multi_socket_action` and rethrow any error the socket callback stashed.
   *
   * @param s The socket to act on, or `CURL_SOCKET_TIMEOUT` to service libcurl's own timeouts.
   * @param ev_bitmask The `CURL_CSELECT_*` bitmask describing what happened on `s`.
   */
  void socket_action(curl_socket_t s, int ev_bitmask);

  /**
   * @brief Fail every transfer this reactor is responsible for and exit the loop. Same contract as
   * `MultiPollReactor::fail_all_pending`.
   */
  void fail_all_pending(std::exception_ptr eptr);

  // libcurl socket callback (CURLMOPT_SOCKETFUNCTION). Called on the I/O thread while inside a
  // libcurl-multi call. `userp` is the owning reactor. Adds, modifies, or removes `s` in the epoll
  // set to match libcurl's interest (`what`).
  static int socket_callback(CURL* easy, curl_socket_t s, int what, void* userp, void* socketp);

  // libcurl timer callback (CURLMOPT_TIMERFUNCTION). Called on the I/O thread. Stores the requested
  // timeout so the next `epoll_wait` uses it. `timeout_ms == -1` deletes the timer.
  static int timer_callback(CURLM* multi, long timeout_ms, void* userp);

  MultiSocketReactorPool* _pool;
  ConcurrentRequestLimiter _request_limiter;
  CURLM* _curl_multi{nullptr};
  int _epfd{-1};
  int _wakefd{-1};
  // Latest timeout libcurl asked for via the timer callback, in ms. -1 means "no active timer".
  // Touched only on the I/O thread.
  long _curl_timeout_ms{-1};
  // Sockets currently registered in the epoll set, used to pick EPOLL_CTL_ADD vs EPOLL_CTL_MOD.
  // Touched only on the I/O thread.
  std::unordered_set<int> _registered_fds;
  // Set by the socket callback when an epoll_ctl fails, rethrown by socket_action so the failure is
  // treated as fatal pool death rather than propagating through libcurl's C frames.
  std::exception_ptr _callback_error;
  std::thread _io_thread;
  std::mutex _submit_mutex;
  std::deque<std::unique_ptr<RemoteMultiTransfer>> _inbox;
  std::deque<std::unique_ptr<RemoteMultiTransfer>> _pending;
  std::unordered_map<CURL*, std::unique_ptr<RemoteMultiTransfer>> _in_flight;
};

/**
 * @brief Process-wide pool that owns N `MultiSocketReactor`s and dispatches sub-range transfers to
 * them.
 *
 * A standalone mirror of `MultiReactorPool` for the `MULTI_SOCKET` backend. It reads the same
 * `KVIKIO_REMOTE_IO_NUM_REACTORS`, `KVIKIO_REMOTE_IO_REACTOR_DISPATCH`, and
 * `KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS` knobs and applies the same `PER_CHUNK` / `PER_PREAD`
 * dispatch and pool-death semantics.
 */
class MultiSocketReactorPool {
 public:
  /**
   * @brief Get the process-wide pool, creating it (and its reactor threads) on first use.
   *
   * @note The returned reference points to a heap-allocated singleton that is intentionally never
   * destroyed, mirroring `MultiReactorPool` and the other leaked singletons. This avoids
   * static-destruction-order coupling between the pool, `LibCurl`, the reactor threads, and CUDA
   * teardown.
   */
  static MultiSocketReactorPool& instance();

  MultiSocketReactorPool(MultiSocketReactorPool const&)            = delete;
  MultiSocketReactorPool& operator=(MultiSocketReactorPool const&) = delete;
  MultiSocketReactorPool(MultiSocketReactorPool&&)                 = delete;
  MultiSocketReactorPool& operator=(MultiSocketReactorPool&&)      = delete;

  /**
   * @brief Submit all sub-range transfers belonging to one `RemoteHandle::pread()` call.
   *
   * Routes each transfer to a reactor according to the captured dispatch policy. The caller must
   * have already obtained the aggregate future before invoking this.
   *
   * @param transfers The sub-range transfers, ownership transferred to the pool.
   */
  void submit_pread(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers);

  /**
   * @brief Whether the pool has been marked dead by a reactor that caught a fatal error.
   */
  [[nodiscard]] bool is_dead() const noexcept;

  /**
   * @brief Get the exception that caused pool death, or a null `exception_ptr` if alive.
   */
  [[nodiscard]] std::exception_ptr death_reason() const noexcept;

  /**
   * @brief Mark the pool dead with the given cause, then wake every reactor. Thread-safe. Only the
   * first call wins.
   *
   * @param eptr The exception that causes pool death.
   */
  void signal_death(std::exception_ptr eptr) noexcept;

 private:
  MultiSocketReactorPool();
  ~MultiSocketReactorPool() noexcept;

  std::vector<std::unique_ptr<MultiSocketReactor>> _reactors;
  RemoteReactorDispatch _dispatch;
  std::atomic<std::size_t> _next_reactor_counter{0};
  std::atomic<bool> _dead{false};
  std::mutex mutable _death_mutex;  // Protects writes to `_death_reason`.
  std::exception_ptr _death_reason;
};

}  // namespace kvikio::detail
