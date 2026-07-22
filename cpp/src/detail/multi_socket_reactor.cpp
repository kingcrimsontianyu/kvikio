/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <curl/curl.h>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/bounce_buffer_cache.hpp>
#include <kvikio/detail/multi_socket_reactor.hpp>
#include <kvikio/detail/remote_multi_transfer.hpp>
#include <kvikio/detail/stream.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/logger_macros.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

namespace {
// Wake a reactor's I/O thread out of epoll_wait by bumping its eventfd. Safe to call from any
// thread, including CUDA driver callback threads. Best-effort: a failed write is harmless because
// the reactor also wakes on its bounded epoll timeout.
void bump_eventfd(int wakefd) noexcept
{
  uint64_t const one = 1;
  std::ignore        = ::write(wakefd, &one, sizeof(one));
}
}  // namespace

MultiSocketReactor::MultiSocketReactor(MultiSocketReactorPool* pool,
                                       std::optional<std::size_t> max_concurrent_requests)
  : _pool{pool}, _request_limiter{max_concurrent_requests}
{
  KVIKIO_EXPECT(
    _pool != nullptr, "MultiSocketReactor requires a non-null pool", std::invalid_argument);
  // Force LibCurl global init before we create the multi handle.
  std::ignore = LibCurl::instance();
  _curl_multi = curl_multi_init();
  KVIKIO_EXPECT(_curl_multi != nullptr, "curl_multi_init() failed", std::runtime_error);

  _epfd = ::epoll_create1(EPOLL_CLOEXEC);
  KVIKIO_EXPECT(
    _epfd >= 0, std::string("epoll_create1: ") + std::strerror(errno), std::runtime_error);

  _wakefd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  KVIKIO_EXPECT(_wakefd >= 0, std::string("eventfd: ") + std::strerror(errno), std::runtime_error);

  epoll_event wake_ev{};
  wake_ev.events  = EPOLLIN;
  wake_ev.data.fd = _wakefd;
  KVIKIO_EXPECT(::epoll_ctl(_epfd, EPOLL_CTL_ADD, _wakefd, &wake_ev) == 0,
                std::string("epoll_ctl(wakefd): ") + std::strerror(errno),
                std::runtime_error);

  auto const sock_mc =
    curl_multi_setopt(_curl_multi, CURLMOPT_SOCKETFUNCTION, &MultiSocketReactor::socket_callback);
  KVIKIO_EXPECT(sock_mc == CURLM_OK,
                std::string("curl_multi_setopt(SOCKETFUNCTION): ") + curl_multi_strerror(sock_mc),
                std::runtime_error);
  curl_multi_setopt(_curl_multi, CURLMOPT_SOCKETDATA, this);
  auto const timer_mc =
    curl_multi_setopt(_curl_multi, CURLMOPT_TIMERFUNCTION, &MultiSocketReactor::timer_callback);
  KVIKIO_EXPECT(timer_mc == CURLM_OK,
                std::string("curl_multi_setopt(TIMERFUNCTION): ") + curl_multi_strerror(timer_mc),
                std::runtime_error);
  curl_multi_setopt(_curl_multi, CURLMOPT_TIMERDATA, this);

  _io_thread = std::thread(&MultiSocketReactor::io_thread_main, this);
}

MultiSocketReactor::~MultiSocketReactor() noexcept
{
  // Intentionally empty. Reactors are owned by the leaked `MultiSocketReactorPool` singleton and
  // never destroyed. This dtor exists only to complete the type for `std::unique_ptr`. Running it
  // would destroy an unjoined `std::thread` and call `std::terminate()`.
}

void MultiSocketReactor::wakeup() noexcept { bump_eventfd(_wakefd); }

int MultiSocketReactor::socket_callback(
  CURL* /*easy*/, curl_socket_t s, int what, void* userp, void* /*socketp*/)
{
  auto* self   = static_cast<MultiSocketReactor*>(userp);
  int const fd = static_cast<int>(s);
  try {
    if (what == CURL_POLL_REMOVE) {
      if (self->_registered_fds.erase(fd) > 0) {
        // The event argument is ignored by EPOLL_CTL_DEL on current kernels.
        if (::epoll_ctl(self->_epfd, EPOLL_CTL_DEL, fd, nullptr) != 0) {
          KVIKIO_FAIL(std::string("epoll_ctl(DEL): ") + std::strerror(errno), std::runtime_error);
        }
      }
    } else {
      epoll_event ev{};
      ev.data.fd = fd;
      if (what == CURL_POLL_IN || what == CURL_POLL_INOUT) { ev.events |= EPOLLIN; }
      if (what == CURL_POLL_OUT || what == CURL_POLL_INOUT) { ev.events |= EPOLLOUT; }
      bool const already = self->_registered_fds.find(fd) != self->_registered_fds.end();
      int const op       = already ? EPOLL_CTL_MOD : EPOLL_CTL_ADD;
      if (::epoll_ctl(self->_epfd, op, fd, &ev) != 0) {
        KVIKIO_FAIL(std::string("epoll_ctl(ADD/MOD): ") + std::strerror(errno), std::runtime_error);
      }
      self->_registered_fds.insert(fd);
    }
  } catch (...) {
    // Never throw through libcurl's C frames. Stash the failure so socket_action rethrows it on the
    // I/O thread and it becomes fatal pool death.
    self->_callback_error = std::current_exception();
  }
  return 0;
}

int MultiSocketReactor::timer_callback(CURLM* /*multi*/, long timeout_ms, void* userp)
{
  // Called on the I/O thread inside a libcurl-multi call, so no synchronization is needed. -1 means
  // libcurl deleted its timer and the reactor should wait on socket activity only.
  static_cast<MultiSocketReactor*>(userp)->_curl_timeout_ms = timeout_ms;
  return 0;
}

void MultiSocketReactor::socket_action(curl_socket_t s, int ev_bitmask)
{
  int running_handles = 0;
  auto const mc       = curl_multi_socket_action(_curl_multi, s, ev_bitmask, &running_handles);
  KVIKIO_EXPECT(mc == CURLM_OK,
                std::string("curl_multi_socket_action: ") + curl_multi_strerror(mc),
                std::runtime_error);
  if (_callback_error) {
    auto eptr       = _callback_error;
    _callback_error = nullptr;
    std::rethrow_exception(eptr);
  }
}

void MultiSocketReactor::submit(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers)
{
  if (transfers.empty()) { return; }
  std::exception_ptr fail_reason;
  {
    std::lock_guard<std::mutex> const lock(_submit_mutex);
    if (_pool->is_dead()) {
      // The pool is dead. Fail the batch immediately instead of pushing into an inbox that will
      // never be drained.
      fail_reason = _pool->death_reason();
    } else {
      for (auto& transfer : transfers) {
        _inbox.push_back(std::move(transfer));
      }
    }
  }
  if (fail_reason) {
    for (auto& transfer : transfers) {
      transfer->aggregate->on_subrange_failed(fail_reason);
    }
    return;
  }
  wakeup();
}

bool MultiSocketReactor::admit_pending()
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;

  // Splice newly submitted transfers out of the inbox (shared with submission threads) to minimize
  // the lock duration.
  {
    std::lock_guard<std::mutex> const lock(_submit_mutex);
    if (_pending.empty()) {
      std::swap(_pending, _inbox);
    } else {
      while (!_inbox.empty()) {
        _pending.push_back(std::move(_inbox.front()));
        _inbox.pop_front();
      }
    }
  }

  bool admitted_any = false;
  // Admission walk over the reactor-private _pending. Each entry is either admitted to libcurl or
  // moved to `deferred_transfers`, which becomes the new `_pending` at the end.
  std::deque<std::unique_ptr<RemoteMultiTransfer>> deferred_transfers;
  // Contexts whose bounce-buffer shard has already missed during this walk. Distinct contexts are
  // assumed few, so a flat vector with linear find suffices.
  std::vector<CUcontext> exhausted_ctxs;
  while (!_pending.empty()) {
    auto transfer = std::move(_pending.front());
    _pending.pop_front();
    try {
      // This ctx already missed the cache this walk, so defer without taking a limiter slot.
      if (transfer->is_device &&
          std::find(exhausted_ctxs.begin(), exhausted_ctxs.end(), transfer->device_ctx) !=
            exhausted_ctxs.end()) {
        deferred_transfers.push_back(std::move(transfer));
        continue;
      }

      // Gate 1 caps network concurrency: the HTTP range requests attached to this reactor's multi
      // handle at once, host and device combined.
      auto slot = _request_limiter.try_acquire();
      if (!slot) {
        deferred_transfers.push_back(std::move(transfer));
        while (!_pending.empty()) {
          deferred_transfers.push_back(std::move(_pending.front()));
          _pending.pop_front();
        }
        break;
      }

      if (transfer->is_device) {
        // Gate 2 caps bounce-buffer use per (reactor thread, CUDA context) across all pipeline
        // phases. A limiter slot freed at libcurl completion does not free the buffer, which stays
        // in-flight until the H2D drains and the recycle callback fires.
        std::optional<CudaPinnedBounceBufferPool::Buffer> bounce_buffer;
        {
          PushAndPopContext c(transfer->device_ctx);
          bounce_buffer = BounceBufferCache::instance().try_get(transfer->device_ctx);
        }
        if (!bounce_buffer) {
          exhausted_ctxs.push_back(transfer->device_ctx);
          deferred_transfers.push_back(std::move(transfer));
          continue;
        }
        transfer->buffer            = std::move(*bounce_buffer);
        transfer->ctx.pinned_buffer = transfer->buffer.get();
      }

      CURL* easy    = transfer->curl->handle();
      auto const mc = curl_multi_add_handle(_curl_multi, easy);
      if (mc != CURLM_OK) {
        // Notify the aggregate to satisfy its sub-range count invariant. Null the local pointer so
        // the catch below skips requeueing this already-failed transfer.
        transfer->aggregate->on_subrange_failed(std::make_exception_ptr(
          std::runtime_error(std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc))));
        transfer.reset();
        KVIKIO_FAIL(std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc),
                    std::runtime_error);
      }
      // Set the guard before the _in_flight.emplace so an emplace failure still detaches the
      // now-attached handle on unwind.
      transfer->attachment = CurlMultiAttachment{_curl_multi, easy};
      transfer->slot       = std::move(slot);
      _in_flight.emplace(easy, std::move(transfer));
      admitted_any = true;
    } catch (...) {
      // Requeue the in-hand transfer (unless already failed above) and the already-deferred
      // entries, so fail_all_pending, which drains `_pending`, resolves their aggregates.
      if (transfer) { _pending.push_front(std::move(transfer)); }
      while (!deferred_transfers.empty()) {
        _pending.push_front(std::move(deferred_transfers.back()));
        deferred_transfers.pop_back();
      }
      throw;
    }
  }
  // The walk drained `_pending`. The deferred entries become the new pending queue.
  std::swap(_pending, deferred_transfers);
  return admitted_any;
}

bool MultiSocketReactor::drain_completions()
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;

  int msgs_left      = 0;
  bool completed_any = false;
  while (auto* msg = curl_multi_info_read(_curl_multi, &msgs_left)) {
    if (msg->msg != CURLMSG_DONE) { continue; }
    completed_any = true;
    auto* easy    = msg->easy_handle;
    auto res      = msg->data.result;

    auto it = _in_flight.find(easy);
    KVIKIO_EXPECT(it != _in_flight.end(),
                  "MultiSocketReactor: completion for unknown handle",
                  std::runtime_error);
    auto transfer = std::move(it->second);
    _in_flight.erase(it);

    std::exception_ptr transfer_err;
    if (res == CURLE_OK && !transfer->ctx.overflow_error) {
      try {
        if (transfer->is_device) {
          // Phase A (network -> pinned) done. Schedule Phase B (pinned -> device) on this
          // (thread, ctx) stream and hand the buffer to a cuLaunchHostFunc recycle callback so the
          // cache slot is returned when the H2D drains. The callback wakes this reactor's epoll
          // loop via the eventfd (curl_multi_wakeup does not break epoll_wait).
          PushAndPopContext c(transfer->device_ctx);
          CUstream stream = StreamCachePerThreadAndContext::get();
          KVIKIO_CUDA_DRIVER_TRY(
            cudaAPI::instance().MemcpyHtoDAsync(convert_void2deviceptr(transfer->device_dst),
                                                transfer->buffer.get(),
                                                transfer->ctx.size,
                                                stream));
          transfer->aggregate->io_event_barrier->record_event(stream);
          BounceBufferCache::instance().recycle_after(
            transfer->device_ctx,
            std::move(transfer->buffer),
            stream,
            [wakefd = _wakefd]() noexcept { bump_eventfd(wakefd); });
        }
        transfer->aggregate->on_subrange_complete(transfer->ctx.size);
      } catch (...) {
        transfer_err = std::current_exception();
      }
    } else {
      // Prefer the handle's recorded error buffer. Fall back to the generic strerror text when
      // libcurl recorded no message.
      auto const errmsg = transfer->curl->error_message();
      std::string desc  = std::string("curl_multi transfer failed (") +
                         (errmsg.empty() ? std::string{curl_easy_strerror(res)} : errmsg) + ")";
      if (transfer->ctx.overflow_error) {
        desc += " [server returned more bytes than requested; maybe range support missing?]";
      }
      transfer_err = std::make_exception_ptr(std::runtime_error(std::move(desc)));
    }
    if (transfer_err) { transfer->aggregate->on_subrange_failed(transfer_err); }
  }
  return completed_any;
}

void MultiSocketReactor::io_thread_main()
{
  constexpr int kMaxEvents = 64;
  try {
    while (!_pool->is_dead()) {
      // (1) Splice the inbox and admit as many transfers as the gates allow.
      bool const admitted = admit_pending();
      // (2) Kick libcurl to start any newly added transfers, which emits their initial
      // socket-interest and timer callbacks.
      if (admitted) { socket_action(CURL_SOCKET_TIMEOUT, 0); }

      // (3) Drain whatever finished so far (including anything the kick above completed).
      bool const completed_any = drain_completions();

      // (4) Wait on the epoll set. 1s idle cap, 10ms while transfers stay deferred in `_pending`,
      // clamped to any timeout libcurl requested via the timer callback, and 0 when a completion
      // this iteration freed a slot and work remains so admission retries at once.
      long timeout_ms = 1000;
      if (!_pending.empty()) { timeout_ms = std::min<long>(timeout_ms, 10); }
      if (_curl_timeout_ms >= 0) { timeout_ms = std::min<long>(timeout_ms, _curl_timeout_ms); }
      if (completed_any && !_pending.empty()) { timeout_ms = 0; }

      std::array<epoll_event, kMaxEvents> events{};
      int const n = ::epoll_wait(_epfd, events.data(), kMaxEvents, static_cast<int>(timeout_ms));
      if (n < 0) {
        if (errno == EINTR) { continue; }
        KVIKIO_FAIL(std::string("epoll_wait: ") + std::strerror(errno), std::runtime_error);
      }

      if (n == 0) {
        // Timed out. Let libcurl service its own timeouts.
        socket_action(CURL_SOCKET_TIMEOUT, 0);
      } else {
        for (int i = 0; i < n; ++i) {
          int const fd = events[i].data.fd;
          if (fd == _wakefd) {
            // A submission or pool-death wakeup. Drain the eventfd counter, the actual work is
            // handled at the loop top.
            uint64_t drain = 0;
            while (::read(_wakefd, &drain, sizeof(drain)) == static_cast<ssize_t>(sizeof(drain))) {}
            continue;
          }
          int ev_bitmask = 0;
          if (events[i].events & EPOLLIN) { ev_bitmask |= CURL_CSELECT_IN; }
          if (events[i].events & EPOLLOUT) { ev_bitmask |= CURL_CSELECT_OUT; }
          if (events[i].events & (EPOLLERR | EPOLLHUP)) { ev_bitmask |= CURL_CSELECT_ERR; }
          socket_action(static_cast<curl_socket_t>(fd), ev_bitmask);
        }
      }

      // (5) Drain completions triggered by the socket activity above.
      drain_completions();
    }
  } catch (...) {
    // Any fatal error caught above declares pool-wide death. The first reactor to signal wins.
    KVIKIO_LOG_ERROR("MultiSocketReactor: fatal error, reactor pool declared dead");
    _pool->signal_death(std::current_exception());
  }
  // Reached by catching the exception above or by noticing _pool->is_dead() at the loop top. Either
  // way, drain our own state with the recorded reason so no caller's future.get() hangs.
  fail_all_pending(_pool->death_reason());
}

void MultiSocketReactor::fail_all_pending(std::exception_ptr eptr)
{
  // Drain the inbox under the submit mutex. submit()'s is_dead() check, already true here, stops
  // new entries from accumulating. Inbox transfers never went through admission, so they hold no
  // buffer and no limiter slot.
  {
    std::lock_guard<std::mutex> const lock(_submit_mutex);
    while (!_inbox.empty()) {
      auto transfer = std::move(_inbox.front());
      _inbox.pop_front();
      transfer->aggregate->on_subrange_failed(eptr);
    }
  }

  // Drain the deferred queue. These transfers normally hold no buffer and no attachment. The
  // exception is one requeued after an _in_flight.emplace failure, which may carry both.
  while (!_pending.empty()) {
    auto transfer = std::move(_pending.front());
    _pending.pop_front();
    transfer->aggregate->on_subrange_failed(eptr);
  }

  // In-flight is touched only by the I/O thread, which is us, so no lock needed.
  for (auto& in_flight_entry : _in_flight) {
    in_flight_entry.second->aggregate->on_subrange_failed(eptr);
  }
  _in_flight.clear();
}

MultiSocketReactorPool::MultiSocketReactorPool() : _dispatch{defaults::remote_io_reactor_dispatch()}
{
  // Force LibCurl global init before any reactor opens a multi handle.
  std::ignore = LibCurl::instance();

  auto const n = defaults::remote_io_num_reactors();
  KVIKIO_EXPECT(n > 0, "remote_io_num_reactors must be a positive integer", std::invalid_argument);

  auto const max_total = defaults::remote_io_max_concurrent_requests();
  std::optional<std::size_t> const per_reactor_max =
    (max_total == 0) ? std::nullopt : std::optional{std::max<std::size_t>(max_total / n, 1)};

  _reactors.reserve(n);
  for (unsigned int i = 0; i < n; ++i) {
    _reactors.emplace_back(std::make_unique<MultiSocketReactor>(this, per_reactor_max));
  }
}

MultiSocketReactorPool::~MultiSocketReactorPool() noexcept
{
  // Intentionally empty. The pool is a leaked singleton, so this dtor is never invoked.
}

MultiSocketReactorPool& MultiSocketReactorPool::instance()
{
  // Heap-leaked singleton. The pool, its reactors, and their `std::thread`s are never destroyed.
  // Resources are cleaned on process exit.
  static MultiSocketReactorPool* inst = new MultiSocketReactorPool();
  return *inst;
}

void MultiSocketReactorPool::submit_pread(
  std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers)
{
  auto const reactor_count = _reactors.size();

  // PER_PREAD: one reactor for the whole pread() call. Preserves per-CURLM connection-pool reuse.
  if (_dispatch == RemoteReactorDispatch::PER_PREAD) {
    auto const idx = _next_reactor_counter.fetch_add(1, std::memory_order_relaxed) % reactor_count;
    _reactors[idx]->submit(std::move(transfers));
    return;
  }

  // PER_CHUNK: round-robin sub-ranges across reactors.
  std::vector<std::vector<std::unique_ptr<RemoteMultiTransfer>>> buckets(reactor_count);
  for (auto& transfer : transfers) {
    auto const idx = _next_reactor_counter.fetch_add(1, std::memory_order_relaxed) % reactor_count;
    buckets[idx].push_back(std::move(transfer));
  }
  for (std::size_t i = 0; i < reactor_count; ++i) {
    if (!buckets[i].empty()) { _reactors[i]->submit(std::move(buckets[i])); }
  }
}

bool MultiSocketReactorPool::is_dead() const noexcept
{
  // This function is on a hot path, so we use atomic instead of a mutex.
  return _dead.load(std::memory_order_acquire);
}

std::exception_ptr MultiSocketReactorPool::death_reason() const noexcept
{
  std::lock_guard<std::mutex> const lock(_death_mutex);
  return _death_reason;
}

void MultiSocketReactorPool::signal_death(std::exception_ptr eptr) noexcept
{
  // The lock serializes _death_reason writes and keeps the _dead store in its scope so the first
  // writer wins, not the last. The store is `release`, pairing with the `acquire` in `is_dead()`.
  {
    std::lock_guard<std::mutex> const lock(_death_mutex);
    // Only the first thread here updates _death_reason and wakes reactors. Later calls early-exit.
    if (_dead.load(std::memory_order_relaxed)) { return; }
    _death_reason = eptr;
    _dead.store(true, std::memory_order_release);
  }

  // Wake every reactor out of epoll_wait so they notice _dead promptly. Including the caller's own
  // reactor is harmless, since it has already left its loop.
  for (auto const& r : _reactors) {
    r->wakeup();
  }
}

}  // namespace kvikio::detail
