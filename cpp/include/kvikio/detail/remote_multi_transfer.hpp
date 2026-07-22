/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <cstddef>
#include <exception>
#include <future>
#include <memory>
#include <mutex>

#include <curl/curl.h>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/detail/concurrent_request_limiter.hpp>
#include <kvikio/detail/io_event_barrier.hpp>
#include <kvikio/detail/remote_callback.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {

// The per-transfer, per-aggregate, and attachment types below are shared by every libcurl multi-API
// remote backend (`MULTI_POLL` and `MULTI_SOCKET`). They carry no reactor-specific logic, so they
// live here rather than in any one reactor header.

/**
 * @brief Collects results from N sub-range transfers and resolves one top-level future once all of
 * them have either succeeded or one has failed.
 *
 * Every sub-range transfer belonging to a single `RemoteHandle::pread()` call holds a
 * `std::shared_ptr<RemoteMultiAggregateContext>`. As completions arrive on the reactor threads
 * (potentially in parallel when `KVIKIO_REMOTE_IO_NUM_REACTORS > 1`), each one calls
 * `on_subrange_complete()` or `on_subrange_failed()`. The thread that decrements `_subranges_left`
 * to zero fulfills `_promise`, with the accumulated byte total on success, or with the first
 * captured exception on failure.
 */
class RemoteMultiAggregateContext {
 public:
  /**
   * @brief Construct an aggregate that expects exactly `num_subranges` completion events.
   *
   * @param num_subranges Number of sub-range transfers the caller has split the read into.
   */
  explicit RemoteMultiAggregateContext(std::size_t num_subranges);

  /**
   * @brief Optional per-pread event watermark for the device-buffer path.
   *
   * Populated by `RemoteHandle::pread` when the destination is device memory and shared by all
   * sub-range transfers belonging to this pread. The reactor records on it after each
   * `cuMemcpyAsync`. The caller's deferred future waits on `sync_all_events()` before returning.
   * Null for host transfers.
   */
  std::shared_ptr<IoEventBarrier> io_event_barrier;

  /**
   * @brief Report that one sub-range transfer succeeded.
   *
   * @param bytes Number of bytes the sub-range delivered.
   */
  void on_subrange_complete(std::size_t bytes);

  /**
   * @brief Report that one sub-range transfer failed. The first exception captured wins.
   *
   * @param eptr The exception describing the failure.
   */
  void on_subrange_failed(std::exception_ptr eptr);

  /**
   * @brief Obtain the future the caller will observe. Must be called exactly once, before any
   * sub-range is submitted to the pool.
   */
  std::future<std::size_t> get_future();

 private:
  std::atomic<std::size_t> _subranges_left;
  std::atomic<std::size_t> _total_bytes{0};
  std::mutex _exception_mutex;
  std::exception_ptr _first_exception;
  std::promise<std::size_t> _promise;
};

/**
 * @brief RAII guard that keeps one libcurl easy handle attached to a multi handle.
 *
 * Set by the reactor right after a successful `curl_multi_add_handle`. Its destructor calls
 * `curl_multi_remove_handle`, so the handle is detached when the owning `RemoteMultiTransfer` is
 * destroyed. A default-constructed or moved-from guard is unset and does nothing on destruction.
 *
 * @note Must be destroyed on the reactor I/O thread that set it, because `CURLM*` is not
 * thread-safe. It is a `RemoteMultiTransfer` member declared after `curl`, so it detaches the
 * handle before `CurlHandle` returns it to the LibCurl pool.
 */
class CurlMultiAttachment {
 public:
  /**
   * @brief Construct an unset guard that holds no attachment.
   */
  CurlMultiAttachment() noexcept = default;

  /**
   * @brief Set a guard for an easy handle already attached to `multi`.
   *
   * @param multi The multi handle the easy handle was added to.
   * @param easy The easy handle to remove on destruction.
   */
  CurlMultiAttachment(CURLM* multi, CURL* easy) noexcept;

  ~CurlMultiAttachment();

  // Move-only.
  CurlMultiAttachment(CurlMultiAttachment&& o) noexcept;
  CurlMultiAttachment& operator=(CurlMultiAttachment&& o) noexcept;
  CurlMultiAttachment(CurlMultiAttachment const&)            = delete;
  CurlMultiAttachment& operator=(CurlMultiAttachment const&) = delete;

 private:
  CURLM* _multi{nullptr};
  CURL* _easy{nullptr};
};

/**
 * @brief Per-transfer state owned by a reactor between submission and completion.
 *
 * One `RemoteMultiTransfer` corresponds to one libcurl easy handle, which corresponds to one HTTP
 * range request. Sub-ranges of the same `pread()` share the same `aggregate`. The `curl` member is
 * held by `std::unique_ptr` because `CurlHandle` is intentionally non-movable.
 *
 * Device-buffer fields (`is_device`, `device_ctx`, `device_dst`, `buffer`) are populated by the
 * pread submitter when the destination is device memory and consumed by the reactor's admission and
 * completion stages. For host transfers, `is_device` is false and the other device fields are
 * unused.
 */
struct RemoteMultiTransfer {
  std::unique_ptr<CurlHandle> curl;

  // Detaches `curl`'s easy handle from the multi handle on destruction. Declared right after `curl`
  // so it runs before `curl` returns the handle to the LibCurl pool. Set by the reactor after a
  // successful `curl_multi_add_handle`, unset until then.
  CurlMultiAttachment attachment;

  CallbackContext ctx;
  std::shared_ptr<RemoteMultiAggregateContext> aggregate;

  // Concurrency slot held from admission until this transfer is destroyed after completion or
  // failure. Empty while the transfer waits in the inbox. Destroying the transfer returns the slot
  // to the reactor's limiter.
  ConcurrentRequestLimiter::Slot slot;

  // Device-path fields. All zeroed/null for host transfers.
  bool is_device{false};
  CUcontext device_ctx{nullptr};
  void* device_dst{nullptr};
  // Pinned bounce buffer checked out from the cache during admission. The reactor moves it into
  // `cache.recycle_after` when the H2D is scheduled at completion. On failure paths where it was
  // not moved, ~RemoteMultiTransfer recycles it via recycle_now.
  CudaPinnedBounceBufferPool::Buffer buffer{nullptr, nullptr, 0};

  // Recycles `buffer` to the bounce-buffer cache if it was not already moved out (failure paths).
  // Must run on the reactor I/O thread that checked the buffer out. See the definition in
  // remote_multi_transfer.cpp for the thread-affinity invariant.
  ~RemoteMultiTransfer();
};

}  // namespace kvikio::detail
