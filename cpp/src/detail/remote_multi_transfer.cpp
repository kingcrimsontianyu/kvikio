/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <curl/curl.h>

#include <kvikio/detail/bounce_buffer_cache.hpp>
#include <kvikio/detail/remote_multi_transfer.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/logger_macros.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

namespace {
void detach_from_multi(CURLM* multi, CURL* easy) noexcept
{
  if (multi == nullptr || easy == nullptr) { return; }
  auto const mc = curl_multi_remove_handle(multi, easy);
  if (mc != CURLM_OK) {
    KVIKIO_LOG_ERROR(std::string("CurlMultiAttachment: curl_multi_remove_handle failed: ") +
                     curl_multi_strerror(mc));
  }
}
}  // namespace

CurlMultiAttachment::CurlMultiAttachment(CURLM* multi, CURL* easy) noexcept
  : _multi{multi}, _easy{easy}
{
}

CurlMultiAttachment::~CurlMultiAttachment()
{
  // Best-effort detach on the reactor I/O thread. If curl_multi_remove_handle fails (rare), the
  // handle stays attached and the owning CurlHandle still returns it to the LibCurl pool, which is
  // undefined behavior in libcurl. A destructor has no better recovery.
  detach_from_multi(_multi, _easy);
}

CurlMultiAttachment::CurlMultiAttachment(CurlMultiAttachment&& o) noexcept
  : _multi{std::exchange(o._multi, nullptr)}, _easy{std::exchange(o._easy, nullptr)}
{
}

CurlMultiAttachment& CurlMultiAttachment::operator=(CurlMultiAttachment&& o) noexcept
{
  if (this != &o) {
    // Detach whatever this guard currently holds before taking over o's handle.
    detach_from_multi(_multi, _easy);
    _multi = std::exchange(o._multi, nullptr);
    _easy  = std::exchange(o._easy, nullptr);
  }
  return *this;
}

RemoteMultiTransfer::~RemoteMultiTransfer()
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;
  // A device transfer still holding its bounce buffer reaches here only on a failure path. The
  // success path moves the buffer into recycle_after, leaving buffer.get() == nullptr.
  //
  // Thread-affinity invariant: the cache is sharded by (this_thread::get_id(), ctx), so a
  // buffer-holding transfer MUST be destroyed on the reactor I/O thread that checked it out during
  // admission. Every such destruction (in-flight drain, admission-walk reset(), completion drop)
  // runs on that thread. Destroying it elsewhere would recycle into the wrong shard and corrupt
  // that shard's accounting.
  if (!is_device || buffer.get() == nullptr) { return; }
  try {
    PushAndPopContext c(device_ctx);
    BounceBufferCache::instance().recycle_now(device_ctx, std::move(buffer));
  } catch (std::exception const& e) {
    KVIKIO_LOG_ERROR(std::string("RemoteMultiTransfer: buffer recycle failed: ") + e.what());
  } catch (...) {
    KVIKIO_LOG_ERROR("RemoteMultiTransfer: buffer recycle failed: unknown exception");
  }
}

RemoteMultiAggregateContext::RemoteMultiAggregateContext(std::size_t num_subranges)
  : _subranges_left{num_subranges}
{
  KVIKIO_EXPECT(num_subranges > 0,
                "RemoteMultiAggregateContext requires at least one sub-range",
                std::invalid_argument);
}

void RemoteMultiAggregateContext::on_subrange_complete(std::size_t bytes)
{
  _total_bytes.fetch_add(bytes, std::memory_order_relaxed);
  // The last thread to decrement _subranges_left to zero fulfills the promise. Its acq_rel
  // decrement acquires every other thread's relaxed _total_bytes writes (each released by that
  // thread's own decrement), so the sum is complete. _first_exception needs no ordering here, since
  // it is written and read under _exception_mutex.
  if (_subranges_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::lock_guard<std::mutex> const lock(_exception_mutex);
    if (_first_exception) {
      _promise.set_exception(_first_exception);
    } else {
      _promise.set_value(_total_bytes.load(std::memory_order_relaxed));
    }
  }
}

void RemoteMultiAggregateContext::on_subrange_failed(std::exception_ptr eptr)
{
  {
    std::lock_guard<std::mutex> const lock(_exception_mutex);
    if (!_first_exception) { _first_exception = eptr; }
  }
  // Last thread to decrement to zero fulfills the promise.
  if (_subranges_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::lock_guard<std::mutex> const lock(_exception_mutex);
    _promise.set_exception(_first_exception);
  }
}

std::future<std::size_t> RemoteMultiAggregateContext::get_future() { return _promise.get_future(); }

}  // namespace kvikio::detail
