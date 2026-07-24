/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <exception>
#include <utility>

#include <kvikio/detail/event.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {

CudaEventPool::CudaEvent::CudaEvent(CudaEventPool* pool,
                                    CUevent event,
                                    CUcontext cuda_context) noexcept
  : _pool(pool), _event(event), _cuda_context(cuda_context)
{
}

CudaEventPool::CudaEvent::~CudaEvent() noexcept
{
  if (_event != nullptr) { _pool->put(_event, _cuda_context); }
}

CudaEventPool::CudaEvent::CudaEvent(CudaEvent&& o) noexcept
  : _pool(std::exchange(o._pool, nullptr)),
    _event(std::exchange(o._event, nullptr)),
    _cuda_context(std::exchange(o._cuda_context, nullptr))
{
}

CudaEventPool::CudaEvent& CudaEventPool::CudaEvent::operator=(CudaEvent&& o) noexcept
{
  if (this != &o) {
    if (_event != nullptr) {
      // Return this event to the pool
      _pool->put(_event, _cuda_context);
    }
    _pool         = std::exchange(o._pool, nullptr);
    _event        = std::exchange(o._event, nullptr);
    _cuda_context = std::exchange(o._cuda_context, nullptr);
  }
  return *this;
}

CUevent CudaEventPool::CudaEvent::get() const noexcept { return _event; }

CUcontext CudaEventPool::CudaEvent::cuda_context() const noexcept { return _cuda_context; }

void CudaEventPool::CudaEvent::record(CUstream stream)
{
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().EventRecord(_event, stream));
}

void CudaEventPool::CudaEvent::synchronize()
{
  KVIKIO_NVTX_FUNC_RANGE();
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().EventSynchronize(_event));
}

bool CudaEventPool::CudaEvent::is_done() const
{
  auto const status = cudaAPI::instance().EventQuery(_event);
  if (status == CUDA_SUCCESS) { return true; }
  if (status == CUDA_ERROR_NOT_READY) { return false; }
  // Any other return code is an error.
  KVIKIO_CUDA_DRIVER_TRY(status);
  // Unreachable. Macro throws on non-success codes.
  return false;
}

CudaEventPool::CudaEvent CudaEventPool::get()
{
  KVIKIO_NVTX_FUNC_RANGE();
  CUcontext ctx{};
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&ctx));
  KVIKIO_EXPECT(ctx != nullptr, "No CUDA context is current");

  // When KVIKIO_TEST_NO_EVENT_POOL=1, bypass the cache and always create a fresh event so we can
  // measure cuEventCreate/cuEventDestroy overhead vs pool reuse.
  static bool const bypass_pool = (std::getenv("KVIKIO_TEST_NO_EVENT_POOL") != nullptr);

  CUevent event{};
  if (!bypass_pool) {
    std::lock_guard const lock(_mutex);
    if (auto it = _pools.find(ctx); it != _pools.end() && !it->second.empty()) {
      event = it->second.back();
      it->second.pop_back();
    }
  }

  if (event == nullptr) {
    KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().EventCreate(&event, CU_EVENT_DISABLE_TIMING));
  }

  return CudaEvent(this, event, ctx);
}

void CudaEventPool::put(CUevent event, CUcontext cuda_context) noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (event == nullptr) { return; }

  static bool const bypass_pool = (std::getenv("KVIKIO_TEST_NO_EVENT_POOL") != nullptr);

  if (bypass_pool) {
    try {
      KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().EventDestroy(event));
    } catch (std::exception const& e) {
      KVIKIO_LOG_ERROR(e.what());
    }
    return;
  }

  try {
    std::lock_guard const lock(_mutex);
    _pools[cuda_context].push_back(event);
  } catch (std::exception const& e) {
    KVIKIO_LOG_ERROR(e.what());
    try {
      KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().EventDestroy(event));
    } catch (std::exception const& e2) {
      KVIKIO_LOG_ERROR(e2.what());
    }
  }
}

std::size_t CudaEventPool::num_free_events(CUcontext cuda_context) const
{
  std::lock_guard const lock(_mutex);
  auto it = _pools.find(cuda_context);
  return (it != _pools.end()) ? it->second.size() : 0;
}

std::size_t CudaEventPool::total_free_events() const
{
  std::lock_guard const lock(_mutex);
  std::size_t total{0};
  for (auto const& [_, events] : _pools) {
    total += events.size();
  }
  return total;
}

CudaEventPool& CudaEventPool::instance()
{
  static CudaEventPool pool;
  return pool;
}

}  // namespace kvikio::detail
