/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstring>

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/remote_upload.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

BounceBufferD2H::BounceBufferD2H(CUstream stream, void const* device_buffer, std::size_t size)
  : _stream{stream},
    _dev{convert_void2deviceptr(device_buffer)},
    _size{size},
    _host_buffer{CudaPinnedBounceBufferPool::instance().get()}
{
  KVIKIO_NVTX_FUNC_RANGE();
}

void BounceBufferD2H::refill()
{
  KVIKIO_NVTX_FUNC_RANGE();
  std::size_t const nbytes = std::min(_size - _dev_offset, _host_buffer.size());
  if (nbytes > 0) {
    KVIKIO_CUDA_DRIVER_TRY(cudaAPI::cuda_memcpy_async(
      convert_void2deviceptr(_host_buffer.get()), _dev + _dev_offset, nbytes, _stream));
    KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(_stream));
  }
  _dev_offset += nbytes;
  _host_offset = 0;
  _host_filled = nbytes;
}

std::size_t BounceBufferD2H::read(char* dst, std::size_t max_size)
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (_host_offset == _host_filled) { refill(); }
  std::size_t const nbytes = std::min(max_size, _host_filled - _host_offset);
  if (nbytes > 0) { std::memcpy(dst, _host_buffer.get(_host_offset), nbytes); }
  _host_offset += nbytes;
  return nbytes;
}

void BounceBufferD2H::seek(std::size_t pos) noexcept
{
  _dev_offset  = std::min(pos, _size);
  _host_offset = 0;
  _host_filled = 0;
}

}  // namespace kvikio::detail
