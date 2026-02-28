/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unistd.h>
#include <cstddef>
#include <cstdlib>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/posix_io.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {
std::size_t posix_device_read_aligned(int fd_direct_on,
                                      void const* devPtr_base,
                                      std::size_t size,
                                      std::size_t file_offset,
                                      std::size_t devPtr_offset)
{
  auto bounce_buffer             = CudaPageAlignedPinnedBounceBufferPool::instance().get();
  CUdeviceptr devPtr             = convert_void2deviceptr(devPtr_base) + devPtr_offset;
  off_t const bounce_buffer_size = convert_size2off(bounce_buffer.size());
  auto const page_size           = get_page_size();
  off_t cur_file_offset          = convert_size2off(file_offset);
  off_t bytes_remaining          = convert_size2off(size);

  // Get a stream for the current CUDA context and thread
  CUstream stream = StreamCachePerThreadAndContext::get();

  while (bytes_remaining > 0) {
    off_t aligned_offset     = align_down(cur_file_offset, page_size);
    std::size_t prefix       = cur_file_offset - aligned_offset;
    std::size_t useful       = std::min<std::size_t>(bytes_remaining, bounce_buffer_size - prefix);
    std::size_t aligned_size = align_up(prefix + useful, page_size);

    // Pure Direct I/O: always aligned offset, aligned buffer, aligned size
    ssize_t nbytes_read = posix_host_io<IOOperationType::READ, PartialIO::YES>(
      -1, bounce_buffer.get(), aligned_size, aligned_offset, fd_direct_on);
    KVIKIO_EXPECT(nbytes_read > static_cast<ssize_t>(prefix),
                  "pread(O_DIRECT): unexpected EOF within the requested range");

    std::size_t actual_useful = std::min<std::size_t>(useful, nbytes_read - prefix);

    CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
      devPtr, static_cast<std::byte*>(bounce_buffer.get()) + prefix, actual_useful, stream));
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));

    cur_file_offset += actual_useful;
    devPtr += actual_useful;
    bytes_remaining -= actual_useful;
  }
  return size;
}

std::size_t posix_device_read(int fd_direct_off,
                              void const* devPtr_base,
                              std::size_t size,
                              std::size_t file_offset,
                              std::size_t devPtr_offset,
                              int fd_direct_on)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  // The bounce buffer must hold at least 2 pages so that a read straddling two adjacent pages can
  // be satisfied in a single aligned pread.
  static std::size_t const lower_bound = 2 * get_page_size();
  // If Direct I/O is supported and requested and bounce buffer is greater than two pages
  if (fd_direct_on != -1 && defaults::auto_direct_io_read() &&
      defaults::bounce_buffer_size() >= lower_bound) {
    return posix_device_read_aligned(fd_direct_on, devPtr_base, size, file_offset, devPtr_offset);
  } else {
    return posix_device_io<IOOperationType::READ>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset);
  }
}

std::size_t posix_device_write(int fd_direct_off,
                               void const* devPtr_base,
                               std::size_t size,
                               std::size_t file_offset,
                               std::size_t devPtr_offset,
                               int fd_direct_on)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  // If Direct I/O is supported and requested
  if (fd_direct_on != -1 && defaults::auto_direct_io_write()) {
    return posix_device_io<IOOperationType::WRITE, CudaPageAlignedPinnedBounceBufferPool>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset, fd_direct_on);
  } else {
    return posix_device_io<IOOperationType::WRITE>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset);
  }
}

}  // namespace kvikio::detail
