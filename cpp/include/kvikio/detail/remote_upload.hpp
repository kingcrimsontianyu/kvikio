/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {

/**
 * @brief Bounce buffer in pinned host memory that stages a device buffer for upload.
 *
 * Libcurl asks for the body in small pieces through `CURLOPT_READFUNCTION`. Copying each piece from
 * device memory on its own would be slow, so the buffer copies `defaults::bounce_buffer_size()`
 * bytes at a time and serves the pieces from host memory.
 *
 * @note Not thread-safe. The caller must have pushed the CUDA context that owns the device buffer.
 */
class BounceBufferD2H {
  CUstream _stream;                                 // The CUDA stream to use.
  CUdeviceptr _dev;                                 // The source device buffer.
  std::size_t _size;                                // Number of bytes in the source device buffer.
  CudaPinnedBounceBufferPool::Buffer _host_buffer;  // The host buffer to bounce data on.
  std::size_t _dev_offset{0};   // Bytes of the source copied to the host buffer so far.
  std::size_t _host_offset{0};  // Bytes of the host buffer handed out so far.
  std::size_t _host_filled{0};  // Bytes of the host buffer holding valid data.

  /**
   * @brief Copy the next piece of the source into the host buffer.
   */
  void refill();

 public:
  /**
   * @brief Create a bounce buffer for a source device buffer.
   *
   * @param stream The CUDA stream used throughout the lifetime of the bounce buffer.
   * @param device_buffer The source device buffer.
   * @param size Number of bytes in the source device buffer.
   */
  BounceBufferD2H(CUstream stream, void const* device_buffer, std::size_t size);

  /**
   * @brief Copy the next bytes of the source into `dst`.
   *
   * @param dst Destination in host memory.
   * @param max_size Capacity of `dst` in bytes.
   * @return Number of bytes copied. Zero once the source is exhausted.
   */
  std::size_t read(char* dst, std::size_t max_size);

  /**
   * @brief Reposition so the next `read()` starts at `pos` bytes into the source.
   *
   * @param pos Position in bytes from the start of the source.
   */
  void seek(std::size_t pos) noexcept;
};

}  // namespace kvikio::detail
