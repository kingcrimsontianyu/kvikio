/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

#include <kvikio/observation.hpp>

namespace kvikio::benchmark {

/**
 * @brief A destination buffer in host memory or in the memory of the current CUDA device.
 *
 * Zero-filled on allocation, so every page is backed before anything is timed. Host memory is
 * page-aligned.
 */
class Buffer {
 public:
  /**
   * @brief Allocate and zero-fill.
   *
   * @param nbytes Size in bytes.
   * @param memory_kind Host or device memory.
   * @param device CUDA device of device memory. The calling thread's current device is left as
   * it was.
   * @exception std::runtime_error if the allocation fails.
   */
  Buffer(std::size_t nbytes, MemoryKind memory_kind, int device = 0);

  ~Buffer() noexcept;

  Buffer(Buffer const&)            = delete;
  Buffer& operator=(Buffer const&) = delete;
  Buffer(Buffer&&)                 = delete;
  Buffer& operator=(Buffer&&)      = delete;

  /**
   * @brief Address of the buffer.
   *
   * @return The address, a device pointer for device memory.
   */
  [[nodiscard]] void* data() const noexcept;

  /**
   * @brief Size of the buffer.
   *
   * @return The size in bytes.
   */
  [[nodiscard]] std::size_t nbytes() const noexcept;

  /**
   * @brief CUDA device of the buffer.
   *
   * @return The device, meaningful for device memory only.
   */
  [[nodiscard]] int device() const noexcept;

 private:
  void* _data{nullptr};
  std::size_t _nbytes{0};
  MemoryKind _memory_kind{MemoryKind::HOST};
  int _device{0};
};

}  // namespace kvikio::benchmark
