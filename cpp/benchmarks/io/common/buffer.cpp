/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer.hpp"

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

namespace kvikio::benchmark {
namespace {

void check_cuda(cudaError_t error, char const* call)
{
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string{call} + " failed: " + cudaGetErrorString(error));
  }
}

/**
 * @brief Make a device current for the lifetime of the object, then restore the previous one.
 */
class CurrentDevice {
 public:
  explicit CurrentDevice(int device)
  {
    check_cuda(cudaGetDevice(&_previous), "cudaGetDevice");
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
  }

  ~CurrentDevice() noexcept { cudaSetDevice(_previous); }

  CurrentDevice(CurrentDevice const&)            = delete;
  CurrentDevice& operator=(CurrentDevice const&) = delete;
  CurrentDevice(CurrentDevice&&)                 = delete;
  CurrentDevice& operator=(CurrentDevice&&)      = delete;

 private:
  int _previous{0};
};

}  // namespace

Buffer::Buffer(std::size_t nbytes, MemoryKind memory_kind, int device)
  : _nbytes{nbytes}, _memory_kind{memory_kind}, _device{device}
{
  if (memory_kind == MemoryKind::DEVICE) {
    CurrentDevice const current{device};
    check_cuda(cudaMalloc(&_data, nbytes), "cudaMalloc");
    check_cuda(cudaMemset(_data, 0, nbytes), "cudaMemset");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    return;
  }

  // std::aligned_alloc() wants the size to be a multiple of the alignment.
  auto const page_size        = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  std::size_t const allocated = (nbytes + page_size - 1) / page_size * page_size;
  _data                       = std::aligned_alloc(page_size, allocated);
  if (_data == nullptr) {
    throw std::runtime_error("cannot allocate " + std::to_string(nbytes) + " bytes of host memory");
  }
  std::memset(_data, 0, allocated);
}

Buffer::~Buffer() noexcept
{
  if (_memory_kind == MemoryKind::DEVICE) {
    try {
      CurrentDevice const current{_device};
      cudaFree(_data);
    } catch (...) {
    }
    return;
  }
  std::free(_data);
}

void* Buffer::data() const noexcept { return _data; }

std::size_t Buffer::nbytes() const noexcept { return _nbytes; }

int Buffer::device() const noexcept { return _device; }

}  // namespace kvikio::benchmark
