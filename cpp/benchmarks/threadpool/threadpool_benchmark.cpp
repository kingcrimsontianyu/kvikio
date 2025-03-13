/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <cstdint>
#include <iostream>

#include <kvikio/defaults.hpp>
#include <nvbench/nvbench.cuh>

namespace kvikio {

void task_compute(std::size_t num_compute_iterations)
{
  [[maybe_unused]] double res{0.0};
  for (std::size_t i = 0u; i < num_compute_iterations; ++i) {
    auto x{static_cast<double>(i)};
    res += std::sqrt(x) + std::cbrt(x) + std::sin(x);
  }
}

void NVB_threadpool_compute(nvbench::state& state)
{
  auto num_threads  = state.get_int64("num_threads");
  auto scaling_type = state.get_string("scaling_type");

  std::string label;
  std::size_t num_compute_tasks;
  if (scaling_type == "strong") {
    num_compute_tasks = 1000;
  } else {  // "weak"
    num_compute_tasks = 1000 * num_threads;
  }

  std::size_t const num_compute_iterations{10000};
  kvikio::defaults::set_thread_pool_nthreads(num_threads);

  state.exec([=](nvbench::launch&) {
    for (std::size_t i = 0u; i < num_compute_tasks; ++i) {
      [[maybe_unused]] auto fut = kvikio::defaults::thread_pool().submit_task(
        [num_compute_iterations = num_compute_iterations] {
          task_compute(num_compute_iterations);
        });
    }
    kvikio::defaults::thread_pool().wait();
  });
}

NVBENCH_BENCH(NVB_threadpool_compute)
  .add_int64_axis("num_threads", {1, 2, 4, 8, 16, 32, 64})
  .add_string_axis("scaling_type", {"strong", "weak"});

}  // namespace kvikio
