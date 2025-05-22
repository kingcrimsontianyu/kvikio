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

#include <stdexcept>

namespace kvikio::detail {
/**
 * @brief Check if a shared library wrapped in a singleton shim class is available
 *
 * @tparam SingletonClass Singleton shim class
 * @return Boolean answer
 */
template <typename SingletonClass>
bool is_available()
{
  static auto result = []() -> bool {
    try {
      SingletonClass::instance();
    } catch (std::runtime_error const&) {
      return false;
    }
    return true;
  }();
  return result;
}
}  // namespace kvikio::detail
