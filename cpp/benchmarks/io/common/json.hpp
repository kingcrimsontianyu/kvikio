/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace kvikio::benchmark {

/**
 * @brief Builds one JSON object, a member at a time, in the order the members are added.
 */
class JsonObject {
 public:
  void add_string(std::string_view key, std::string_view value);
  void add_integer(std::string_view key, std::uint64_t value);
  /// A number that is not finite is written as `null`, which JSON has no number for.
  void add_number(std::string_view key, double value);
  void add_numbers(std::string_view key, std::vector<double> const& values);
  void add_null(std::string_view key);
  void add_object(std::string_view key, JsonObject const& value);
  /// Add JSON that is already serialized, such as the output of `Summary::to_json()`.
  void add_json(std::string_view key, std::string_view json);

  /**
   * @brief Serialize.
   *
   * @return The object, on one line.
   */
  [[nodiscard]] std::string str() const;

 private:
  /// Start a member: the separator from the previous one, the key, and the colon.
  void add_key(std::string_view key);

  std::string _members;
};

}  // namespace kvikio::benchmark
