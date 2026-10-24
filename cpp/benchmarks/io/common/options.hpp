/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace kvikio::benchmark {

/**
 * @brief A command-line flag a benchmark accepts.
 */
struct Flag {
  /// Name without the leading dashes, such as `"read-size"`.
  std::string name;
  /// What the value is, shown in the help, such as `"BYTES"`. Empty for a switch, which takes no
  /// value.
  std::string value_name{};
  /// One line of help.
  std::string help{};
  /// Whether the flag must be given.
  bool required{false};
  /// The value used when the flag is not given, shown in the help.
  std::optional<std::string> default_value{};
};

/**
 * @brief The flags given on the command line.
 *
 * Accepts `--name value`, `--name=value`, and `--name` alone for a switch. An undeclared flag, a
 * missing value, a flag given twice, a missing required flag, and a positional argument are all
 * errors. `--help` and `-h` are always accepted.
 *
 * A getter returns the given value, or else the flag's default, or else nothing.
 */
class Options {
 public:
  /**
   * @brief Declare the flags.
   *
   * @param usage What the help prints before the flags.
   * @param flags Every flag the benchmark accepts.
   */
  Options(std::string usage, std::vector<Flag> flags);

  /**
   * @brief Parse the command line.
   *
   * @param argc Argument count, as passed to `main()`.
   * @param argv Arguments, as passed to `main()`.
   * @exception std::invalid_argument if the command line is not valid.
   */
  void parse(int argc, char** argv);

  /**
   * @brief The help text, listing every flag.
   *
   * @return The text, newline-terminated.
   */
  [[nodiscard]] std::string help() const;

  /**
   * @brief Whether a flag was given on the command line. Meant for switches.
   *
   * @param name The flag's name.
   * @return True if it was given.
   */
  [[nodiscard]] bool is_set(std::string const& name) const;

  /**
   * @brief A flag's value as text.
   *
   * @param name The flag's name.
   * @return The value, or nothing if the flag was not given and has no default.
   */
  [[nodiscard]] std::optional<std::string> get_string(std::string const& name) const;

  /**
   * @brief A flag's value as a non-negative integer, such as `"32"`.
   *
   * @param name The flag's name.
   * @exception std::invalid_argument if the value is not one.
   * @return The value, or nothing if the flag was not given and has no default.
   */
  [[nodiscard]] std::optional<std::size_t> get_count(std::string const& name) const;

  /**
   * @brief A flag's value as a byte count, such as `"8388608"`, `"64MiB"` or `"1GB"`.
   *
   * @param name The flag's name.
   * @exception std::invalid_argument if the value is not one.
   * @return The value, or nothing if the flag was not given and has no default.
   */
  [[nodiscard]] std::optional<std::size_t> get_bytes(std::string const& name) const;

  /**
   * @brief A flag's value as a non-negative number of seconds, such as `"30"` or `"0.5"`.
   *
   * @param name The flag's name.
   * @exception std::invalid_argument if the value is not one.
   * @return The value, or nothing if the flag was not given and has no default.
   */
  [[nodiscard]] std::optional<double> get_seconds(std::string const& name) const;

 private:
  /**
   * @brief Look up a declared flag.
   *
   * @param name The flag's name.
   * @return The flag, or null if no flag has that name.
   */
  [[nodiscard]] Flag const* find_flag(std::string_view name) const;

  std::string _usage;
  std::vector<Flag> _flags;
  /// The values given on the command line, by flag name. A switch maps to an empty string.
  std::map<std::string, std::string> _values;
};

}  // namespace kvikio::benchmark
