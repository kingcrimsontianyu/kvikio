/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "options.hpp"

#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace kvikio::benchmark {
namespace {

struct ByteUnit {
  std::string_view suffix;
  std::uint64_t multiplier;
};

// Matched against the whole suffix, lowercased.
constexpr std::array<ByteUnit, 10> byte_units{{
  {"", 1},
  {"b", 1},
  {"kb", 1000ULL},
  {"mb", 1000ULL * 1000},
  {"gb", 1000ULL * 1000 * 1000},
  {"tb", 1000ULL * 1000 * 1000 * 1000},
  {"kib", 1024ULL},
  {"mib", 1024ULL * 1024},
  {"gib", 1024ULL * 1024 * 1024},
  {"tib", 1024ULL * 1024 * 1024 * 1024},
}};

std::invalid_argument invalid_value(std::string const& name,
                                    std::string_view what,
                                    std::string const& text)
{
  return std::invalid_argument("--" + name + ": invalid " + std::string{what} + " '" + text + "'");
}

std::size_t parse_count(std::string const& name, std::string const& text)
{
  std::uint64_t value{};
  auto const* const first = text.data();
  auto const* const last  = text.data() + text.size();
  auto const [end, error] = std::from_chars(first, last, value);
  if (error != std::errc{} || end != last) { throw invalid_value(name, "count", text); }
  return value;
}

std::size_t parse_bytes(std::string const& name, std::string const& text)
{
  std::uint64_t number{};
  auto const* const first        = text.data();
  auto const* const last         = text.data() + text.size();
  auto const [digits_end, error] = std::from_chars(first, last, number);
  if (error != std::errc{} || digits_end == first) {
    throw invalid_value(name, "byte count", text);
  }

  std::string suffix;
  for (auto const* p = digits_end; p != last; ++p) {
    suffix.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
  }

  for (auto const& unit : byte_units) {
    if (unit.suffix != suffix) { continue; }
    if (number > std::numeric_limits<std::uint64_t>::max() / unit.multiplier) {
      throw invalid_value(name, "byte count", text);
    }
    return number * unit.multiplier;
  }
  throw invalid_value(name, "byte count", text);
}

double parse_seconds(std::string const& name, std::string const& text)
{
  double value{};
  auto const* const first = text.data();
  auto const* const last  = text.data() + text.size();
  auto const [end, error] = std::from_chars(first, last, value);
  if (error != std::errc{} || end != last || !std::isfinite(value) || value < 0) {
    throw invalid_value(name, "number of seconds", text);
  }
  return value;
}

}  // namespace

Options::Options(std::string usage, std::vector<Flag> flags)
  : _usage{std::move(usage)}, _flags{std::move(flags)}
{
  _flags.push_back(Flag{.name = "help", .help = "Print this help and exit"});
}

void Options::parse(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i) {
    std::string_view const arg{argv[i]};
    if (arg == "-h") {
      _values["help"] = "";
      continue;
    }
    if (!arg.starts_with("--")) {
      throw std::invalid_argument("unexpected argument '" + std::string{arg} + "'");
    }

    // Split "--name=value". Without "=", the value, if the flag takes one, is the next argument.
    std::string_view name = arg.substr(2);
    std::optional<std::string> value;
    auto const equals = name.find('=');
    if (equals != std::string_view::npos) {
      value = std::string{name.substr(equals + 1)};
      name  = name.substr(0, equals);
    }

    Flag const* const flag = find_flag(name);
    if (flag == nullptr) { throw std::invalid_argument("unknown flag --" + std::string{name}); }
    if (_values.contains(flag->name)) {
      throw std::invalid_argument("--" + flag->name + " is given more than once");
    }

    if (flag->value_name.empty()) {
      if (value.has_value()) {
        throw std::invalid_argument("--" + flag->name + " does not take a value");
      }
      _values[flag->name] = "";
      continue;
    }

    if (!value.has_value()) {
      if (i + 1 >= argc) { throw std::invalid_argument("--" + flag->name + " needs a value"); }
      ++i;
      value = argv[i];
    }
    _values[flag->name] = value.value();
  }

  // Asking for help is not an error, whatever else is missing.
  if (is_set("help")) { return; }
  for (auto const& flag : _flags) {
    if (flag.required && !_values.contains(flag.name)) {
      throw std::invalid_argument("--" + flag.name + " is required");
    }
  }
}

std::string Options::help() const
{
  std::ostringstream os;
  os << _usage << "\n\nflags:\n";
  for (auto const& flag : _flags) {
    std::string left = "--" + flag.name;
    if (!flag.value_name.empty()) { left += " " + flag.value_name; }
    os << "  " << left;
    // Align the help in a column, or start it on the next line when the flag is too long.
    std::size_t constexpr column = 34;
    if (left.size() + 2 < column) {
      os << std::string(column - 2 - left.size(), ' ');
    } else {
      os << "\n" << std::string(column, ' ');
    }
    os << flag.help;
    if (flag.required) { os << " (required)"; }
    if (flag.default_value.has_value()) {
      os << " (default: " << flag.default_value.value() << ")";
    }
    os << "\n";
  }
  return os.str();
}

bool Options::is_set(std::string const& name) const
{
  if (find_flag(name) == nullptr) { throw std::logic_error("undeclared flag --" + name); }
  return _values.contains(name);
}

std::optional<std::string> Options::get_string(std::string const& name) const
{
  Flag const* const flag = find_flag(name);
  if (flag == nullptr) { throw std::logic_error("undeclared flag --" + name); }
  auto const it = _values.find(name);
  if (it != _values.end()) { return it->second; }
  return flag->default_value;
}

std::optional<std::size_t> Options::get_count(std::string const& name) const
{
  auto const text = get_string(name);
  if (!text.has_value()) { return std::nullopt; }
  return parse_count(name, text.value());
}

std::optional<std::size_t> Options::get_bytes(std::string const& name) const
{
  auto const text = get_string(name);
  if (!text.has_value()) { return std::nullopt; }
  return parse_bytes(name, text.value());
}

std::optional<double> Options::get_seconds(std::string const& name) const
{
  auto const text = get_string(name);
  if (!text.has_value()) { return std::nullopt; }
  return parse_seconds(name, text.value());
}

Flag const* Options::find_flag(std::string_view name) const
{
  for (auto const& flag : _flags) {
    if (flag.name == name) { return &flag; }
  }
  return nullptr;
}

}  // namespace kvikio::benchmark
