/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "json.hpp"

#include <cmath>
#include <format>

namespace kvikio::benchmark {
namespace {

std::string quote(std::string_view text)
{
  std::string out = "\"";
  for (char const c : text) {
    if (c == '"') {
      out += "\\\"";
    } else if (c == '\\') {
      out += "\\\\";
    } else if (static_cast<unsigned char>(c) < 0x20) {
      out += std::format("\\u{:04x}", static_cast<unsigned char>(c));
    } else {
      out.push_back(c);
    }
  }
  out += "\"";
  return out;
}

std::string number(double value)
{
  if (!std::isfinite(value)) { return "null"; }
  // The shortest text that reads back as the same double.
  return std::format("{}", value);
}

}  // namespace

void JsonObject::add_string(std::string_view key, std::string_view value)
{
  add_key(key);
  _members += quote(value);
}

void JsonObject::add_integer(std::string_view key, std::uint64_t value)
{
  add_key(key);
  _members += std::to_string(value);
}

void JsonObject::add_number(std::string_view key, double value)
{
  add_key(key);
  _members += number(value);
}

void JsonObject::add_numbers(std::string_view key, std::vector<double> const& values)
{
  add_key(key);
  _members += "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) { _members += ", "; }
    _members += number(values[i]);
  }
  _members += "]";
}

void JsonObject::add_null(std::string_view key)
{
  add_key(key);
  _members += "null";
}

void JsonObject::add_object(std::string_view key, JsonObject const& value)
{
  add_key(key);
  _members += value.str();
}

void JsonObject::add_json(std::string_view key, std::string_view json)
{
  add_key(key);
  _members += json;
}

std::string JsonObject::str() const { return "{" + _members + "}"; }

void JsonObject::add_key(std::string_view key)
{
  if (!_members.empty()) { _members += ", "; }
  _members += quote(key);
  _members += ": ";
}

}  // namespace kvikio::benchmark
