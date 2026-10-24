/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "latency_monitor.hpp"

#include <algorithm>

namespace kvikio::benchmark {
namespace {

/**
 * @brief The nearest-rank percentile of sorted durations.
 *
 * @param sorted Durations in ascending order, at least one.
 * @param percent The percentile, from 1 to 100.
 * @return The duration at rank ceil(percent / 100 * n), counting ranks from 1.
 */
Duration nearest_rank(std::vector<Duration> const& sorted, std::size_t percent)
{
  std::size_t const rank = (percent * sorted.size() + 99) / 100;
  return sorted[rank - 1];
}

}  // namespace

LatencyMonitor::LatencyMonitor(ObservationKind kind)
{
  _registration_id = register_monitor(this, kind);
}

LatencyMonitor::~LatencyMonitor() { unregister_monitor(_registration_id); }

std::optional<TimePoint> LatencyMonitor::first_finish() const
{
  std::lock_guard const lock{_mutex};
  return _first_finish;
}

Latency LatencyMonitor::latency(TimePoint start, TimePoint end) const
{
  std::vector<Duration> durations;
  {
    std::lock_guard const lock{_mutex};
    for (auto const& sample : _samples) {
      if (sample.end >= start && sample.end < end) { durations.push_back(sample.duration); }
    }
  }

  Latency result;
  if (durations.empty()) { return result; }

  std::sort(durations.begin(), durations.end());
  Duration total{};
  for (auto const duration : durations) {
    total += duration;
  }
  auto const count = static_cast<Duration::rep>(durations.size());
  result.count     = durations.size();
  result.mean      = total / count;
  result.p50       = nearest_rank(durations, 50);
  result.p90       = nearest_rank(durations, 90);
  result.p99       = nearest_rank(durations, 99);
  result.max       = durations.back();
  return result;
}

void LatencyMonitor::on_start(Observation const&) noexcept {}

void LatencyMonitor::on_finish(Observation const& observation) noexcept
{
  if (!observation.ok) { return; }
  std::lock_guard const lock{_mutex};
  // push_back() fails only when memory runs out, which terminates the benchmark, since this
  // callback is noexcept.
  _samples.push_back(Sample{.end = observation.end, .duration = observation.duration()});
  if (!_first_finish.has_value() || observation.end < _first_finish.value()) {
    _first_finish = observation.end;
  }
}

}  // namespace kvikio::benchmark
