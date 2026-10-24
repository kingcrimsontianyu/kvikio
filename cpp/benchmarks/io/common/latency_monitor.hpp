/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include <kvikio/observation.hpp>

namespace kvikio::benchmark {

/**
 * @brief Latency of the operations that finished in a span.
 *
 * Percentiles are nearest-rank: the smallest duration that at least that share of the operations
 * did not exceed. All zero when no operation finished.
 */
struct Latency {
  std::size_t count{};
  Duration mean{};
  Duration p50{};
  Duration p90{};
  Duration p99{};
  Duration max{};
};

/**
 * @brief Keeps the duration of every successful operation KvikIO reports, for exact percentiles.
 *
 * Every duration is stored, and sorted only when percentiles are asked for. A sketch such as a
 * t-digest bounds memory at the cost of exactness, and an I/O benchmark does not produce enough
 * operations, at 16 bytes each, to need one.
 *
 * Like any `kvikio::Monitor`, it sees the I/O of the whole process.
 */
class LatencyMonitor final : private Monitor {
 public:
  /**
   * @brief Register and begin recording.
   *
   * @param kind `LOGICAL` for one duration per user-facing call, `PHYSICAL` for one per transfer.
   */
  explicit LatencyMonitor(ObservationKind kind);

  /**
   * @brief Unregister, which waits for callbacks in progress.
   */
  ~LatencyMonitor() override;

  // Not copyable or movable: the registry holds this object's address.
  LatencyMonitor(LatencyMonitor const&)            = delete;
  LatencyMonitor& operator=(LatencyMonitor const&) = delete;
  LatencyMonitor(LatencyMonitor&&)                 = delete;
  LatencyMonitor& operator=(LatencyMonitor&&)      = delete;

  /**
   * @brief When the first successful operation finished.
   *
   * @return The time, or nothing if none has finished yet.
   */
  [[nodiscard]] std::optional<TimePoint> first_finish() const;

  /**
   * @brief Latency of the successful operations that finished in `[start, end)`.
   *
   * @param start Start of the span.
   * @param end End of the span.
   * @return The latency.
   */
  [[nodiscard]] Latency latency(TimePoint start, TimePoint end) const;

 private:
  void on_start(Observation const& observation) noexcept override;
  void on_finish(Observation const& observation) noexcept override;

  struct Sample {
    TimePoint end;
    Duration duration;
  };

  std::uint64_t _registration_id{};
  mutable std::mutex _mutex;
  std::vector<Sample> _samples;
  std::optional<TimePoint> _first_finish;
};

}  // namespace kvikio::benchmark
