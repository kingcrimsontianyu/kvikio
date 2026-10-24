/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <kvikio/observation.hpp>

namespace kvikio::benchmark {

/**
 * @brief CPU time of the whole machine, summed over all cores, from `/proc/stat`.
 *
 * Unlike the process's own CPU time, this includes the softirq time the network stack spends
 * receiving on the process's behalf.
 */
struct MachineCpuTime {
  /// Time in user mode, including niced processes.
  double user_seconds{};
  double system_seconds{};
  double irq_seconds{};
  double softirq_seconds{};
};

/**
 * @brief A reading of CPU time and network traffic.
 *
 * Only differences between two readings mean anything.
 */
struct SystemUsage {
  /// When the reading was taken.
  TimePoint time{};
  /// CPU time of this process.
  double process_user_seconds{};
  double process_system_seconds{};
  /// Empty if `/proc/stat` cannot be read.
  std::optional<MachineCpuTime> machine;
  /// Bytes the network interface has received. Empty if it cannot be read.
  std::optional<std::uint64_t> nic_received_bytes;
};

/**
 * @brief Cores the whole machine kept busy between two readings, on average.
 */
struct MachineCores {
  double user{};
  double system{};
  double softirq{};
  /// User, system, irq and softirq together.
  double busy{};
};

/**
 * @brief Cores kept busy between two readings, on average.
 */
struct CpuCores {
  double process_user{};
  double process_system{};
  /// Empty unless both readings have the machine's CPU time.
  std::optional<MachineCores> machine;
};

/**
 * @brief Read CPU time and network traffic now.
 *
 * @param interface The network interface to count, or nothing to skip it.
 * @return The reading.
 */
[[nodiscard]] SystemUsage read_system_usage(std::optional<std::string> const& interface);

/**
 * @brief Cores kept busy between two readings.
 *
 * @param earlier The earlier reading.
 * @param later The later reading.
 * @return The cores, on average over the time between the readings.
 */
[[nodiscard]] CpuCores cpu_cores_between(SystemUsage const& earlier, SystemUsage const& later);

/**
 * @brief The network interface of the IPv4 default route, from `/proc/net/route`.
 *
 * @return The interface name, or nothing if there is no default route.
 */
[[nodiscard]] std::optional<std::string> default_route_interface();

}  // namespace kvikio::benchmark
