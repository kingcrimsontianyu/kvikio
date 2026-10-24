/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "system_usage.hpp"

#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <sstream>

namespace kvikio::benchmark {
namespace {

double to_seconds(timeval const& time)
{
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec) / 1e6;
}

std::optional<MachineCpuTime> read_machine_cpu_time()
{
  std::ifstream file{"/proc/stat"};
  // The first line sums all cores: "cpu user nice system idle iowait irq softirq ...", in ticks.
  std::string label;
  std::uint64_t user{};
  std::uint64_t nice{};
  std::uint64_t system{};
  std::uint64_t idle{};
  std::uint64_t iowait{};
  std::uint64_t irq{};
  std::uint64_t softirq{};
  file >> label >> user >> nice >> system >> idle >> iowait >> irq >> softirq;
  if (!file || label != "cpu") { return std::nullopt; }

  auto const ticks_per_second = static_cast<double>(sysconf(_SC_CLK_TCK));
  MachineCpuTime result;
  result.user_seconds    = static_cast<double>(user + nice) / ticks_per_second;
  result.system_seconds  = static_cast<double>(system) / ticks_per_second;
  result.irq_seconds     = static_cast<double>(irq) / ticks_per_second;
  result.softirq_seconds = static_cast<double>(softirq) / ticks_per_second;
  return result;
}

std::optional<std::uint64_t> read_nic_received_bytes(std::string const& interface)
{
  std::ifstream file{"/proc/net/dev"};
  // After two header lines, one line per interface: "  name: rx_bytes rx_packets ...".
  std::string line;
  while (std::getline(file, line)) {
    auto const colon = line.find(':');
    if (colon == std::string::npos) { continue; }
    std::string name = line.substr(0, colon);
    name.erase(0, name.find_first_not_of(' '));
    if (name != interface) { continue; }
    std::istringstream fields{line.substr(colon + 1)};
    std::uint64_t received_bytes{};
    fields >> received_bytes;
    if (!fields) { return std::nullopt; }
    return received_bytes;
  }
  return std::nullopt;
}

}  // namespace

SystemUsage read_system_usage(std::optional<std::string> const& interface)
{
  SystemUsage result;
  result.time = Clock::now();

  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  result.process_user_seconds   = to_seconds(usage.ru_utime);
  result.process_system_seconds = to_seconds(usage.ru_stime);

  result.machine = read_machine_cpu_time();
  if (interface.has_value()) { result.nic_received_bytes = read_nic_received_bytes(*interface); }
  return result;
}

CpuCores cpu_cores_between(SystemUsage const& earlier, SystemUsage const& later)
{
  double const seconds = std::chrono::duration<double>(later.time - earlier.time).count();
  CpuCores result;
  result.process_user   = (later.process_user_seconds - earlier.process_user_seconds) / seconds;
  result.process_system = (later.process_system_seconds - earlier.process_system_seconds) / seconds;

  if (earlier.machine.has_value() && later.machine.has_value()) {
    MachineCpuTime const& a = earlier.machine.value();
    MachineCpuTime const& b = later.machine.value();
    MachineCores machine;
    machine.user     = (b.user_seconds - a.user_seconds) / seconds;
    machine.system   = (b.system_seconds - a.system_seconds) / seconds;
    machine.softirq  = (b.softirq_seconds - a.softirq_seconds) / seconds;
    double const irq = (b.irq_seconds - a.irq_seconds) / seconds;
    machine.busy     = machine.user + machine.system + irq + machine.softirq;
    result.machine   = machine;
  }
  return result;
}

std::optional<std::string> default_route_interface()
{
  std::ifstream file{"/proc/net/route"};
  // After a header line, one route per line: "Iface Destination Gateway ...", hexadecimal.
  std::string line;
  std::getline(file, line);
  while (std::getline(file, line)) {
    std::istringstream fields{line};
    std::string interface;
    std::string destination;
    fields >> interface >> destination;
    if (fields && destination == "00000000") { return interface; }
  }
  return std::nullopt;
}

}  // namespace kvikio::benchmark
