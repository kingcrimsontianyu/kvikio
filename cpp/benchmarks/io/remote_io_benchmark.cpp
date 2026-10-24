/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Read bandwidth and latency of KvikIO remote I/O: S3, HTTP and WebHDFS.
//
// Threads keep --reads-in-flight preads going, one each. Every pread reads a slice of --read-size
// bytes from one of the objects in --url-file, and KvikIO splits it into range requests of
// --task-size. Connection setup can take seconds over TLS, so the window opens --settle-seconds
// after the first range request completes rather than at a fixed time. Bandwidth is read once
// every --interval-seconds inside the window.
//
// The same loop measures bandwidth and latency, with different parameters:
//   bandwidth: --read-size 1GiB --task-size 64MiB --reads-in-flight 32
//   latency:   --read-size 64KiB --reads-in-flight 1 --window-seconds 300
//
// With --destination device, --devices spreads the buffers over several GPUs of this one process,
// buffer i on the i-th listed device in turn, so the reads land on all of them at once. The threads
// never make a device current: KvikIO finds each buffer's GPU from its address. --verify then reads
// one slice into each GPU, with another GPU current, and compares it with a host read.

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <kvikio/defaults.hpp>
#include <kvikio/observation.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/statistics/counters.hpp>
#include <kvikio/statistics/summary.hpp>

#include "common/buffer.hpp"
#include "common/json.hpp"
#include "common/latency_monitor.hpp"
#include "common/options.hpp"
#include "common/system_usage.hpp"

namespace kvikio::benchmark {
namespace {

constexpr std::size_t KiB      = 1024;
constexpr std::size_t MiB      = 1024 * KiB;
constexpr std::size_t GiB      = 1024 * MiB;
constexpr double bytes_per_gib = static_cast<double>(GiB);

// -------------------------------------------------------------------------------------------------
// Command line
// -------------------------------------------------------------------------------------------------

constexpr char const* usage =
  "Read bandwidth and latency of KvikIO remote I/O.\n"
  "\n"
  "usage: REMOTE_IO_BENCHMARK --url-file PATH --reads-in-flight N [flag]...\n"
  "\n"
  "A flag named after a KvikIO setting overrides its KVIKIO_* environment variable. Every other\n"
  "KvikIO setting is taken from the environment.";

std::vector<Flag> make_flags()
{
  return {
    {.name       = "url-file",
     .value_name = "PATH",
     .help       = "Object URLs, one per line. Blank lines and # comments are skipped",
     .required   = true},
    {.name          = "endpoint-type",
     .value_name    = "TYPE",
     .help          = "auto, s3, s3_public, s3_presigned_url, webhdfs or http",
     .default_value = "auto"},
    {.name          = "read-size",
     .value_name    = "BYTES",
     .help          = "Bytes per pread. Objects smaller than this are skipped",
     .default_value = "256MiB"},
    {.name       = "reads-in-flight",
     .value_name = "N",
     .help       = "Preads kept in flight, one thread each",
     .required   = true},
    {.name       = "max-buffers",
     .value_name = "N",
     .help       = "Destination buffers, shared by the reads if fewer than --reads-in-flight "
                   "(default: --reads-in-flight)"},
    {.name          = "destination",
     .value_name    = "MEMORY",
     .help          = "host, or device for the GPUs in --devices",
     .default_value = "host"},
    {.name          = "devices",
     .value_name    = "LIST",
     .help          = "CUDA devices the device buffers are spread over in turn, such as 0,1",
     .default_value = "0"},
    {.name = "verify",
     .help = "After the run, check a slice read into each device against a host read"},
    {.name          = "settle-seconds",
     .value_name    = "S",
     .help          = "Time from the first completed range request to the window",
     .default_value = "10"},
    {.name          = "window-seconds",
     .value_name    = "S",
     .help          = "Length of the window, a whole number of intervals",
     .default_value = "30"},
    {.name          = "interval-seconds",
     .value_name    = "S",
     .help          = "Bandwidth is read once per interval",
     .default_value = "1"},
    {.name          = "startup-timeout-seconds",
     .value_name    = "S",
     .help          = "Give up if no range request completes within this time",
     .default_value = "180"},
    {.name       = "interface",
     .value_name = "NAME",
     .help       = "Network interface whose received bytes are counted (default: the default "
                   "route's)"},
    {.name       = "seed",
     .value_name = "N",
     .help       = "Seed of the slice order (default: random, and printed)"},
    {.name = "json", .help = "Also print the result on one line, starting with \"JSON \""},
    {.name       = "backend",
     .value_name = "NAME",
     .help       = "easy_threadpool or multi_poll (KVIKIO_REMOTE_IO_BACKEND)"},
    {.name       = "nthreads",
     .value_name = "N",
     .help       = "Thread pool size, the concurrency cap of EASY_THREADPOOL (KVIKIO_NTHREADS)"},
    {.name       = "num-reactors",
     .value_name = "N",
     .help       = "MULTI_POLL reactor threads (KVIKIO_REMOTE_IO_NUM_REACTORS)"},
    {.name       = "max-concurrent-requests",
     .value_name = "N",
     .help       = "Concurrency cap of MULTI_POLL, 0 for none "
                   "(KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS)"},
    {.name       = "task-size",
     .value_name = "BYTES",
     .help       = "Size of a range request (KVIKIO_TASK_SIZE)"},
    {.name       = "bounce-buffer-size",
     .value_name = "BYTES",
     .help       = "Staging buffer for device reads (KVIKIO_BOUNCE_BUFFER_SIZE)"},
  };
}

std::string to_lower(std::string_view text)
{
  std::string result;
  for (char const c : text) {
    result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return result;
}

struct EndpointTypeName {
  RemoteEndpointType type;
  std::string_view name;
};

constexpr std::array<EndpointTypeName, 6> endpoint_type_names{{
  {RemoteEndpointType::AUTO, "auto"},
  {RemoteEndpointType::S3, "s3"},
  {RemoteEndpointType::S3_PUBLIC, "s3_public"},
  {RemoteEndpointType::S3_PRESIGNED_URL, "s3_presigned_url"},
  {RemoteEndpointType::WEBHDFS, "webhdfs"},
  {RemoteEndpointType::HTTP, "http"},
}};

RemoteEndpointType parse_endpoint_type(std::string const& text)
{
  std::string const name = to_lower(text);
  for (auto const& entry : endpoint_type_names) {
    if (entry.name == name) { return entry.type; }
  }
  throw std::invalid_argument("--endpoint-type: unknown type '" + text + "'");
}

std::string_view endpoint_type_name(RemoteEndpointType type)
{
  for (auto const& entry : endpoint_type_names) {
    if (entry.type == type) { return entry.name; }
  }
  return "unknown";
}

RemoteIOBackend parse_backend(std::string const& text)
{
  std::string const name = to_lower(text);
  if (name == "easy_threadpool") { return RemoteIOBackend::EASY_THREADPOOL; }
  if (name == "multi_poll") { return RemoteIOBackend::MULTI_POLL; }
  throw std::invalid_argument("--backend: unknown backend '" + text + "'");
}

std::string_view backend_name(RemoteIOBackend backend)
{
  switch (backend) {
    case RemoteIOBackend::EASY_THREADPOOL: return "EASY_THREADPOOL";
    case RemoteIOBackend::MULTI_POLL: return "MULTI_POLL";
  }
  return "UNKNOWN";
}

MemoryKind parse_destination(std::string const& text)
{
  std::string const name = to_lower(text);
  if (name == "host") { return MemoryKind::HOST; }
  if (name == "device") { return MemoryKind::DEVICE; }
  throw std::invalid_argument("--destination: expected host or device, got '" + text + "'");
}

/**
 * @brief How the benchmark runs, from the command line.
 */
struct Config {
  std::string url_file;
  RemoteEndpointType endpoint_type{RemoteEndpointType::AUTO};
  std::size_t read_size{};
  std::size_t reads_in_flight{};
  std::size_t num_buffers{};
  MemoryKind destination{MemoryKind::HOST};
  /// Devices of the device buffers, empty for host buffers.
  std::vector<int> devices;
  bool verify{false};
  double settle_seconds{};
  double interval_seconds{};
  std::size_t num_intervals{};
  double startup_timeout_seconds{};
  std::optional<std::string> interface;
  std::uint64_t seed{};
  bool json{false};
};

std::vector<int> parse_devices(std::string const& text)
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    throw std::runtime_error("--devices: cudaGetDeviceCount failed");
  }
  std::vector<int> devices;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    std::size_t const end  = std::min(text.find(',', begin), text.size());
    std::string const item = text.substr(begin, end - begin);
    std::size_t parsed     = 0;
    int device             = -1;
    try {
      device = std::stoi(item, &parsed);
    } catch (std::exception const&) {
      parsed = 0;
    }
    if (parsed == 0 || parsed != item.size() || device < 0 || device >= count) {
      throw std::invalid_argument(
        std::format("--devices: '{}' is not one of the {} visible devices", item, count));
    }
    if (std::find(devices.begin(), devices.end(), device) != devices.end()) {
      throw std::invalid_argument(std::format("--devices: device {} is listed twice", device));
    }
    devices.push_back(device);
    begin = end + 1;
  }
  return devices;
}

std::string join_devices(std::vector<int> const& devices)
{
  std::string text;
  for (int const device : devices) {
    if (!text.empty()) { text += ","; }
    text += std::to_string(device);
  }
  return text;
}

Config make_config(Options const& options)
{
  Config config;
  config.url_file      = options.get_string("url-file").value();
  config.endpoint_type = parse_endpoint_type(options.get_string("endpoint-type").value());

  config.read_size = options.get_bytes("read-size").value();
  if (config.read_size == 0) { throw std::invalid_argument("--read-size must be positive"); }

  config.reads_in_flight = options.get_count("reads-in-flight").value();
  if (config.reads_in_flight == 0) {
    throw std::invalid_argument("--reads-in-flight must be positive");
  }

  config.num_buffers     = config.reads_in_flight;
  auto const max_buffers = options.get_count("max-buffers");
  if (max_buffers.has_value()) {
    if (max_buffers.value() == 0) { throw std::invalid_argument("--max-buffers must be positive"); }
    config.num_buffers = std::min(config.reads_in_flight, max_buffers.value());
  }

  config.destination = parse_destination(options.get_string("destination").value());
  if (config.destination == MemoryKind::DEVICE) {
    config.devices = parse_devices(options.get_string("devices").value());
    if (config.num_buffers < config.devices.size()) {
      throw std::invalid_argument("--max-buffers must be at least the number of --devices");
    }
  }
  config.verify         = options.is_set("verify");
  config.settle_seconds = options.get_seconds("settle-seconds").value();

  // Every interval has the same length, so the window must hold a whole number of them.
  double const window_seconds = options.get_seconds("window-seconds").value();
  config.interval_seconds     = options.get_seconds("interval-seconds").value();
  if (config.interval_seconds <= 0) {
    throw std::invalid_argument("--interval-seconds must be positive");
  }
  double const intervals = window_seconds / config.interval_seconds;
  config.num_intervals   = static_cast<std::size_t>(std::llround(intervals));
  if (config.num_intervals == 0 ||
      std::abs(intervals - static_cast<double>(config.num_intervals)) > 1e-9 * intervals) {
    throw std::invalid_argument(
      "--window-seconds must be a whole, positive number of --interval-seconds");
  }

  config.startup_timeout_seconds = options.get_seconds("startup-timeout-seconds").value();

  config.interface = options.get_string("interface");
  if (!config.interface.has_value()) { config.interface = default_route_interface(); }

  auto const seed = options.get_count("seed");
  if (seed.has_value()) {
    config.seed = seed.value();
  } else {
    config.seed = std::random_device{}();
  }

  config.json = options.is_set("json");
  return config;
}

// KvikIO's settings are taken as flags as well as KVIKIO_* variables because the variables' parser
// stops at the first character it cannot read, without an error: KVIKIO_TASK_SIZE=64MiB means 64
// bytes. The flags understand units. Either way, the settings in effect are printed.
void apply_kvikio_settings(Options const& options)
{
  auto const backend = options.get_string("backend");
  if (backend.has_value()) { defaults::set_remote_io_backend(parse_backend(backend.value())); }

  auto const nthreads = options.get_count("nthreads");
  if (nthreads.has_value()) {
    if (nthreads.value() > std::numeric_limits<unsigned int>::max()) {
      throw std::invalid_argument("--nthreads is too large");
    }
    defaults::set_thread_pool_nthreads(static_cast<unsigned int>(nthreads.value()));
  }

  auto const num_reactors = options.get_count("num-reactors");
  if (num_reactors.has_value()) {
    if (num_reactors.value() > std::numeric_limits<unsigned int>::max()) {
      throw std::invalid_argument("--num-reactors is too large");
    }
    defaults::set_remote_io_num_reactors(static_cast<unsigned int>(num_reactors.value()));
  }

  auto const max_concurrent_requests = options.get_count("max-concurrent-requests");
  if (max_concurrent_requests.has_value()) {
    defaults::set_remote_io_max_concurrent_requests(max_concurrent_requests.value());
  }

  auto const task_size = options.get_bytes("task-size");
  if (task_size.has_value()) { defaults::set_task_size(task_size.value()); }

  auto const bounce_buffer_size = options.get_bytes("bounce-buffer-size");
  if (bounce_buffer_size.has_value()) {
    defaults::set_bounce_buffer_size(bounce_buffer_size.value());
  }
}

// -------------------------------------------------------------------------------------------------
// Settings and workload
// -------------------------------------------------------------------------------------------------

/**
 * @brief The KvikIO settings in effect, read back after the flags are applied.
 */
struct KvikioSettings {
  RemoteIOBackend backend{};
  unsigned int nthreads{};
  unsigned int num_reactors{};
  std::size_t max_concurrent_requests{};
  std::size_t task_size{};
  std::size_t bounce_buffer_size{};
  std::size_t http_max_attempts{};
  long http_timeout_seconds{};
};

KvikioSettings read_kvikio_settings()
{
  KvikioSettings settings;
  settings.backend                 = defaults::remote_io_backend();
  settings.nthreads                = defaults::thread_pool_nthreads();
  settings.num_reactors            = defaults::remote_io_num_reactors();
  settings.max_concurrent_requests = defaults::remote_io_max_concurrent_requests();
  settings.task_size               = defaults::task_size();
  settings.bounce_buffer_size      = defaults::bounce_buffer_size();
  settings.http_max_attempts       = defaults::http_max_attempts();
  settings.http_timeout_seconds    = defaults::http_timeout();
  return settings;
}

/**
 * @brief The most range requests KvikIO keeps on the wire at once.
 *
 * @param settings The settings in effect.
 * @return The cap, or 0 for none.
 */
std::size_t concurrency_cap(KvikioSettings const& settings)
{
  if (settings.backend == RemoteIOBackend::EASY_THREADPOOL) { return settings.nthreads; }
  if (settings.max_concurrent_requests == 0) { return 0; }
  // Every reactor gets at least one request, whatever the budget.
  return std::max<std::size_t>(settings.max_concurrent_requests, settings.num_reactors);
}

/**
 * @brief The environment variables that change what a run measures. Never credentials.
 *
 * @return Name and value pairs, sorted by name.
 */
std::vector<std::pair<std::string, std::string>> relevant_environment()
{
  std::vector<std::pair<std::string, std::string>> result;
  for (char** entry = environ; *entry != nullptr; ++entry) {
    std::string_view const text{*entry};
    auto const equals = text.find('=');
    if (equals == std::string_view::npos) { continue; }
    std::string const name{text.substr(0, equals)};
    bool const relevant = name.starts_with("KVIKIO_") || name == "AWS_ENDPOINT_URL" ||
                          name == "AWS_DEFAULT_REGION" || name == "AWS_REGION" ||
                          name == "CUDA_VISIBLE_DEVICES";
    if (relevant) { result.emplace_back(name, std::string{text.substr(equals + 1)}); }
  }
  std::sort(result.begin(), result.end());
  return result;
}

std::vector<std::string> read_urls(std::string const& path)
{
  std::ifstream file{path};
  if (!file) { throw std::runtime_error("cannot open --url-file " + path); }
  std::vector<std::string> urls;
  std::string line;
  while (std::getline(file, line)) {
    auto const first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos) { continue; }
    auto const last       = line.find_last_not_of(" \t\r");
    std::string const url = line.substr(first, last - first + 1);
    if (url.starts_with("#")) { continue; }
    urls.push_back(url);
  }
  if (urls.empty()) { throw std::runtime_error("no URL in " + path); }
  return urls;
}

/**
 * @brief The part of an object that one pread reads.
 */
struct Slice {
  std::size_t object_index{};
  std::size_t offset{};
};

/**
 * @brief Cut every object into slices of `read_size`, in shuffled order.
 *
 * The tail of an object shorter than `read_size` is left out, so every read is the same size. The
 * order is shuffled so that the reads in flight spread over the objects, and so that runs with
 * different seeds do not all read the same objects first.
 *
 * @param objects The open objects.
 * @param read_size Bytes per read.
 * @param seed Seed of the order.
 * @return The slices.
 */
std::vector<Slice> make_slices(std::vector<RemoteHandle> const& objects,
                               std::size_t read_size,
                               std::uint64_t seed)
{
  std::vector<Slice> slices;
  for (std::size_t i = 0; i < objects.size(); ++i) {
    std::size_t const nbytes = objects[i].nbytes();
    for (std::size_t offset = 0; offset + read_size <= nbytes; offset += read_size) {
      slices.push_back(Slice{.object_index = i, .offset = offset});
    }
  }
  std::mt19937_64 generator{seed};
  std::shuffle(slices.begin(), slices.end(), generator);
  return slices;
}

// -------------------------------------------------------------------------------------------------
// Running
// -------------------------------------------------------------------------------------------------

/**
 * @brief The threads that keep the reads in flight, each running one pread at a time.
 *
 * The threads take slices in order from a shared counter, and start over once every slice has been
 * read. The first failed read stops every thread and is kept. Buffer `i` belongs to destination `i`
 * modulo `num_destinations`, and the bytes read into each destination are counted.
 */
class SubmitterThreads {
 public:
  SubmitterThreads(std::vector<RemoteHandle>& objects,
                   std::vector<Slice> const& slices,
                   std::vector<std::unique_ptr<Buffer>> const& buffers,
                   std::size_t read_size,
                   std::size_t num_destinations)
    : _objects{objects},
      _slices{slices},
      _buffers{buffers},
      _read_size{read_size},
      _bytes_read(num_destinations)
  {
  }

  ~SubmitterThreads() { stop(); }

  SubmitterThreads(SubmitterThreads const&)            = delete;
  SubmitterThreads& operator=(SubmitterThreads const&) = delete;
  SubmitterThreads(SubmitterThreads&&)                 = delete;
  SubmitterThreads& operator=(SubmitterThreads&&)      = delete;

  /**
   * @brief Start the threads. Thread `i` reads into buffer `i` modulo the number of buffers.
   *
   * @param count Number of threads.
   */
  void start(std::size_t count)
  {
    for (std::size_t i = 0; i < count; ++i) {
      std::size_t const buffer_index = i % _buffers.size();
      void* const buffer             = _buffers[buffer_index]->data();
      std::size_t const destination  = buffer_index % _bytes_read.size();
      _threads.emplace_back([this, buffer, destination] { run(buffer, destination); });
    }
  }

  /**
   * @brief Stop every thread after its current read, and wait for them.
   */
  void stop()
  {
    _stop.store(true);
    for (auto& thread : _threads) {
      if (thread.joinable()) { thread.join(); }
    }
  }

  [[nodiscard]] bool failed() const
  {
    std::lock_guard const lock{_failure_mutex};
    return _failure != nullptr;
  }

  /**
   * @brief Throw the first failed read's exception, if a read failed.
   */
  void rethrow_failure() const
  {
    std::lock_guard const lock{_failure_mutex};
    if (_failure != nullptr) { std::rethrow_exception(_failure); }
  }

  /**
   * @brief Reads started so far, counting any in flight.
   *
   * @return The count.
   */
  [[nodiscard]] std::size_t reads_started() const { return _next_slice.load(); }

  /**
   * @brief Bytes of finished reads so far, per destination.
   *
   * @return One count per destination.
   */
  [[nodiscard]] std::vector<std::size_t> bytes_read() const
  {
    std::vector<std::size_t> result;
    for (auto const& count : _bytes_read) {
      result.push_back(count.load());
    }
    return result;
  }

 private:
  void run(void* buffer, std::size_t destination)
  {
    try {
      while (!_stop.load()) {
        std::size_t const index = _next_slice.fetch_add(1);
        Slice const& slice      = _slices[index % _slices.size()];
        _bytes_read[destination] +=
          _objects[slice.object_index].pread(buffer, _read_size, slice.offset).get();
      }
    } catch (...) {
      std::lock_guard const lock{_failure_mutex};
      if (_failure == nullptr) { _failure = std::current_exception(); }
      _stop.store(true);
    }
  }

  std::vector<RemoteHandle>& _objects;
  std::vector<Slice> const& _slices;
  std::vector<std::unique_ptr<Buffer>> const& _buffers;
  std::size_t _read_size;
  std::atomic<bool> _stop{false};
  std::atomic<std::size_t> _next_slice{0};
  std::vector<std::atomic<std::size_t>> _bytes_read;
  mutable std::mutex _failure_mutex;
  std::exception_ptr _failure;
  std::vector<std::thread> _threads;
};

/**
 * @brief What the window is measured with, read at the start and at the end of every interval.
 */
struct Reading {
  statistics::Summary transfers;
  SystemUsage usage;
  /// Bytes of finished reads into each destination.
  std::vector<std::size_t> destination_bytes;
};

Duration to_duration(double seconds)
{
  return std::chrono::duration_cast<Duration>(std::chrono::duration<double>(seconds));
}

double to_seconds(Duration duration) { return std::chrono::duration<double>(duration).count(); }

/**
 * @brief Payload bandwidth of each interval of the window.
 */
struct IntervalBandwidth {
  /// One per interval, in order.
  std::vector<double> gib_per_s;
  double min{};
  double median{};
  double max{};
  /// Coefficient of variation: the standard deviation over the mean.
  double cv{};
  /// Intervals that ran at less than half the median, such as a stall while connections reopen.
  std::size_t below_half_median{};
};

IntervalBandwidth interval_bandwidth(std::vector<Reading> const& readings)
{
  IntervalBandwidth result;
  for (std::size_t i = 1; i < readings.size(); ++i) {
    statistics::Summary const interval = readings[i].transfers.since(readings[i - 1].transfers);
    double const bytes                 = static_cast<double>(interval.bytes_read);
    result.gib_per_s.push_back(bytes / bytes_per_gib / to_seconds(interval.wall()));
  }

  std::vector<double> sorted = result.gib_per_s;
  std::sort(sorted.begin(), sorted.end());
  std::size_t const n = sorted.size();
  result.min          = sorted.front();
  result.max          = sorted.back();
  if (n % 2 == 1) {
    result.median = sorted[n / 2];
  } else {
    result.median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  }

  double sum = 0;
  for (double const rate : sorted) {
    sum += rate;
  }
  double const mean = sum / static_cast<double>(n);
  double squares    = 0;
  for (double const rate : sorted) {
    squares += (rate - mean) * (rate - mean);
  }
  if (mean > 0) { result.cv = std::sqrt(squares / static_cast<double>(n)) / mean; }

  for (double const rate : sorted) {
    if (rate < 0.5 * result.median) { ++result.below_half_median; }
  }
  return result;
}

/**
 * @brief Everything a run reports.
 */
struct Result {
  KvikioSettings settings;
  std::vector<std::pair<std::string, std::string>> environment;

  std::size_t num_objects{};
  std::size_t num_objects_skipped{};
  /// Number of objects of each endpoint type, by type name.
  std::map<std::string, std::size_t> endpoint_types;
  double open_seconds{};
  statistics::Counters open_counters;
  std::size_t num_slices{};
  std::size_t requests_per_read{};

  double time_to_first_completion_seconds{};
  double window_seconds{};
  std::size_t reads_started{};

  double payload_gib_per_s{};
  /// Finished reads per device over the window, in the order of --devices.
  std::vector<double> device_gib_per_s;
  IntervalBandwidth intervals;
  std::optional<double> nic_gib_per_s;
  double requests_in_flight{};
  /// Reads that finished in the window, per second.
  double reads_per_second{};

  Latency read_latency;
  Latency request_latency;
  CpuCores cpu;

  /// From the start of the reads to the window.
  statistics::Counters startup_counters;
  /// Range requests in the window, with the counters of the window.
  statistics::Summary window;
};

// -------------------------------------------------------------------------------------------------
// Report
// -------------------------------------------------------------------------------------------------

std::string format_bytes(std::size_t nbytes)
{
  auto const value = static_cast<double>(nbytes);
  if (nbytes >= GiB) { return std::format("{:.2f} GiB", value / static_cast<double>(GiB)); }
  if (nbytes >= MiB) { return std::format("{:.2f} MiB", value / static_cast<double>(MiB)); }
  if (nbytes >= KiB) { return std::format("{:.2f} KiB", value / static_cast<double>(KiB)); }
  return std::format("{} B", nbytes);
}

std::string format_rate(double gib_per_s)
{
  double const gbps = gib_per_s * bytes_per_gib * 8 / 1e9;
  if (gib_per_s < 1) { return std::format("{:.2f} MiB/s  {:.3f} Gbps", gib_per_s * 1024, gbps); }
  return std::format("{:.2f} GiB/s  {:.1f} Gbps", gib_per_s, gbps);
}

std::string format_milliseconds(Duration duration)
{
  return std::format("{:.2f}", to_seconds(duration) * 1e3);
}

std::string format_counters(statistics::Counters const& counters)
{
  return std::format("{} opened ({:.3f} s dns, {:.3f} s tcp, {:.3f} s tls, summed), {} retries",
                     counters.http_connections,
                     to_seconds(counters.http_dns),
                     to_seconds(counters.http_tcp),
                     to_seconds(counters.http_tls),
                     counters.http_retries);
}

void print_row(std::string_view label, std::string const& value)
{
  std::cout << std::format("    {:<26}{}\n", label, value);
}

void print_settings(Config const& config, Result const& result)
{
  KvikioSettings const& settings = result.settings;
  std::cout << "--> settings\n";
  print_row("backend", std::string{backend_name(settings.backend)});
  if (settings.backend == RemoteIOBackend::EASY_THREADPOOL) {
    print_row("threads", std::to_string(settings.nthreads));
  } else {
    print_row("reactors", std::to_string(settings.num_reactors));
    print_row("max concurrent requests", std::to_string(settings.max_concurrent_requests));
  }
  print_row("task size", format_bytes(settings.task_size));
  if (config.destination == MemoryKind::DEVICE) {
    print_row("bounce buffer size", format_bytes(settings.bounce_buffer_size));
  }
  print_row("http max attempts", std::to_string(settings.http_max_attempts));
  print_row("http timeout", std::format("{} s", settings.http_timeout_seconds));
  for (auto const& [name, value] : result.environment) {
    print_row("environment", name + "=" + value);
  }

  std::cout << "--> workload\n";
  std::string types;
  for (auto const& [name, count] : result.endpoint_types) {
    if (!types.empty()) { types += ", "; }
    types += std::format("{} {}", count, name);
  }
  print_row("objects",
            std::format("{} ({}), {} skipped as smaller than a read",
                        result.num_objects,
                        types,
                        result.num_objects_skipped));
  print_row(
    "opening",
    std::format(
      "{:.2f} s, {} size probes", result.open_seconds, result.open_counters.remote_size_probes));
  print_row("slices",
            std::format(
              "{} of {}, seed {}", result.num_slices, format_bytes(config.read_size), config.seed));
  std::string destination{to_string(config.destination)};
  if (config.destination == MemoryKind::DEVICE) {
    destination += ", devices " + join_devices(config.devices);
  }
  print_row("destination", destination);
  print_row("reads in flight", std::to_string(config.reads_in_flight));
  print_row("range requests per read", std::to_string(result.requests_per_read));
  std::size_t const cap = concurrency_cap(settings);
  std::string cap_text  = "no cap";
  if (cap > 0) { cap_text = std::format("cap {}", cap); }
  print_row("range requests queued",
            std::format("{}, {}", config.reads_in_flight * result.requests_per_read, cap_text));
  print_row("buffers",
            std::format("{} of {}, {} in all",
                        config.num_buffers,
                        format_bytes(config.read_size),
                        format_bytes(config.num_buffers * config.read_size)));
  std::cout << std::flush;
}

void print_latency_row(std::string_view label, Latency const& latency)
{
  if (latency.count == 0) {
    print_row(label, "none finished in the window");
    return;
  }
  print_row(label,
            std::format("{:<10}{:<10}{:<10}{:<10}{:<10}{}",
                        latency.count,
                        format_milliseconds(latency.mean),
                        format_milliseconds(latency.p50),
                        format_milliseconds(latency.p90),
                        format_milliseconds(latency.p99),
                        format_milliseconds(latency.max)));
}

void print_result(Config const& config, Result const& result)
{
  std::cout << "--> bandwidth\n";
  print_row("payload", format_rate(result.payload_gib_per_s));
  for (std::size_t i = 0; i < result.device_gib_per_s.size(); ++i) {
    print_row(std::format("device {} (finished reads)", config.devices[i]),
              format_rate(result.device_gib_per_s[i]));
  }
  IntervalBandwidth const& intervals = result.intervals;
  // In the unit the median is printed in.
  double scale          = 1;
  std::string_view unit = "GiB/s";
  if (intervals.median < 1) {
    scale = 1024;
    unit  = "MiB/s";
  }
  print_row("per interval",
            std::format("{:.2f} min, {:.2f} median, {:.2f} max {}, cv {:.1f}%",
                        intervals.min * scale,
                        intervals.median * scale,
                        intervals.max * scale,
                        unit,
                        intervals.cv * 100));
  print_row("intervals below half",
            std::format("{} of {}", intervals.below_half_median, intervals.gib_per_s.size()));
  if (result.nic_gib_per_s.has_value()) {
    print_row("nic " + config.interface.value(), format_rate(result.nic_gib_per_s.value()));
  } else {
    print_row("nic", "unavailable");
  }
  std::size_t const cap = concurrency_cap(result.settings);
  std::string cap_text  = "no cap";
  if (cap > 0) { cap_text = std::format("cap {}", cap); }
  print_row("range requests in flight",
            std::format("{:.1f} on average, {}", result.requests_in_flight, cap_text));
  print_row("reads", std::format("{:.1f} per second", result.reads_per_second));

  std::cout << std::format("{:<30}{:<10}{:<10}{:<10}{:<10}{:<10}{}\n",
                           "--> latency (ms)",
                           "count",
                           "mean",
                           "p50",
                           "p90",
                           "p99",
                           "max");
  print_latency_row("read", result.read_latency);
  print_latency_row("range request", result.request_latency);

  std::cout << "--> connections\n";
  print_row("startup", format_counters(result.startup_counters));
  print_row("window", format_counters(result.window.counters));

  std::cout << "--> cpu\n";
  CpuCores const& cpu = result.cpu;
  print_row("process",
            std::format("{:.2f} user + {:.2f} system cores", cpu.process_user, cpu.process_system));
  if (cpu.machine.has_value()) {
    MachineCores const& machine = cpu.machine.value();
    print_row("machine",
              std::format("{:.2f} user + {:.2f} system + {:.2f} softirq, {:.2f} busy of {} cores",
                          machine.user,
                          machine.system,
                          machine.softirq,
                          machine.busy,
                          std::thread::hardware_concurrency()));
    if (result.payload_gib_per_s > 0) {
      print_row(
        "cost",
        std::format("{:.2f} busy cores per GiB/s", machine.busy / result.payload_gib_per_s));
    }
  }
}

JsonObject latency_json(Latency const& latency)
{
  JsonObject json;
  json.add_integer("count", latency.count);
  json.add_integer("mean_ns", static_cast<std::uint64_t>(latency.mean.count()));
  json.add_integer("p50_ns", static_cast<std::uint64_t>(latency.p50.count()));
  json.add_integer("p90_ns", static_cast<std::uint64_t>(latency.p90.count()));
  json.add_integer("p99_ns", static_cast<std::uint64_t>(latency.p99.count()));
  json.add_integer("max_ns", static_cast<std::uint64_t>(latency.max.count()));
  return json;
}

JsonObject result_json(Config const& config, Result const& result)
{
  KvikioSettings const& settings = result.settings;
  JsonObject environment;
  for (auto const& [name, value] : result.environment) {
    environment.add_string(name, value);
  }
  JsonObject settings_json;
  settings_json.add_string("backend", backend_name(settings.backend));
  settings_json.add_integer("nthreads", settings.nthreads);
  settings_json.add_integer("num_reactors", settings.num_reactors);
  settings_json.add_integer("max_concurrent_requests", settings.max_concurrent_requests);
  settings_json.add_integer("task_size", settings.task_size);
  settings_json.add_integer("bounce_buffer_size", settings.bounce_buffer_size);
  settings_json.add_integer("http_max_attempts", settings.http_max_attempts);
  settings_json.add_integer("http_timeout_seconds",
                            static_cast<std::uint64_t>(settings.http_timeout_seconds));
  settings_json.add_object("environment", environment);

  JsonObject endpoint_types;
  for (auto const& [name, count] : result.endpoint_types) {
    endpoint_types.add_integer(name, count);
  }
  JsonObject workload;
  workload.add_integer("objects", result.num_objects);
  workload.add_integer("objects_skipped", result.num_objects_skipped);
  workload.add_object("endpoint_types", endpoint_types);
  workload.add_number("open_seconds", result.open_seconds);
  workload.add_json("open_counters", result.open_counters.to_json());
  workload.add_integer("slices", result.num_slices);
  workload.add_integer("read_size", config.read_size);
  workload.add_integer("seed", config.seed);
  workload.add_string("destination", to_string(config.destination));
  workload.add_string("devices", join_devices(config.devices));
  workload.add_integer("reads_in_flight", config.reads_in_flight);
  workload.add_integer("requests_per_read", result.requests_per_read);
  workload.add_integer("concurrency_cap", concurrency_cap(settings));
  workload.add_integer("buffers", config.num_buffers);

  JsonObject intervals;
  intervals.add_numbers("gib_per_s", result.intervals.gib_per_s);
  intervals.add_number("min", result.intervals.min);
  intervals.add_number("median", result.intervals.median);
  intervals.add_number("max", result.intervals.max);
  intervals.add_number("cv", result.intervals.cv);
  intervals.add_integer("below_half_median", result.intervals.below_half_median);

  JsonObject cpu;
  cpu.add_number("process_user", result.cpu.process_user);
  cpu.add_number("process_system", result.cpu.process_system);
  if (result.cpu.machine.has_value()) {
    MachineCores const& machine = result.cpu.machine.value();
    cpu.add_number("machine_user", machine.user);
    cpu.add_number("machine_system", machine.system);
    cpu.add_number("machine_softirq", machine.softirq);
    cpu.add_number("machine_busy", machine.busy);
  } else {
    cpu.add_null("machine_user");
    cpu.add_null("machine_system");
    cpu.add_null("machine_softirq");
    cpu.add_null("machine_busy");
  }
  cpu.add_integer("machine_cores", std::thread::hardware_concurrency());

  JsonObject json;
  json.add_object("settings", settings_json);
  json.add_object("workload", workload);
  json.add_number("time_to_first_completion_seconds", result.time_to_first_completion_seconds);
  json.add_number("settle_seconds", config.settle_seconds);
  json.add_number("window_seconds", result.window_seconds);
  json.add_integer("reads_started", result.reads_started);
  json.add_number("payload_gib_per_s", result.payload_gib_per_s);
  json.add_numbers("device_gib_per_s", result.device_gib_per_s);
  json.add_object("intervals", intervals);
  if (result.nic_gib_per_s.has_value()) {
    json.add_string("nic_interface", config.interface.value());
    json.add_number("nic_gib_per_s", result.nic_gib_per_s.value());
  } else {
    json.add_null("nic_interface");
    json.add_null("nic_gib_per_s");
  }
  json.add_number("requests_in_flight", result.requests_in_flight);
  json.add_number("reads_per_second", result.reads_per_second);
  json.add_object("read_latency", latency_json(result.read_latency));
  json.add_object("request_latency", latency_json(result.request_latency));
  json.add_object("cpu_cores", cpu);
  json.add_json("startup_counters", result.startup_counters.to_json());
  json.add_json("window_summary", result.window.to_json());
  return json;
}

// -------------------------------------------------------------------------------------------------
// Verification
// -------------------------------------------------------------------------------------------------

/**
 * @brief Read one slice into each device, with another device current, and compare it with a host
 * read of the same slice.
 *
 * The other device is made current on purpose: KvikIO must find the destination's GPU from the
 * buffer, not from the calling thread.
 */
void verify_devices(Config const& config,
                    std::vector<RemoteHandle>& objects,
                    std::vector<Slice> const& slices)
{
  std::cout << "--> verify\n";
  for (std::size_t i = 0; i < config.devices.size(); ++i) {
    int const device     = config.devices[i];
    int const current    = config.devices[(i + 1) % config.devices.size()];
    Slice const& slice   = slices[i % slices.size()];
    RemoteHandle& object = objects[slice.object_index];

    Buffer const expected{config.read_size, MemoryKind::HOST};
    object.pread(expected.data(), config.read_size, slice.offset).get();

    Buffer const on_device{config.read_size, MemoryKind::DEVICE, device};
    if (cudaSetDevice(current) != cudaSuccess) { throw std::runtime_error("cudaSetDevice failed"); }
    object.pread(on_device.data(), config.read_size, slice.offset).get();

    std::vector<std::byte> actual(config.read_size);
    if (cudaMemcpy(actual.data(), on_device.data(), config.read_size, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      throw std::runtime_error("cudaMemcpy failed");
    }
    if (std::memcmp(actual.data(), expected.data(), config.read_size) != 0) {
      throw std::runtime_error(
        std::format("verify: the slice read into device {} differs from the host read", device));
    }
    print_row(std::format("device {}", device),
              std::format("{} match a host read, read with device {} current",
                          format_bytes(config.read_size),
                          current));
  }
}

// -------------------------------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------------------------------

int run(int argc, char** argv)
{
  Options options{usage, make_flags()};
  options.parse(argc, argv);
  if (options.is_set("help")) {
    std::cout << options.help();
    return 0;
  }

  Config const config = make_config(options);
  apply_kvikio_settings(options);

  Result result;
  result.settings    = read_kvikio_settings();
  result.environment = relevant_environment();

  // Open every object before anything is measured. Opening asks each object for its size.
  std::vector<std::string> const urls    = read_urls(config.url_file);
  statistics::Counters const before_open = statistics::counters();
  TimePoint const open_start             = Clock::now();
  std::vector<RemoteHandle> objects;
  for (auto const& url : urls) {
    objects.push_back(RemoteHandle::open(url, config.endpoint_type));
  }
  result.open_seconds  = to_seconds(Clock::now() - open_start);
  result.open_counters = statistics::counters().since(before_open);

  result.num_objects = objects.size();
  for (auto const& object : objects) {
    result.endpoint_types[std::string{endpoint_type_name(object.remote_endpoint_type())}] += 1;
    if (object.nbytes() < config.read_size) { ++result.num_objects_skipped; }
  }

  std::vector<Slice> const slices = make_slices(objects, config.read_size, config.seed);
  if (slices.empty()) { throw std::runtime_error("no object is as large as --read-size"); }
  result.num_slices = slices.size();
  result.requests_per_read =
    (config.read_size + result.settings.task_size - 1) / result.settings.task_size;

  // Buffer i goes on the i-th device of --devices in turn, so the threads alternate between them.
  std::vector<std::unique_ptr<Buffer>> buffers;
  for (std::size_t i = 0; i < config.num_buffers; ++i) {
    int const device = config.devices.empty() ? 0 : config.devices[i % config.devices.size()];
    buffers.push_back(std::make_unique<Buffer>(config.read_size, config.destination, device));
  }
  std::size_t const num_destinations = std::max<std::size_t>(config.devices.size(), 1);

  print_settings(config, result);

  // Request latencies do not compare between the backends when requests are retried. MULTI_POLL
  // reports each attempt as a range request of its own and a failed one is left out, while
  // EASY_THREADPOOL reports every attempt of a request and the backoff between them as one.
  statistics::SummaryMonitor const transfers{ObservationKind::PHYSICAL};
  LatencyMonitor const read_latency{ObservationKind::LOGICAL};
  LatencyMonitor const request_latency{ObservationKind::PHYSICAL};
  // Declared last, so that its threads are stopped before anything they use is destroyed.
  SubmitterThreads submitters{objects, slices, buffers, config.read_size, num_destinations};

  std::cout << "--> running" << std::endl;
  TimePoint const started = Clock::now();
  submitters.start(config.reads_in_flight);

  // The time of the first completion comes from its observation, so checking for it every 10 ms
  // costs no accuracy.
  TimePoint const give_up = started + to_duration(config.startup_timeout_seconds);
  std::optional<TimePoint> first_completion;
  while (true) {
    first_completion = request_latency.first_finish();
    if (first_completion.has_value() || submitters.failed() || Clock::now() >= give_up) { break; }
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
  }
  submitters.rethrow_failure();
  if (!first_completion.has_value()) {
    throw std::runtime_error(
      std::format("no range request completed within {} s", config.startup_timeout_seconds));
  }
  result.time_to_first_completion_seconds = to_seconds(first_completion.value() - started);
  print_row("first range request done",
            std::format("{:.2f} s after the start", result.time_to_first_completion_seconds));
  print_row("window",
            std::format("{} intervals of {} s, from {} s after that",
                        config.num_intervals,
                        config.interval_seconds,
                        config.settle_seconds));
  std::cout << std::flush;

  TimePoint const window_start = first_completion.value() + to_duration(config.settle_seconds);
  Duration const interval      = to_duration(config.interval_seconds);
  std::vector<Reading> readings;
  for (std::size_t i = 0; i <= config.num_intervals; ++i) {
    std::this_thread::sleep_until(window_start + interval * static_cast<Duration::rep>(i));
    if (submitters.failed()) { break; }
    readings.push_back(Reading{.transfers         = transfers.get(),
                               .usage             = read_system_usage(config.interface),
                               .destination_bytes = submitters.bytes_read()});
  }
  submitters.stop();
  submitters.rethrow_failure();
  result.reads_started = submitters.reads_started();

  double const passes =
    static_cast<double>(result.reads_started) / static_cast<double>(result.num_slices);
  print_row("passes over the slices", std::format("{:.2f}", passes));
  if (passes > 1) {
    std::cerr << "warning: some slices were read more than once. Add objects, or shorten the run\n";
  }

  Reading const& first  = readings.front();
  Reading const& last   = readings.back();
  result.window         = last.transfers.since(first.transfers);
  result.window_seconds = to_seconds(result.window.wall());
  result.payload_gib_per_s =
    static_cast<double>(result.window.bytes_read) / bytes_per_gib / result.window_seconds;
  result.intervals = interval_bandwidth(readings);
  for (std::size_t i = 0; i < config.devices.size(); ++i) {
    auto const bytes = static_cast<double>(last.destination_bytes[i] - first.destination_bytes[i]);
    result.device_gib_per_s.push_back(bytes / bytes_per_gib / result.window_seconds);
  }

  if (first.usage.nic_received_bytes.has_value() && last.usage.nic_received_bytes.has_value()) {
    auto const received = static_cast<double>(last.usage.nic_received_bytes.value() -
                                              first.usage.nic_received_bytes.value());
    result.nic_gib_per_s =
      received / bytes_per_gib / to_seconds(last.usage.time - first.usage.time);
  }

  // The range requests' durations added up, over the window, is how many were on the wire on
  // average (Little's law). Subtracting them from the reads' durations does not give the time the
  // reads spent queueing, as the example in kvikio/statistics/summary.hpp suggests, since the
  // range requests of one read overlap.
  result.requests_in_flight = to_seconds(result.window.total_duration) / result.window_seconds;

  result.read_latency     = read_latency.latency(first.transfers.end, last.transfers.end);
  result.request_latency  = request_latency.latency(first.transfers.end, last.transfers.end);
  result.reads_per_second = static_cast<double>(result.read_latency.count) / result.window_seconds;
  result.cpu              = cpu_cores_between(first.usage, last.usage);
  result.startup_counters = first.transfers.counters;

  print_result(config, result);
  if (config.verify && config.destination == MemoryKind::DEVICE) {
    verify_devices(config, objects, slices);
  }
  if (config.json) { std::cout << "JSON " << result_json(config, result).str() << "\n"; }
  return 0;
}

}  // namespace
}  // namespace kvikio::benchmark

int main(int argc, char** argv)
{
  try {
    return kvikio::benchmark::run(argc, argv);
  } catch (std::exception const& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
