/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstddef>
#include <cstdint>

#include <kvikio/detail/direct_receive_stats.hpp>
#include <kvikio/remote_direct_receive.hpp>

namespace kvikio {
namespace {

struct AtomicDirectReceiveStats {
  std::atomic<std::uint64_t> transfers_requested{};
  std::atomic<std::uint64_t> strict_rx_transfers_activated{};
  std::atomic<std::uint64_t> strict_rx_transfers_completed{};
  std::atomic<std::uint64_t> copied_stream_transfers_completed{};
  std::atomic<std::uint64_t> transfers_fallback{};
  std::atomic<std::uint64_t> fallback_capability_unavailable{};
  std::atomic<std::uint64_t> fallback_ineligible_request{};
  std::atomic<std::uint64_t> transfers_failed{};
  std::atomic<std::uint64_t> protocol_validation_failures{};
  std::atomic<std::uint64_t> retries{};
  std::atomic<std::uint64_t> strict_rx_raw_received_bytes{};
  std::atomic<std::uint64_t> strict_rx_body_bytes{};
  std::atomic<std::uint64_t> copied_stream_raw_received_bytes{};
  std::atomic<std::uint64_t> copied_stream_body_bytes{};
  std::atomic<std::uint64_t> direct_placement_bytes{};
  std::atomic<std::uint64_t> framing_compaction_bytes{};
};

AtomicDirectReceiveStats& atomic_stats() noexcept
{
  static AtomicDirectReceiveStats stats;
  return stats;
}

std::uint64_t load(std::atomic<std::uint64_t> const& value) noexcept
{
  return value.load(std::memory_order_relaxed);
}

void clear(std::atomic<std::uint64_t>& value) noexcept
{
  value.store(0, std::memory_order_relaxed);
}

void increment(std::atomic<std::uint64_t>& value, std::uint64_t amount = 1) noexcept
{
  value.fetch_add(amount, std::memory_order_relaxed);
}

}  // namespace

RemoteDirectReceiveStats remote_direct_receive_stats() noexcept
{
  auto const& s = atomic_stats();
  RemoteDirectReceiveStats result;
  result.transfers_requested               = load(s.transfers_requested);
  result.strict_rx_transfers_activated     = load(s.strict_rx_transfers_activated);
  result.strict_rx_transfers_completed     = load(s.strict_rx_transfers_completed);
  result.copied_stream_transfers_completed = load(s.copied_stream_transfers_completed);
  result.transfers_fallback                = load(s.transfers_fallback);
  result.fallback_capability_unavailable   = load(s.fallback_capability_unavailable);
  result.fallback_ineligible_request       = load(s.fallback_ineligible_request);
  result.transfers_failed                  = load(s.transfers_failed);
  result.protocol_validation_failures      = load(s.protocol_validation_failures);
  result.retries                           = load(s.retries);
  result.strict_rx_raw_received_bytes      = load(s.strict_rx_raw_received_bytes);
  result.strict_rx_body_bytes              = load(s.strict_rx_body_bytes);
  result.copied_stream_raw_received_bytes  = load(s.copied_stream_raw_received_bytes);
  result.copied_stream_body_bytes          = load(s.copied_stream_body_bytes);
  result.direct_placement_bytes            = load(s.direct_placement_bytes);
  result.framing_compaction_bytes          = load(s.framing_compaction_bytes);
  return result;
}

void reset_remote_direct_receive_stats() noexcept
{
  auto& s = atomic_stats();
  clear(s.transfers_requested);
  clear(s.strict_rx_transfers_activated);
  clear(s.strict_rx_transfers_completed);
  clear(s.copied_stream_transfers_completed);
  clear(s.transfers_fallback);
  clear(s.fallback_capability_unavailable);
  clear(s.fallback_ineligible_request);
  clear(s.transfers_failed);
  clear(s.protocol_validation_failures);
  clear(s.retries);
  clear(s.strict_rx_raw_received_bytes);
  clear(s.strict_rx_body_bytes);
  clear(s.copied_stream_raw_received_bytes);
  clear(s.copied_stream_body_bytes);
  clear(s.direct_placement_bytes);
  clear(s.framing_compaction_bytes);
}

namespace detail {

void direct_receive_record_requested() noexcept { increment(atomic_stats().transfers_requested); }

void direct_receive_record_strict_activated() noexcept
{
  increment(atomic_stats().strict_rx_transfers_activated);
}

void direct_receive_record_strict_completion(std::size_t raw_bytes, std::size_t body_bytes) noexcept
{
  auto& s = atomic_stats();
  increment(s.strict_rx_transfers_completed);
  increment(s.strict_rx_raw_received_bytes, raw_bytes);
  increment(s.strict_rx_body_bytes, body_bytes);
}

void direct_receive_record_copied_completion(std::size_t raw_bytes, std::size_t body_bytes) noexcept
{
  auto& s = atomic_stats();
  increment(s.copied_stream_transfers_completed);
  increment(s.copied_stream_raw_received_bytes, raw_bytes);
  increment(s.copied_stream_body_bytes, body_bytes);
}

void direct_receive_record_placement(std::size_t direct_bytes,
                                     std::size_t framing_compaction_bytes) noexcept
{
  auto& s = atomic_stats();
  increment(s.direct_placement_bytes, direct_bytes);
  increment(s.framing_compaction_bytes, framing_compaction_bytes);
}

void direct_receive_record_fallback(DirectReceiveFallbackReason reason) noexcept
{
  auto& s = atomic_stats();
  increment(s.transfers_fallback);
  if (reason == DirectReceiveFallbackReason::capability_unavailable) {
    increment(s.fallback_capability_unavailable);
  } else {
    increment(s.fallback_ineligible_request);
  }
}

void direct_receive_record_failed(DirectReceiveFailureReason reason) noexcept
{
  auto& s = atomic_stats();
  increment(s.transfers_failed);
  if (reason == DirectReceiveFailureReason::protocol_validation) {
    increment(s.protocol_validation_failures);
  }
}

void direct_receive_record_retry() noexcept { increment(atomic_stats().retries); }

}  // namespace detail
}  // namespace kvikio
