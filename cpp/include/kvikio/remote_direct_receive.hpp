/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

#include <kvikio/shim/utils.hpp>

namespace kvikio {

/**
 * @brief Policy for the experimental remote direct receive path.
 *
 * With direct receive, libcurl reads a remote range into memory lent by KvikIO instead of its own
 * buffer. For host destinations, the response body lands at its final offset in the caller's
 * buffer, so KvikIO performs no copy after validating the response headers. Over HTTPS, strict
 * direct receive additionally lets the Linux kernel decrypt TLS records (RX kTLS) straight into that
 * memory.
 */
enum class RemoteDirectReceiveMode : std::uint8_t {
  OFF     = 0,  ///< Use the ordinary remote receive path.
  PREFER  = 1,  ///< Use direct receive where eligible. Fall back only before body bytes arrive.
  REQUIRE = 2,  ///< Fail unless strict RX kTLS direct receive activates for every transfer.
};

/**
 * @brief Counters for the experimental remote direct receive path.
 *
 * Counters are cumulative within one loaded KvikIO library. They separate strict RX kTLS work from
 * copied-stream work, which is direct receive through libcurl's ordinary TLS or cleartext receive,
 * so a benchmark can detect a silent change of data path. A transfer is one internal sub-range
 * request, not one user-facing `pread()`.
 *
 * Raw bytes include HTTP framing received into lent buffers. Body bytes include only accepted range
 * payload. Byte and completion totals count successful final attempts only. Direct-placement bytes
 * were received at their final destination offset. Framing-compaction bytes shared a small receive
 * window with the response headers and were copied to their final offset.
 */
struct RemoteDirectReceiveStats {
  std::uint64_t transfers_requested{};
  std::uint64_t strict_rx_transfers_activated{};
  std::uint64_t strict_rx_transfers_completed{};
  std::uint64_t copied_stream_transfers_completed{};
  std::uint64_t transfers_fallback{};
  std::uint64_t fallback_capability_unavailable{};
  std::uint64_t fallback_ineligible_request{};
  std::uint64_t transfers_failed{};
  std::uint64_t protocol_validation_failures{};
  std::uint64_t retries{};

  std::uint64_t strict_rx_raw_received_bytes{};
  std::uint64_t strict_rx_body_bytes{};
  std::uint64_t copied_stream_raw_received_bytes{};
  std::uint64_t copied_stream_body_bytes{};
  std::uint64_t direct_placement_bytes{};
  std::uint64_t framing_compaction_bytes{};
};

/**
 * @brief Whether this KvikIO build can use remote direct receive.
 *
 * Direct receive needs a libcurl that provides caller-owned receive buffers and strict RX kTLS. With
 * any other libcurl, `PREFER` uses the ordinary path and `REQUIRE` fails.
 *
 * @return `true` if the linked libcurl provides the direct receive API.
 */
[[nodiscard]] KVIKIO_EXPORT bool remote_direct_receive_supported() noexcept;

/**
 * @brief Take a snapshot of the direct receive counters.
 *
 * Each field is read atomically, but the structure as a whole is not a transactional snapshot of
 * concurrent transfers. It is meant for before and after deltas around a benchmark.
 *
 * @return The current counter values.
 */
[[nodiscard]] KVIKIO_EXPORT RemoteDirectReceiveStats remote_direct_receive_stats() noexcept;

/**
 * @brief Reset all direct receive counters to zero.
 *
 * Call only while no remote reads are in flight when exact deltas are required.
 */
KVIKIO_EXPORT void reset_remote_direct_receive_stats() noexcept;

}  // namespace kvikio
