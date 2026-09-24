/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <kvikio/detail/direct_receive_stats.hpp>
#include <kvikio/remote_direct_receive.hpp>

TEST(RemoteDirectReceiveStats, records_every_counter_and_resets)
{
  using kvikio::detail::DirectReceiveFailureReason;
  using kvikio::detail::DirectReceiveFallbackReason;
  kvikio::reset_remote_direct_receive_stats();

  kvikio::detail::direct_receive_record_requested();
  kvikio::detail::direct_receive_record_requested();
  kvikio::detail::direct_receive_record_strict_activated();
  kvikio::detail::direct_receive_record_strict_completion(110, 100);
  kvikio::detail::direct_receive_record_copied_completion(22, 20);
  kvikio::detail::direct_receive_record_placement(115, 5);
  kvikio::detail::direct_receive_record_fallback(DirectReceiveFallbackReason::capability_unavailable);
  kvikio::detail::direct_receive_record_fallback(DirectReceiveFallbackReason::ineligible_request);
  kvikio::detail::direct_receive_record_failed(DirectReceiveFailureReason::protocol_validation);
  kvikio::detail::direct_receive_record_failed(DirectReceiveFailureReason::other);
  kvikio::detail::direct_receive_record_retry();

  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.transfers_requested, 2);
  EXPECT_EQ(stats.strict_rx_transfers_activated, 1);
  EXPECT_EQ(stats.strict_rx_transfers_completed, 1);
  EXPECT_EQ(stats.copied_stream_transfers_completed, 1);
  EXPECT_EQ(stats.transfers_fallback, 2);
  EXPECT_EQ(stats.fallback_capability_unavailable, 1);
  EXPECT_EQ(stats.fallback_ineligible_request, 1);
  EXPECT_EQ(stats.transfers_failed, 2);
  EXPECT_EQ(stats.protocol_validation_failures, 1);
  EXPECT_EQ(stats.retries, 1);
  EXPECT_EQ(stats.strict_rx_raw_received_bytes, 110);
  EXPECT_EQ(stats.strict_rx_body_bytes, 100);
  EXPECT_EQ(stats.copied_stream_raw_received_bytes, 22);
  EXPECT_EQ(stats.copied_stream_body_bytes, 20);
  EXPECT_EQ(stats.direct_placement_bytes, 115);
  EXPECT_EQ(stats.framing_compaction_bytes, 5);

  kvikio::reset_remote_direct_receive_stats();
  auto const cleared = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(cleared.transfers_requested, 0);
  EXPECT_EQ(cleared.strict_rx_transfers_activated, 0);
  EXPECT_EQ(cleared.strict_rx_transfers_completed, 0);
  EXPECT_EQ(cleared.copied_stream_transfers_completed, 0);
  EXPECT_EQ(cleared.transfers_fallback, 0);
  EXPECT_EQ(cleared.fallback_capability_unavailable, 0);
  EXPECT_EQ(cleared.fallback_ineligible_request, 0);
  EXPECT_EQ(cleared.transfers_failed, 0);
  EXPECT_EQ(cleared.protocol_validation_failures, 0);
  EXPECT_EQ(cleared.retries, 0);
  EXPECT_EQ(cleared.strict_rx_raw_received_bytes, 0);
  EXPECT_EQ(cleared.strict_rx_body_bytes, 0);
  EXPECT_EQ(cleared.copied_stream_raw_received_bytes, 0);
  EXPECT_EQ(cleared.copied_stream_body_bytes, 0);
  EXPECT_EQ(cleared.direct_placement_bytes, 0);
  EXPECT_EQ(cleared.framing_compaction_bytes, 0);
}
