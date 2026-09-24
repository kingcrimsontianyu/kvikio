/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace kvikio::detail {

/**
 * @brief Why a direct receive transfer used the ordinary or copied-stream path instead.
 */
enum class DirectReceiveFallbackReason : std::uint8_t {
  capability_unavailable,  ///< Strict RX kTLS could not activate before any body byte arrived.
  ineligible_request,      ///< The request never qualified for strict direct receive.
};

/**
 * @brief Classification of a failed direct receive transfer.
 */
enum class DirectReceiveFailureReason : std::uint8_t {
  other,
  protocol_validation,  ///< The response failed exact range or object-version validation.
};

// Counter increments shared by the remote implementation and the public stats API.
void direct_receive_record_requested() noexcept;
void direct_receive_record_strict_activated() noexcept;
void direct_receive_record_strict_completion(std::size_t raw_bytes,
                                             std::size_t body_bytes) noexcept;
void direct_receive_record_copied_completion(std::size_t raw_bytes,
                                             std::size_t body_bytes) noexcept;
void direct_receive_record_placement(std::size_t direct_bytes,
                                     std::size_t framing_compaction_bytes) noexcept;
void direct_receive_record_fallback(DirectReceiveFallbackReason reason) noexcept;
void direct_receive_record_failed(DirectReceiveFailureReason reason) noexcept;
void direct_receive_record_retry() noexcept;

}  // namespace kvikio::detail
