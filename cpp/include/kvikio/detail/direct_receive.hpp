/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <curl/curl.h>

// Direct receive needs libcurl's caller-owned receive buffers and strict RX kTLS. Both are
// experimental libcurl extensions, so the implementation is compiled only when the headers provide
// them. The data layout of every type below is independent of this macro.
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
#define KVIKIO_HAS_CURL_DIRECT_RECEIVE 1
#endif

namespace kvikio {
class CurlHandle;
}

namespace kvikio::detail {

/**
 * @brief Whether response body bytes may be accepted as the requested payload.
 *
 * Direct receive must make this decision from a complete HTTP header block before it records any
 * body spans. HTTP/1.1 informational and redirect bodies are consumed but discarded. An exact
 * HTTP/1.1 206 response is accepted only after all Range metadata has been validated.
 */
enum class DirectReceiveBodyDisposition : std::uint8_t {
  undecided,
  discard,
  accept_range,
  reject,
};

/**
 * @brief One contiguous run of body bytes inside a receive buffer.
 */
struct DirectReceiveSpan {
  std::size_t source_offset{};       ///< Offset of the run in the receive buffer.
  std::size_t destination_offset{};  ///< Offset of the run in the requested range.
  std::size_t size{};                ///< Length of the run.
};

// A receive callback must never allocate. Bound each buffer's retained scatter description so an
// adversarial callback sequence cannot grow memory.
inline constexpr std::size_t direct_receive_max_spans_per_buffer = 256;

// Keep entity-tag processing allocation-free in libcurl callbacks. S3 ETags are much smaller.
// Oversized or malformed validators are rejected before the destination sees response bytes.
inline constexpr std::size_t direct_receive_max_entity_tag_size = 1024;

/**
 * @brief Smallest buffer lent to libcurl for one receive.
 *
 * At least one TLS record (16 KiB) and one libcurl write callback (`CURL_MAX_WRITE_SIZE`), so a
 * receive never has to split a record. This is also the size of the header window of a host read.
 *
 * @return The minimum receive size in bytes.
 */
[[nodiscard]] std::size_t direct_receive_minimum_receive_size() noexcept;

/**
 * @brief Object identity shared by every range and retry of one remote handle.
 *
 * The first exact Range response establishes a strong entity tag. Concurrent responses must
 * converge on the same tag before any body is accepted. Eligible later requests send it in
 * If-Match. The expected size is the object size recorded when the handle was opened.
 */
class DirectReceiveObjectSnapshot {
 public:
  DirectReceiveObjectSnapshot(std::size_t expected_size,
                              bool require_entity_tag,
                              bool send_if_match) noexcept;

  DirectReceiveObjectSnapshot(DirectReceiveObjectSnapshot const&)            = delete;
  DirectReceiveObjectSnapshot& operator=(DirectReceiveObjectSnapshot const&) = delete;

  [[nodiscard]] std::size_t expected_size() const noexcept;
  [[nodiscard]] bool requires_entity_tag() const noexcept;

  /**
   * @brief Learn or validate a response's strong entity tag.
   *
   * Safe to call from a noexcept libcurl callback. It performs no allocation and rejects on any
   * synchronization failure.
   *
   * @param entity_tag The response's ETag, if it had exactly one valid one.
   * @return Whether the response belongs to the snapshot's object version.
   */
  [[nodiscard]] bool accept_entity_tag(std::optional<std::string_view> entity_tag) noexcept;

  /**
   * @brief The learned tag to send in If-Match, if the endpoint sends one and it is known.
   */
  [[nodiscard]] std::optional<std::string> if_match_entity_tag() const;

 private:
  std::size_t _expected_size;
  bool _require_entity_tag;
  bool _send_if_match;
  mutable std::mutex _mutex;
  std::array<char, direct_receive_max_entity_tag_size> _entity_tag{};
  std::size_t _entity_tag_size{};
};

/**
 * @brief Fixed-capacity description of one released receive buffer, ready for placement.
 *
 * The descriptor is fixed-size so taking a buffer from the libcurl callback state cannot allocate.
 * Its destination offsets are relative to the complete requested range, not to this buffer.
 */
struct DirectReceiveReleasedBuffer {
  std::array<DirectReceiveSpan, direct_receive_max_spans_per_buffer> spans{};
  std::size_t span_count{};
  std::size_t raw_bytes{};
  std::size_t body_bytes{};
};

/**
 * @brief How many body bytes a placement left in place and how many it copied.
 */
struct DirectReceiveHostPlacement {
  std::size_t direct_bytes{};  ///< Received at their final offset. No copy.
  std::size_t staged_bytes{};  ///< Copied out of the header window.
};

/**
 * @brief Validate one released receive buffer and publish its body into a host destination.
 *
 * A header-window buffer is copied only after its complete scatter description validates. A
 * destination-backed buffer needs no copy, but it must hold only a contiguous body prefix at the
 * exact offset assigned by the response tracker. All validation completes before the destination
 * is modified.
 *
 * @param source The released receive buffer.
 * @param source_capacity Capacity of `source` in bytes.
 * @param released Scatter description of the body bytes in `source`.
 * @param destination Start of the requested range in the caller's buffer.
 * @param destination_extent Size of the requested range.
 * @param expected_destination_offset Body bytes placed before this buffer.
 * @param direct_destination_buffer Whether `source` is the destination itself.
 * @return How many bytes were placed directly and how many were copied.
 * @exception std::logic_error if the description is inconsistent. The destination is untouched.
 */
[[nodiscard]] DirectReceiveHostPlacement place_direct_receive_on_host(
  void const* source,
  std::size_t source_capacity,
  DirectReceiveReleasedBuffer const& released,
  void* destination,
  std::size_t destination_extent,
  std::size_t expected_destination_offset,
  bool direct_destination_buffer);

/**
 * @brief Tracks body spans that libcurl exposes from a lent receive buffer.
 *
 * HTTP headers and body may share a receive buffer. The write callback therefore cannot assume that
 * the body starts at offset zero. It records source offsets while destination offsets stay densely
 * packed. Adjacent spans are coalesced without copying.
 */
class DirectReceiveSpanTracker {
 public:
  void set_buffer(void* buffer, std::size_t capacity);
  [[nodiscard]] bool record_body(void const* data, std::size_t size) noexcept;
  [[nodiscard]] bool advance_raw(std::size_t raw_bytes) noexcept;
  void reset() noexcept;

  [[nodiscard]] std::span<DirectReceiveSpan const> spans() const noexcept;
  [[nodiscard]] std::size_t body_bytes() const noexcept;
  [[nodiscard]] std::size_t raw_bytes() const noexcept;
  [[nodiscard]] bool contains(void const* data, std::size_t size) const noexcept;
  [[nodiscard]] bool span_capacity_exhausted() const noexcept;

 private:
  std::byte* _buffer{};
  std::size_t _capacity{};
  std::size_t _body_bytes{};
  std::size_t _raw_bytes{};
  std::size_t _span_count{};
  bool _span_capacity_exhausted{};
  std::array<DirectReceiveSpan, direct_receive_max_spans_per_buffer> _spans{};
};

/**
 * @brief Per-attempt HTTP response validation for direct receive.
 */
class DirectReceiveResponse {
 public:
  void consume_header(std::string_view line) noexcept;
  void reset() noexcept;
  [[nodiscard]] bool transfer_encoding_seen() const noexcept;
  [[nodiscard]] bool is_partial_content_response() const noexcept;
  [[nodiscard]] DirectReceiveBodyDisposition body_disposition(
    std::size_t requested_offset,
    std::size_t requested_size,
    std::size_t expected_object_size,
    bool require_entity_tag) const noexcept;
  [[nodiscard]] std::optional<std::string> validate(long response_code,
                                                    long http_version,
                                                    curl_off_t content_length,
                                                    std::size_t requested_offset,
                                                    std::size_t requested_size,
                                                    std::size_t expected_object_size,
                                                    bool require_entity_tag) const;
  [[nodiscard]] std::optional<std::string_view> entity_tag() const noexcept;

 private:
  std::optional<std::size_t> _content_length;
  std::optional<std::size_t> _content_range_start;
  std::optional<std::size_t> _content_range_end;
  std::optional<std::size_t> _content_range_total;
  std::optional<unsigned int> _response_code;
  std::array<char, direct_receive_max_entity_tag_size> _entity_tag{};
  std::size_t _entity_tag_size{};
  bool _content_range_seen{};
  bool _entity_tag_seen{};
  bool _entity_tag_invalid{};
  bool _content_encoding_identity{true};
  bool _transfer_encoding_seen{};
  bool _http11{};
  bool _header_block_complete{};
  bool _malformed{};
};

/**
 * @brief Whether a failed strict direct receive attempt may retry through libcurl's ordinary TLS.
 *
 * Only libcurl's pre-handoff capability result qualifies. Security, integrity, authentication,
 * transient I/O, and any failure after strict receive became active must stay fatal, or follow the
 * ordinary retry policy, rather than silently change the measured path.
 *
 * @param required Whether the policy is `REQUIRE`.
 * @param result The transfer's libcurl result.
 * @param direct_status The transfer's `CURLINFO_KTLS_DIRECT_RX_STATUS`.
 * @param body_bytes Body bytes already accepted by the attempt.
 * @return Whether a copied-stream retry is allowed.
 */
[[nodiscard]] bool direct_receive_can_fallback(bool required,
                                               CURLcode result,
                                               long direct_status,
                                               std::size_t body_bytes) noexcept;

/**
 * @brief Bridge between libcurl's receive-buffer callbacks and one transfer.
 *
 * Defined only when `KVIKIO_HAS_CURL_DIRECT_RECEIVE` is set.
 */
class CurlDirectReceiveState;

/**
 * @brief Reactor-owned direct receive state of one host-destination sub-range transfer.
 *
 * The first buffer lent to libcurl is a small header window, so HTTP framing never lands in the
 * caller's buffer. Once the final response headers pass validation, the rest of the caller's
 * destination is lent directly and the body lands at its final offset.
 */
struct DirectReceiveTransfer {
  DirectReceiveTransfer() noexcept;
  ~DirectReceiveTransfer();
  DirectReceiveTransfer(DirectReceiveTransfer const&)            = delete;
  DirectReceiveTransfer& operator=(DirectReceiveTransfer const&) = delete;

  std::shared_ptr<DirectReceiveObjectSnapshot> object_snapshot;

  // Whether this attempt requests strict RX kTLS. False means the copied stream.
  bool strict_attempt{};

  // Whether a strict attempt may fall back to the copied stream (`PREFER`).
  bool fallback_allowed{};

  // Whether If-Match has been added to the request. Kept across retries.
  bool if_match_applied{};

  // Whether this transfer's strict activation has been counted.
  bool strict_activation_recorded{};

  // Whether the buffer currently lent to libcurl is the destination rather than the header window.
  bool buffer_is_destination{};

  // Body offset within the range at which the currently lent buffer starts.
  std::size_t buffer_body_offset{};

  // Placement totals of the current attempt.
  std::size_t direct_bytes{};
  std::size_t staged_bytes{};

  // Header window. Allocated at admission and released before a retry backoff.
  std::vector<std::byte> header_window;

  std::unique_ptr<CurlDirectReceiveState> callbacks;
};

#if defined(KVIKIO_HAS_CURL_DIRECT_RECEIVE)

/**
 * @brief One-transfer bridge between libcurl and the buffers KvikIO lends it.
 *
 * A transfer lends consecutive unused regions of one receive buffer. A release advances the
 * allocation cursor by the raw bytes consumed, so a later receive cannot overwrite spans that have
 * not been placed yet. The callbacks never allocate or copy.
 */
class CurlDirectReceiveState {
 public:
  CurlDirectReceiveState(std::size_t requested_offset,
                         std::size_t requested_size,
                         std::shared_ptr<DirectReceiveObjectSnapshot> object_snapshot);
  CurlDirectReceiveState(CurlDirectReceiveState const&)            = delete;
  CurlDirectReceiveState& operator=(CurlDirectReceiveState const&) = delete;
  CurlDirectReceiveState(CurlDirectReceiveState&&)                 = delete;
  CurlDirectReceiveState& operator=(CurlDirectReceiveState&&)      = delete;

  /**
   * @brief Configure libcurl to stream one validated Range response through lent buffers.
   *
   * @param curl Handle to configure.
   * @param require_ktls When true, request strict Linux RX kTLS. When false, use the same lent
   * buffers with libcurl's ordinary TLS or cleartext receive. The latter is the copied stream.
   */
  void configure(CurlHandle& curl, bool require_ktls);

  /**
   * @brief Lend a buffer of at least `direct_receive_minimum_receive_size()` bytes.
   *
   * The buffer is released for placement once too little room is left for another full receive.
   */
  void install_buffer(void* receive_buffer, std::size_t capacity);

  /**
   * @brief Lend a buffer, choosing whether to release it early when little room is left.
   *
   * A destination-backed buffer passes `rotate_before_short_loan = false`, so it keeps receiving
   * until the range is complete.
   */
  void install_buffer(void* receive_buffer, std::size_t capacity, bool rotate_before_short_loan);

  [[nodiscard]] bool buffer_ready() const noexcept;
  [[nodiscard]] bool needs_buffer() const noexcept;
  [[nodiscard]] bool body_complete() const noexcept;
  [[nodiscard]] bool response_body_accepted() noexcept;
  void finalize_current_buffer() noexcept;
  [[nodiscard]] DirectReceiveReleasedBuffer take_released_buffer();
  [[nodiscard]] std::optional<std::string> validate(CurlHandle& curl) const;

  [[nodiscard]] std::size_t raw_bytes() const noexcept;
  [[nodiscard]] std::size_t body_bytes() const noexcept;
  [[nodiscard]] bool callback_failed() const noexcept;
  [[nodiscard]] bool callback_protocol_validation_failed() const noexcept;
  [[nodiscard]] std::string_view callback_error() const noexcept;

  // Public to permit deterministic lifecycle tests without a socket. libcurl reaches these through
  // the static C callbacks below.
  [[nodiscard]] curl_recv_buffer_result acquire_buffer(std::size_t suggested_size,
                                                       curl_recv_buffer* buffer) noexcept;
  void release_buffer(curl_recv_buffer const* buffer, std::size_t used) noexcept;
  [[nodiscard]] std::size_t consume_body(char* data, std::size_t size, std::size_t nmemb) noexcept;
  [[nodiscard]] std::size_t consume_header(char* data,
                                           std::size_t size,
                                           std::size_t nmemb) noexcept;

 private:
  static curl_recv_buffer_result acquire(CURL* easy,
                                         std::size_t suggested_size,
                                         curl_recv_buffer* buffer,
                                         void* userdata) noexcept;
  static void release(CURL* easy,
                      curl_recv_buffer const* buffer,
                      std::size_t used,
                      void* userdata) noexcept;
  static std::size_t write_body(char* data,
                                std::size_t size,
                                std::size_t nmemb,
                                void* userdata) noexcept;
  static std::size_t write_header(char* data,
                                  std::size_t size,
                                  std::size_t nmemb,
                                  void* userdata) noexcept;

  enum class CallbackError : std::uint8_t {
    none,
    size_overflow,
    invalid_callback_data,
    invalid_acquire,
    invalid_release,
    body_outside_loan,
    body_length_exceeded,
    span_capacity_exhausted,
    transfer_encoding,
    response_not_accepted,
    object_snapshot_mismatch,
  };

  void fail_callback(CallbackError error) noexcept;
  [[nodiscard]] static std::string_view callback_error_message(CallbackError error) noexcept;
  [[nodiscard]] bool accept_response_snapshot() noexcept;

  void* _receive_buffer{};
  std::size_t _capacity{};
  std::size_t _requested_offset;
  std::size_t _requested_size;
  std::shared_ptr<DirectReceiveObjectSnapshot> _object_snapshot;
  std::size_t _completed_raw_bytes{};
  std::size_t _completed_body_bytes{};
  DirectReceiveSpanTracker _tracker;
  DirectReceiveResponse _response;
  bool _loan_outstanding{};
  bool _loan_ever_released{};
  bool _rotate_before_short_loan{true};
  void* _loan_buffer{};
  std::size_t _loan_capacity{};
  std::size_t _loan_body_high_water{};
  bool _buffer_ready{};
  bool _callback_failed{};
  bool _snapshot_validation_attempted{};
  bool _snapshot_accepted{};
  CallbackError _callback_error{CallbackError::none};
};

#endif

}  // namespace kvikio::detail
