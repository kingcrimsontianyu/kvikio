/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <cuda_runtime_api.h>
#include <curl/curl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/direct_receive.hpp>
#include <kvikio/hdfs.hpp>
#include <kvikio/remote_direct_receive.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

namespace {

// A single-threaded HTTP/1.1 server on the loopback interface. It serves one configurable body,
// optionally as exact 206 range responses, and can pause mid-body so a test can observe a
// transfer in flight.
class LocalHttpServer {
 public:
  explicit LocalHttpServer(std::string body                   = {},
                           bool exact_range                   = false,
                           std::size_t transient_failures     = 0,
                           std::size_t pause_after_body_bytes = 0,
                           std::size_t successful_requests    = 1,
                           bool honor_range_requests          = false,
                           bool pause_before_body             = false,
                           std::string entity_tag             = {},
                           std::vector<int> response_statuses = {})
    : _body{std::move(body)},
      _exact_range{exact_range},
      _transient_failures{transient_failures},
      _pause_after_body_bytes{pause_after_body_bytes},
      _successful_requests{successful_requests},
      _honor_range_requests{honor_range_requests},
      _pause_before_body{pause_before_body},
      _entity_tag{std::move(entity_tag)},
      _response_statuses{std::move(response_statuses)}
  {
    _listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_listen_fd < 0) { throw std::runtime_error(std::strerror(errno)); }

    int reuse = 1;
    if (::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
      auto const message = std::string{std::strerror(errno)};
      ::close(_listen_fd);
      _listen_fd = -1;
      throw std::runtime_error(message);
    }

    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (::bind(_listen_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(_listen_fd, 1) != 0) {
      auto const message = std::string{std::strerror(errno)};
      ::close(_listen_fd);
      _listen_fd = -1;
      throw std::runtime_error(message);
    }

    socklen_t address_length = sizeof(address);
    if (::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&address), &address_length) != 0) {
      auto const message = std::string{std::strerror(errno)};
      ::close(_listen_fd);
      _listen_fd = -1;
      throw std::runtime_error(message);
    }
    _port   = ntohs(address.sin_port);
    _thread = std::thread{[this] { serve(); }};
  }

  ~LocalHttpServer()
  {
    resume_body();
    if (_listen_fd >= 0) {
      ::shutdown(_listen_fd, SHUT_RDWR);
      ::close(_listen_fd);
      _listen_fd = -1;
    }
    if (_thread.joinable()) { _thread.join(); }
  }

  [[nodiscard]] uint16_t port() const noexcept { return _port; }

  [[nodiscard]] bool wait_until_accepted(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock{_state_mutex};
    return _state_cv.wait_for(lock, timeout, [this] { return _accepted; });
  }

  [[nodiscard]] bool wait_until_body_paused(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock{_state_mutex};
    return _state_cv.wait_for(lock, timeout, [this] { return _body_paused; });
  }

  void resume_body() noexcept
  {
    try {
      {
        std::lock_guard lock{_state_mutex};
        _resume_body = true;
      }
      _state_cv.notify_all();
    } catch (...) {
    }
  }

  void replace_body(std::string body)
  {
    std::lock_guard lock{_state_mutex};
    if (body.size() != _body.size()) {
      throw std::invalid_argument("replacement HTTP body must preserve Content-Length");
    }
    _body = std::move(body);
  }

  [[nodiscard]] std::string const& request()
  {
    if (_thread.joinable()) { _thread.join(); }
    return _request;
  }

 private:
  static bool send_all(int fd, char const* data, std::size_t size)
  {
    while (size != 0) {
      auto const sent = ::send(fd, data, size, MSG_NOSIGNAL);
      if (sent <= 0) { return false; }
      data += sent;
      size -= static_cast<std::size_t>(sent);
    }
    return true;
  }

  [[nodiscard]] static std::optional<std::pair<std::size_t, std::size_t>> requested_range(
    std::string_view request)
  {
    constexpr std::string_view prefix{"Range: bytes="};
    auto const prefix_begin = request.find(prefix);
    if (prefix_begin == std::string_view::npos) { return std::nullopt; }
    auto const value_begin = prefix_begin + prefix.size();
    auto const value_end   = request.find("\r\n", value_begin);
    if (value_end == std::string_view::npos) { return std::nullopt; }
    auto const value = request.substr(value_begin, value_end - value_begin);
    auto const dash  = value.find('-');
    if (dash == std::string_view::npos) { return std::nullopt; }

    std::size_t first{};
    std::size_t last{};
    auto const [first_end, first_error] = std::from_chars(value.data(), value.data() + dash, first);
    auto const [last_end, last_error] =
      std::from_chars(value.data() + dash + 1, value.data() + value.size(), last);
    if (first_error != std::errc{} || last_error != std::errc{} ||
        first_end != value.data() + dash || last_end != value.data() + value.size() ||
        first > last) {
      return std::nullopt;
    }
    return std::pair{first, last};
  }

  void serve()
  {
    auto const request_count = _response_statuses.empty()
                                 ? _transient_failures + _successful_requests
                                 : _response_statuses.size();
    for (std::size_t attempt = 0; attempt < request_count; ++attempt) {
      auto const client = ::accept(_listen_fd, nullptr, nullptr);
      if (client < 0) { return; }
      {
        std::lock_guard lock{_state_mutex};
        _accepted = true;
      }
      _state_cv.notify_all();

      std::string request;
      char buffer[4096];
      while (request.find("\r\n\r\n") == std::string::npos) {
        auto const received = ::recv(client, buffer, sizeof(buffer), 0);
        if (received <= 0) { break; }
        request.append(buffer, static_cast<std::size_t>(received));
      }
      _request += request;

      auto response_header       = std::string{};
      std::size_t response_first = 0;
      std::size_t response_size  = _body.size();
      auto const response_status =
        _response_statuses.empty()
          ? (attempt < _transient_failures ? 503 : (_exact_range ? 206 : 200))
          : _response_statuses[attempt];
      bool const send_body = response_status == 200 || response_status == 206;
      if (response_status == 503) {
        response_header =
          "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: "
          "close\r\n\r\n";
      } else if (response_status == 412) {
        response_header =
          "HTTP/1.1 412 Precondition Failed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
      } else if (response_status == 206) {
        if (_honor_range_requests) {
          auto const range = requested_range(request);
          if (!range.has_value() || range->second >= _body.size()) {
            ::close(client);
            return;
          }
          response_first = range->first;
          response_size  = range->second - range->first + 1;
        }
        response_header = std::string{"HTTP/1.1 206 Partial Content\r\nContent-Length: "} +
                          std::to_string(response_size) + "\r\nContent-Range: bytes " +
                          std::to_string(response_first) + "-" +
                          std::to_string(response_first + response_size - 1) + "/" +
                          std::to_string(_body.size()) + "\r\n";
        if (!_entity_tag.empty()) { response_header += "ETag: " + _entity_tag + "\r\n"; }
        response_header += "Connection: close\r\n\r\n";
      } else if (response_status == 200) {
        response_header = std::string{"HTTP/1.1 200 OK\r\nContent-Length: "} +
                          std::to_string(_body.size()) + "\r\nConnection: close\r\n\r\n";
      } else {
        ::close(client);
        return;
      }
      if (!send_all(client, response_header.data(), response_header.size())) {
        ::close(client);
        return;
      }
      if (send_body) {
        if (_pause_before_body) {
          std::unique_lock lock{_state_mutex};
          _body_paused = true;
          _state_cv.notify_all();
          _state_cv.wait(lock, [this] { return _resume_body; });
        }
        auto const prefix = std::min(_pause_after_body_bytes, response_size);
        if (!send_all(client, _body.data() + response_first, prefix)) {
          ::close(client);
          return;
        }
        if (prefix != 0 && prefix < response_size) {
          std::unique_lock lock{_state_mutex};
          _body_paused = true;
          _state_cv.notify_all();
          _state_cv.wait(lock, [this] { return _resume_body; });
        }
        std::ignore =
          send_all(client, _body.data() + response_first + prefix, response_size - prefix);
      }
      ::shutdown(client, SHUT_RDWR);
      ::close(client);
    }
  }

  int _listen_fd = -1;
  uint16_t _port = 0;
  std::thread _thread;
  std::string _request;
  std::string _body;
  bool _exact_range{};
  std::size_t _transient_failures{};
  std::size_t _pause_after_body_bytes{};
  std::size_t _successful_requests{};
  bool _honor_range_requests{};
  bool _pause_before_body{};
  std::string _entity_tag;
  std::vector<int> _response_statuses;
  std::mutex _state_mutex;
  std::condition_variable _state_cv;
  bool _accepted{};
  bool _body_paused{};
  bool _resume_body{};
};

template <typename Predicate>
[[nodiscard]] bool wait_until(Predicate&& predicate, std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return true;
}

// An endpoint that records calls instead of configuring requests.
class CountingEndpoint : public kvikio::RemoteEndpoint {
 public:
  explicit CountingEndpoint(
    kvikio::RemoteEndpointType endpoint_type = kvikio::RemoteEndpointType::HTTP,
    std::string url                          = "http://example.com/test")
    : RemoteEndpoint{endpoint_type}, _url{std::move(url)}
  {
  }

  void setopt(kvikio::CurlHandle&) override { ++setopt_calls; }

  std::string str() const override { return _url; }

  [[nodiscard]] bool supports_exact_http_range() const noexcept override
  {
    return remote_endpoint_type() != kvikio::RemoteEndpointType::WEBHDFS;
  }

  [[nodiscard]] bool uses_origin_tls() const noexcept override
  {
    return _url.starts_with("https://");
  }

  std::size_t get_file_size() override { return file_size; }

  void setup_range_request(kvikio::CurlHandle&, std::size_t, std::size_t) override
  {
    ++range_request_calls;
  }

  std::size_t file_size{100};
  int setopt_calls{};
  int range_request_calls{};

 private:
  std::string _url;
};

class RestoreRemoteIoBackend {
 public:
  RestoreRemoteIoBackend() : _backend{kvikio::defaults::remote_io_backend()} {}
  ~RestoreRemoteIoBackend() { kvikio::defaults::set_remote_io_backend(_backend); }

 private:
  kvikio::RemoteIOBackend _backend;
};

class RestoreRemoteDirectReceiveMode {
 public:
  RestoreRemoteDirectReceiveMode() : _mode{kvikio::defaults::remote_direct_receive_mode()} {}
  ~RestoreRemoteDirectReceiveMode() { kvikio::defaults::set_remote_direct_receive_mode(_mode); }

 private:
  kvikio::RemoteDirectReceiveMode _mode;
};

class RestoreHttpRetryPolicy {
 public:
  RestoreHttpRetryPolicy()
    : _max_attempts{kvikio::defaults::http_max_attempts()},
      _timeout{kvikio::defaults::http_timeout()},
      _status_codes{kvikio::defaults::http_status_codes()}
  {
  }

  ~RestoreHttpRetryPolicy()
  {
    kvikio::defaults::set_http_max_attempts(_max_attempts);
    kvikio::defaults::set_http_timeout(_timeout);
    kvikio::defaults::set_http_status_codes(std::move(_status_codes));
  }

 private:
  std::size_t _max_attempts;
  long _timeout;
  std::vector<int> _status_codes;
};

std::string patterned_body(std::size_t size, unsigned multiplier, unsigned increment)
{
  std::string body(size, '\0');
  for (std::size_t i = 0; i < body.size(); ++i) {
    body[i] = static_cast<char>((i * multiplier + increment) & 0xffU);
  }
  return body;
}

std::string local_url(LocalHttpServer const& server)
{
  return "http://127.0.0.1:" + std::to_string(server.port()) + "/object";
}

bool all_equal(char const* begin, char const* end, char value)
{
  return std::all_of(begin, end, [value](char c) { return c == value; });
}

}  // namespace

TEST(RemoteDirectReceive, curl_handle_composes_and_owns_s3_request_headers)
{
  LocalHttpServer server;
  auto curl = create_curl_handle();
  {
    // The endpoint is destroyed before the request. The handle must own its header storage.
    kvikio::S3Endpoint endpoint(
      local_url(server), "us-east-1", "ASIACUSTOMKEY", "secret-access-key", "session-token");
    endpoint.setopt(curl);
  }
  curl.append_http_header("If-Match: \"snapshot-etag\"");
  curl.setopt(CURLOPT_NOBODY, 1L);
  curl.setopt(CURLOPT_PROXY, "");
  curl.perform();

  auto const& request = server.request();
  for (auto const expected : {std::string_view{"x-amz-security-token: session-token\r\n"},
                              std::string_view{"If-Match: \"snapshot-etag\"\r\n"}}) {
    auto const first = request.find(expected);
    ASSERT_NE(first, std::string::npos) << request;
    EXPECT_EQ(request.find(expected, first + expected.size()), std::string::npos) << request;
  }
  // SigV4 signs both headers.
  auto const authorization_begin = request.find("Authorization:");
  ASSERT_NE(authorization_begin, std::string::npos) << request;
  auto const authorization_end = request.find("\r\n", authorization_begin);
  auto const authorization =
    std::string_view{request}.substr(authorization_begin, authorization_end - authorization_begin);
  EXPECT_NE(authorization.find("if-match"), std::string_view::npos) << authorization;
  EXPECT_NE(authorization.find("x-amz-security-token"), std::string_view::npos) << authorization;
}

TEST(RemoteDirectReceive, endpoints_report_capabilities_and_object_identity_policy)
{
  kvikio::HttpEndpoint cleartext{"http://example.com/object"};
  EXPECT_TRUE(cleartext.supports_exact_http_range());
  EXPECT_FALSE(cleartext.uses_origin_tls());
  EXPECT_FALSE(cleartext.direct_receive_requires_entity_tag());
  EXPECT_FALSE(cleartext.direct_receive_sends_if_match());

  kvikio::HttpEndpoint tls{"HTTPS://example.com/object"};
  EXPECT_TRUE(tls.uses_origin_tls());

  kvikio::S3Endpoint authenticated_s3{
    "https://bucket.s3.us-east-1.amazonaws.com/object", "us-east-1", "access-key", "secret-key"};
  EXPECT_TRUE(authenticated_s3.supports_exact_http_range());
  EXPECT_TRUE(authenticated_s3.uses_origin_tls());
  EXPECT_TRUE(authenticated_s3.direct_receive_requires_entity_tag());
  EXPECT_TRUE(authenticated_s3.direct_receive_sends_if_match());

  kvikio::S3PublicEndpoint public_s3{"https://bucket.s3.us-east-1.amazonaws.com/object"};
  EXPECT_TRUE(public_s3.direct_receive_requires_entity_tag());
  EXPECT_TRUE(public_s3.direct_receive_sends_if_match());

  kvikio::S3EndpointWithPresignedUrl presigned_s3{
    "https://bucket.s3.us-east-1.amazonaws.com/object?"
    "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=signature&"
    "X-Amz-Credential=credential&X-Amz-SignedHeaders=host"};
  EXPECT_TRUE(presigned_s3.direct_receive_requires_entity_tag());
  EXPECT_FALSE(presigned_s3.direct_receive_sends_if_match());

  kvikio::WebHdfsEndpoint webhdfs{"https://host:1234/webhdfs/v1/data.bin"};
  EXPECT_FALSE(webhdfs.supports_exact_http_range());
}

TEST(RemoteDirectReceive, require_rejects_cleartext_before_any_request)
{
  if (!kvikio::remote_direct_receive_supported()) { GTEST_SKIP() << "no libcurl support"; }
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::REQUIRE);

  auto endpoint      = std::make_unique<CountingEndpoint>();
  auto* endpoint_ptr = endpoint.get();
  kvikio::RemoteHandle remote_handle(std::move(endpoint), endpoint_ptr->file_size);
  std::vector<char> output(1);

  EXPECT_THAT([&] { std::ignore = remote_handle.pread(output.data(), 1, 0, 1); },
              ThrowsMessage<std::runtime_error>(HasSubstr("REQUIRE needs an HTTPS endpoint")));
  EXPECT_EQ(endpoint_ptr->setopt_calls, 0);
  EXPECT_EQ(endpoint_ptr->range_request_calls, 0);
}

TEST(RemoteDirectReceive, require_rejects_an_ineligible_endpoint_before_any_request)
{
  if (!kvikio::remote_direct_receive_supported()) { GTEST_SKIP() << "no libcurl support"; }
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::REQUIRE);

  auto endpoint      = std::make_unique<CountingEndpoint>(kvikio::RemoteEndpointType::WEBHDFS,
                                                     "https://example.com/webhdfs/v1/test");
  auto* endpoint_ptr = endpoint.get();
  kvikio::RemoteHandle remote_handle(std::move(endpoint), endpoint_ptr->file_size);
  std::vector<char> output(1);

  EXPECT_THAT(
    [&] { std::ignore = remote_handle.pread(output.data(), 1, 0, 1); },
    ThrowsMessage<std::runtime_error>(HasSubstr("needs an exact-range HTTP or S3 endpoint")));
  EXPECT_EQ(endpoint_ptr->setopt_calls, 0);
  EXPECT_EQ(endpoint_ptr->range_request_calls, 0);
}

TEST(RemoteDirectReceive, require_rejects_a_device_destination_before_any_request)
{
  if (!kvikio::remote_direct_receive_supported()) { GTEST_SKIP() << "no libcurl support"; }
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::REQUIRE);

  auto endpoint =
    std::make_unique<CountingEndpoint>(kvikio::RemoteEndpointType::HTTP, "https://example.com/x");
  auto* endpoint_ptr = endpoint.get();
  kvikio::RemoteHandle remote_handle(std::move(endpoint), endpoint_ptr->file_size);
  void* device_output{nullptr};
  ASSERT_EQ(cudaMalloc(&device_output, 1), cudaSuccess);

  EXPECT_THAT([&] { std::ignore = remote_handle.pread(device_output, 1, 0, 1); },
              ThrowsMessage<std::runtime_error>(HasSubstr("host destinations only")));
  EXPECT_EQ(endpoint_ptr->setopt_calls, 0);
  EXPECT_EQ(endpoint_ptr->range_request_calls, 0);
  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
}

TEST(RemoteDirectReceive, prefer_counts_an_easy_threadpool_read_as_ineligible)
{
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::EASY_THREADPOOL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);

  auto const body = patterned_body(300, 97U, 5U);
  LocalHttpServer server{body, true, 0, 0, 3, true};
  kvikio::RemoteHandle remote_handle(std::make_unique<kvikio::HttpEndpoint>(local_url(server)),
                                     body.size());
  std::vector<char> output(body.size());
  kvikio::reset_remote_direct_receive_stats();

  EXPECT_EQ(remote_handle.pread(output.data(), output.size(), 0, 100).get(), body.size());
  EXPECT_TRUE(std::equal(body.begin(), body.end(), output.begin()));
  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.transfers_requested, 3);
  EXPECT_EQ(stats.fallback_ineligible_request, 3);
  EXPECT_EQ(stats.copied_stream_transfers_completed, 0);
}

TEST(RemoteDirectReceive, host_receive_preserves_boundaries_and_places_validated_tails)
{
  if (!kvikio::remote_direct_receive_supported()) { GTEST_SKIP() << "no libcurl support"; }
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);

  auto const window = kvikio::detail::direct_receive_minimum_receive_size();
  constexpr std::size_t guard_size = 31;
  constexpr char guard_value       = static_cast<char>(0x5a);
  for (auto const body_size :
       std::vector<std::size_t>{1, window - 1, window, window + 1, 3 * window + 137}) {
    auto const body = patterned_body(body_size, 193U, 29U);
    LocalHttpServer server{body, true};
    kvikio::RemoteHandle remote_handle(std::make_unique<kvikio::HttpEndpoint>(local_url(server)),
                                       body.size());
    std::vector<char> output(body.size() + 2 * guard_size, guard_value);
    auto* const destination = output.data() + guard_size;
    kvikio::reset_remote_direct_receive_stats();

    EXPECT_EQ(remote_handle.pread(destination, body.size(), 0, body.size()).get(), body.size());
    EXPECT_TRUE(std::equal(body.begin(), body.end(), destination)) << "size " << body_size;
    EXPECT_TRUE(all_equal(output.data(), destination, guard_value));
    EXPECT_TRUE(all_equal(destination + body.size(), output.data() + output.size(), guard_value));

    auto const stats = kvikio::remote_direct_receive_stats();
    EXPECT_EQ(stats.transfers_requested, 1);
    EXPECT_EQ(stats.copied_stream_transfers_completed, 1);
    EXPECT_EQ(stats.copied_stream_body_bytes, body.size());
    EXPECT_EQ(stats.direct_placement_bytes + stats.framing_compaction_bytes, body.size());
    if (body.size() > window) { EXPECT_GT(stats.direct_placement_bytes, 0); }
  }
}

TEST(RemoteDirectReceive, host_receive_waits_on_a_one_byte_direct_tail)
{
  if (!kvikio::remote_direct_receive_supported()) { GTEST_SKIP() << "no libcurl support"; }
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);

  // The server sends the headers, then pauses before the only body byte.
  LocalHttpServer server{"q", true, 0, 0, 1, false, true};
  kvikio::RemoteHandle remote_handle(std::make_unique<kvikio::HttpEndpoint>(local_url(server)), 1);
  std::array<char, 1> output{};
  kvikio::reset_remote_direct_receive_stats();

  auto completion        = remote_handle.pread(output.data(), output.size(), 0, output.size());
  auto const body_paused = server.wait_until_body_paused(std::chrono::seconds{5});
  EXPECT_TRUE(body_paused);
  if (body_paused) {
    EXPECT_EQ(completion.wait_for(std::chrono::milliseconds{100}), std::future_status::timeout);
  }
  server.resume_body();

  EXPECT_EQ(completion.get(), 1);
  EXPECT_EQ(output[0], 'q');
  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.copied_stream_transfers_completed, 1);
  EXPECT_EQ(stats.direct_placement_bytes, 1);
  EXPECT_EQ(stats.framing_compaction_bytes, 0);
}

TEST(RemoteDirectReceive, host_receive_places_multiple_nonzero_ranges)
{
  if (!kvikio::remote_direct_receive_supported()) { GTEST_SKIP() << "no libcurl support"; }
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);

  auto const window      = kvikio::detail::direct_receive_minimum_receive_size();
  auto const task_size   = window + 83;
  auto const read_size   = 2 * task_size + 157;
  auto const file_offset = std::size_t{113};
  auto const num_ranges  = 1 + (read_size - 1) / task_size;
  auto const body        = patterned_body(file_offset + read_size + window, 157U, 41U);
  LocalHttpServer server{body, true, 0, 0, num_ranges, true};
  kvikio::RemoteHandle remote_handle(std::make_unique<kvikio::HttpEndpoint>(local_url(server)),
                                     body.size());
  constexpr std::size_t guard_size = 29;
  constexpr char guard_value       = static_cast<char>(0x6d);
  std::vector<char> output(read_size + 2 * guard_size, guard_value);
  auto* const destination = output.data() + guard_size;
  kvikio::reset_remote_direct_receive_stats();

  EXPECT_EQ(remote_handle.pread(destination, read_size, file_offset, task_size).get(), read_size);
  EXPECT_TRUE(
    std::equal(body.begin() + file_offset, body.begin() + file_offset + read_size, destination));
  EXPECT_TRUE(all_equal(output.data(), destination, guard_value));
  EXPECT_TRUE(all_equal(destination + read_size, output.data() + output.size(), guard_value));

  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.transfers_requested, num_ranges);
  EXPECT_EQ(stats.copied_stream_transfers_completed, num_ranges);
  EXPECT_EQ(stats.copied_stream_body_bytes, read_size);
  EXPECT_EQ(stats.direct_placement_bytes + stats.framing_compaction_bytes, read_size);
  EXPECT_GT(stats.direct_placement_bytes, 0);

  auto const& request = server.request();
  EXPECT_THAT(request, HasSubstr("Range: bytes=113-"));
  EXPECT_THAT(request, HasSubstr("Range: bytes=" + std::to_string(file_offset + task_size) + "-"));
  EXPECT_THAT(request,
              HasSubstr("Range: bytes=" + std::to_string(file_offset + 2 * task_size) + "-"));
}

TEST(RemoteDirectReceive, host_failure_waits_for_sibling_destination_ownership)
{
  if (!kvikio::remote_direct_receive_supported()) { GTEST_SKIP() << "no libcurl support"; }
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  RestoreHttpRetryPolicy const restore_retry;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);
  kvikio::defaults::set_http_max_attempts(1);
  kvikio::defaults::set_http_status_codes({503});

  auto const window      = kvikio::detail::direct_receive_minimum_receive_size();
  auto const task_size   = window + 101;
  auto const read_size   = 3 * task_size;
  auto const file_offset = std::size_t{17};
  auto const body        = patterned_body(file_offset + read_size + window, 139U, 53U);
  // One sub-range receives a terminal 503. A successful sibling pauses after a body prefix, while
  // the third connection waits behind it in this serial test server.
  LocalHttpServer server{body, true, 1, 257, 2, true};
  kvikio::RemoteHandle remote_handle(std::make_unique<kvikio::HttpEndpoint>(local_url(server)),
                                     body.size());
  constexpr std::size_t guard_size = 37;
  constexpr char guard_value       = static_cast<char>(0x59);
  std::vector<char> output(read_size + 2 * guard_size, guard_value);
  auto* const destination = output.data() + guard_size;
  kvikio::reset_remote_direct_receive_stats();

  auto completion           = remote_handle.pread(destination, read_size, file_offset, task_size);
  auto const sibling_paused = server.wait_until_body_paused(std::chrono::seconds{5});
  EXPECT_TRUE(sibling_paused);
  auto const failure_observed =
    wait_until([] { return kvikio::remote_direct_receive_stats().transfers_failed == 1; },
               std::chrono::seconds{5});
  EXPECT_TRUE(failure_observed);
  // The failed read must not complete while a sibling still writes into the destination.
  if (sibling_paused && failure_observed) {
    EXPECT_EQ(completion.wait_for(std::chrono::milliseconds{100}), std::future_status::timeout);
  }

  server.resume_body();
  EXPECT_THROW(std::ignore = completion.get(), std::runtime_error);
  EXPECT_TRUE(all_equal(output.data(), destination, guard_value));
  EXPECT_TRUE(all_equal(destination + read_size, output.data() + output.size(), guard_value));

  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.transfers_requested, 3);
  EXPECT_EQ(stats.copied_stream_transfers_completed, 2);
  EXPECT_EQ(stats.transfers_failed, 1);
  EXPECT_EQ(stats.copied_stream_body_bytes, 2 * task_size);
}

TEST(RemoteDirectReceive, host_receive_requeues_a_retryable_http_failure)
{
  if (!kvikio::remote_direct_receive_supported()) { GTEST_SKIP() << "no libcurl support"; }
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  RestoreHttpRetryPolicy const restore_retry;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);
  kvikio::defaults::set_http_max_attempts(2);
  kvikio::defaults::set_http_status_codes({503});

  std::string const body(2 * kvikio::detail::direct_receive_minimum_receive_size() + 251, 'h');
  LocalHttpServer server{body, true, 1};
  kvikio::RemoteHandle remote_handle(std::make_unique<kvikio::HttpEndpoint>(local_url(server)),
                                     body.size());
  std::vector<char> output(body.size(), '\0');
  kvikio::reset_remote_direct_receive_stats();

  EXPECT_EQ(remote_handle.pread(output.data(), output.size(), 0, output.size()).get(),
            body.size());
  EXPECT_TRUE(std::equal(body.begin(), body.end(), output.begin()));

  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.transfers_requested, 1);
  EXPECT_EQ(stats.retries, 1);
  EXPECT_EQ(stats.copied_stream_transfers_completed, 1);
  EXPECT_EQ(stats.transfers_failed, 0);
  EXPECT_EQ(stats.direct_placement_bytes + stats.framing_compaction_bytes, body.size());
}

TEST(RemoteDirectReceive, host_retry_overwrites_an_accepted_partial_body)
{
  if (!kvikio::remote_direct_receive_supported()) { GTEST_SKIP() << "no libcurl support"; }
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  RestoreHttpRetryPolicy const restore_retry;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);
  kvikio::defaults::set_http_max_attempts(2);
  kvikio::defaults::set_http_timeout(1);

  auto const window       = kvikio::detail::direct_receive_minimum_receive_size();
  auto const body_size    = 2 * window + 311;
  auto const first_prefix = window + 73;
  std::string const first_body(body_size, 'x');
  auto const final_body = patterned_body(body_size, 211U, 17U);
  // The first attempt stalls after a body prefix and times out. The retry gets the final body.
  LocalHttpServer server{first_body, true, 0, first_prefix, 2};
  kvikio::RemoteHandle remote_handle(std::make_unique<kvikio::HttpEndpoint>(local_url(server)),
                                     body_size);
  constexpr std::size_t guard_size = 23;
  constexpr char guard_value       = static_cast<char>(0x65);
  std::vector<char> output(body_size + 2 * guard_size, guard_value);
  auto* const destination = output.data() + guard_size;
  kvikio::reset_remote_direct_receive_stats();

  auto completion                 = remote_handle.pread(destination, body_size, 0, body_size);
  auto const first_attempt_paused = server.wait_until_body_paused(std::chrono::seconds{5});
  EXPECT_TRUE(first_attempt_paused);
  auto const retry_observed =
    first_attempt_paused &&
    wait_until([] { return kvikio::remote_direct_receive_stats().retries == 1; },
               std::chrono::seconds{5});
  EXPECT_TRUE(retry_observed);
  if (retry_observed) {
    EXPECT_TRUE(all_equal(destination, destination + first_prefix, 'x'));
    server.replace_body(final_body);
  }
  // Always unblock the server before the destination can be destroyed.
  server.resume_body();

  EXPECT_EQ(completion.get(), body_size);
  EXPECT_TRUE(std::equal(final_body.begin(), final_body.end(), destination));
  EXPECT_TRUE(all_equal(output.data(), destination, guard_value));
  EXPECT_TRUE(all_equal(destination + body_size, output.data() + output.size(), guard_value));

  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.transfers_requested, 1);
  EXPECT_EQ(stats.retries, 1);
  EXPECT_EQ(stats.copied_stream_transfers_completed, 1);
  EXPECT_EQ(stats.copied_stream_body_bytes, body_size);
  EXPECT_EQ(stats.transfers_failed, 0);
}

TEST(RemoteDirectReceive, s3_retry_uses_learned_if_match_and_412_is_terminal)
{
  if (!kvikio::remote_direct_receive_supported()) { GTEST_SKIP() << "no libcurl support"; }
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  RestoreHttpRetryPolicy const restore_retry;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);
  kvikio::defaults::set_http_max_attempts(3);
  // A failed object-version precondition stays terminal even if it is listed as retryable.
  kvikio::defaults::set_http_status_codes({412, 503});
  kvikio::reset_remote_direct_receive_stats();

  std::string const body(257, 'v');
  LocalHttpServer server{
    body, true, 0, 0, 1, false, false, "\"snapshot-etag\"", std::vector<int>{206, 503, 412}};
  auto endpoint = std::make_unique<kvikio::S3Endpoint>(
    local_url(server), "us-east-1", "ASIACUSTOMKEY", "secret-access-key", "session-token");
  kvikio::RemoteHandle remote_handle(std::move(endpoint), body.size());
  std::vector<char> output(body.size(), '\0');

  // The first read establishes the snapshot ETag.
  ASSERT_EQ(remote_handle.pread(output.data(), body.size(), 0, body.size()).get(), body.size());
  ASSERT_TRUE(std::equal(body.begin(), body.end(), output.begin()));

  // The second read sends If-Match, is retried after a 503, then fails for good on 412.
  auto changed_object = remote_handle.pread(output.data(), body.size(), 0, body.size());
  EXPECT_THAT([&] { std::ignore = changed_object.get(); },
              ThrowsMessage<std::runtime_error>(HasSubstr("object changed")));

  auto const& requests   = server.request();
  auto count_occurrences = [&requests](std::string_view needle) {
    std::size_t count{};
    for (auto offset = requests.find(needle); offset != std::string::npos;
         offset      = requests.find(needle, offset + needle.size())) {
      ++count;
    }
    return count;
  };
  EXPECT_EQ(count_occurrences("If-Match: \"snapshot-etag\"\r\n"), 2);
  EXPECT_EQ(count_occurrences("x-amz-security-token: session-token\r\n"), 3);

  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.transfers_requested, 2);
  EXPECT_EQ(stats.retries, 1);
  EXPECT_EQ(stats.copied_stream_transfers_completed, 1);
  EXPECT_EQ(stats.transfers_failed, 1);
  EXPECT_EQ(stats.protocol_validation_failures, 1);
}

TEST(RemoteDirectReceive, s3_rejects_a_response_without_an_etag)
{
  if (!kvikio::remote_direct_receive_supported()) { GTEST_SKIP() << "no libcurl support"; }
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);
  kvikio::reset_remote_direct_receive_stats();

  std::string const body(64, 'e');
  LocalHttpServer server{body, true};
  auto endpoint = std::make_unique<kvikio::S3Endpoint>(
    local_url(server), "us-east-1", "access-key", "secret-access-key");
  kvikio::RemoteHandle remote_handle(std::move(endpoint), body.size());
  std::vector<char> output(body.size(), '\0');

  EXPECT_THROW(std::ignore = remote_handle.pread(output.data(), body.size(), 0, body.size()).get(),
               std::runtime_error);
  // Validation fails before any body byte is accepted, so the destination is untouched.
  EXPECT_TRUE(all_equal(output.data(), output.data() + output.size(), '\0'));
  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.transfers_failed, 1);
  EXPECT_EQ(stats.protocol_validation_failures, 1);
}
