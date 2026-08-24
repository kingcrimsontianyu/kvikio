/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <mutex>

#include <curl/curl.h>

namespace kvikio::detail {

/**
 * @brief A process-wide libcurl share holding the caches that are safe to share across threads.
 *
 * Each `CURLM` keeps its own DNS cache, so `MULTI_POLL` resolves the same hostname once per reactor
 * instead of once per process, and repeats it whenever an entry ages out. Against a rate-limited
 * resolver such as the VPC's on EC2, a large reactor count turns that into a burst of identical
 * queries and requests start failing with "Could not resolve host". One shared cache makes it one
 * lookup. The TLS session cache is shared for the same reason.
 *
 * libcurl calls the lock and unlock callbacks around every access, so this is safe to use from all
 * reactor threads at once.
 *
 * @note Deliberately does **not** share `CURL_LOCK_DATA_CONNECT`: libcurl documents sharing
 * connections between concurrent threads as unsupported. Connections stay per multi handle.
 *
 * @note Never destroyed, matching the leaked-singleton convention used by `MultiReactorPool`,
 * whose reactor threads outlive static destruction.
 */
class CurlShare {
 public:
  /**
   * @brief Get the process-wide share, creating it on first use.
   */
  static CurlShare& instance();

  CurlShare(CurlShare const&)            = delete;
  CurlShare& operator=(CurlShare const&) = delete;
  CurlShare(CurlShare&&)                 = delete;
  CurlShare& operator=(CurlShare&&)      = delete;

  /**
   * @brief The underlying share handle, for `curl_easy_setopt(CURLOPT_SHARE, ...)`.
   */
  [[nodiscard]] CURLSH* handle() const noexcept { return _share; }

 private:
  CurlShare();
  ~CurlShare() = default;

  static void lock_cb(CURL* handle, curl_lock_data data, curl_lock_access access, void* userptr);
  static void unlock_cb(CURL* handle, curl_lock_data data, void* userptr);

  CURLSH* _share{nullptr};
  // One mutex per shared data kind, so the DNS cache and the session cache do not serialize
  // against each other.
  std::array<std::mutex, CURL_LOCK_DATA_LAST> _mutexes;
};

}  // namespace kvikio::detail
