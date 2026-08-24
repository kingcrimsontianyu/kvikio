/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>

#include <curl/curl.h>

#include <kvikio/detail/curl_share.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {

CurlShare::CurlShare()
{
  // The share holds libcurl-owned caches, so the global state has to exist first.
  std::ignore = LibCurl::instance();

  _share = curl_share_init();
  KVIKIO_EXPECT(_share != nullptr, "curl_share_init() failed", std::runtime_error);

  auto set = [this](CURLSHoption option, auto value) {
    auto const sc = curl_share_setopt(_share, option, value);
    KVIKIO_EXPECT(sc == CURLSHE_OK,
                  std::string("curl_share_setopt: ") + curl_share_strerror(sc),
                  std::runtime_error);
  };
  set(CURLSHOPT_LOCKFUNC, &CurlShare::lock_cb);
  set(CURLSHOPT_UNLOCKFUNC, &CurlShare::unlock_cb);
  set(CURLSHOPT_USERDATA, this);
  set(CURLSHOPT_SHARE, CURL_LOCK_DATA_DNS);
  set(CURLSHOPT_SHARE, CURL_LOCK_DATA_SSL_SESSION);
}

CurlShare& CurlShare::instance()
{
  // Leaked on purpose: reactor threads run past static destruction and would otherwise be left
  // holding a freed share.
  static CurlShare* inst = new CurlShare();
  return *inst;
}

void CurlShare::lock_cb(CURL* handle, curl_lock_data data, curl_lock_access access, void* userptr)
{
  // A single exclusive lock per data kind. libcurl distinguishes shared from exclusive access, but
  // what is guarded here is a cache lookup that is already cheap next to the transfer it serves, so
  // a reader-writer lock would not pay for itself.
  static_cast<void>(handle);
  static_cast<void>(access);
  auto* self = static_cast<CurlShare*>(userptr);
  self->_mutexes[static_cast<std::size_t>(data)].lock();
}

void CurlShare::unlock_cb(CURL* handle, curl_lock_data data, void* userptr)
{
  static_cast<void>(handle);
  auto* self = static_cast<CurlShare*>(userptr);
  self->_mutexes[static_cast<std::size_t>(data)].unlock();
}

}  // namespace kvikio::detail
