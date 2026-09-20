/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <regex>
#include <string>

#include <curl/curl.h>

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/remote_callback.hpp>
#include <kvikio/detail/remote_upload.hpp>

namespace kvikio::detail {

void CallbackContext::reset_for_retry() noexcept
{
  offset         = 0;
  overflow_error = false;
}

std::size_t callback_host_memory(char* data, std::size_t size, std::size_t nmemb, void* context)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto ctx                 = reinterpret_cast<CallbackContext*>(context);
  std::size_t const nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_FUNC_RANGE(nbytes);
  std::memcpy(ctx->buf + ctx->offset, data, nbytes);
  ctx->offset += nbytes;
  return nbytes;
}

std::size_t callback_pinned_buffer(char* data, std::size_t size, std::size_t nmemb, void* context)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto ctx                 = reinterpret_cast<CallbackContext*>(context);
  std::size_t const nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_FUNC_RANGE(nbytes);
  std::memcpy(static_cast<char*>(ctx->pinned_buffer) + ctx->offset, data, nbytes);
  ctx->offset += nbytes;
  return nbytes;
}

std::size_t callback_get_string_response(char* data,
                                         std::size_t size,
                                         std::size_t num_bytes,
                                         void* userdata)
{
  auto new_data_size = size * num_bytes;
  auto* response     = reinterpret_cast<std::string*>(userdata);
  response->append(data, new_data_size);
  return new_data_size;
}

void UploadContext::reset_for_retry() noexcept
{
  offset = 0;
  if (bounce_buffer != nullptr) { bounce_buffer->seek(0); }
}

std::size_t callback_read_host_memory(char* dst, std::size_t size, std::size_t nmemb, void* context)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto ctx                 = reinterpret_cast<UploadContext*>(context);
  std::size_t const nbytes = std::min(size * nmemb, ctx->size - ctx->offset);
  if (nbytes > 0) { std::memcpy(dst, ctx->buf + ctx->offset, nbytes); }
  ctx->offset += nbytes;
  return nbytes;
}

std::size_t callback_read_device_memory(char* dst,
                                        std::size_t size,
                                        std::size_t nmemb,
                                        void* context)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto ctx                 = reinterpret_cast<UploadContext*>(context);
  std::size_t const nbytes = ctx->bounce_buffer->read(dst, size * nmemb);
  ctx->offset += nbytes;
  return nbytes;
}

int callback_seek_upload(void* context, curl_off_t offset, int origin)
{
  if (origin != SEEK_SET || offset < 0) { return CURL_SEEKFUNC_CANTSEEK; }
  auto ctx = reinterpret_cast<UploadContext*>(context);
  if (static_cast<std::size_t>(offset) > ctx->size) { return CURL_SEEKFUNC_FAIL; }
  ctx->offset = static_cast<std::size_t>(offset);
  if (ctx->bounce_buffer != nullptr) { ctx->bounce_buffer->seek(ctx->offset); }
  return CURL_SEEKFUNC_OK;
}

std::size_t callback_header_etag(char* data,
                                 std::size_t size,
                                 std::size_t num_bytes,
                                 void* userdata)
{
  auto const new_data_size = size * num_bytes;
  auto* etag               = reinterpret_cast<std::string*>(userdata);
  std::string const header_line{data, new_data_size};
  std::regex static const pattern(R"(^ETag:\s*(\S+))", std::regex::icase);
  std::smatch match_result;
  if (std::regex_search(header_line, match_result, pattern)) { *etag = match_result[1].str(); }
  return new_data_size;
}
}  // namespace kvikio::detail
