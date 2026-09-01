/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

#include <curl/curl.h>

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/remote_callback.hpp>

namespace kvikio::detail {

void CallbackContext::reset_for_retry() noexcept
{
  offset         = 0;
  overflow_error = false;
  seg_idx        = 0;
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

  // Easy-backend path: the whole span goes to one buffer and there are no holes.
  if (ctx->segments.empty()) {
    std::memcpy(ctx->buf + ctx->offset, data, nbytes);
    ctx->offset += nbytes;
    return nbytes;
  }

  // `ctx->offset` is the position in the span, so it compares directly against `span_offset`.
  auto remaining = nbytes;
  auto* src      = data;
  while (remaining > 0 && ctx->seg_idx < ctx->segments.size()) {
    auto const& segment = ctx->segments[ctx->seg_idx];
    auto const position = static_cast<std::size_t>(ctx->offset);

    if (position < segment.span_offset) {  // In a hole. Advance without copying.
      auto const skipped = std::min(remaining, segment.span_offset - position);
      src += skipped;
      ctx->offset += static_cast<std::ptrdiff_t>(skipped);
      remaining -= skipped;
      continue;
    }

    auto const filled = position - segment.span_offset;
    auto const n      = std::min(remaining, segment.length - filled);
    std::memcpy(static_cast<std::byte*>(segment.dst) + filled, src, n);
    src += n;
    ctx->offset += static_cast<std::ptrdiff_t>(n);
    remaining -= n;
    if (filled + n == segment.length) { ++ctx->seg_idx; }
  }

  // A span ends on wanted bytes, so nothing should be left. Count anything that is, to keep
  // `offset` equal to the bytes received.
  ctx->offset += static_cast<std::ptrdiff_t>(remaining);
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
}  // namespace kvikio::detail
