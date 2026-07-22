/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <kvikio/defaults.hpp>
#include <kvikio/remote_handle.hpp>

#include "utils/env.hpp"

namespace {

constexpr std::string_view kBackendEnv = "KVIKIO_REMOTE_IO_BACKEND";

}  // namespace

TEST(MultiSocketBackendParse, RecognizedNames)
{
  // Only the canonical name is accepted. Case-insensitive plus leading/trailing whitespace is fine.
  for (auto const& v : {"multi_socket", "MULTI_SOCKET", "Multi_Socket", "  multi_socket  "}) {
    kvikio::test::EnvVarContext ctx{{kBackendEnv, v}};
    EXPECT_EQ(kvikio::getenv_or(kBackendEnv, kvikio::RemoteIOBackend::EASY_THREADPOOL),
              kvikio::RemoteIOBackend::MULTI_SOCKET)
      << "value: " << v;
  }
}

TEST(MultiSocketBackendParse, CoexistsWithOtherBackends)
{
  // Adding multi_socket must not perturb parsing of the other two backend names.
  {
    kvikio::test::EnvVarContext ctx{{kBackendEnv, "easy_threadpool"}};
    EXPECT_EQ(kvikio::getenv_or(kBackendEnv, kvikio::RemoteIOBackend::MULTI_SOCKET),
              kvikio::RemoteIOBackend::EASY_THREADPOOL);
  }
  {
    kvikio::test::EnvVarContext ctx{{kBackendEnv, "multi_poll"}};
    EXPECT_EQ(kvikio::getenv_or(kBackendEnv, kvikio::RemoteIOBackend::MULTI_SOCKET),
              kvikio::RemoteIOBackend::MULTI_POLL);
  }
}

TEST(MultiSocketBackendParse, BadValueThrows)
{
  // The short alias "socket" and a few near-misses are deliberately rejected.
  for (auto const& v : {"socket", "multisocket", "multi-socket", "multi_sockets"}) {
    kvikio::test::EnvVarContext ctx{{kBackendEnv, v}};
    EXPECT_THROW(kvikio::getenv_or(kBackendEnv, kvikio::RemoteIOBackend::EASY_THREADPOOL),
                 std::invalid_argument)
      << "value: " << v;
  }
}
