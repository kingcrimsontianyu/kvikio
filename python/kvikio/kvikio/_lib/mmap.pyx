# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

import os
from typing import Any, Optional

from posix cimport fcntl, stat

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.optional cimport nullopt, optional
from libcpp.string cimport string
from libcpp.utility cimport move, pair

from kvikio._lib.arr cimport parse_buffer_argument
from kvikio._lib.future cimport IOFuture, _wrap_io_future, future

from kvikio._lib import defaults


cdef extern from "<kvikio/mmap.hpp>" namespace "kvikio" nogil:
    cdef cppclass CppMmapHandle "kvikio::MmapHandle":
        CppMmapHandle() noexcept
        CppMmapHandle(string file_path, string flags, optional[size_t] initial_map_size,
                      size_t initial_map_offset, fcntl.mode_t mode,
                      optional[int] map_flags) except +
        size_t initial_map_size() noexcept
        size_t initial_map_offset() noexcept
        size_t file_size() except +
        void close() noexcept
        bool closed() noexcept
        size_t read(void* buf, optional[size_t] size, size_t offset) except +
        future[size_t] pread(void* buf, optional[size_t] size, size_t offset,
                             size_t task_size) except +

cdef class InternalMmapHandle:
    cdef CppMmapHandle _handle

    def __init__(self, file_path: os.PathLike,
                 flags: str = "r",
                 initial_map_size: Optional[int] = None,
                 initial_map_offset: int = 0,
                 mode: int = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
                 map_flags: Optional[int] = None):
        if not os.path.exists(file_path):
            raise RuntimeError("Unable to open file")

        cdef optional[size_t] cpp_initial_map_size
        if initial_map_size is None:
            cpp_initial_map_size = nullopt
        else:
            cpp_initial_map_size = <size_t>(initial_map_size)

        path_bytes = os.fsencode(file_path)
        flags_bytes = str(flags).encode()

        cdef optional[int] cpp_map_flags
        if map_flags is None:
            cpp_map_flags = nullopt
        else:
            cpp_map_flags = <int>(map_flags)

        self._handle = move(CppMmapHandle(path_bytes,
                                          flags_bytes,
                                          cpp_initial_map_size,
                                          initial_map_offset,
                                          mode,
                                          cpp_map_flags))

    def initial_map_size(self) -> int:
        return self._handle.initial_map_size()

    def initial_map_offset(self) -> int:
        return self._handle.initial_map_offset()

    def file_size(self) -> int:
        return self._handle.file_size()

    def close(self) -> None:
        self._handle.close()

    def closed(self) -> bool:
        return self._handle.closed()

    def read(self, buf: Any, size: Optional[int] = None, offset: int = 0) -> int:
        cdef optional[size_t] cpp_size
        if size is None:
            cpp_size = nullopt
        else:
            cpp_size = <size_t>(size)
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)
        return self._handle.read(<void*>info.first,
                                 cpp_size,
                                 offset)

    def pread(self, buf: Any, size: Optional[int] = None, offset: int = 0,
              task_size: Optional[int] = None) -> IOFuture:
        cdef optional[size_t] cpp_size
        if size is None:
            cpp_size = nullopt
        else:
            cpp_size = <size_t>(size)
        cdef pair[uintptr_t, size_t] info = parse_buffer_argument(buf, size, True)

        if task_size is None:
            cpp_task_size = defaults.task_size()
        else:
            cpp_task_size = task_size

        return _wrap_io_future(self._handle.pread(<void*>info.first,
                               cpp_size,
                               offset,
                               cpp_task_size))
