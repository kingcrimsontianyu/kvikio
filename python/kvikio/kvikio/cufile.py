# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import os
import pathlib
from typing import Optional, Union, overload

from kvikio._lib import file_handle  # type: ignore


class IOFutureStream:
    """Future for CuFile async IO

    This class shouldn't be used directly, instead non-blocking async IO operations
    such as `CuFile.raw_read_async` and `CuFile.raw_write_async` returns an instance
    of this class.

    The instance must be kept alive alive until all data has been read from disk. One
    way to do this, is by calling `StreamFuture.check_bytes_done()`, which will
    synchronize the associated stream and return the number of bytes read.
    """

    __slots__ = "_handle"

    def __init__(self, handle):
        self._handle = handle

    def check_bytes_done(self) -> int:
        return self._handle.check_bytes_done()


class IOFuture:
    """Future for CuFile IO

    This class shouldn't be used directly, instead non-blocking IO operations such
    as `CuFile.pread` and `CuFile.pwrite` returns an instance of this class. Use
    `.get()` to wait on the completion of the IO operation and retrieve the result.
    """

    __slots__ = "_handle"

    def __init__(self, handle):
        self._handle = handle

    def get(self) -> int:
        """Retrieve the result of the IO operation that created this future

        This call blocks until the IO operation finishes.

        Returns
        -------
        int
            The size of bytes that were read or written successfully.
        """
        return self._handle.get()

    def done(self) -> bool:
        """Return True if the future is done.

        Returns
        -------
        bool
            Whether the future is done or not
        """
        return self._handle.done()


class CuFile(io.RawIOBase):
    """File handle for GPUDirect Storage (GDS)"""

    def __init__(
        self, file: Union[pathlib.Path, str], flags: str = "r", seekable: bool = True
    ):
        """Open and register file for GDS IO operations

        CuFile opens the file twice and maintains two file descriptors.
        One file is opened with the specified `flags` and the other file is
        opened with the `flags` plus the `O_DIRECT` flag.

        Parameters
        ----------
        file: pathlib.Path or str
            Path-like object giving the pathname (absolute or relative to the current
            working directory) of the file to be opened and registered.
        flags: str, optional
            "r" -> "open for reading (default)"
            "w" -> "open for writing, truncating the file first"
            "a" -> "open for writing, appending to the end of file if it exists"
            "+" -> "open for updating (reading and writing)"
        """
        super().__init__()
        self._handle = file_handle.CuFile(file, flags)
        self._seekable = seekable
        self._position = 0
        self._file_size = self._handle.nbytes()

    def close(self) -> None:
        """Deregister the file and close the file"""
        self._handle.close()

    @property
    def closed(self) -> bool:
        return self._handle.closed()

    def _check_closed(self) -> None:
        """Internal: raise a ValueError if file is closed"""
        if self.closed:
            raise ValueError("Cannot perform I/O operation on closed file")

    def fileno(self) -> int:
        """Get the file descriptor of the open file"""
        self._check_closed()
        return self._handle.fileno()

    def open_flags(self) -> int:
        """Get the flags of the file descriptor (see open(2))"""
        self._check_closed()
        return self._handle.open_flags()

    def __enter__(self) -> "CuFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def readable(self) -> bool:
        """Return whether the file is open for reading"""
        self._check_closed()
        flags = self.open_flags()
        return (flags & os.O_ACCMODE) in (os.O_RDONLY, os.O_RDWR)

    def writable(self) -> bool:
        """Return whether the file is open for writing"""
        self._check_closed()
        flags = self.open_flags()
        return (flags & os.O_ACCMODE) in (os.O_WRONLY, os.O_RDWR)

    def seekable(self) -> bool:
        """Return whether the file supports seek and tell operations"""
        self._check_closed()
        return self._seekable

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Change the stream position to the given byte offset, interpreted
        relative to the position indicated by whence, and return the new absolute
        position.

        Parameters
        ----------
        offset: int
            The offset to seek to, relative to whence.
        whence: int, optional
            - io.SEEK_SET or 0 (default): seek from the start of the stream. offset
              should be zero or positive.
            - io.SEEK_CUR or 1: seek relative to current position. offset may be
              negative
            - io.SEEK_END or 2: seek relative to end of file. offset is usually
              negative

        Returns
        -------
        int
            The new absolute position.

        Raises
        ------
        ValueError
            If the file is closed.
        OSError
            If the file is not seekable or invalid whence value.
        """
        self._check_closed()
        if not self._seekable:
            raise OSError("File is not seekable")

        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._position + offset
        elif whence == io.SEEK_END:
            new_pos = self._file_size + offset
        else:
            raise OSError(f"Invalid whence value: {whence}")

        if new_pos < 0:
            raise OSError(f"Negative seek position: {new_pos}")

        self._position = new_pos
        return self._position

    def tell(self) -> int:
        """Return current stream position.

        Returns
        -------
        int
            Current position in the file.

        Raises
        ------
        ValueError
            If the file is closed.
        OSError
            If the file is not seekable.
        """
        self._check_closed()
        if not self._seekable:
            raise OSError("File is not seekable")
        return self._position

    def pread(
        self,
        buf,
        size: Optional[int] = None,
        file_offset: int = 0,
        task_size: Optional[int] = None,
    ) -> IOFuture:
        """Reads specified bytes from the file into device or host memory in parallel

        `pread` reads the data from a specified file at a specified offset and size
        bytes into `buf`. The API works correctly for unaligned offsets and any data
        size, although the performance might not match the performance of aligned reads.
        See additional details in the notes below.

        `pread` is non-blocking and returns a `IOFuture` that can be waited upon. It
        partitions the operation into tasks of size `task_size` for execution in the
        default thread pool.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device or host buffer to read into.
        size: int, optional
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to read from.
        task_size: int, default=kvikio.defaults.task_size()
            Size of each task in bytes.

        Returns
        -------
        IOFuture
            Future that on completion returns the size of bytes that were successfully
            read.

        Notes
        -----
        KvikIO can only make use of GDS for reads that are aligned to a page boundary.
        For unaligned reads, KvikIO has to split the reads into aligned and unaligned
        parts. The GPU page size is 4kB, so all reads should be at an offset that is a
        multiple of 4096 bytes. If the desired `file_offset` is not a multiple of 4096,
        it is likely desirable to round down to the nearest multiple of 4096 and discard
        any undesired bytes from the resulting data. Similarly, it is optimal for `size`
        to be a multiple of 4096 bytes. When GDS isn't used, this is less critical.
        """
        return IOFuture(self._handle.pread(buf, size, file_offset, task_size))

    def pwrite(
        self,
        buf,
        size: Optional[int] = None,
        file_offset: int = 0,
        task_size: Optional[int] = None,
    ) -> IOFuture:
        """Writes specified bytes from device or host memory into the file in parallel

        `pwrite` writes the data from `buf` to the file at a specified offset and size.
        The API works correctly for unaligned offset and data sizes, although the
        performance is not on-par with aligned writes. See additional details in the
        notes below.

        `pwrite` is non-blocking and returns a `IOFuture` that can be waited upon. It
        partitions the operation into tasks of size `task_size` for execution in the
        default thread pool.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device or host buffer to write to.
        size: int, optional
            Size in bytes to write.
        file_offset: int, optional
            Offset in the file to write from.
        task_size: int, default=kvikio.defaults.task_size()
            Size of each task in bytes.

        Returns
        -------
        IOFuture
            Future that on completion returns the size of bytes that were successfully
            written.

        Notes
        -----
        KvikIO can only make use of GDS for writes that are aligned to a page boundary.
        For unaligned writes, KvikIO has to split the writes into aligned and unaligned
        parts. The GPU page size is 4kB, so all writes should be at an offset that is a
        multiple of 4096 bytes. If the desired `file_offset` is not a multiple of 4096,
        it is likely desirable to round down to the nearest multiple of 4096 and discard
        any undesired bytes from the resulting data. Similarly, it is optimal for `size`
        to be a multiple of 4096 bytes. When GDS isn't used, this is less critical.
        """
        return IOFuture(self._handle.pwrite(buf, size, file_offset, task_size))

    @overload
    def read(self, size: int = -1) -> bytes:
        """Read and return up to size bytes (Python io.RawIOBase interface)"""
        ...

    @overload
    def read(
        self,
        buf,
        size: Optional[int] = None,
        file_offset: int = 0,
        task_size: Optional[int] = None,
    ) -> int:
        """Reads specified bytes from the file into the device memory in parallel
        (KvikIO C++ interface)."""
        ...

    def read(self, *args, **kwargs) -> Union[bytes, int]:
        """Reads bytes from the file

        This method supports two signatures:

        1. Python io.RawIOBase interface: read(size=-1) -> bytes
           Reads and returns up to size bytes from the current position.
           If size is -1, reads until EOF.

        2. KvikIO C++ interface: read(buf, size=None, file_offset=0, task_size=None) -> int
           Reads specified bytes from the file into device or host memory in parallel.

        The method automatically detects which interface is being used based on the
        arguments provided.
        """

        self._check_closed()

        # Detect which signature is being used
        if len(args) == 0 or (len(args) == 1 and isinstance(args[0], int)):
            return self._read_python_rawio(*args, **kwargs)
        else:
            return self._read_cpp_kvikio(*args, **kwargs)

    def _read_python_rawio(self, size: int = -1) -> bytes:
        """Read and return up to size bytes (Python io.RawIOBase interface)

        Parameters
        ----------
        size: int, optional
            Number of bytes to read. If -1, read until EOF.

        Returns
        -------
        bytes
            Bytes read from the file.

        Raises
        ------
        ValueError
            If the file is closed or not readable.
        """
        if not self.readable():
            raise ValueError("File not open for reading")

        if size == -1:
            size = self._file_size

        if size <= 0 or self._position == self._file_size:
            return b""

        fd = self.fileno()

        current_position = 0
        if self._seekable:
            current_position = self._position

        adjusted_size = min(size, self._file_size - current_position)
        self._position += adjusted_size

        num_bytes_to_read = adjusted_size
        raw_data = bytes()
        while current_position < self._position:
            current_raw_data = os.pread(
                fd, num_bytes_to_read, current_position)
            current_bytes_read = len(current_raw_data)

            raw_data += current_raw_data
            current_position += current_bytes_read

            num_bytes_to_read = self._position - current_position
        return raw_data

    def _read_cpp_kvikio(
        self,
        buf,
        size: Optional[int] = None,
        file_offset: int = 0,
        task_size: Optional[int] = None,
    ) -> int:
        """Reads specified bytes from the file into device or host memory in parallel

        This is a blocking version of `.pread`.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device or host buffer to read into.
        size: int, optional
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to read from.
        task_size: int, default=kvikio.defaults.task_size()
            Size of each task in bytes.

        Returns
        -------
        int
            The size of bytes that were successfully read.

        Notes
        -----
        KvikIO can only make use of GDS for reads that are aligned to a page boundary.
        For unaligned reads, KvikIO has to split the reads into aligned and unaligned
        parts. The GPU page size is 4kB, so all reads should be at an offset that is a
        multiple of 4096 bytes. If the desired `file_offset` is not a multiple of 4096,
        it is likely desirable to round down to the nearest multiple of 4096 and discard
        any undesired bytes from the resulting data. Similarly, it is optimal for `size`
        to be a multiple of 4096 bytes. When GDS isn't used, this is less critical.
        """
        return self.pread(buf, size, file_offset, task_size).get()

    def write(
        self,
        buf,
        size: Optional[int] = None,
        file_offset: int = 0,
        task_size: Optional[int] = None,
    ) -> int:
        """Writes specified bytes from the device memory into the file in parallel

        This is a blocking version of `.pwrite`.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to write to.
        size: int, optional
            Size in bytes to write.
        file_offset: int, optional
            Offset in the file to write from.
        task_size: int, default=kvikio.defaults.task_size()
            Size of each task in bytes.

        Returns
        -------
        int
            The size of bytes that were successfully written.

        Notes
        -----
        KvikIO can only make use of GDS for writes that are aligned to a page boundary.
        For unaligned writes, KvikIO has to split the writes into aligned and unaligned
        parts. The GPU page size is 4kB, so all writes should be at an offset that is a
        multiple of 4096 bytes. If the desired `file_offset` is not a multiple of 4096,
        it is likely desirable to round down to the nearest multiple of 4096 and discard
        any undesired bytes from the resulting data. Similarly, it is optimal for `size`
        to be a multiple of 4096 bytes. When GDS isn't used, this is less critical.
        """
        return self.pwrite(buf, size, file_offset, task_size).get()

    def raw_read_async(
        self,
        buf,
        stream,
        size: Optional[int] = None,
        file_offset: int = 0,
        dev_offset: int = 0,
    ) -> IOFutureStream:
        """Reads specified bytes from the file into the device memory asynchronously

        This is an async version of `.raw_read` that doesn't use threads and
        does not support host memory.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to read into.
        stream: cuda.Stream
            CUDA stream to perform the read operation asynchronously.
        size: int, optional
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to read from.

        Returns
        -------
        IOFutureStream
            Future that when executed ".check_bytes_done()" returns the size of bytes
            that were successfully read. The instance must be kept alive until
            all data has been read from disk. One way to do this, is by calling
            `IOFutureStream.check_bytes_done()`, which will synchronize the associated
            stream and return the number of bytes read.
        """
        return self._handle.read_async(buf, size, file_offset, dev_offset, stream)

    def raw_write_async(
        self,
        buf,
        stream,
        size: Optional[int] = None,
        file_offset: int = 0,
        dev_offset: int = 0,
    ) -> IOFutureStream:
        """Writes specified bytes from the device memory into the file asynchronously

        This is an async version of `.raw_write` that doesn't use threads and
        does not support host memory.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to write to.
        stream: cuda.Stream
            CUDA stream to perform the write operation asynchronously.
        size: int, optional
            Size in bytes to write.
        file_offset: int, optional
            Offset in the file to write from.

        Returns
        -------
        IOFutureStream
            Future that when executed ".check_bytes_done()" returns the size of bytes
            that were successfully written. The instance must be kept alive until
            all data has been written to disk. One way to do this, is by calling
            `IOFutureStream.check_bytes_done()`, which will synchronize the associated
            stream and return the number of bytes written.
        """
        return self._handle.write_async(buf, size, file_offset, dev_offset, stream)

    def raw_read(
        self,
        buf,
        size: Optional[int] = None,
        file_offset: int = 0,
        dev_offset: int = 0,
    ) -> int:
        """Reads specified bytes from the file into the device memory

        This is a low-level version of `.read` that doesn't use threads and
        does not support host memory.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to read into.
        size: int, optional
            Size in bytes to read.
        file_offset: int, optional
            Offset in the file to read from.
        dev_offset: int, optional
            Offset in the `buf` to read from.

        Returns
        -------
        int
            The size of bytes that were successfully read.

        Notes
        -----
        KvikIO can only make use of GDS for reads that are aligned to a page boundary.
        For unaligned reads, KvikIO has to split the reads into aligned and unaligned
        parts. The GPU page size is 4kB, so all reads should be at an offset that is a
        multiple of 4096 bytes. If the desired `file_offset` is not a multiple of 4096,
        it is likely desirable to round down to the nearest multiple of 4096 and discard
        any undesired bytes from the resulting data. Similarly, it is optimal for `size`
        to be a multiple of 4096 bytes. When GDS isn't used, this is less critical.
        """
        return self._handle.read(buf, size, file_offset, dev_offset)

    def raw_write(
        self,
        buf,
        size: Optional[int] = None,
        file_offset: int = 0,
        dev_offset: int = 0,
    ) -> int:
        """Writes specified bytes from the device memory into the file

        This is a low-level version of `.write` that doesn't use threads and
        does not support host memory.

        Parameters
        ----------
        buf: buffer-like or array-like
            Device buffer to write to.
        size: int, optional
            Size in bytes to write.
        file_offset: int, optional
            Offset in the file to write from.
        dev_offset: int, optional
            Offset in the `buf` to write from.

        Returns
        -------
        int
            The size of bytes that were successfully written.

        Notes
        -----
        KvikIO can only make use of GDS for writes that are aligned to a page boundary.
        For unaligned writes, KvikIO has to split the writes into aligned and unaligned
        parts. The GPU page size is 4kB, so all writes should be at an offset that is a
        multiple of 4096 bytes. If the desired `file_offset` is not a multiple of 4096,
        it is likely desirable to round down to the nearest multiple of 4096 and discard
        any undesired bytes from the resulting data. Similarly, it is optimal for `size`
        to be a multiple of 4096 bytes. When GDS isn't used, this is less critical.
        """
        return self._handle.write(buf, size, file_offset, dev_offset)


def get_page_cache_info(
    file: Union[os.PathLike, str, int, io.IOBase],
) -> tuple[int, int]:
    """Obtain the page cache residency information for a given file

    Example:

    .. code-block:: python

       num_pages_in_page_cache, num_pages = kvikio.get_page_cache_info(my_file)
       percent_in_page_cache = num_pages_in_page_cache / num_pages

    Parameters
    ----------
    file: a path-like object, or string, or file descriptor, or file object
        File to check.

    Returns
    -------
    tuple[int, int]
        A pair containing the number of pages resident in the page cache
        and the total number of pages.
    """
    return file_handle.get_page_cache_info(file)


def clear_page_cache(
    reclaim_dentries_and_inodes: bool = True, clear_dirty_pages: bool = True
) -> bool:
    """Clear the page cache

    Parameters
    ----------
    reclaim_dentries_and_inodes: bool, optional
        Whether to free reclaimable slab objects which include dentries and inodes.

        - If `true`, equivalent to executing `/sbin/sysctl vm.drop_caches=3`;
        - If `false`, equivalent to executing `/sbin/sysctl vm.drop_caches=1`.
    clear_dirty_pages: bool, optional
        Whether to trigger the writeback process to clear the dirty pages. If `true`,
        `sync` will be called prior to cache dropping.

    Returns
    -------
    bool
        Whether the page cache has been successfully cleared.
    """
    return file_handle.clear_page_cache(reclaim_dentries_and_inodes, clear_dirty_pages)
