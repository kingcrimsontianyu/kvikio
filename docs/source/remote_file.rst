Remote File
===========

KvikIO provides direct access to remote files, including AWS S3, WebHDFS, and generic HTTP/HTTPS.

Example
-------

.. literalinclude:: ../../python/kvikio/examples/http_io.py
    :language: python

Writing to AWS S3
-----------------

:py:meth:`kvikio.RemoteFile.write` and :py:meth:`kvikio.RemoteFile.pwrite` replace an S3 object with the content of a host or device buffer. Only S3 files opened with credentials support writes. S3 has no byte-range writes, so there is no file offset argument.

An object that does not exist yet has no size to probe, so open it with ``nbytes=0``. This also skips the connectivity probe of :py:meth:`kvikio.RemoteFile.open` in AUTO mode, which would otherwise fall back to the read-only public S3 endpoint.

:py:meth:`kvikio.RemoteFile.pwrite` uses an S3 multipart upload. The buffer is split into parts of ``task_size`` bytes, and each part is uploaded by a thread of the default thread pool. S3 requires every part but the last to be at least 5 MiB and allows at most 10,000 parts, so the part size is raised above ``task_size`` when needed. When a single part covers the buffer, one PUT request is used instead. If any part fails, the multipart upload is aborted and the error is raised. Writes always run in the thread pool. ``KVIKIO_REMOTE_IO_BACKEND`` only applies to reads.

.. code-block:: python

    import cupy
    import kvikio

    a = cupy.arange(100)
    with kvikio.RemoteFile.open_s3("my-bucket", "my-object", nbytes=0) as f:
        f.write(a)

AWS S3 object naming requirement
--------------------------------

KvikIO imposes the following naming requirements derived from the `AWS object naming guidelines <https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html>`_ .

 - ``!``, ``*``, ``'``, ``(``, ``)``, ``&``, ``$``, ``@``, ``=``, ``;``, ``:``, ``+``, ``,``: These special characters are automatically encoded by KvikIO, and are safe for use in key names.

 - ``-``, ``_``, ``.``: These special characters are **not** automatically encoded by KvikIO, but are still safe for use in key names.

 - ``/`` is used as path separator and must not appear in the object name itself.

 - Space character must be explicitly encoded (``%20``) because it will otherwise render the URL malformed.

 - ``?`` must be explicitly encoded (``%3F``) because it will otherwise cause ambiguity with the query string.

 - Control characters ``0x00`` ~ ``0x1F`` hexadecimal (0~31 decimal) and ``0x7F`` (127) are automatically encoded by KvikIO, and are safe for use in key names.

 - Other printable special characters must be avoided, such as ``\``, ``{``, ``^``, ``}``, ``%``, `````, ``]``, ``"``, ``>``, ``[``, ``~``, ``<``, ``#``, ``|``.

 - Non-ASCII characters ``0x80`` ~ ``0xFF`` (128~255) must be avoided.
