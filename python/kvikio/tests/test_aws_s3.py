# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import multiprocessing as mp
import socket
import time
import warnings
from contextlib import contextmanager

import pytest

import kvikio
import kvikio.defaults

pytestmark = pytest.mark.skipif(
    not kvikio.is_remote_file_available(),
    reason="cannot test remote IO, please build KvikIO with with AWS S3 support",
)

# Notice, we import boto and moto after the `is_remote_file_available` check.
import boto3  # noqa: E402
import moto  # noqa: E402
import moto.server  # noqa: E402


@pytest.fixture(scope="session")
def endpoint_ip():
    return "127.0.0.1"


@pytest.fixture(scope="session")
def endpoint_port():
    # Return a free port per worker session.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def start_s3_server(ip_address, port):
    server = moto.server.ThreadedMotoServer(ip_address=ip_address, port=port)
    server.start()
    time.sleep(600)
    print("ThreadedMotoServer shutting down because of timeout (10min)")


@pytest.fixture(scope="session")
def s3_base(endpoint_ip, endpoint_port):
    """Fixture to set up moto server in separate process"""
    with pytest.MonkeyPatch.context() as monkeypatch:
        # Use fake aws credentials
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "foobar_key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "foobar_secret")
        monkeypatch.setenv("AWS_SECURITY_TOKEN", "foobar_security_token")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "foobar_session_token")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        p = mp.Process(target=start_s3_server, args=(endpoint_ip, endpoint_port))
        p.start()
        yield f"http://{endpoint_ip}:{endpoint_port}"
        p.kill()


@contextmanager
def s3_context(s3_base, bucket, files=None):
    if files is None:
        files = {}
    with warnings.catch_warnings():
        # boto3 calls `datetime.datetime.utcnow()`, which is deprecated
        # in Python v3.12.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        client = boto3.client("s3", endpoint_url=s3_base)
        client.create_bucket(Bucket=bucket, ACL="public-read-write")
        for f, data in files.items():
            client.put_object(Bucket=bucket, Key=f, Body=data)
        yield kvikio.S3Context(s3_base)
        for f, data in files.items():
            try:
                client.delete_object(Bucket=bucket, Key=f)
            except Exception:
                pass


@pytest.mark.parametrize("size", [10, 100, 1000])
@pytest.mark.parametrize("nthreads", [1, 3])
@pytest.mark.parametrize("tasksize", [99, 999])
@pytest.mark.parametrize("buffer_size", [101, 1001])
def test_read(s3_base, xp, size, nthreads, tasksize, buffer_size):
    bucket_name = "test_read"
    object_name = "a1"
    a = xp.arange(size)
    with s3_context(
        s3_base=s3_base, bucket=bucket_name, files={object_name: bytes(a)}
    ) as ctx:
        with kvikio.defaults.set_num_threads(nthreads):
            with kvikio.defaults.set_task_size(tasksize):
                with kvikio.defaults.set_bounce_buffer_size(buffer_size):
                    with kvikio.RemoteFile(ctx, bucket_name, object_name) as f:
                        assert f.nbytes() == a.nbytes
                        b = xp.empty_like(a)
                        assert f.read(buf=b) == a.nbytes
                        xp.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "start,end",
    [
        (0, 10 * 4096),
        (1, int(1.3 * 4096)),
        (int(2.1 * 4096), int(5.6 * 4096)),
        (42, int(2**20)),
    ],
)
def test_read_with_file_offset(s3_base, xp, start, end):
    bucket_name = "test_read_with_file_offset"
    object_name = "a1"
    a = xp.arange(end, dtype=xp.int64)
    with s3_context(
        s3_base=s3_base, bucket=bucket_name, files={object_name: bytes(a)}
    ) as ctx:
        with kvikio.RemoteFile(ctx, bucket_name, object_name) as f:
            b = xp.zeros(shape=(end - start,), dtype=xp.int64)
            assert f.read(b, file_offset=start * a.itemsize) == b.nbytes
            xp.testing.assert_array_equal(a[start:end], b)

        with kvikio.RemoteFile.from_url(ctx, f"s3://{bucket_name}/{object_name}") as f:
            b = xp.zeros(shape=(end - start,), dtype=xp.int64)
            assert f.read(b, file_offset=start * a.itemsize) == b.nbytes
            xp.testing.assert_array_equal(a[start:end], b)


def test_remote_path_error(s3_base):
    bucket_name = "test_remote_path_error"
    with s3_context(s3_base=s3_base, bucket=bucket_name) as ctx:
        with pytest.raises(ValueError, match="No response body"):
            kvikio.RemoteFile.from_url(ctx, "s3://unknown-bucket/unknown-object")

        with pytest.raises(ValueError, match="No response body"):
            kvikio.RemoteFile.from_url(ctx, f"s3://{bucket_name}/unknown-object")

        with pytest.raises(ValueError, match="path must start with the S3 scheme"):
            kvikio.RemoteFile.from_url(ctx, f"s3:/{bucket_name}/")

        with pytest.raises(ValueError, match="path does not contain a bucket name"):
            kvikio.RemoteFile.from_url(ctx, "s3:///unknown-object")

        with pytest.raises(ValueError, match="path does not contain an object name"):
            kvikio.RemoteFile.from_url(ctx, f"s3://{bucket_name}/")

        with pytest.raises(ValueError, match="path does not contain an object name"):
            kvikio.RemoteFile.from_url(ctx, f"s3://{bucket_name}")
