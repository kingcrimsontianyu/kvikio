import kvikio
import kvikio.defaults
import subprocess
import os.path
import numpy as np
import time


class TestManager:
    def __init__(self):
        self.arena_dir = "/mnt/nvme_ubuntu_test"
        # self.arena_dir = "."
        self.filename = os.path.join(self.arena_dir, "parallel_io.bin")
        self.my_data = None
        self.Mi = 1024 * 1024
        self.num_elements = 1024 * self.Mi / 8
        self.num_threads = 8
        self.task_size_B = 128 * 1024

    def _drop_file_cache(self):
        full_command = "sudo /sbin/sysctl vm.drop_caches=3"
        subprocess.run(full_command.split())

    def _do_it(self):
        # Write
        # [start, stop)
        self.my_data = np.arange(0, self.num_elements, dtype=np.float64)
        self.my_data.tofile(self.filename)

        # Read
        f = kvikio.CuFile(self.filename, "r")
        buf = np.empty_like(self.my_data)

        repetition = 4
        elapsed_total = 0

        for i in range(repetition):
            self._drop_file_cache()

            print(f"--> {i}, ", end="")
            start = time.time()

            fut = f.pread(buf)
            fut.get()

            elapsed = time.time() - start
            elapsed_total += elapsed

            bandwidth = self.num_elements * 8 / elapsed / self.Mi
            print("time: {:.2f} s, bandwidth = {:.2f} MiB/s".format(
                elapsed, bandwidth))

        bandwidth_total = self.num_elements * 8 * repetition / elapsed_total / self.Mi
        print(
            "--> time: {:.2f} s, bandwidth: {:.2f} MiB/s".format(elapsed_total, bandwidth_total))

    def do_it(self):
        with kvikio.defaults.set_num_threads(self.num_threads):
            with kvikio.defaults.set_task_size(self.task_size_B):
                self._do_it()


if __name__ == "__main__":
    tm = TestManager()
    tm.do_it()
