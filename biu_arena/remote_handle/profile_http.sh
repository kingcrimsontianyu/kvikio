#!/usr/bin/env bash

# Run this script with sudo in order to collect performance counters

my_python=/home/coder/.local/share/venvs/rapids/bin/python
my_program=test_http.py

nsys profile \
-o http \
-t nvtx,cuda,osrt \
-f true \
-b none \
-d 10 \
--gpu-metrics-devices=0 \
--cpuctxsw=none \
--gpuctxsw=true \
--cuda-memory-usage=true \
-e KVIKIO_COMPAT_MODE=ON,KVIKIO_NTHREADS=4 \
$my_python $my_program

