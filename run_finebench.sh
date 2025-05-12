#!/bin/bash

# Generate a random master port to avoid conflicts
MASTER_PORT=$(shuf -i 10000-65535 -n 1)
echo "Starting torchrun with master port $MASTER_PORT"


# Make sure the process only sees the GPUs we specified
export CUDA_VISIBLE_DEVICES=1
torchrun --master_port $MASTER_PORT --nproc-per-node=1 run.py --config finebench_config.json --work-dir output_v2.1_sub --reuse