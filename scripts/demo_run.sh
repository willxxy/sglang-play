#!/usr/bin/env bash
# Use the user-space GCC 13 for runtime JIT kernel compilation, if installed
# (see "JIT host compiler fix" in README.md).
if [ -f "$(dirname "${BASH_SOURCE[0]}")/toolchain_env.sh" ]; then
    source "$(dirname "${BASH_SOURCE[0]}")/toolchain_env.sh"
fi

# sglang serve --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --port 30000 --disable-cuda-graph --disable-piecewise-cuda-graph --sampling-backend pytorch --attention-backend triton --disable-overlap-schedule

sglang serve --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --port 30000