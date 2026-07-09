#!/usr/bin/env bash
# Use the user-space GCC 13 for runtime JIT kernel compilation, if installed
# (see "JIT host compiler fix" in README.md).
if [ -f "$(dirname "${BASH_SOURCE[0]}")/toolchain_env.sh" ]; then
    source "$(dirname "${BASH_SOURCE[0]}")/toolchain_env.sh"
fi

# Exempt loopback from the corporate proxy: sglang's warmup polls its own
# http://127.0.0.1:30000/model_info via requests, which honors http_proxy —
# the McAfee gateway answers 502 and sglang kills itself after 120 tries
# (see "Corporate proxy vs localhost fix" in README.md).
export no_proxy="localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

# sglang serve --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --port 30000 --disable-cuda-graph --disable-piecewise-cuda-graph --sampling-backend pytorch --attention-backend triton --disable-overlap-schedule

sglang serve --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --port 30000