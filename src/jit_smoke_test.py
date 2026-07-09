"""Compile the exact JIT kernel that crashed the server (fused RoPE, bf16).

Run after `source scripts/toolchain_env.sh`:

    uv run python3 src/jit_smoke_test.py

Succeeds in a minute or so if the host compiler is set up correctly; fails
with the same nvcc/ninja error as the server otherwise. The result is cached
in ~/.cache/tvm-ffi, so the server won't need to recompile it.
"""

import torch

from sglang.jit_kernel.rope import _jit_fused_rope_module

if __name__ == "__main__":
    # Same parameters as the failing build dir:
    # sgl_kernel_jit_fused_rope_true_64_true_bf16_t_*
    module = _jit_fused_rope_module(True, 64, torch.bfloat16)
    print("JIT fused RoPE kernel compiled and loaded OK:", module)

    '''
    Output
    (sglang-play) bash-4.4$ uv run python3 src/jit_smoke_test.py 
    JIT fused RoPE kernel compiled and loaded OK: ffi.Module(imports_=())
    '''