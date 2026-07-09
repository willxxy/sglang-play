# Source this file (do not execute) before running sglang:
#     source scripts/toolchain_env.sh
#
# Points SGLang's runtime JIT compilation (tvm-ffi + nvcc) at the user-space
# GCC 13 installed by scripts/setup_toolchain.sh:
#   - CXX/CC:              used by tvm-ffi for C++ compiles and linking
#   - NVCC_PREPEND_FLAGS:  injects -ccbin so nvcc uses GCC 13 as host compiler
#                          instead of the system GCC 8.5 (documented nvcc env
#                          var, CUDA >= 11.5)
#   - LD_LIBRARY_PATH:     the JIT-built .so needs the matching newer
#                          libstdc++.so.6 at runtime

_sglang_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_sglang_tc="${SGLANG_TOOLCHAIN_PREFIX:-$_sglang_env_dir/.toolchain}/gcc13"

_sglang_cxx=""
for _cand in "$_sglang_tc/bin/g++" "$_sglang_tc/bin/x86_64-conda-linux-gnu-g++"; do
    if [ -x "$_cand" ]; then _sglang_cxx="$_cand"; break; fi
done
_sglang_cc=""
for _cand in "$_sglang_tc/bin/gcc" "$_sglang_tc/bin/x86_64-conda-linux-gnu-gcc"; do
    if [ -x "$_cand" ]; then _sglang_cc="$_cand"; break; fi
done

if [ -z "$_sglang_cxx" ] || [ -z "$_sglang_cc" ]; then
    echo "Toolchain not found under $_sglang_tc — run: bash scripts/setup_toolchain.sh" >&2
else
    export CXX="$_sglang_cxx"
    export CC="$_sglang_cc"
    export NVCC_PREPEND_FLAGS="-ccbin $CXX"
    export LD_LIBRARY_PATH="$_sglang_tc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "SGLang JIT host compiler: $CXX"
fi

unset _sglang_env_dir _sglang_tc _sglang_cxx _sglang_cc _cand
