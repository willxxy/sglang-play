#!/usr/bin/env bash
# One-time, no-sudo setup of a modern host compiler for SGLang's JIT kernels.
#
# Why: sglang.jit_kernel compiles CUDA kernels at runtime with -std=c++20.
# nvcc uses the system gcc as host compiler; on RHEL8 that is GCC 8.5, which
# has no C++20 and no <version> header -> "fatal error: version: No such file
# or directory". This installs GCC 13 (what sglang tests against) from
# conda-forge into ./.toolchain — entirely in user space.
#
# Usage:
#   bash scripts/setup_toolchain.sh          # one time
#   source scripts/toolchain_env.sh          # in every shell that runs sglang

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${SGLANG_TOOLCHAIN_PREFIX:-$REPO_DIR/.toolchain}"
MAMBA="$PREFIX/micromamba"
ENV_DIR="$PREFIX/gcc13"

mkdir -p "$PREFIX"

if [ ! -x "$MAMBA" ]; then
    echo "==> Downloading micromamba (static binary, no install needed)..."
    if ! curl -fsSL -o "$MAMBA" \
        "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64"; then
        echo "==> GitHub download failed, trying micro.mamba.pm tarball..."
        curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-64/latest" \
            | tar -xj -C "$PREFIX" --strip-components=1 bin/micromamba
    fi
    chmod +x "$MAMBA"
fi
"$MAMBA" --version >/dev/null

if [ ! -e "$ENV_DIR/bin/g++" ] && [ ! -e "$ENV_DIR/bin/x86_64-conda-linux-gnu-g++" ]; then
    echo "==> Installing GCC 13 toolchain into $ENV_DIR (~1.5 GB, a few minutes)..."
    "$MAMBA" create -y -p "$ENV_DIR" -c conda-forge --no-rc "gcc=13" "gxx=13"
fi

# Resolve the g++ binary name (conda-forge ships prefixed binaries; the
# gcc/gxx meta packages usually add unprefixed symlinks too).
CXX_BIN=""
for cand in "$ENV_DIR/bin/g++" "$ENV_DIR/bin/x86_64-conda-linux-gnu-g++"; do
    if [ -x "$cand" ]; then CXX_BIN="$cand"; break; fi
done
if [ -z "$CXX_BIN" ]; then
    echo "ERROR: g++ not found under $ENV_DIR/bin after install" >&2
    exit 1
fi

echo "==> Self-test: compiling a C++20 snippet that includes <version>..."
TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT
cat > "$TMPD/t.cpp" <<'EOF'
#include <version>
#if !defined(__cpp_lib_source_location)
#include <cstdio>
#endif
int main() { return 0; }
EOF
"$CXX_BIN" -std=c++20 "$TMPD/t.cpp" -o "$TMPD/t"
echo "    OK: $("$CXX_BIN" --version | head -1)"

if command -v nvcc >/dev/null 2>&1; then
    echo "==> Self-test: nvcc with this host compiler..."
    cat > "$TMPD/t.cu" <<'EOF'
#include <version>
__global__ void k() {}
int main() { return 0; }
EOF
    nvcc -std=c++20 -ccbin "$CXX_BIN" -c "$TMPD/t.cu" -o "$TMPD/t.o"
    echo "    OK: nvcc accepts GCC 13 as host compiler with -std=c++20"
else
    echo "==> nvcc not on PATH here; skipping nvcc self-test."
fi

echo
echo "Done. Before running sglang, in every shell do:"
echo "    source scripts/toolchain_env.sh"
echo "and clear any failed JIT builds once:"
echo "    rm -rf ~/.cache/tvm-ffi"
