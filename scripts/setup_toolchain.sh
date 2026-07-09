#!/usr/bin/env bash
# We have older GCC (8.5) so we need to download and override it with a different version via micromamba
# One time usage needed only

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${SGLANG_TOOLCHAIN_PREFIX:-$REPO_DIR/.toolchain}"
MAMBA="$PREFIX/micromamba"
ENV_DIR="$PREFIX/gcc13"

# Keep micromamba's package cache / root prefix inside the toolchain dir too,
# so nothing lands in $HOME.
export MAMBA_ROOT_PREFIX="$PREFIX/mamba-root"

# Pinned micromamba release with checksum verification.
# Tags: https://github.com/mamba-org/micromamba-releases/releases
MM_VERSION="${MICROMAMBA_VERSION:-2.8.1-0}"
MM_URL="https://github.com/mamba-org/micromamba-releases/releases/download/${MM_VERSION}/micromamba-linux-64"

mkdir -p "$PREFIX"

if [ ! -x "$MAMBA" ]; then
    echo "==> Downloading micromamba ${MM_VERSION} (static binary, no install needed)..."
    if ! curl -fsSL -o "$MAMBA.tmp" "$MM_URL"; then
        echo "ERROR: download failed for $MM_URL" >&2
        echo "Pick a release tag from https://github.com/mamba-org/micromamba-releases/releases" >&2
        echo "and re-run with: MICROMAMBA_VERSION=<tag> bash scripts/setup_toolchain.sh" >&2
        exit 1
    fi

    echo "==> Verifying SHA256..."
    if [ -n "${MICROMAMBA_SHA256:-}" ]; then
        EXPECTED_SHA="$MICROMAMBA_SHA256"
    else
        EXPECTED_SHA="$(curl -fsSL "$MM_URL.sha256" | awk '{print $1}')"
    fi
    echo "$EXPECTED_SHA  $MAMBA.tmp" | sha256sum -c -
    chmod +x "$MAMBA.tmp"
    mv "$MAMBA.tmp" "$MAMBA"
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
echo "and clear the failed JIT builds once:"
echo "    rm -rf ~/.cache/tvm-ffi/sgl_kernel_jit_*"
