#!/usr/bin/env bash
# Build & install SGLang 0.5.14 from source on a glibc-2.28 host (RHEL8/CentOS8),
# where the prebuilt manylinux_2_34 wheels won't install. See INSTALL_0.5.14.md.
#
# From 0.5.11 on, sglang bundles a compiled Rust/PyO3 gRPC extension, so its wheels
# are manylinux_2_34 (glibc >= 2.34) with no sdist. Building locally links sglang
# against THIS host's glibc 2.28. Every dependency already has a glibc-2.28 wheel,
# so only sglang itself needs building.
#
# Prereqs (one-time, see INSTALL_0.5.14.md):
#   * an ACTIVE uv venv (Python 3.11)          <- source .venv-0514/bin/activate
#   * rustc/cargo >= 1.85 (edition 2024)        <- rustup update stable
#   * protoc (tonic-build invokes it)           <- micromamba install protobuf
#
# Usage:
#   source .venv-0514/bin/activate
#   bash scripts/build_sglang_0514.sh
set -euo pipefail

SGLANG_REF="${SGLANG_REF:-v0.5.14}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${SGLANG_SRC_DIR:-$REPO_DIR/.sglang-src}"
CU_INDEX="${SGLANG_CU_INDEX:-https://docs.sglang.ai/whl/cu129/}"
TORCH_INDEX="${TORCH_CU_INDEX:-https://download.pytorch.org/whl/cu129}"

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- 0. environment checks ------------------------------------------------
command -v uv    >/dev/null || die "uv not found on PATH"
command -v git   >/dev/null || die "git not found on PATH"
command -v cargo >/dev/null || die "cargo/rustc not found. Install rustup; need rustc >= 1.85 (edition 2024)."

# protoc (tonic-build shells out to it)
if ! command -v protoc >/dev/null && ! { [ -n "${PROTOC:-}" ] && [ -x "${PROTOC:-}" ]; }; then
    die "protoc not found (tonic-build needs it). Install it (see INSTALL_0.5.14.md), then re-run."
fi

# driver advisory (non-fatal): 0.5.14's CUDA-13 deps want driver >= 580
if command -v nvidia-smi >/dev/null; then
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    dmaj="${drv%%.*}"
    if [ -n "${dmaj:-}" ] && [ "$dmaj" -lt 580 ]; then
        log "WARNING: driver $drv (< 580). 0.5.14 pulls CUDA-13 deps (cuda-python>=13, cutlass-dsl[cu13])."
        printf '    The build/install will still complete, but cu13 kernels may fail at runtime.\n'
        printf '    If the JIT smoke test or serve errors on CUDA, bump the driver to >= 580.\n'
    fi
fi

# --- 1. fetch source ------------------------------------------------------
if [ ! -d "$SRC_DIR/.git" ]; then
    log "Cloning sglang into $SRC_DIR"
    git clone https://github.com/sgl-project/sglang.git "$SRC_DIR"
fi
log "Checking out $SGLANG_REF"
git -C "$SRC_DIR" fetch --tags --quiet
git -C "$SRC_DIR" checkout --quiet "$SGLANG_REF"

# --- 2. build the wheel with the SYSTEM compiler --------------------------
# Clear any gcc13/JIT overrides for the build only, so the Rust/C extension links
# against the base system libs (glibc 2.28) and stays portable. gcc13 is for the
# runtime JIT step later, not for this build.
log "Building sglang $SGLANG_REF wheel (system compiler; Rust + protoc)"
uv pip install --quiet --upgrade build wheel
( cd "$SRC_DIR/python" \
  && env -u CC -u CXX -u CFLAGS -u CXXFLAGS -u LDFLAGS -u NVCC_CCBIN -u NVCC_PREPEND_FLAGS \
        python -m build --wheel --outdir "$SRC_DIR/dist" )

WHEEL="$(ls -t "$SRC_DIR"/dist/sglang-*.whl 2>/dev/null | head -1 || true)"
[ -n "$WHEEL" ] || die "wheel build produced no sglang-*.whl in $SRC_DIR/dist"
log "Built: $WHEEL"

# --- 3. install wheel + deps ---------------------------------------------
log "Installing sglang wheel (+ deps; prerelease allowed for flash-attn-4 beta)"
uv pip install --prerelease=allow "$WHEEL"

log "Re-pinning cu129 binaries (torch / sglang-kernel / sgl-deep-gemm)"
uv pip install --force-reinstall torch==2.11.0 torchaudio==2.11.0 torchvision --index-url "$TORCH_INDEX"
uv pip install --force-reinstall sglang-kernel==0.4.4 --index-url "$CU_INDEX"
uv pip install --force-reinstall "sgl-deep-gemm==0.1.3" --index-url "$CU_INDEX" --no-deps

# --- 4. clear stale JIT cache --------------------------------------------
log "Clearing ~/.cache/tvm-ffi (flashinfer/xgrammar/transformers all changed)"
rm -rf "$HOME/.cache/tvm-ffi"

# --- 5. extra project deps (llm-augment) ---------------------------------
# Installed LAST, on top of the pinned sglang stack. requirements.txt holds
# only lower bounds, so uv leaves the cu129 torch / sglang-kernel / sgl-deep-gemm
# pins from step 3 untouched (they already satisfy the bounds). sglang stays
# highest priority; these deps go on top.
REQ_FILE="$REPO_DIR/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    log "Installing extra project deps from requirements.txt (sglang pins kept)"
    uv pip install -r "$REQ_FILE"
fi

cat <<'NEXT'

Done building & installing sglang 0.5.14 (+ requirements.txt on top).

Next (the runtime JIT uses your gcc13 toolchain, NOT the system compiler above):
  bash scripts/setup_toolchain.sh          # one-time, if not already done
  source scripts/toolchain_env.sh          # in every shell that runs sglang
  uv run python3 src/check_environment.py  # confirm  sglang: 0.5.14
  uv run python3 src/jit_smoke_test.py     # must print OK
  bash scripts/demo_run.sh
NEXT