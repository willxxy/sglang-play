export MAMBA_ROOT_PREFIX="$PWD/.toolchain/mamba-root"
./.toolchain/micromamba install -y -p ./.toolchain/gcc13 -c conda-forge libprotobuf
export PROTOC="$PWD/.toolchain/gcc13/bin/protoc"