#!/usr/bin/env bash
# build.sh  –  compile the CUDA kernel and then the Rust binary
set -e

echo "=== 1/2  Compiling CUDA kernel to PTX ==="
nvcc -ptx sha1_kernel.cu -o sha1_kernel.ptx -arch=sm_121 --use_fast_math -O3

mkdir -p target/release
cp sha1_kernel.ptx target/release/

echo "=== 2/2  Building Rust binary ==="
cargo build --release

cp sha1_kernel.ptx target/release/

echo ""
echo "=== Done — run with: ==="
echo "cd target/release && ./rust_sha1_cracker"