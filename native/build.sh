#!/bin/sh
# Build the native kernels into a shared lib next to the radar package.
# Used by the Docker build stage; run manually for local dev if cargo is present.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$DIR/../radar/libradar_native.so}"
( cd "$DIR" && cargo build --release )
cp "$DIR/target/release/libradar_native.so" "$OUT"
echo "built $OUT"
