#!/bin/sh
# Compile the reflectivity CMAX kernel to a shared lib next to the radar package.
# Used by the Docker build stage; run manually for local dev if rustc is present.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$DIR/../radar/libcmax.so}"
rustc -C opt-level=3 -C target-cpu=native --crate-type=cdylib \
    "$DIR/cmax.rs" -o "$OUT"
echo "built $OUT"
