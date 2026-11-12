#!/usr/bin/env bash
# Configures and builds SimPart in build/ (Release). Extra arguments go to CMake.
#
# The compilers can be chosen with the usual CMake environment variables, e.g.
#   CUDACXX=/usr/local/cuda-12.3/bin/nvcc CXX=g++-12 CUDAHOSTCXX=g++-12 scripts/build.sh
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-$REPO_DIR/build}

cmake -S "$REPO_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$BUILD_DIR" -j "$(nproc)"
echo "built $BUILD_DIR/app/simpart"
