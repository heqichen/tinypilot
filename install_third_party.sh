#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/third_party"

download_and_extract() {
    local url="$1"
    local dest_dir="$2"
    local file_name="$3"
    mkdir -p "$dest_dir/output"
    if [ -f "$dest_dir/$file_name" ]; then
        echo "$dest_dir/$file_name already exists, skip download and extract."
    else
        wget -O "$dest_dir/$file_name" "$url"
        tar -xzf "$dest_dir/$file_name" -C "$dest_dir/output"
    fi
}

build_googletest() {
    local src_dir="$1"
    local install_dir="$2"
    local build_dir="$src_dir/build"
    echo "src dir = $src_dir" 
    echo "install dir = $install_dir"
    echo "build dir = $build_dir"

    mkdir -p "$build_dir"
    mkdir -p "$install_dir"

    cd "$build_dir"
    cmake -DCMAKE_INSTALL_PREFIX:PATH="$install_dir" ..
    make -j"$(nproc)"
    make install
    cd -
}

arch=$(uname -m)
DEST_DIR="$SCRIPT_DIR/third_party/ArmNN"
if [ "$arch" = "x86_64" ]; then
    ARTIFACT_URL="https://github.com/ARM-software/armnn/releases/download/v25.02/ArmNN-linux-x86_64.tar.gz"
    FILE_NAME="ArmNN.tar.gz"
    download_and_extract "$ARTIFACT_URL" "$DEST_DIR" "$FILE_NAME"
elif [ "$arch" = "aarch64" ]; then
    ARTIFACT_URL="https://github.com/ARM-software/armnn/releases/download/v25.02/MULTI_ISA-GCC11-ArmNN+ACL-linux-armv8a.tar.gz"
    FILE_NAME="ArmNN.tar.gz"
    download_and_extract "$ARTIFACT_URL" "$DEST_DIR" "$FILE_NAME"
else
    echo "Error: Unsupported architecture: $arch" >&2
    exit 1
fi

ARTIFACT_URL="https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz"
FILE_NAME="googletest-1.17.0.tar.gz"
DEST_DIR="$SCRIPT_DIR/third_party/googletest"
INSTALL_DIR="$DEST_DIR/install"
download_and_extract "$ARTIFACT_URL" "$DEST_DIR" "$FILE_NAME"
build_googletest "$DEST_DIR/output/googletest-1.17.0" "$INSTALL_DIR"

ARTIFACT_URL="https://github.com/PINTO0309/onnx2tf/archive/refs/tags/1.28.2.tar.gz"
FILE_NAME="onnx2tf-1.28.2.tar.gz"
DEST_DIR="$SCRIPT_DIR/third_party/onnx2tf"
download_and_extract "$ARTIFACT_URL" "$DEST_DIR" "$FILE_NAME"


