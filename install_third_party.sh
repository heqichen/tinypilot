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
        return 0
    fi
    wget -O "$dest_dir/$file_name" "$url"
    tar -xzf "$dest_dir/$file_name" -C "$dest_dir/output"
}

arch=$(uname -m)
DEST_DIR="$SCRIPT_DIR/third_party/ArmNN"
if [ "$arch" = "x86_64" ]; then
    ARMNN_URL="https://github.com/ARM-software/armnn/releases/download/v25.02/ArmNN-linux-x86_64.tar.gz"
    FILE_NAME="ArmNN.tar.gz"
    download_and_extract "$ARMNN_URL" "$DEST_DIR" "$FILE_NAME"
elif [ "$arch" = "aarch64" ]; then
    ARMNN_URL="https://github.com/ARM-software/armnn/releases/download/v25.02/MULTI_ISA-GCC11-ArmNN+ACL-linux-armv8a.tar.gz"
    FILE_NAME="ArmNN.tar.gz"
    download_and_extract "$ARMNN_URL" "$DEST_DIR" "$FILE_NAME"
else
    echo "Error: Unsupported architecture: $arch" >&2
    exit 1
fi

