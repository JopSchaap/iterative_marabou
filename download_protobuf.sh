#!/bin/bash

get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

curdir=$pwd
mydir="${0%/*}"
installdir=$(get_abs_filename $mydir)/protobuf-3.19.2/installed

# Compatibility with MacOS
if [[ $OSTYPE == 'darwin'* ]]; then
    export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
fi

cd $mydir
echo "downloading protobuf-3.19.2"
wget -q https://github.com/protocolbuffers/protobuf/releases/download/v3.19.2/protobuf-cpp-3.19.2.tar.gz -O protobuf-cpp-3.19.2.tar.gz
echo "unzipping protobuf-3.19.2"
tar -xzf protobuf-cpp-3.19.2.tar.gz # >> /dev/null
echo "installing protobuf-3.19.2"
cd protobuf-3.19.2
mkdir -p $installdir
# CXXFLAGS=-fPIC ./configure --disable-shared --prefix=$installdir --enable-fast-install
./configure "CXXFLAGS=-fPIC" "CFLAGS=-fPIC" --prefix=$installdir
make -j$(nproc)
make install

cd $curdir 