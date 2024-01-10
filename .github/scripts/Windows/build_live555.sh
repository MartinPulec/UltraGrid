#!/bin/bash -ex

export CC=gcc
export CXX=g++

cd /c
rm -rf live555
git clone https://github.com/xanview/live555/
cd live555
./genMakefiles mingw

# ensure binutils ld is used (not lld)
pacman -Sy --noconfirm binutils
PATH=/usr/bin:$PATH

make -j "$(nproc)" CPLUSPLUS_COMPILER="c++ -DNO_GETIFADDRS -DNO_OPENSSL"
pacman -Rs --noconfirm binutils

