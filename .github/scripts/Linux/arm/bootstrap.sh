#!/bin/sh -eu

BUILD_DIR=.
sudo apt install binfmt-support qemu qemu-user-static debootstrap
wget http://archive.raspbian.org/raspbian.public.key -O - | sudo apt-key add -q
mkdir -p $BUILD_DIR
sudo qemu-debootstrap --keyring=/etc/apt/trusted.gpg --arch armhf buster $BUILD_DIR http://mirrordirector.raspbian.org/raspbian/
chroot $BUILD_DIR sh -c 'apt install build-essential pkg-config autoconf automake libtool'
chroot $BUILD_DIR sh -c 'portaudio19-dev libsdl2-dev libglib2.0-dev libglew-dev libcurl4-openssl-dev freeglut3-dev libssl-dev libjack-dev libavcodec-dev libasound2-dev'
chroot $BUILD_DIR sh -c 'apt install desktop-file-utils git-core libfuse-dev libcairo2-dev cmake wget zsync' # to build appimagetool
chroot $BUILD_DIR sh -c 'git clone https://github.com/AppImage/AppImageKit.git && cd AppImageKit && ./build.sh && cd build && cmake -DAUXILIARY_FILES_DESTINATION= .. && make install'
