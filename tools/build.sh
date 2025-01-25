#!/bin/bash

ARCHITECTURE=$1
LIB_DIR=$2

VERSION="1.0.0"

PACKAGE_DIR="dist/chisei_${VERSION}_${ARCHITECTURE}"
DEBIAN_DIR="${PACKAGE_DIR}/DEBIAN"

USR_DIR="${PACKAGE_DIR}/usr"
INCLUDE_DIR="${USR_DIR}/include"
BUILD_DIR="dist/build"
SO_FILE="${BUILD_DIR}/libchisei.so"

sudo apt install -y         \
    g++-riscv64-linux-gnu   \
    gcc-riscv64-linux-gnu   \
    g++-arm-linux-gnueabihf \
    gcc-arm-linux-gnueabihf 

case "$ARCHITECTURE" in
    amd64)
        CROSS_COMPILE=""
        ;;
    armhf)
        CROSS_COMPILE="arm-linux-gnueabihf-"
        ;;
    riscv64)
        CROSS_COMPILE="riscv64-linux-gnu-"
        ;;
    *)
        echo -e "\033[93m[-]\033[0m Unsupported architecture: $ARCHITECTURE"
        exit 1
        ;;
esac

mkdir -p "${DEBIAN_DIR}"
mkdir -p "${INCLUDE_DIR}/chisei"
mkdir -p "${USR_DIR}/lib/${LIB_DIR}"
mkdir -p "${BUILD_DIR}"

echo -e "\033[92m[+]\033[0m Building shared library for ${ARCHITECTURE}..."
if [ "$ARCHITECTURE" = "amd64" ]; then
    g++ -fPIC -shared -o "${SO_FILE}" -Iinclude src/chisei/*.cpp
else
    ${CROSS_COMPILE}g++ -fPIC -shared -o "${SO_FILE}" -Iinclude src/chisei/*.cpp
fi

cp -r include/chisei/* "${INCLUDE_DIR}/chisei/"
cp "${SO_FILE}" "${USR_DIR}/lib/${LIB_DIR}/"

cat <<EOF > "${DEBIAN_DIR}/control"
Package: chisei
Version: ${VERSION}
Section: libs
Priority: optional
Architecture: ${ARCHITECTURE}
Maintainer: Nathanne Isip <nathanneisip@gmail.com>
Description: chisei
 A lightweight, efficient, and user-friendly C++ library for creating,
 training, and deploying fully connected neural networks; designed with
 simplicity and performance in mind.
EOF

chmod 755 "${DEBIAN_DIR}"
chmod 755 "${USR_DIR}"
chmod 755 "${INCLUDE_DIR}"
chmod 755 "${USR_DIR}/lib/${LIB_DIR}"

dpkg-deb --build "${PACKAGE_DIR}" > /dev/null

rm -rf "${PACKAGE_DIR}"
rm -rf "${BUILD_DIR}"

echo -e "\033[92m[+]\033[0m Debian package for ${ARCHITECTURE} created successfully!"
