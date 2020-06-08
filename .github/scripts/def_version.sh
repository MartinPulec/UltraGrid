#!/bin/sh

if expr $GITHUB_REF : 'refs/heads/release/'; then
  VERSION=${GITHUB_REF#refs/heads/release/}
  TAG=v$VERSION
else
  VERSION=continuous
  TAG=continuous
fi

echo "::set-env name=VERSION::$VERSION"
echo "::set-env name=TAG::$TAG"
