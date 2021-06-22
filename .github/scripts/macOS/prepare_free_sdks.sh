#!/bin/sh -eux

XIMEA_DOWNLOAD_URL=https://www.ximea.com/downloads/recent/XIMEA_OSX_SP.dmg

if [ $# -ge 1 ] && [ x$1 = x-e ]; then
        curl -LI $XIMEA_DOWNLOAD_URL | grep -i '^etag' | sed 's/.*"\(.*\)".*/\1/'
        return 0
fi

rm -rf /var/tmp/sdks-free
mkdir -p /var/tmp/sdks-free
cd /var/tmp/sdks-free
curl -S -LO $XIMEA_DOWNLOAD_URL

