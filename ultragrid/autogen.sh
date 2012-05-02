#! /bin/sh

udt_init() {
        cd ../udt4
        make clean
        if [ `uname` = "Linux" ]; then
                if [ `uname -p` = "x86_64" ]; then
                        make -e arch=AMD64
                else
                        exit
                        make
                fi
        else
                make -e os=OSX arch=IA32
        fi
        cd -
}

libgpujpeg_init() {
        cd ../libgpujpeg
        make clean
        make
        cd -
}


srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

ORIGDIR=`pwd`
cd $srcdir

#udt_init
#libgpujpeg_init

aclocal && \
autoheader && \
autoconf && \
$srcdir/configure $@
STATUS=$?

make clean

cd $ORIGDIR

exit $STATUS

