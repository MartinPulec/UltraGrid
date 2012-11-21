#
# EBCOT MAKEFILE
#

# Configure
TARGET := mqc.a
CXXFILES := mqc.cpp
CUFILES := gpu/common.cu gpu/final.cu

# Include Common Makefile
include ../../../common.mk

# Headers for files
mqc.cpp.o: mqc.cpp mqc.h
gpu/common.cu.o: gpu/common.cu gpu/common.h
gpu/final.cu.o: gpu/final.cu gpu/final.h


# Explicit cleanup
clean: localclean

localclean:
	rm -f `find . -name '*.o'`
	rm -f `find . -name '*.a'`
