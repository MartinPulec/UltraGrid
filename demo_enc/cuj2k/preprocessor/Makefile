#
# PREPROCESSOR MAKEFILE
#

# Configure
TARGET := preprocessor.a
CFILES := preprocessor.c
CUFILES := preprocessor_kernel.cu preprocessor_fmt.cu

# Include Common Makefile
include ../../common.mk

# Headers for files
preprocessor.c.o: preprocessor.c preprocessor.h
preprocessor_kernel.cu.o: preprocessor_kernel.cu preprocessor.h preprocessor_ct.h preprocessor_store.h
preprocessor_fmt.cu.o: preprocessor_fmt.cu preprocessor_fmt.h
