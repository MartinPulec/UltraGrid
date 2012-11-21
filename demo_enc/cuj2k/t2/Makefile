#
# T2 MAKEFILE
#

# Configure
TARGET := t2.a
CXXFILES := t2.cpp
MODULES := cpu/t2_cpu.a gpu/t2_gpu.a

# Include Common Makefile
include ../../common.mk

# Headers for files
t2.cpp.o: t2.cpp t2.h t2_type.h cpu/t2_cpu.h gpu/t2_gpu.h ../j2k_encoder.h

