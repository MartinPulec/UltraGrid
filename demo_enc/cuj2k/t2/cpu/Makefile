#
# T2 CPU MAKEFILE
#

# Configure
TARGET := t2_cpu.a
CXXFILES := t2_cpu.cpp t2_cpu_tag_tree.cpp

# Include Common Makefile
include ../../../common.mk

# Headers for files
t2_cpu.cpp.o: t2_cpu.cpp t2_cpu.h t2_cpu_output.h t2_cpu_tag_tree.h t2_cpu_codewords.h ../../j2k_encoder.h
t2_cpu_tag_tree.cpp.o: t2_cpu_tag_tree.cpp t2_cpu_tag_tree.h t2_cpu_output.h
