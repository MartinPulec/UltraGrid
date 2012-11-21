#
# DWT Makefile
#

# Configure
TARGET := dwt.a
CFILES := dwt.c
CUFILES := fdwt53.cu rdwt53.cu fdwt97.cu rdwt97.cu copyback.cu

# Include Common Makefile
include ../../../common.mk

# Redefine optimization.
OPTFLAGS := -O3
NVCCFLAGS += --ptxas-options=-v

# Dependencies for source files
fdwt53.cu.o: fdwt53.cu dwt.h common.h transform_buffer.h io.h
rdwt53.cu.o: rdwt53.cu dwt.h common.h transform_buffer.h io.h
fdwt97.cu.o: fdwt97.cu dwt.h common.h transform_buffer.h io.h
rdwt97.cu.o: rdwt97.cu dwt.h common.h transform_buffer.h io.h
copyback.cu.o: copyback.cu dwt.h
dwt.c.o: dwt.c dwt.h

