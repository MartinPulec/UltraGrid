#
# CUDA JPEG2000 MAKEFILE
#

# Configure
TARGET := cuj2k.a
CFILES := j2k_encoder.c j2k_encoder_init.c j2k_timer.c
CXXFILES := j2k_encoder_progression.cpp
MODULES := preprocessor/preprocessor.a dwt/gpu/dwt.a quantizer/quantizer.a \
           ebcot/ebcot.a t2/t2.a rate_control/rate_control.a \
           compaction/compaction.a

# Include Common Makefile
include ../common.mk

# Headers for files
j2k_encoder.c.o: j2k_encoder.c j2k_encoder.h t2/t2.h t2/t2_type.h j2k_timer.h
j2k_encoder_init.c.o: j2k_encoder_init.c j2k_encoder_init.h j2k_encoder_progression.h
j2k_encoder_progression.cpp.o: j2k_encoder_progression.cpp j2k_encoder_progression.h
j2k_timer.c.o: j2k_timer.c j2k_timer.h
