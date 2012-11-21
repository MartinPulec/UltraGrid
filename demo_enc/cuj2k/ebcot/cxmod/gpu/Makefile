#
# Context modeller Makefile
#

# Configure
TARGET := cxmod.a
CUFILES := cxmod.cu

# Include Common Makefile
include ../../../../common.mk

# Redefine optimization.
#OPTFLAGS := -O3 --ptxas-options=-v -keep
OPTFLAGS := -O3 --ptxas-options=-v

# Dependencies for files
cxmod.cu.o: cxmod.cu \
    ../../../j2k_encoder.h \
    cxmod_bitplane_init.h \
    cxmod_border_init.h \
    cxmod_coding.h \
    cxmod_device_types.h \
    cxmod_finalize_precomp.h \
    cxmod_interface.h \
    cxmod_kernel.h \
    cxmod_loading.h \
    cxmod_lut_interfaces.h \
    cxmod_prefix_sum.h \
    cxmod_reduction.h \
    cxmod_sc_precomp.h \
    cxmod_specialized.h \
    cxmod_spp_sigma_precomp.h \
    cxmod_util.h \
    lookup_tables/cxmod_sc_lut.h \
    lookup_tables/cxmod_zc_mrc_hh_lut.h \
    lookup_tables/cxmod_zc_mrc_hl_lut.h \
    lookup_tables/cxmod_zc_mrc_ll_lh_lut.h
    

