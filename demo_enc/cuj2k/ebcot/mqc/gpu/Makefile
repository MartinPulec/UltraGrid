#
# MQ-CODER GPU MAKEFILE
#

# Configure
TARGET := gpu.a
CUFILES := configuration.cu common.cu develop.cu final.cu \
           impr_renorm.cu impr_renorm_table.cu loop_unrolling.cu \
           original.cu prefix_sum.cu register.cu shared.cu

# Include Common Makefile
include ../../../../common.mk
OPTFLAGS := -O3

# Headers for files
configuration.cu.o: configuration.cu configuration.h
common.cu.o: common.cu common.h
develop.cu.o: develop.cu develop.h common.h configuration.h
final.cu.o: final.cu final.h common.h
loop_unrolling.cu.o: loop_unrolling.cu loop_unrolling.h common.h
original.cu.o: original.cu original.h common.h
impr_renorm.cu.o: impr_renorm.cu impr_renorm.h common.h
impr_renorm_table.cu.o: impr_renorm_table.cu impr_renorm_table.h common.h
prefix_sum.cu.o: prefix_sum.cu prefix_sum.h common.h
shared.cu.o: shared.cu shared.h common.h configuration.h
register.cu.o: register.cu register.h common.h configuration.h

