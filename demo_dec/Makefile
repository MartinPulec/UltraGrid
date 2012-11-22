#
# CUDA JPEG2000 DECODER MAKEFILE
#

# Configure
TARGET := demo_dec.a
CU_FILES := $(wildcard *.cu)
CPP_FILES := $(wildcard *.cpp)
H_FILES := $(wildcard *.h)
O_FILES := $(sort $(addsuffix .o,$(CU_FILES) $(CPP_FILES)))


# OPTFLAGS := -O0 -g
OPTFLAGS := -O3
INCLDIRS := -I/usr/local/cuda/include/
COMPILER_OPTS := -ansi -Wextra -Wno-long-long

# Dependencies for sources
%.cpp.o: %.cpp $(H_FILES)
	@echo Compiling $<
	@g++ $(OPTFLAGS) $(INCLDIRS) $(COMPILER_OPTS) -pedantic -c -o $@ $<

%.cu.o: %.cu $(H_FILES)
	@echo Compiling $<
	@nvcc $(OPTFLAGS) $(INCLDIRS) --ptxas-options=-v -arch sm_20 -Xcompiler "$(COMPILER_OPTS)" -c $< -o $@

%.cu:    # this avoids implicit rule, which causes circular dependencies


$(TARGET): $(O_FILES)
	@echo Creating archive $(TARGET)
	@ar rcs $(TARGET) $(O_FILES)

clean:
	rm -f $(TARGET) $(O_FILES)
