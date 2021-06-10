# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda
INCLUDE_DIR = /usr/include
LIB_DIR = /usr/lib/aarch64-linux-gnu
TEGRA_LIB_DIR = /usr/lib/aarch64-linux-gnu/tegra

# For hardfp
#LIB_DIR = /usr/lib/arm-linux-gnueabihf
#TEGRA_LIB_DIR = /usr/lib/arm-linux-gnueabihf/tegra

OSUPPER = $(shell uname -s 2>/dev/null | tr "[:lower:]" "[:upper:]")
OSLOWER = $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

GCC ?= g++
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(GCC)

# internal flags
NVCCFLAGS   := --shared
CCFLAGS     := -fPIC
LDFLAGS     :=

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?=

override abi := aarch64
LDFLAGS += --dynamic-linker=/lib/ld-linux-aarch64.so.1

# For hardfp
#override abi := gnueabihf
#LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
#CCFLAGS += -mfloat-abi=hard

ifeq ($(ARMv7),1)
NVCCFLAGS += -target-cpu-arch ARM
ifneq ($(TARGET_FS),)
CCFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += -rpath-link=$(TARGET_FS)/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/$(abi)-linux-gnu

# For hardfp
#LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)

endif
endif

# Debug build flags
dbg = 0
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      TARGET := debug
else
      TARGET := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I./
LIBRARIES := -L$(LIB_DIR) -lEGL -lGLESv2
LIBRARIES += -L$(TEGRA_LIB_DIR) -lcuda -lrt

################################################################################

# CUDA code generation flags
ifneq ($(OS_ARCH),armv7l)
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
endif
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM32    := -gencode arch=compute_32,code=sm_32
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SMXX    := -gencode arch=compute_50,code=compute_50
GENCODE_SM53    := -gencode arch=compute_53,code=sm_53
GENCODE_SM62    := -gencode arch=compute_62,code=sm_62
GENCODE_SM72    := -gencode arch=compute_72,code=sm_72
GENCODE_SM_PTX  := -gencode arch=compute_72,code=compute_72
ifeq ($(OS_ARCH),armv7l)
GENCODE_FLAGS   ?= $(GENCODE_SM32)
else
GENCODE_FLAGS   ?= $(GENCODE_SM30) $(GENCODE_SM32) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM53) $(GENCODE_SM62) $(GENCODE_SM72) $(GENCODE_SMXX) $(GENCODE_SM_PTX)
endif

# Target rules
all: build

build: libnvsample_cudaprocess.so

nvsample_cudaprocess.o : nvsample_cudaprocess.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

libnvsample_cudaprocess.so : nvsample_cudaprocess.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

clean:
	rm libnvsample_cudaprocess.so nvsample_cudaprocess.o

clobber: clean
