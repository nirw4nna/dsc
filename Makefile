CXX			=	g++
NVCC		=	nvcc
HIPCC		=	hipcc

HIPCCFLAGS	=	-std=c++20 -I$(ROCM)/include -I./dsc/include/ --offload-arch=native -Wall -Wextra -Wformat \
                -Wcast-qual -Wcast-align -Wstrict-aliasing -Wpointer-arith -Wunused -Wdouble-promotion \
                -Wno-missing-braces -Wcast-align -fno-exceptions -fno-rtti
NVCCFLAGS	=	-std=c++20 -I$(CUDA)/include -I./dsc/include/ -ccbin=$(CXX) -arch=native \
				-forward-unknown-opts -Wall -Wextra -Wformat -Wnoexcept  \
                -Wcast-qual -Wcast-align -Wstrict-aliasing -Wpointer-arith -Wunused -Wdouble-promotion \
                -Wlogical-op -Wcast-align -fno-exceptions -fno-rtti
CXXFLAGS	=	-std=c++20 -I./dsc/include/ -Wall -Wextra -Wformat -Wnoexcept  \
 				-Wcast-qual -Wcast-align -Wstrict-aliasing -Wpointer-arith -Wunused -Wdouble-promotion \
 				-Wlogical-op -Wcast-align -fno-exceptions -fno-rtti -pthread
LDFLAGS		=	-lm -fuse-ld=mold

UNAME_M		:=	$(shell uname -m)
UNAME_S		:=	$(shell uname -s)

ifdef DSC_GPU
	# Try to detect the GPU vendor based on the available compiler
	DSC_CUDA := $(shell which $(NVCC) 2>/dev/null)
	DSC_HIP := $(shell which $(HIPCC) 2>/dev/null)
	ifdef DSC_CUDA
		ifneq ($(wildcard /opt/cuda),)
			CUDA ?= /opt/cuda
		else
			CUDA ?= /usr/local/cuda
		endif
	else
		# Check for HIP only if CUDA is not defined
		ifdef DSC_HIP
			ifneq ($(wildcard /opt/rocm),)
				ROCM ?=	/opt/rocm
			else
				ROCM ?=	/usr/local/rocm
			endif
		endif
	endif
endif

# Make sure only one GPU platform is defined
ifdef DSC_CUDA
	ifdef DSC_HIP
		$(error ERROR: both DSC_CUDA and DSC_HIP are defined - this is not supported)
	endif
endif

ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64))
	# Use all available CPU extensions, x86 only
	CXXFLAGS	+= 	-march=native -mtune=native
endif

ifndef DSC_LOG_LEVEL
	ifdef DSC_FAST
		DSC_LOG_LEVEL	:=	1
	else
		DSC_LOG_LEVEL	:=	0
	endif
endif

CXXFLAGS	+=	-DDSC_LOG_LEVEL=$(DSC_LOG_LEVEL)
NVCCFLAGS	+=	-DDSC_LOG_LEVEL=$(DSC_LOG_LEVEL)
HIPCCFLAGS	+=	-DDSC_LOG_LEVEL=$(DSC_LOG_LEVEL)

ifdef DSC_FAST
	# -Ofast turns on all the unsafe math optimizations, including -ffinite-math-only this is an issue when testing
	# because Inf and NaN have different meaning but will be treated as equals when using -ffinite-math-only.
	# When inferencing assuming only finite numbers is correct but since it's doesn't actually hurt performance
	# let's keep this flag so we can run our tests without worrying about denormal numbers.
	CXXFLAGS	+=	-Ofast -fno-finite-math-only -ffp-contract=fast -funroll-loops -flto=auto -fuse-linker-plugin
	NVCCFLAGS	+=	-O3
	HIPCCFLAGS	+=	-O3
else
	CXXFLAGS	+=	-O0 -fno-omit-frame-pointer -g
	NVCCFLAGS	+=	-O0 -fno-omit-frame-pointer -g -G
	HIPCCFLAGS	+=	-O0 -fno-omit-frame-pointer -g
endif

ifdef DSC_TRACING
	CXXFLAGS	+=	-DSC_TRACING=1
	NVCCFLAGS	+=	-DSC_TRACING=1
	HIPCCFLAGS	+=	-DSC_TRACING=1
endif

# If we are not compiling the shared object and are in debug mode then run in ASAN mode
ifeq ($(MAKECMDGOALS),shared)
	CXXFLAGS	+=	-fPIC
	NVCCFLAGS	+=	-fPIC
	HIPCCFLAGS	+=	-fPIC
endif

GPU_SRCS	:=	$(wildcard dsc/src/gpu/*.cpp)
GPU_OBJS	:=	$(GPU_SRCS:.cpp=.o)

# Enable CUDA support
ifdef DSC_CUDA
	# BF16 is supported in compute capability >= 8.0 (Ampere)
	HAS_BF16_GPU := $(shell compute_major=$$(nvidia-smi --query-gpu=compute_cap --format=noheader | cut -d. -f1); \
						if [ "$${compute_major}" -ge 8 ]; then echo 1; fi)
	ifeq ($(HAS_BF16_GPU), 1)
		NVCCFLAGS	+=	-DDSC_BF16
		CXXFLAGS	+=	-DDSC_BF16
	endif

	CXXFLAGS	+=	-I$(CUDA)/include -DDSC_CUDA
	NVCCFLAGS	+=	-x cu -DDSC_CUDA
	LDFLAGS		+=	-L$(CUDA)/lib64 -lcudart -lcublas

	OBJS		+=	$(GPU_OBJS)

$(GPU_OBJS): %.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif

# Enable HIP support
ifdef DSC_HIP
	GPU_TARGETS := $(shell ${ROCM_PATH}/bin/rocm_agent_enumerator)
    HAS_BF16_GPU := $(shell echo '${GPU_TARGETS}' | grep -q -E "gfx90a|gfx94[0-2]|gfx103[0-6]" && echo 1)
    ifeq ($(HAS_BF16_GPU), 1)
		HIPCCFLAGS	+=	-DDSC_BF16
		CXXFLAGS	+=	-DDSC_BF16
    endif

	# TODO: is -D__HIP_PLATFORM_AMD__ required?
	CXXFLAGS	+=	-I$(ROCM)/include -DDSC_HIP -D__HIP_PLATFORM_AMD__
	HIPCCFLAGS	+=	-DDSC_HIP
	LDFLAGS		+=	-L$(ROCM)/lib -lamdhip64 -lrocrand -lrocblas

	OBJS		+=	$(GPU_OBJS)

$(GPU_OBJS): %.o: %.cpp
	$(HIPCC) $(HIPCCFLAGS) -c $< -o $@
endif


$(info dsc build info: )
$(info   OS:		$(UNAME_S))
$(info   ARCH:		$(UNAME_M))
$(info   CXX:		$(shell $(CXX) --version | head -n 1))
$(info   CXXFLAGS:	$(CXXFLAGS))

ifdef DSC_CUDA
$(info   NVCC:		$(shell $(NVCC) --version | head -n 4 | tail -n 1))
$(info   NVCCFLAGS:	$(NVCCFLAGS))
endif

ifdef DSC_HIP
$(info   HIPCC:		$(shell $(HIPCC) --version | head -n 1 | tail -n 1))
$(info   HIPCCFLAGS:	$(HIPCCFLAGS))
endif

$(info   LDFLAGS:	$(LDFLAGS))
$(info )

SRCS		=	$(wildcard dsc/src/*.cpp)
SRCS		+=	$(wildcard dsc/src/cpu/*.cpp)
OBJS		+=	$(SRCS:.cpp=.o)

SHARED_LIB	=	python/dsc/libdsc.so

.PHONY: clean shared

clean:
	rm -rf *.o *.so *.old $(OBJS) $(GPU_OBJS) $(SHARED_LIB)

shared: $(OBJS)
	$(CXX) $(CXXFLAGS) -shared $(OBJS) -o $(SHARED_LIB) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
