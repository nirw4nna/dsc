CUDA		?=	/usr/local/cuda
CXX			=	g++
NVCC		=	nvcc
NVCCFLAGS	=	-std=c++20 -I$(CUDA)/include -I./dsc/include/ -ccbin=$(CXX) -arch=native \
				--forward-unknown-to-host-compiler -Wall -Wextra -Wformat -Wnoexcept \
				-Wcast-qual -Wunused -Wdouble-promotion -Wlogical-op -Wcast-align -fno-exceptions -fno-rtti
CXXFLAGS	=	-std=c++20 -I./dsc/include/ -I./dsc/api/ -Wall -Wextra -Wformat -Wnoexcept  \
 				-Wcast-qual -Wunused -Wdouble-promotion -Wlogical-op -Wcast-align -fno-exceptions -fno-rtti -pthread
LDFLAGS		=	-lm

UNAME_M		=	$(shell uname -m)
UNAME_S		=	$(shell uname -s)

ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64))
	# Use all available CPU extensions, x86 only
	CXXFLAGS	+= 	-march=native -mtune=native
endif

# Defining the __FAST_MATH__ macro makes computations faster even without adding any actual -ffast-math like flag.
# On the other hand, -ffast-math makes the FFTs slower.
ifdef DSC_FAST
	CXXFLAGS	+= -DDSC_FAST -O3 -D__FAST_MATH__ -ffp-contract=fast -funroll-loops -flto=auto -fuse-linker-plugin
	NVCCFLAGS	+= -DDSC_FAST -O3 --use_fast_math
else
	CXXFLAGS	+= -DDSC_DEBUG -O0 -fno-omit-frame-pointer -g
	NVCCFLAGS	+= -DDSC_DEBUG -O0 -fno-omit-frame-pointer -G
endif

ifdef DSC_ENABLE_TRACING
	CXXFLAGS	+= -DDSC_ENABLE_TRACING
endif

ifdef DSC_MAX_TRACES
	CXXFLAGS	+= -DDSC_MAX_TRACES=$(DSC_MAX_TRACES)
endif

# If we are not compiling the shared object and are in debug mode then run in ASAN mode
ifeq ($(MAKECMDGOALS),shared)
	CXXFLAGS	+= -fPIC
	NVCCFLAGS	+= -fPIC
else
	ifndef DSC_FAST
		CXXFLAGS	+= -fsanitize=address
		NVCCFLAGS	+= -fsanitize=address
	endif
endif

CUDA_SRCS	= $(wildcard dsc/src/cuda/*.cu)
CUDA_OBJS	= $(CUDA_SRCS:.cu=.o)

# Enable CUDA support
ifdef DSC_CUDA
	CXXFLAGS	+= -I$(CUDA)/include -DDSC_CUDA
	NVCCFLAGS	+= -DDSC_CUDA
	LDFLAGS		+= -L$(CUDA)/lib64 -lcudart

	OBJS		+= $(CUDA_OBJS)
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
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

$(info   LDFLAGS:	$(LDFLAGS))
$(info )

SRCS		= $(wildcard dsc/src/*.cpp)
SRCS		+= $(wildcard dsc/src/cpu/*.cpp)
OBJS		+= $(SRCS:.cpp=.o)

SHARED_LIB	= python/dsc/libdsc.so

.PHONY: clean shared

clean:
	rm -rf *.o *.so *.old $(OBJS) $(CUDA_OBJS) $(SHARED_LIB)

shared: $(OBJS)
	$(CXX) $(CXXFLAGS) -shared $(OBJS) -o $(SHARED_LIB) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

