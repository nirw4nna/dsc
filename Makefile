CUDA		?=	/usr/local/cuda
CC			?=	gcc
CXX			?=	g++
NVCC		?=	nvcc
NVCCFLAGS	=	-std=c++20 -I$(CUDA)/include -I./dsc/include/ -ccbin=$(CXX) -code=sm_60 -arch=compute_60 \
				--forward-unknown-to-host-compiler -Wall -Wextra -Wformat -Wnoexcept \
				-Wcast-qual -Wunused -Wdouble-promotion -Wlogical-op -Wcast-align -fno-exceptions -fno-rtti
CXXFLAGS	=	-std=c++20 -I$(CUDA)/include -I./dsc/include/ -I./dsc/api/ -Wall -Wextra -Wformat -Wnoexcept  \
 				-Wcast-qual -Wunused -Wdouble-promotion -Wlogical-op -Wcast-align -fno-exceptions -fno-rtti -pthread
LDFLAGS		=	-lm -L$(CUDA)/lib64 -lcudart

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

ifdef DSC_MAX_FFT_PLANS
	CXXFLAGS	+= -DDSC_MAX_FFT_PLANS=$(DSC_MAX_FFT_PLANS)
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

$(info dsc build info: )
$(info   OS:		$(UNAME_S))
$(info   ARCH:		$(UNAME_M))
$(info   CXXFLAGS:	$(CXXFLAGS))
$(info   LDFLAGS:	$(LDFLAGS))
$(info   CXX:		$(shell $(CXX) --version | head -n 1))
$(info   CUDA:		$(DSC_CUDA))
$(info )

SRCS		= $(wildcard dsc/src/*.cpp)
SRCS		+= $(wildcard dsc/src/cpu/*.cpp)
CUDA_SRCS	= $(wildcard dsc/src/cuda/*.cu)
OBJS		= $(SRCS:.cpp=.o)
CUDA_OBJS	= $(CUDA_SRCS:.cu=.o)

SHARED_LIB	= python/dsc/libdsc.so

.PHONY: clean shared

clean:
	rm -rf *.o *.so *.old $(OBJS) $(CUDA_OBJS) $(SHARED_LIB)

shared: $(OBJS) $(CUDA_OBJS)
	$(CXX) $(CXXFLAGS) -shared $(OBJS) $(CUDA_OBJS) -o $(SHARED_LIB) $(LDFLAGS)

#test_simple: dsc/tests/test_simple.cpp $(OBJS)
#	$(CXX) $(CXXFLAGS) $< -o $@ $(OBJS) $(LDFLAGS)

#test_src: dsc/tests/test_src.cpp $(OBJS)
#	$(CXX) $(CXXFLAGS) -sEXPORTED_FUNCTIONS='["_dsc_src","_malloc","_free"]' -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=2200MB -sMAXIMUM_MEMORY=2200MB -sENVIRONMENT=web $< -o dsrc.js $(OBJS) $(LDFLAGS)
#	cp dsrc.js /home/lowl/Scrivania/projects/dspcraft/website/src/
#	cp dsrc.wasm /home/lowl/Scrivania/projects/dspcraft/website/src/

#dsc/src/cpu/dsc_cpu_impl.o: dsc/src/cpu/dsc_cpu.cpp
#	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

