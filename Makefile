CXX			=	g++
AR			=	ar

CXXFLAGS	=	-std=c++20 -I./dsc/include/ -I./dsc/api/ -Wall -Wextra -Wformat -Wnoexcept  \
 				-Wcast-qual -Wcast-align -Wstrict-aliasing -Wpointer-arith -Wunused -Wdouble-promotion \
 				-Wlogical-op -Wcast-align -fno-exceptions -fno-rtti
LDFLAGS		=	-lm

UNAME_M		=	$(shell uname -m)
UNAME_S		=	$(shell uname -s)

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

ifdef DSC_FAST
	CXXFLAGS	+=	-Ofast -ffp-contract=fast -funroll-loops -flto=auto -fuse-linker-plugin
else
	CXXFLAGS	+=	-O0 -fno-omit-frame-pointer -g
endif

ifdef DSC_ENABLE_TRACING
	CXXFLAGS	+=	-DDSC_ENABLE_TRACING -pthread
endif

# If we are not compiling the shared object and are in debug mode then run in ASAN mode
ifeq ($(MAKECMDGOALS),shared)
	CXXFLAGS	+=	-fPIC
endif


$(info dsc build info: )
$(info   OS:		$(UNAME_S))
$(info   ARCH:		$(UNAME_M))
$(info   CXX:		$(shell $(CXX) --version | head -n 1))
$(info   CXXFLAGS:	$(CXXFLAGS))
$(info   LDFLAGS:	$(LDFLAGS))
$(info )

SRCS		=	$(wildcard dsc/src/*.cpp)
SRCS		+=	$(wildcard dsc/src/cpu/*.cpp)
OBJS		+=	$(SRCS:.cpp=.o)

SHARED_LIB	=	python/dsc/libdsc.so
STATIC_LIB	=	libdsc.a

.PHONY: clean shared static

clean:
	rm -rf *.o *.so *.old $(OBJS) $(CUDA_OBJS) $(SHARED_LIB) $(STATIC_LIB)

static: $(OBJS)
	$(AR) -rcs $(STATIC_LIB) $(OBJS)

shared: $(OBJS)
	$(CXX) $(CXXFLAGS) -shared $(OBJS) -o $(SHARED_LIB) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
