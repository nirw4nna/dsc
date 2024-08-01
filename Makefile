CC			=	gcc
CXX			=	g++
CXXFLAGS	=	-std=c++20 -I./dsc/include/ -Wall -Wextra -Wshadow -Wformat -Wnoexcept  \
 				-Wcast-qual -Wunused -Wdouble-promotion -Wlogical-op -Wcast-align -fno-exceptions -fno-rtti -pthread
CFLAGS		=	-std=c99 -W -Wall -I./dsc/tests/
LDFLAGS		=	-lm

UNAME_M		=	$(shell uname -m)
UNAME_S		=	$(shell uname -s)

ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64))
	# Use all available CPU extensions, x86 only
	CXXFLAGS	+= 	-march=native -mtune=native
	CFLAGS		+= 	-march=native -mtune=native
endif

ifdef DSC_FAST
	CXXFLAGS	+= -DDSC_FAST -O3 -ffp-contract=fast -funroll-loops -flto=auto -fuse-linker-plugin
	CFLAGS		+= -DDSC_FAST -O3 -ffp-contract=fast -funroll-loops -flto=auto -fuse-linker-plugin
else
	CXXFLAGS	+= -DDSC_DEBUG -O0 -fno-omit-frame-pointer -g
	CFLAGS		+= -DDSC_DEBUG -O0 -fno-omit-frame-pointer -g
endif

# If we are not compiling the shared object and are in debug mode then run in ASAN mode
ifeq ($(MAKECMDGOALS),shared)
	CXXFLAGS	+= -fPIC
else
	ifndef DSC_FAST
		CXXFLAGS	+= -fsanitize=address
	endif
endif

$(info dsc build info: )
$(info   OS:		$(UNAME_S))
$(info   ARCH:		$(UNAME_M))
$(info   CXXFLAGS:	$(CXXFLAGS))
$(info   LDFLAGS:	$(LDFLAGS))
$(info   CXX:		$(shell $(CXX) --version | head -n 1))
$(info )

SRCS		= $(wildcard dsc/src/*.cpp)
OBJS		= $(SRCS:.cpp=.o)
SHARED_LIB	= python/dsc/libdsc.so

.PHONY: clean shared

clean:
	rm -rf *.o *.so *.old $(OBJS) $(SHARED_LIB)

shared: $(OBJS)
	$(CXX) $(CXXFLAGS) -shared $(OBJS) -o $(SHARED_LIB)

#pocketfft.o: dsc/tests/pocketfft.c
#	$(CC) $(CFLAGS) -c $< -o $@
#
#test_fft: dsc/tests/test_fft.cpp $(OBJS) pocketfft.o
#	$(CXX) $(CXXFLAGS) $< -o $@ $(OBJS) pocketfft.o $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

