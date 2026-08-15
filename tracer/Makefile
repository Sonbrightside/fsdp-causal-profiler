# GapScope CUPTI tracer build
# Requires: CUDA toolkit with CUPTI (set CUDA_HOME or load your site's cuda module)

CUDA_HOME ?= $(or $(CUDA_PATH),/usr/local/cuda)

CXX      ?= g++
CXXFLAGS  = -std=c++17 -O2 -fPIC -Wall
INCLUDES  = -I$(CUDA_HOME)/include \
            -I$(CUDA_HOME)/extras/CUPTI/include
LDFLAGS   = -shared \
            -L$(CUDA_HOME)/extras/CUPTI/lib64 \
            -L$(CUDA_HOME)/lib64
LDLIBS    = -lcupti -lcuda -lsqlite3 -lpthread

TARGET = libcupti_trace.so
SRCS   = cupti_trace.cpp clock_anchor.cpp

$(TARGET): $(SRCS) clock_anchor.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean