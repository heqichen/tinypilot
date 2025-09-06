# use "make CLANG=1" to use clang, default is g++
# use "make DEBUG=1" to add debug symbol

# GoogleTest include directory
THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))
WORKSPACE_ROOT := $(abspath $(dir $(THIS_MAKEFILE)))


GTEST_INCLUDE_DIR := $(WORKSPACE_ROOT)/third_party/googletest/install/include
GTEST_LIB_DIR := $(WORKSPACE_ROOT)/third_party/googletest/install/lib

CXXFLAGS := -I$(WORKSPACE_ROOT)

ifeq ($(CLANG),1)
CXX := clang++
CXXFLAGS += -std=c++2b -I /usr/include/c++/11/ -I /usr/include/x86_64-linux-gnu/c++/11/
LDFLAGS := -L /usr/lib/gcc/x86_64-linux-gnu/11/ 
LD := clang++
else
CXX := g++
CXXFLAGS += -std=c++23
LDFLAGS :=
LD := g++
endif


ifeq ($(DEBUG),1)
CXXFLAGS += -g -O0
LDFLAGS += -g -O0
else
endif

CP := cp



# MAKE := DEBUG=$(DEBUG) CLANG=$(CLANG) make
# cleanall:
# 	$(MAKE) -C $(WORKSPACE_ROOT) clean
# .PHONY: cleanall

