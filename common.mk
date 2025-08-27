# use "make CLANG=1" to use clang, default is g++
# use "make DEBUG=1" to add debug symbol

ifeq ($(CLANG),1)
CXX = clang++
CXXFLAGS = -std=c++2b -I /usr/include/c++/11/ -I /usr/include/x86_64-linux-gnu/c++/11/
LDFLAGS = -L /usr/lib/gcc/x86_64-linux-gnu/11/ 
else
CXX = g++
CXXFLAGS = -std=c++23
LDFLAGS =
endif

ifeq ($(DEBUG),1)
CXXFLAGS += -g -O0
endif