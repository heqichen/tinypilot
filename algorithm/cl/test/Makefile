CXX = g++
CXXFLAGS = -std=c++11 -O2
LDFLAGS = -lOpenCL
TARGET = simple_cl_test 2
SRC = simple_cl_test.cpp

all: $(TARGET)

cl_info: cl_info.cpp
	clang++ cl_info.cpp -I /usr/include/c++/11/ -I /usr/include/x86_64-linux-gnu/c++/11/  -L /usr/lib/gcc/x86_64-linux-gnu/11/ -lOpenCL -o cl_info

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
