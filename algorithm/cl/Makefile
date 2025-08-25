
CXXFLAGS = -std=c++11 -O2 -I /usr/include/c++/11/ -I /usr/include/x86_64-linux-gnu/c++/11/ 
CXX = clang++

LDFLAGS = -L /usr/lib/gcc/x86_64-linux-gnu/11/ -lOpenCL 


mm_layout: mm_layout.o
	$(CXX) $^ -o $@ $(LDFLAGS)

mm_layout.o: mm_layout.cpp mm_layout.cl
	$(CXX) $(CXXFLAGS) -c $< -o $@