include ../../common.mk


LDFLAGS += -lOpenCL

all: mm_layout.o transform.o cl_util.o

# mm_layout: mm_layout.o
# 	$(CXX) $^ -o $@ $(LDFLAGS)

mm_layout.o: mm_layout.cpp mm_layout.h Makefile | mm_layout.cl
transform.o: transform.cpp transform.h Makefile | transform.cl
cl_util.o: cl_util.cpp cl_util.h Makefile

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o

.PHONY: clean all
