include ../../common.mk


LDFLAGS += -lOpenCL

all: mm_layout.o transform.o

# mm_layout: mm_layout.o
# 	$(CXX) $^ -o $@ $(LDFLAGS)

mm_layout.o: mm_layout.cpp mm_layout.h Makefile | mm_layout.cl
transform.o: transform.cpp transform.h Makefile | transform.cl

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f mm_layout.o transform.o

.PHONY: clean all
