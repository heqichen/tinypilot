include ../../common.mk


LDFLAGS += -lOpenCL

all: mm_layout.o transform.o

# mm_layout: mm_layout.o
# 	$(CXX) $^ -o $@ $(LDFLAGS)

mm_layout.o: mm_layout.cpp mm_layout.cl | Makefile
transform.o: transform.cpp transform.cl | Makefile

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f mm_layout mm_layout.o

.PHONY: clean all