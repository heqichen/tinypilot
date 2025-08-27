include ../../common.mk

LDFLAGS += -lOpenCL

mm_layout: mm_layout.o
	$(CXX) $^ -o $@ $(LDFLAGS)

mm_layout.o: mm_layout.cpp mm_layout.cl Makefile
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f mm_layout mm_layout.o

.PHONY: clean