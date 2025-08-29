include ../../../common.mk

CXXFLAGS += -I$(GTEST_INCLUDE_DIR)
LDFLAGS += -lOpenCL -L$(GTEST_LIB_DIR) 

TARGET = simple_cl_test mm_layout_test cl_info

all: $(TARGET)

mm_layout_test: | CHECK_DEPS

simple_cl_test: simple_cl_test.o
mm_layout_test: mm_layout_test.o ../mm_layout.o | mm_layout.cl

cl_info: cl_info.o

test: $(TARGET)
	./mm_layout_test

CHECK_DEPS:
	$(MAKE) -C ../ all

mm_layout_test: ../mm_layout.o mm_layout_test.o
	$(CXX) $^ $(LDFLAGS) -lgtest_main -lgtest -o $@

mm_layout.cl: ../mm_layout.cl
	cp $< $@

%: %.o
	$(CXX) $^ $(LDFLAGS) -o $@ 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) simple_cl_test.o mm_layout_test.o cl_info.o

.PHONY: clean test CHECK_DEPS