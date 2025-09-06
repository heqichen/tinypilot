include ../../../common.mk

CXXFLAGS += -I$(GTEST_INCLUDE_DIR)
LDFLAGS += -lOpenCL -L$(GTEST_LIB_DIR) -lgtest_main -lgtest
TARGET = simple_cl_test mm_layout_test cl_info transform_test

all: $(TARGET)
transform_test: | CHECK_DEPS
mm_layout_test: | CHECK_DEPS
transform_test.o: transform_test.cpp transform_test.h

transform_test: transform_test.o ../transform.o | Makefile transform.cl
mm_layout_test: mm_layout_test.o ../mm_layout.o | Makefile mm_layout.cl

$(TARGET):
	$(LD) $^ $(LDFLAGS) -o $@

simple_cl_test: simple_cl_test.o | Makefile
	$(CXX) $^ -lOpenCL -o $@	

# %_test: %_test.o ../%.o | %.cl 

cl_info: cl_info.o

test: $(TARGET)
	./simple_cl_test
	./mm_layout_test
	./cl_info
	./transform_test

CHECK_DEPS:
	$(MAKE) -C ../ all

%.cl: ../%.cl
	$(CP) $< $@

# %: %.o
# 	$(CXX) $^ $(LDFLAGS) -o $@ 

# %.o: %.cpp
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) simple_cl_test.o mm_layout_test.o cl_info.o transform_test.o mm_layout.cl transform.cl module_smoke_test.o

.PHONY: all clean test CHECK_DEPS