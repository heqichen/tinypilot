include ../../../common.mk

CXXFLAGS += -I$(GTEST_INCLUDE_DIR)
LDFLAGS += -lOpenCL -L$(GTEST_LIB_DIR) -lgtest_main -lgtest

TARGET = simple_cl_test mm_layout_test

all: $(TARGET)
	$(MAKE) -C ../ all

simple_cl_test: simple_cl_test.o
mm_layout_test: mm_layout_test.o ../mm_layout.o

test: $(TARGET)
	./mm_layout_test

%: %.o
	$(CXX) $^ $(LDFLAGS) -o $@ 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) simple_cl_test.o mm_layout_test.o
