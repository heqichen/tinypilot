include ../../../common.mk

CXXFLAGS += -I$(GTEST_INCLUDE_DIR)
LDFLAGS += -L$(GTEST_LIB_DIR) -lOpenCL

TARGET = module_smoke_test

all: $(TARGET)

module_smoke_test: ../prepare.o $(WORKSPACE_ROOT)/algorithm/cl/transform.o |  transform.cl CHECK_DEPS

test: $(TARGET)
	./module_smoke_test

CHECK_DEPS:
	$(MAKE) -C ../ all
	$(MAKE) -C $(WORKSPACE_ROOT)/algorithm/cl all

module_smoke_test: module_smoke_test.o
	$(CXX) $^ $(LDFLAGS) -lgtest_main -lgtest -o $@

%.cl: $(WORKSPACE_ROOT)/algorithm/cl/%.cl
	cp $< $@

%: %.o
	$(CXX) $^ $(LDFLAGS) -o $@ 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o

.PHONY: clean test CHECK_DEPS