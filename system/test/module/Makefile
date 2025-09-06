include ../../../common.mk

CXXFLAGS += -I$(GTEST_INCLUDE_DIR)
LDFLAGS += -L$(GTEST_LIB_DIR) -lgtest_main -lgtest -lOpenCL

TARGETS := vision_smoke_test

OBJS := $(WORKSPACE_ROOT)/perception/vision/prepare.o
OBJS += $(WORKSPACE_ROOT)/algorithm/cl/transform.o

vision_smoke_test: vision_smoke_test.o $(OBJS) | CHECK_DEPS transform.cl 

vision_smoke_test.o: vision_smoke_test.cpp | vision_smoke_test.h Makefile

%.o:
	$(CXX) $(CXXFLAGS) -c $^ -o $@

%:
	$(CXX) $^ $(LDFLAGS) -o $@

transform.cl:
	$(CP) $(WORKSPACE_ROOT)/algorithm/cl/transform.cl $@

CHECK_DEPS:
	$(MAKE) -C $(WORKSPACE_ROOT)/perception/vision all
	$(MAKE) -C $(WORKSPACE_ROOT)/algorithm/cl all

clean:
	rm -f $(TARGETS) *.o *.cl

.PHONY: CHECK_DEPS clean