include ../../../common.mk

CXXFLAGS += $(GTEST_CXXFLAGS)
LDFLAGS += -lOpenCL $(GTEST_LDFLAGS)
TARGET = prepare_test

all: $(TARGET)
$(TARGET): | CHECK_DEPS

PREPARE_TEST_DEP_OBJS :=  ../prepare.o
PREPARE_TEST_DEP_OBJS += $(WORKSPACE_ROOT)/algorithm/cl/transform.o
PREPARE_TEST_DEP_OBJS += $(WORKSPACE_ROOT)/algorithm/cl/mm_layout.o
PREPARE_TEST_DEP_OBJS += $(WORKSPACE_ROOT)/algorithm/cl/cl_util.o



prepare_test: prepare_test.o  $(PREPARE_TEST_DEP_OBJS) | Makefile transform.cl mm_layout.cl

test: $(TARGET)
	./prepare_test



%.cl: $(WORKSPACE_ROOT)/algorithm/cl/%.cl
	$(CP) $< $@

$(TARGET):
	$(LD) $^ $(LDFLAGS) -o $@

CHECK_DEPS:
	$(MAKE) -C ../ all
	$(MAKE) -C  $(WORKSPACE_ROOT)/algorithm/cl all

clean:
	rm -f $(TARGET) *.o *.cl

.PHONY: all clean test CHECK_DEPS

