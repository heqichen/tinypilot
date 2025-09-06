include ../../../common.mk

CXXFLAGS += -I$(GTEST_INCLUDE_DIR)
LDFLAGS += -lOpenCL -L$(GTEST_LIB_DIR) -lgtest_main -lgtest
TARGET = prepare_test

all: $(TARGET)
$(TARGET): | CHECK_DEPS
prepare_test: prepare_test.o ../prepare.o $(WORKSPACE_ROOT)/algorithm/cl/transform.o | Makefile transform.cl

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

