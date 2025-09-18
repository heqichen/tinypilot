include ../../common.mk

CXXFLAGS += $(GTEST_CXXFLAGS)
LDFLAGS += $(GTEST_LDFLAGS) $(ARMNN_LDFLAGS)
TARGET = model_runner_test



all: $(TARGET)
$(TARGET): | CHECK_DEPS

model_runner_test: model_runner_test.o ../model_runner.o

test: $(TARGET)
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./model_runner_test


$(TARGET):
	$(LD) $^ $(LDFLAGS) -o $@

CHECK_DEPS:
	$(MAKE) -C ../ all

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean test CHECK_DEPS

