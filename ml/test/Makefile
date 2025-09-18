include ../../common.mk

CXXFLAGS += $(GTEST_CXXFLAGS) $(ARMNN_CXXFLAGS)
LDFLAGS += $(GTEST_LDFLAGS) $(ARMNN_LDFLAGS)
TARGET = vision_runner_test policy_runner_test


VISION_TFLITE_FILE_PATH := $(WORKSPACE_ROOT)/mlutils/modelgen/models_tflite/test_model_pb/driving_vision_fp32_float32.tflite
# POLICY_TFLITE_FILE_PATH := $(WORKSPACE_ROOT)/mlutils/modelgen/models_tflite/test_model_pb/driving_policy_fp32_float32.tflite
POLICY_TFLITE_FILE_PATH := /home/heqichen/workspace/tinypilot/POC/fix-gather/saved_model/output_float32.tflite
DEFFLAGS += -DVISION_TFLITE_FILE_PATH=\"$(VISION_TFLITE_FILE_PATH)\" -DPOLICY_TFLITE_FILE_PATH=\"$(POLICY_TFLITE_FILE_PATH)\"

all: $(TARGET)
$(TARGET): | CHECK_DEPS


vision_runner_test: vision_runner_test.o ../vision_runner.o ../model_runtime.o
policy_runner_test: policy_runner_test.o ../policy_runner.o ../model_runtime.o

test: $(TARGET)
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./policy_runner_test

$(TARGET):
	$(LD) $^ $(LDFLAGS) -o $@

CHECK_DEPS:
	$(MAKE) -C ../ all

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean test CHECK_DEPS

