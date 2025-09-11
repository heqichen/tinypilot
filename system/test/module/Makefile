include ../../../common.mk

CXXFLAGS += -I$(GTEST_INCLUDE_DIR) $(ARMNN_CXXFLAGS)
LDFLAGS += -L$(GTEST_LIB_DIR) -lgtest_main -lgtest -lOpenCL $(ARMNN_LDFLAGS)

TARGETS := vision_smoke_test video_to_nn_test

OBJS := $(WORKSPACE_ROOT)/perception/vision/prepare.o
OBJS += $(WORKSPACE_ROOT)/perception/vision/ml/model_runner.o
OBJS += $(WORKSPACE_ROOT)/algorithm/cl/transform.o
OBJS += $(WORKSPACE_ROOT)/algorithm/cl/cl_util.o
OBJS += $(WORKSPACE_ROOT)/algorithm/cl/mm_layout.o

$(TARGETS) : | CHECK_DEPS
vision_smoke_test: vision_smoke_test.o $(OBJS) | transform.cl mm_layout.cl models/onnx2py_fp32_float32.tflite
video_to_nn_test: video_to_nn_test.o $(OBJS) | transform.cl mm_layout.cl models/onnx2py_fp32_float32.tflite
vision_smoke_test.o: vision_smoke_test.cpp | vision_smoke_test.h Makefile
video_to_nn_test.o: video_to_nn_test.cpp | video_to_nn_test.h Makefile


all:$(TARGETS)

run_test: $(TARGETS)
	LD_LIBRARY_PATH=$(WORKSPACE_ROOT)/third_party/ArmNN/output ./video_to_nn_test

%.o:
	$(CXX) $(CXXFLAGS) -c $^ -o $@

%: %.o
	$(LD) $^ $(LDFLAGS) -o $@

transform.cl:
	$(CP) $(WORKSPACE_ROOT)/algorithm/cl/transform.cl $@

mm_layout.cl:
	$(CP) $(WORKSPACE_ROOT)/algorithm/cl/mm_layout.cl $@

models/onnx2py_fp32_float32.tflite:
	$(CP) -r $(WORKSPACE_ROOT)/perception/vision/ml/models ./

CHECK_DEPS:
	$(MAKE) -C $(WORKSPACE_ROOT)/perception/vision all
	$(MAKE) -C $(WORKSPACE_ROOT)/perception/vision/ml all
	$(MAKE) -C $(WORKSPACE_ROOT)/algorithm/cl all

clean:
	rm -f $(TARGETS) *.o *.cl

.PHONY: CHECK_DEPS clean