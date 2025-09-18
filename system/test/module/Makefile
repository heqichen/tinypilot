include ../../../common.mk

CXXFLAGS += -I$(GTEST_INCLUDE_DIR) $(ARMNN_CXXFLAGS) $(OPENCV_CXXFLAGS)
LDFLAGS += -L$(GTEST_LIB_DIR) -lgtest_main -lgtest -lOpenCL $(ARMNN_LDFLAGS) $(OPENCV_LDFLAGS)

TARGETS := vision_smoke_test video_to_nn_test

OBJS := $(WORKSPACE_ROOT)/perception/vision/prepare.o
OBJS += $(WORKSPACE_ROOT)/perception/vision/ml/vision_runner.o
OBJS += $(WORKSPACE_ROOT)/algorithm/cl/transform.o
OBJS += $(WORKSPACE_ROOT)/algorithm/cl/cl_util.o
OBJS += $(WORKSPACE_ROOT)/algorithm/cl/mm_layout.o
OBJS += $(WORKSPACE_ROOT)/sensor/vision/input/video_input.o

$(TARGETS) : | CHECK_DEPS
vision_smoke_test: vision_smoke_test.o $(OBJS) | transform.cl mm_layout.cl models/onnx2py_fp32_float32.tflite
video_to_nn_test: video_to_nn_test.o $(OBJS) | transform.cl mm_layout.cl models/onnx2py_fp32_float32.tflite assets/mono_color.mp4 assets/driving_clip.mp4
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

assets/mono_color.mp4:
	mkdir -p assets
	$(CP) $(WORKSPACE_ROOT)/sensor/vision/test/mono_color.mp4 $@

assets/driving_clip.mp4: 
	mkdir -p assets
	$(CP) $(WORKSPACE_ROOT)/sensor/vision/test/driving_clip.mp4 $@

models/onnx2py_fp32_float32.tflite:
	$(CP) -r $(WORKSPACE_ROOT)/perception/vision/ml/models ./

CHECK_DEPS:
	mkdir -p output
	$(MAKE) -C $(WORKSPACE_ROOT)/perception/vision all
	$(MAKE) -C $(WORKSPACE_ROOT)/perception/vision/ml all
	$(MAKE) -C $(WORKSPACE_ROOT)/algorithm/cl all
	$(MAKE) -C $(WORKSPACE_ROOT)/sensor/vision/input all

clean:
	rm -f $(TARGETS) *.o *.cl
	rm -rf output assets

.PHONY: CHECK_DEPS clean