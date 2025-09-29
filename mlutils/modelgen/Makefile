include ../../common.mk

all: models_fp32/driving_vision.onnx models_fp32/driving_policy.onnx

models_fp32/driving_vision.onnx: models_py/driving_vision.py
	mkdir -p models_fp32
	python models_py/driving_vision.py models_fp32/driving_vision_fp32.onnx

models_fp32/driving_policy.onnx: models_py/driving_policy.py
	mkdir -p models_fp32
	python models_py/driving_policy.py models_fp32/driving_policy_fp32.onnx

# Make patch diff driving_policy_orig.py driving_policy.py

models_py/driving_vision.py: $(WORKSPACE_ROOT)/POC/resources/models/driving_vision.onnx
	mkdir -p models_py
	python -m onnxconverter_common.onnx2py $^ $@
	patch models_py/driving_vision.py driving_vision.patch

models_py/driving_policy.py: $(WORKSPACE_ROOT)/POC/resources/models/driving_policy.onnx
	mkdir -p models_py
	python -m onnxconverter_common.onnx2py $^ $@
	patch models_py/driving_policy.py driving_policy.patch

# python convert_vision.py -m models_fp32/ -o models_tflite/ -v
# This command run overnight
#

clean:
	rm -rf models_py models_fp32

.PHONY: clean
