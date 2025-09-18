include ../common.mk

CXXFLAGS += $(ARMNN_CXXFLAGS)
all: vision_runner.o model_runtime.o

vision_runner.o: vision_runner.cpp vision_runner.h Makefile
model_runtime.o: model_runtime.cpp model_runtime.h Makefile
policy.o: policy.cpp policy.h Makefile

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o

.PHONY: clean all
