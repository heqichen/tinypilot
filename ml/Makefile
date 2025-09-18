include ../common.mk

CXXFLAGS += $(ARMNN_CXXFLAGS)
all: model_runner.o

model_runner.o: model_runner.cpp model_runner.h Makefile

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o

.PHONY: clean all
