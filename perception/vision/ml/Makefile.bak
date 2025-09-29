LD = g++
CPP = g++

CFLAGS += -I ../../../third_party/ArmNN/output/include

LDFLAGS += -L ../../../third_party/ArmNN/output -larmnn -larmnnTfLiteParser -lprotobuf

run: run_tflite
	LD_LIBRARY_PATH=../../../third_party/ArmNN/output ./$<

run_tflite: run_tflite.o
	$(LD) -o $@ $< $(LDFLAGS) $(LIBS)

run_tflite.o: run_tflite.cpp
	$(CPP) -c $< $(CFLAGS) -o $@



clean:
	$(RM) run_tflite run_tflite.o

.PHONY: clean