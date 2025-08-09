
LD = g++
CPP = g++
CFLAGS += -g $(shell pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0)
LDFLAGS += $(shell pkg-config --libs gstreamer-1.0 gstreamer-app-1.0)

all: gst_test
	./gst_test

gst_test: gst_test.o
	$(LD) -o $@ $< $(LDFLAGS) $(LIBS)

gst_test.o: gst_test.cpp
	$(CPP) -c $< $(CFLAGS) -o $@

clean:
	$(RM) gst_test gst_test.o

.PHONY: clean all