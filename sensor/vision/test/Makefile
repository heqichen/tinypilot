
LD = g++
CPP = g++
CFLAGS += -g $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkg-config --libs opencv4)

all: video_file_loader
	./video_file_loader

video_file_loader: video_file_loader.o
	$(LD) -o $@ $< $(LDFLAGS) $(LIBS)

video_file_loader.o: video_file_loader.cpp
	$(CPP) -c $< $(CFLAGS) -o $@

clean:
	$(RM) video_file_loader video_file_loader.o

.PHONY: clean all