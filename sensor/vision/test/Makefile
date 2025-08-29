include ../../../common.mk

# OpenCV does not support C++23 yet
CXXFLAGS = -std=c++17

CXXFLAGS += $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkg-config --libs opencv4)

all: video_file_loader
	./video_file_loader

video_file_loader: video_file_loader.o
	$(CXX) -o $@ $< $(LDFLAGS) $(LIBS)

video_file_loader.o: video_file_loader.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

clean:
	$(RM) video_file_loader video_file_loader.o

.PHONY: clean all