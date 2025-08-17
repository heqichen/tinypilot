CXX = g++
CXXFLAGS = -std=c++11 -O2
LDFLAGS = -lOpenCL
TARGET = simple_cl_test
SRC = simple_cl_test.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
