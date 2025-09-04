include ../../common.mk


LDFLAGS += -lOpenCL

all: prepare.o

prepare.o: prepare.cpp | prepare.h

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f prepare.o

.PHONY: clean all
