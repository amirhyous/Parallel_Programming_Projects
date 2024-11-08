CXX = g++
CXXFLAGS = -std=c++11 
LIBS = `pkg-config --cflags --libs opencv4`

all: main run
main: Q4.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

.PHONY: clean
run:
	./main>output.txt
clean:
	rm -f main
