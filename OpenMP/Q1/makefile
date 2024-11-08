CC=g++-13

CFLAGS=-c -fopenmp
LNKFLAGS=-I/opt/homebrew/opt/libomp/include
INCDIR=
LIBDIR=-L/opt/homebrew/opt/libomp/lib
LIBFILE=-lomp 

all: main

main: main.o 
	$(CC) $(LNKFLAGS) $(LIBDIR) $(LIBFILE) main.o -o main

main.o: main.cpp
	$(CC) $(CFLAGS) $(INCDIR) main.cpp

clean:
	rm -f *o main