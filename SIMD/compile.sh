g++ -msse -O2 -march=native -o obj/Q1 src/Q1.cpp `pkg-config --cflags --libs opencv4`
g++ -msse -O2 -march=native -o obj/Q2 src/Q2.cpp `pkg-config --cflags --libs opencv4`
g++ -msse -O2 -march=native -o obj/Q3 src/Q3.cpp `pkg-config --cflags --libs opencv4`
g++ -msse -O2 -march=native -o obj/Q4 src/Q4.cpp `pkg-config --cflags --libs opencv4`

echo "Done compiling"
