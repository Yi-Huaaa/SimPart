g++ src/main.cpp src/graph.cpp src/fault.cpp src/pattern.cpp src/simulation.cpp  -std=c++11 -O3 -Wall -o main.out
./main.out benchmark/simple.ckt benchmark/simple.flst benchmark/simple.ptn
rm main.out
