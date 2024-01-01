g++ src/main.cpp src/graph.cpp src/simulation.cpp src/fault.cpp  -std=c++11 -O3 -Wall -o main.out
./main.out benchmark/simple.ckt benchmark/simple.flst benchmark/simple.ptn
rm main.out



# generate pattern and fault files under the src file
# python3 generate_fault_patterns.py 7 14 ../benchmark/simple.flst 3 8 simple.ptn  && echo "../benchmark/simple.flst" && cat ../benchmark/simple.flst && echo "../benchmark/simple.ptn" && cat ../benchmark/simple.ptn