# ANN
In mac os

brew install libomp

clang++ -Xpreprocessor -fopenmp main.cpp ANN.cpp -o a.out -lomp 

./a.out thread_count
