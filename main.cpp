#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <time.h>
//#include <windows.h>
#include <algorithm>

#include <omp.h>  

// Custom
#include "ANN.h"

using namespace std;

int main(int argc, const char * argv[])
{  
    // Throw if thread number out of range
    if (argc != 2)
    {
        cout << "Usage: enter thread count from 1 - 4" << endl;
        return 1;
    }

    int thread_count = stoi(argv[1]);
    if (thread_count < 1 || thread_count > 4)
    {
        cout << "Usage: enter thread count from 1 - 4" << endl;
        return 1;
    }

    // Load data
    vector< vector<float> > X_train;
    vector<float> y_train;

    ifstream myfile("data/test.txt");

    if (myfile.is_open())
    {
        cout << "Loading data ...\n";
        string line;
        while (getline(myfile, line))
        {
            int x, y;
            vector<float> X;
            stringstream ss(line);
            ss >> y;
            y_train.push_back(y);
            for (int i = 0; i < 28 * 28; i++) {
                ss >> x;
                X.push_back(x/255.0);
            }
            X_train.push_back(X);
        }

        myfile.close();
        cout << "Loading data finished.\n";
    }
    else
        cout << "Unable to open file" << '\n';

    // X_train.erase(X_train.begin());
    // y_train.erase(y_train.begin());

    double start = omp_get_wtime();

    // Test ann class
    ANN ann(2, 16, 12); // 4 layer, neuron: n, 16, 12, 10
    ann.Train(X_train, y_train, 0.1, 12, 4, thread_count, 4); // r = 0.1, 12 sample/mini-batch, 4 epoch
    ann.predict(X_train, y_train);

    double end = omp_get_wtime();

    cout << "Total running time: " << (end - start) << " second" << endl;
    
    // // Save weight
    // ann.Save();

    // Load weight
    // ANN ann2(2, 16, 12);
    // ann2.Load();
    // ann2.predict(X_train, y_train);

    return 0;
}