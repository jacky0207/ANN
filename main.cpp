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

// Custom
#include "ANN.h"

using namespace std;

int main(int argc, const char * argv[]) {
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

    // Test ann class
    ANN ann(2, 16, 12);
    ann.Train(X_train, y_train, 0.1, 1, 1);
    ann.predict(X_train, y_train);
    
    // Save weight
    ann.SaveWeight();

    // Load weight
    ANN ann2(2, 16, 12);
    ann2.LoadWeight();
    ann2.predict(X_train, y_train);

    return 0;
}