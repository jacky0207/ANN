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

    vector< vector<float> > X_train;
    vector<float> y_train;

    ifstream myfile("data/train_small.txt");

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

    ANN ann(2, 10, 16);

    return 0;
}

//        vector< vector <float> >::iterator rowIteratorX = X_train.begin();
//        int rowX = 0;
//
//        // Predict X for each row
//        while(rowIteratorX != X_train.end())	// Loop every row
//        {
//            vector< float >::iterator columnIteratorX = (*rowIteratorX).begin();
//            int columnX = 0;
//
//            // Predict X for each row
//            while(columnIteratorX != (*rowIteratorX).end())	// Loop every row
//            {
//                cout << (*rowIteratorX).at(columnX) * 255 << "\t";
//
//                // rowIteratorX
//                columnIteratorX++;
//                columnX++;
//            }
//
//            cout << endl << endl;
//
//            // rowIteratorX
//            rowIteratorX++;
//            rowX++;
//        }

//        vector< float >::iterator iteratorY = y_train.begin();
//        int indexY = 0;
//
//        // Predict X for each row
//        while(iteratorY != y_train.end())	// Loop every row
//        {
//            cout << y_train.at(indexY) << "\t";
//
//            // rowIteratorX
//            iteratorY++;
//            indexY++;
//        }
//
//        cout << endl << endl;