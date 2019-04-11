#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <time.h>
// #include <windows.h>
#include <algorithm>

using namespace std;

vector< vector<float> > X_train;
vector<float> y_train;

void LoadXY();

int main(int argc, const char * argv[]) {

//	vector< vector<float> > X_train;
//	vector<float> y_train;
    LoadXY();

	return 0;
}

void LoadXY()
{
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
}
