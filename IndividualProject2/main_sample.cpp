#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h> 

// #include "your header file"
#include "../Perceptron.h"

using namespace std;

vector< vector<float> > getIrisX();
vector<float> getIrisy();

int main()
{
	// load the training data 
	vector< vector<float> > X = getIrisX();
	vector<float> y = getIrisy();

	// create and train your perceptron model using X and y
	float r = 0.001;
	int epoch = 10;

	// Object
	Perceptron perceptron(r, epoch);
	
	cout << "************************* 1. Training with existing r and epoch without loading weight *************************" << endl;
	perceptron.Train(X, y);
	cout << "************************* 2. Training with existing modified r and epoch without loading weight *************************" << endl;
	perceptron.SetLearningRate(0.002);
	perceptron.SetEpoch(5);
	perceptron.Train(X, y);	
	cout << "************************* 3. Training with input r and epoch without loading weight *************************" << endl;
	perceptron.Train(X, y, 0.0005, 40);

	remove("weight.txt");

	cout << "************************* 4. Training with existing r and epoch with loading weight *************************" << endl;
	perceptron = Perceptron(r, epoch, true);
	perceptron.Train(X, y);
	
	cout << "************************* 5. Get prediction of X using trained weight *************************" << endl;
	perceptron.PrintPredictResult(X, y);

	cout << "************************* 6. Training with input r and epoch with loading weight *************************" << endl;
	perceptron.Train(X, y, 0.0005, 40);
	
	// Will throw exception since haven't trained
	// cout << "************************* 7.1. Get prediction of X without any trained weight *************************" << endl;
	// Perceptron(r, epoch).PrintPredictResult(X, y);
	
	cout << "************************* 7.2 Get prediction of X using trained weight *************************" << endl;
	perceptron.PrintPredictResult(X, y);

	// // test case 1:
	// vector<float> test1;
	// test1.push_back(5.0);
	// test1.push_back(3.5);
	// test1.push_back(1.3);
	// test1.push_back(0.2);
	// // test on test1


	// // test case 2:
	// vector<float> test2;
	// test2.push_back(6.0);
	// test2.push_back(2.1);
	// test2.push_back(5.2);
	// test2.push_back(1.4);
	// // test on test2

	return 0;
}

vector<float> getIrisy()
{
	vector<float> y;

	ifstream inFile;
	inFile.open("y.data");
	string sampleClass;
	for (int i = 0; i < 100; i++)
	{
		inFile >> sampleClass;
		if (sampleClass == "Iris-setosa")
		{
			y.push_back(-1);
		}
		else
		{
			y.push_back(1);
		}
	}

	return y;
}

vector< vector<float> > getIrisX()
{
	ifstream af;
	ifstream bf;
	ifstream cf;
	ifstream df;
	af.open("a.data");
	bf.open("b.data");
	cf.open("c.data");
	df.open("d.data");

	vector< vector<float> > X;

	for (int i = 0; i < 100; i++)
	{
		char scrap;
		int scrapN;
		af >> scrapN;
		bf >> scrapN;
		cf >> scrapN;
		df >> scrapN;

		af >> scrap;
		bf >> scrap;
		cf >> scrap;
		df >> scrap;
		float a, b, c, d;
		af >> a;
		bf >> b;
		cf >> c;
		df >> d;
		// X.push_back(vector < float > {a, b, c, d});

		vector < float > abcd;
		abcd.push_back(a);
		abcd.push_back(b);
		abcd.push_back(c);
		abcd.push_back(d);
		X.push_back(abcd);
	}

	af.close();
	bf.close();
	cf.close();
	df.close();

	return X;
}