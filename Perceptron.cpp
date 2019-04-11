#include "iostream"
#include "fstream"
using namespace std;

// Custom file
#include "Perceptron.h"
#include "globalOperation.cpp"

Perceptron::Perceptron(float r, int epoch, bool saveAndLoad)
{
	W = 0;
	rowLength = 0;

	if (r <= 0) __throw_out_of_range("Usage: r > 0"); this->r = r;
	if (epoch <= 0) __throw_out_of_range("Usage: epoch > 0");this->epoch = epoch;
	this->saveAndLoad = saveAndLoad;
}

void Perceptron::SetLearningRate(float r)
{
	this->r = r;
}

void Perceptron::SetEpoch(int epoch)
{
	this->epoch = epoch;
}

void Perceptron::SetInputWeight(bool inputWeight)
{
	this->saveAndLoad = inputWeight;
}

// Use existing r and epoch
void Perceptron::Train(vector< vector < float > > X,
						vector<float> Y)
{
	Train(X, Y, r, epoch);
}

/*
input: a set of N data sample Xi, and a set of labels yi, 0 ≤ i ≤ N-1
learning rate r, number of epochs epoch
*/
// Use input r and epoch
void Perceptron::Train(vector< vector < float > > X,
						vector<float> Y,
						float r,
						int epoch)
{
	InitializeW(X);	// Initialize weight

	cout << "r = " << r << ", epoch = " << epoch << endl;
	
    for(int epochIndex = 0; epochIndex < epoch; epochIndex++)
	{
		cout << "-------------------------Epoch" << epochIndex + 1 << "-------------------------" << endl;

        int errors = 0;  // record the number of wrong predictions
	
		vector< vector <float> >::iterator iteratorX = X.begin();
		int indexX = 0;

		// Predict X for each row
		while(iteratorX != X.end())	// Loop every row
		{
			// Get Vector<float> form
			Vector<float> rowX = GetRowX(iteratorX);

			// Get predict and real of Y
            int predictY = Predict(rowX);	// Predict current row
			int realY = Y.at(indexX);

			// Fixing weight
            if (predictY != realY)	// if prediction is wrong
			{
				float update = r * (realY - predictY);

				// W = W + update * Xi;  // element-wise operation, including the bias
				W = VectorAddition<float>(rowX, *W, update);
				errors++;

				// Print error row
				cout << "row: " << indexX << ", Compare: " << predictY << " " << realY << endl;
			}

			// Increment
			iteratorX++;
			indexX++;
        }

        cout << "Error: " << errors << endl; // to see how good the model is in this epoch
		cout << "-------------------------end Epoch" << epochIndex << "-------------------------\n" << endl;

		// Stop training when no errors
		if (errors == 0)
		{
			cout << "Training complete at Epoch " << epochIndex + 1 << "\n" << endl;
			break;
		}
		// Have errors but training is completed
		else if (epochIndex == epoch - 1)
		{
			cout << "Training complete but still have errors\n" << endl;
			break;
		}
    }

	// Save weight
	if (saveAndLoad)
	{
		SaveWeight();
	}
}

#pragma region Training function
// Initialize bias to 1
// Initialize all weights to 0
// Get rowLength
void Perceptron::InitializeW(vector< vector < float > > X)
{
	cout << "Initialize W" << endl;

	// Get length of row
	vector< vector <float> >::iterator iteratorX = X.begin();
	vector<float>::iterator iteratorXData = (*iteratorX).begin();
	int indexXData = 0;	

	while(iteratorXData != (*iteratorX).end())
	{
		// Increment
		iteratorXData++;
		indexXData++;
	}

	rowLength = indexXData + 1;	// Set row length, including bias

	// Load Weight
	if (!saveAndLoad || !LoadWeight())
	{
		// Initialize if cannot load weight
		float WArray[indexXData];	// Set W array to all 0, except 1st is 1 for bias

		WArray[0] = 1;	// bias
		
		for (int index = 1; index < rowLength; index++)
		{
			WArray[index] = 0;	// weight
		}				

		W = new Vector<float>(WArray, rowLength);	// (b, 0, ..., 0)					
	}
}

// Return Vector<float> from vector<float> data
Vector<float> Perceptron::GetRowX(vector< vector <float> >::iterator iteratorX)
{
	float datas[rowLength];	// include bias

	datas[0] = 1;	// bias multiplier

	for (int indexXDataCount = 1; indexXDataCount < rowLength; indexXDataCount++)
	{
		datas[indexXDataCount] = (*iteratorX).at(indexXDataCount-1);	// value
	}

	return Vector<float>(datas, rowLength);	// (1, x0, ..., xn-1)
}
#pragma endregion

/*
Input: a vector x
Output: 1 or -1 
*/
int Perceptron::Predict(Vector<float> x)
{
	// z = b + w1*x1 + w2*x2 + ... wn*xn
	float z = VectorDotProduct(*W, x);
	return z >= 0 ? 1 : -1;
}

// Return a dataset of predict Y in a dataset X
vector <int> Perceptron::Predict(vector< vector < float > > X)
{
	// Throw if no training before
	if (!W)
	{
		__throw_logic_error("The weight is not trained");
	}

	vector<int> Y;

	// Loop X
	vector< vector <float> >::iterator iteratorX = X.begin();
	int indexX = 0;

	// Predict X for each row
	while(iteratorX != X.end())	// Loop every row
	{
		// Get Vector<float> form
		Vector<float> rowX = GetRowX(iteratorX);

		// Get predict and real of Y
		int predictY = Predict(rowX);	// Predict current row

		// Push result to Y
		Y.push_back(predictY);

		// Increment
		iteratorX++;
		indexX++;
	}

	return Y;
}

void Perceptron::PrintPredictResult(vector< vector < float > > X, vector < float > Y)
{
	cout << "-------------------------" << "Print PredictResult" << "-------------------------" << endl;

	vector<int> prediction = Predict(X);
	vector<int>::iterator iteratorPrediction = prediction.begin();
	int indexPrediction = 0;
	int errors = 0;

	cout << "Row\tReal Y\tPredict Y" << endl;	// Row header 
	// Print predict result of vector<int>
	while(iteratorPrediction != prediction.end())	// Loop every row
	{
		int trueY = Y.at(indexPrediction);
		int predictY = prediction.at(indexPrediction);
		bool wrong = trueY != predictY;

		// Print predict result in current row
		cout << indexPrediction << "\t" << trueY << "\t" << predictY << (wrong ? "\terror" : "") << endl;

		// Add errors count
		if (wrong)
		{
			errors += 1;
		}

		// Increment
		iteratorPrediction++;
		indexPrediction++;
	}
	cout << "Error: " << errors << endl;

	cout << "-------------------------" << "end Print PredictResult" << "-------------------------" << endl;
}

// Save and Load
void Perceptron::SaveWeight()
{
	cout << "-------------------------" << "Save Weight" << "-------------------------" << endl;

	ofstream myfile;
	myfile.open ("weight.txt");

	// Convert W to text
	Vector<float> w = *W;
	string weight = w.GetData();
	myfile << weight;

	cout << "Output weight: \n" << weight << endl;

	myfile.close();

	cout << "-------------------------" << "End Save Weight" << "-------------------------\n" << endl;
}

bool Perceptron::LoadWeight()
{
	ifstream myfile;
	myfile.open ("weight.txt");

	// Convert text to W
	string weight;
	
	if (myfile.is_open())
	{
		cout << "-------------------------" << "Load Weight" << "-------------------------" << endl;

		float WArray[rowLength];	// Set W array to all 0, except 1st is 1 for bias
		int index = 0;

		// Set weight to W
		while(getline(myfile, weight))
		{
			WArray[index++] = stof(weight);	// bias
		}

		W = new Vector<float>(WArray, rowLength);	// (b, 0, ..., 0)

		myfile.close();

		// Print input weight
		cout << "Input weight: ";
		Vector<float> w = *W;
		w.Print();

		cout << "-------------------------" << "End Load Weight" << "-------------------------\n" << endl;

		return true;
	}
	else
	{
		cout << "No input weight" << endl;

		return false;
	}
}