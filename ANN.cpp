//
// Created by Jacky Lam on 2019-04-11.
//

#include "ANN.h"

#include "iostream"
#include <vector>
#include <stdarg.h>     /* va_list, va_start, va_arg, va_end */
using namespace std;

ANN::ANN(int hiddenLayer, ...)
    : layer(hiddenLayer + 2)
    , W(0)
    , b(0)
    , a(0)
{
    cout << "-------------------------" << "Initialize neuron" << "-------------------------" << endl;

    // Initialize hidden layer neuron for each hidden layer
    neuron = new int[layer];   // also include input and output

    // input layer
    neuron[0] = 0;

    // hidden layer
    va_list vl;
    va_start(vl, hiddenLayer);

    for (int layerIndex = 1; layerIndex < layer - 1; layerIndex++)
    {
        neuron[layerIndex] = va_arg(vl, int);
    }

    va_end(vl);

    // Output layer
    neuron[layer - 1] = 10;


    // Debug layer neuron
    PrintNeuron();

    cout << "-------------------------" << "end Initialize neuron" << "-------------------------" << endl;
}

void ANN::Train(vector<vector<float> > X,
                vector<float> Y,
                float r,
                int miniBatchSize,
                int epoch)
{
    // Initialize W
    InitializeW(X);

    // Start train
//    int runCount = 0;
//    while (runCount < epoch) {
//
//        // using built-in random generator:
//        std::random_shuffle ( X.begin(), X.end() );
//
//
//
//
//
//        runCount++;
//    }
}

void ANN::InitializeW(vector<vector<float> > X)
{
    cout << "-------------------------" << "Initialize W" << "-------------------------" << endl;

    // Initialize input layer
    neuron[0] = 32;

    // Debug layer neuron
    PrintNeuron();

    vector < vector < vector < float > > > W;  // temp W

    for (int layerIndex = 1; layerIndex < layer; layerIndex++)  // l = 2-L, loop hidden layer
    {
        vector <  vector < float > > layerW;    // W of current layer
        int lastLayerNeuron = neuron[layerIndex - 1];   // last layer neuron number

        for (int layerNeuron = 0; layerNeuron < neuron[layerIndex]; layerNeuron++)   // row = neuron number
        {
            layerW.push_back(vector< float >(lastLayerNeuron, 0)); // column = last layer neuron number
        }

        W.push_back(layerW);    // push to W
    }

    this->W = &W;

    // Debug W
    PrintWeight();

    cout << "-------------------------" << "end Initialize W" << "-------------------------" << endl;

    cout << endl;

    cout << "-------------------------" << "Initialize b" << "-------------------------" << endl;

    vector < vector < float > > b;  // temp b

    for (int layerIndex = 1; layerIndex < layer; layerIndex++)  // l = 2-L, loop hidden layer
    {
        b.push_back(vector < float > (neuron[layerIndex], 1));
    }

    this->b = &b;

    // Debug b
    PrintBias();

    cout << "-------------------------" << "end Initialize b" << "-------------------------" << endl;


//    // Get length of row
//    vector< vector <float> >::iterator iteratorX = X.begin();
//    vector<float>::iterator iteratorXData = (*iteratorX).begin();
//    int indexXData = 0;
//
//    while(iteratorXData != (*iteratorX).end())
//    {
//        // Increment
//        iteratorXData++;
//        indexXData++;
//    }
//
//    rowLength = indexXData + 1;	// Set row length, including bias

//    // Load Weight
//    if (!saveAndLoad || !LoadWeight())
//    {
//        // Initialize if cannot load weight
//        float WArray[indexXData];	// Set W array to all 0, except 1st is 1 for bias
//
//        WArray[0] = 1;	// bias
//
//        for (int index = 1; index < rowLength; index++)
//        {
//            WArray[index] = 0;	// weight
//        }
//
//        W = new Vector<float>(WArray, rowLength);	// (b, 0, ..., 0)
//    }
}

void ANN::PrintNeuron()
{
    cout << "Number o layer: " << layer << endl;

    cout << "Number of neuron in hidden layer: ";
    for (int layerIndex = 0; layerIndex < layer; layerIndex++)
    {
        cout << neuron[layerIndex] << " ";
    }

    cout << endl;
}

void ANN::PrintWeight()
{
    cout << "-------------------------" << "Print Weight" << "-------------------------" << endl;

    for (int layerIndex = 1; layerIndex < W->size() + 1; layerIndex++)
    {
        cout << "layer " << layerIndex << endl;

        vector < vector < float > > layerW = W->at(layerIndex - 1);    // W of current layer

        for (int neuronNumber = 0; neuronNumber < layerW.size(); neuronNumber++)   // row = neuron number
        {
            vector < float > rowW = layerW.at(layerIndex - 1);    // W of current layer

            for (int row = 0; row < rowW.size(); row++)
            {
                cout << rowW.at(row) << " ";
            }

            cout << endl;
        }

        cout << endl;
    }

    cout << "-------------------------" << "end Print Weight" << "-------------------------" << endl;
}

void ANN::PrintBias()
{
    for (int layerIndex = 1; layerIndex < layer; layerIndex++)
    {
        vector < float > layerb = b->at(layerIndex - 1);    // W of current layer
        cout << "layer " << layerIndex << endl;

        for (int neuronNumber = 0; neuronNumber < layerb.size(); neuronNumber++)   // length = neuron number
        {
            cout << layerb.at(neuronNumber) << " ";
        }

        cout << endl;
    }
}