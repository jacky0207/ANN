//
// Created by Jacky Lam on 2019-04-11.
//

#include "ANN.h"

#include "iostream"
#include <vector>
#include <cmath>
#include <stdarg.h>     /* va_list, va_start, va_arg, va_end */
using namespace std;

ANN::ANN(int hiddenLayer, ...)
    : layer(hiddenLayer + 2)
    , WList(0)
    , bList(0)
    , aList(0)
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

    cout << "-------------------------" << "Initialize a" << "-------------------------" << endl;

    // Start train
    for (int indexX = 0; indexX < X.size(); indexX++)   // Loop every sample in X
    {
        for (int layerIndex = 1; layerIndex < layer; layerIndex++)  // Loop every layer except input layer
        {
            FeedForward(layerIndex, X.at(indexX));
        }
    }

    // Debug aList
    PrintActivation();

    cout << "-------------------------" << "end Initialize a" << "-------------------------" << endl;


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
    neuron[0] = X.at(0).size();

    // Debug layer neuron
    PrintNeuron();

    // Clear W if trained before
    if (WList.size() != 0)
    {
        WList.clear();
    }

    for (int layerIndex = 1; layerIndex < layer; layerIndex++)  // l = 2-L, loop hidden layer
    {
        vector<vector<float> > W;    // W of current layer
        int lastLayerNeuron = neuron[layerIndex - 1];   // last layer neuron number

        for (int neuronIndex = 0; neuronIndex < neuron[layerIndex]; neuronIndex++)   // row = neuron number
        {
            W.push_back(vector<float>(lastLayerNeuron, 0)); // column = last layer neuron number
        }

        WList.push_back(W);    // push to W
    }

    // Debug W
    PrintWeight();

    cout << "-------------------------" << "end Initialize W" << "-------------------------" << endl;

    cout << endl;

    cout << "-------------------------" << "Initialize b" << "-------------------------" << endl;

    // Clear b if trained before
    if (bList.size() != 0)
    {
        bList.clear();
    }

    for (int layerIndex = 1; layerIndex < layer; layerIndex++)  // l = 2-L, loop hidden layer
    {
        bList.push_back(vector<float>(neuron[layerIndex], 1));
    }

    // Debug b
    PrintBias();

    cout << "-------------------------" << "end Initialize b" << "-------------------------" << endl;

    // Clear a if trained before
    if (aList.size() != 0)
    {
        aList.clear();
    }
}

// --------------------------------------------------------------------------- Feed Forward Function ---------------------------------------------------------------------------
void ANN::FeedForward(int l, vector<float> sample)
{
    vector< float > a = SigmoidActivation(l, sample);
    aList.push_back(a);
}

vector<float> ANN::SigmoidActivation(int l, vector<float> sample)
{
    vector<float> a, z, previousA, b;
    vector<vector<float> > w;

    // Retrieve w, previous a and b
    w = WList.at(l - 1);
    previousA = l == 1 ? sample : aList.at(l - 2); // get sample or previous a
    b = bList.at(l - 1);

    // Calculate z and a
    // zl = wla(l-1) + bl
    for (int neuronIndex = 0; neuronIndex < w.size(); neuronIndex++)    // every neuron
    {
        vector<float> neuronWeight = w.at(neuronIndex);

        float summation = 0;    // summation

        summation += b.at(neuronIndex);

        for (int previousAIndex = 0; previousAIndex < previousA.size(); previousAIndex++)   // every previous layer neuron
        {
//            cout << neuronWeight.at(previousAIndex) << "*" << previousA.at(previousAIndex) << endl;
            summation += neuronWeight.at(previousAIndex) * previousA.at(previousAIndex);
        }

        z.push_back(summation);
    }

    // al = sgm(zl)
    // sgm(x) = 1 / (1 + exp(x))
    for (int neuronIndex = 0; neuronIndex < w.size(); neuronIndex++)
    {
        a.push_back(1 / (1 + exp(z.at(neuronIndex))));
//        a.push_back(z.at(neuronIndex));
    }

    return a;
}
// --------------------------------------------------------------------------- Feed Forward Function ---------------------------------------------------------------------------

// --------------------------------------------------------------------------- Debugging ---------------------------------------------------------------------------
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

    for (int layerIndex = 1; layerIndex < WList.size() + 1; layerIndex++)
    {
        cout << "layer " << layerIndex << endl;

        vector<vector<float> > layerW = WList.at(layerIndex - 1);    // W of current layer

        for (int neuronIndex = 0; neuronIndex < layerW.size(); neuronIndex++)   // row = neuron number
        {
            vector<float> rowW = layerW.at(layerIndex - 1);    // W of current layer

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
        vector<float> b = bList.at(layerIndex - 1);    // b of current layer
        cout << "layer " << layerIndex << endl;

        for (int neuronIndex = 0; neuronIndex < b.size(); neuronIndex++)   // length = neuron number
        {
            cout << b.at(neuronIndex) << " ";
        }

        cout << endl;
    }
}

void ANN::PrintActivation()
{
    for (int layerIndex = 1; layerIndex < layer; layerIndex++)
    {
        vector<float> a = aList.at(layerIndex - 1);    // a of current layer
        cout << "layer " << layerIndex << endl;

        for (int neuronIndex = 0; neuronIndex < a.size(); neuronIndex++)   // length = neuron number
        {
            cout << a.at(neuronIndex) << " ";
        }

        cout << endl;
    }
}