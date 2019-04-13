//
// Created by Jacky Lam on 2019-04-11.
//

#include "ANN.h"

#include "iostream"
#include <stdarg.h>     /* va_list, va_start, va_arg, va_end */
using namespace std;

ANN::ANN(int hiddenLayer, ...)
    : hiddenLayer(hiddenLayer)
    , inputNeuron(0)
    , outputNeuron(10)
    , W(0)
    , b(0)
    , a(0)
{
    // Initialize hidden layer neuron for each hidden layer
    hiddenLayerNeuron = new int[hiddenLayer];

    va_list vl;
    va_start(vl, hiddenLayer);

    for (int hiddenLayerIndex = 0; hiddenLayerIndex < hiddenLayer; hiddenLayerIndex++)
    {
        hiddenLayerNeuron[hiddenLayerIndex] = va_arg(vl, int);
    }

    va_end(vl);

    // Debug hidden layer neuron
    cout << "Number of hidden layer: " << hiddenLayer << endl;
    cout << "Number of neuron in hidden layer: ";
    for (int hiddenLayerIndex = 0; hiddenLayerIndex < hiddenLayer; hiddenLayerIndex++)
    {
        cout << hiddenLayerNeuron[hiddenLayerIndex] << " ";
    }
    cout << endl;
}

//void ANN::Train(std::vector <float> <vector> X, float r,int miniBatchSize,int epoch){
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
//}