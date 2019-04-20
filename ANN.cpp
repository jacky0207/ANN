//
// Created by Jacky Lam on 2019-04-11.
//

#include "ANN.h"
#include <time.h>
#include "iostream"
#include "string"
#include <vector>
#include <cmath>
#include <stdarg.h>     /* va_list, va_start, va_arg, va_end */

using namespace std;

vector<int> labelToVector(int y) {
    vector<int> trueY;
    for (int i = 0; i < 10; i++) {
        if (i == y) {
            trueY.push_back(1);
        } else {
            trueY.push_back(0);
        }
    }
    return trueY;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float sigmoid_derivative(float x) {
    double fx = sigmoid(x);
    return fx * (1 - fx);
}

void transpose(vector<vector<float> > &b) {
    if (b.size() == 0)
        return;

    vector<vector<float> > trans_vec(b[0].size(), vector<float>());

    for (int i = 0; i < b.size(); i++) {
        for (int j = 0; j < b[i].size(); j++) {
            trans_vec[j].push_back(b[i][j]);
        }
    }

    b = trans_vec;    // <--- reassign here
}


ANN::ANN(int hiddenLayer, ...)
        : layer(hiddenLayer + 2), WList(0), bList(0), aList(0), inputSumList(0) {
    cout << "-------------------------" << "Initialize neuron" << "-------------------------" << endl;

    // Initialize hidden layer neuron for each hidden layer
    neuron = new int[layer];   // also include input and output

    // input layer
    neuron[0] = 0;

    // hidden layer
    va_list vl;
            va_start(vl, hiddenLayer);

    for (int layerIndex = 1; layerIndex < layer - 1; layerIndex++) {
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
                int epoch) {
    // Initialize W
    InitializeW(X);

    cout << endl;

    cout << "--------------------------------------------------" << "Start Train" << "--------------------------------------------------" << endl;

    // using built-in random generator:
    std::random_shuffle ( X.begin(), X.end() );

    int numberOfMiniBatch = ceil(X.size() / miniBatchSize);

    cout << "Size of X: " << X.size() << endl;
    cout << "Mini-batch size: " << miniBatchSize << endl;
    cout << "Number of mini-batch: " << numberOfMiniBatch << endl;

    // int runCount = 0;

    // Output data
    // Store list of all samples in the mini-batch
    // sample->layer->value
    vector<vector<vector<float> > > miniBatchErrorList; // all error list in a mini-batch
    vector<vector<vector<float> > > miniBatchAList; // all a list in a mini-batch

    // numberOfMiniBatch * miniBatchSize = X.size()
    for (int epochIndex = 0; epochIndex < epoch; ++epochIndex)  // Loop epoch
    {
        cout << "--------------------------------------------------" << "Epoch " << epochIndex << "--------------------------------------------------" << endl;

        for (int miniBatchIndex = 0; miniBatchIndex < numberOfMiniBatch; ++miniBatchIndex)  // Loop mini-batch
        {
            cout << "--------------------------------------------------" << "Mini-batch " << miniBatchIndex << "--------------------------------------------------" << endl;

            cout << "Before" << endl;
            PrintWeight();
            PrintBias();

            if (miniBatchErrorList.size() != 0)
            {
                miniBatchErrorList.clear();
                miniBatchAList.clear();
            }

            for (int correlatedIndexX = 0; correlatedIndexX < miniBatchSize; ++correlatedIndexX)    // Loop samples in mini-batch
            {
                int indexX = miniBatchIndex * miniBatchSize + correlatedIndexX; // actual index of X
                cout << "Index X: " << indexX << endl;

                cout << "-------------------------" << "Initialize a" << "-------------------------" << endl;

                clearList();

                for (int layerIndex = 1; layerIndex < layer; layerIndex++)  // Loop every layer except input layer
                {
                    FeedForward(layerIndex, X.at(indexX), indexX);
                    // cout << "processing layer : " << layerIndex << endl;
                }

                // PrintActivation();            
                // PrintBias();

                cout << "-------------------------" << "end Initialize a" << "-------------------------" << endl;

                float totalError = calculate_total_error(Y.at(indexX));

                cout << "Sample : " << indexX << " ; Total Error : " << totalError << " in epoch : " <<  epochIndex << endl;

                // Find Last layer error
                vector<float> error = lastLayerError(Y.at(indexX));
                errorsList.insert(errorsList.begin(), error);

                // Find Hidden layer error
                for (int layerIndex = layer - 1; layerIndex > 1; layerIndex--) {    // l = L-1 to 2
                    // cout << "Hidden layer error : " << error.size() << endl;
                    vector<float> layerError = Error(layerIndex, error);
                    errorsList.insert(errorsList.begin(), layerError);
                    error = layerError;
                }

                // Push the input sample for update weight use first
                // m samples have m input
                aList.insert(aList.begin(), X.at(indexX));

                // Store list to temp list
                miniBatchAList.push_back(aList);
                miniBatchErrorList.push_back(errorsList);
            }

            // // Debug mini batch list
            // cout << endl;
            // cout << "-------------------------" << "Mini batch a list" << "-------------------------" << endl;
            // for (int i1 = 0; i1 < miniBatchSize; i1++)
            // {
            //     cout << "-------------------------" << "Sample " << i1 << "-------------------------" << endl;
            //     vector<vector<float> > aList = miniBatchAList.at(i1);
            //     for (int i2 = 0; i2 < aList.size(); i2++)
            //     {
            //         vector<float> a = aList.at(i2);
            //         for (int i3 = 0; i3 < a.size(); i3++)
            //         {
            //             cout << a.at(i3) << " ";                        
            //         }   
            //         cout << endl;
            //     }
            //     cout << "-------------------------" << "end Sample " << i1 << "-------------------------" << endl;
            //     cout << endl;
            // }
            // cout << "-------------------------" << "end Mini batch a list" << "-------------------------" << endl;
            // cout << endl;

            // // Debug mini batch list
            // cout << endl;
            // cout << "-------------------------" << "Mini batch loss list" << "-------------------------" << endl;
            // for (int i1 = 0; i1 < miniBatchSize; i1++)
            // {
            //     cout << "-------------------------" << "Sample " << i1 << "-------------------------" << endl;
            //     vector<vector<float> > errorList = miniBatchErrorList.at(i1);
            //     for (int i2 = 0; i2 < errorList.size(); i2++)
            //     {
            //         vector<float> error = errorList.at(i2);
            //         for (int i3 = 0; i3 < error.size(); i3++)
            //         {
            //             cout << error.at(i3) << " ";                        
            //         }   
            //         cout << endl;
            //     }   
            //     cout << "-------------------------" << "end Sample " << i1 << "-------------------------" << endl;
            //     cout << endl;
            // }
            // cout << "-------------------------" << "end Mini batch loss list" << "-------------------------" << endl;
            // cout << endl;
                     
            // Adjust Weights
            updateWeights(r, miniBatchAList, miniBatchErrorList);
                        
            cout << "--------------------------------------------------" << "end Mini-batch " << miniBatchIndex << "--------------------------------------------------" << endl;
            cout << endl;
        }

        cout << "--------------------------------------------------" << "end Epoch " << epochIndex << "--------------------------------------------------" << endl;
    }

    cout << "--------------------------------------------------" << "end Train" << "--------------------------------------------------" << endl;


    cout << "*******************************************************" << "Trained result" << "*******************************************************" << endl;
    PrintWeight();
    PrintBias();

    cout << "Final loss";

    // Start train
//     while (runCount < epoch) {

// //        using built-in random generator:
// //        std::random_shuffle ( X.begin(), X.end() );
//         cout << endl;
//         cout << "--------------------------------------------------" << "Epoch " << runCount << "--------------------------------------------------" << endl;

//         for (int indexX = 0; indexX < 3; indexX++)   // Loop every sample in X
//         {
//             cout << "-------------------------" << "Initialize a" << "-------------------------" << endl;

//             clearList();

//             for (int layerIndex = 1; layerIndex < layer; layerIndex++)  // Loop every layer except input layer
//             {
//                 FeedForward(layerIndex, X.at(indexX), indexX);
//                 // cout << "processing layer : " << layerIndex << endl;
//             }

//             PrintActivation();            
//             PrintBias();

//             cout << "-------------------------" << "end Initialize a" << "-------------------------" << endl;

//             float totalError = calculate_total_error(Y.at(indexX));

//             // cout << "Sample : " << indexX << " ; Total Error : " << totalError << " in epoch : " <<  runCount << endl;

//             // Find Last layer error
//             vector<float> error = lastLayerError(Y.at(indexX));
//             errorsList.insert(errorsList.begin(), error);

//             // Find Hidden layer error
//             for (int layerIndex = layer - 1; layerIndex > 1; layerIndex--) {
//                 // cout << "Hidden layer error : " << error.size() << endl;
//                 vector<float> layerError = Error(layerIndex, error);
//                 errorsList.insert(errorsList.begin(), layerError);
//                 error = layerError;
//             }

//             // Adjust Weights
//             updateWeights(r, X.size(), X.at(indexX));
//             // cout << endl;
//         }
        
//         cout << "--------------------------------------------------" << "end Epoch " << runCount << "--------------------------------------------------" << endl;
//         cout << endl;
        
//         runCount++;        
//     }

    cout << "--------------------------------------------------" << "end Start Train" << "--------------------------------------------------" << endl;

    // Debug aList


//    PrintinputSumList();


//    PrintErrorList();



}

void ANN::InitializeW(vector<vector<float> > X) {
    cout << "-------------------------" << "Initialize W" << "-------------------------" << endl;

    // Initialize input layer
    neuron[0] = X.at(0).size();

    // Debug layer neuron
    PrintNeuron();

    // Clear W if trained before
    if (WList.size() != 0) {
        WList.clear();
    }

    for (int layerIndex = 1; layerIndex < layer; layerIndex++)  // l = 2-L, loop hidden layer
    {

        vector<vector<float> > W;    // W of current layer
        int lastLayerNeuron = neuron[layerIndex - 1];   // preivous layer neuron number

        for (int neuronIndex = 0; neuronIndex < neuron[layerIndex]; neuronIndex++)   // row = neuron number
        {
            vector<float> wRow;


            for (int j = 0; j < lastLayerNeuron; j++) {

                float min = 0.1;
                float max = 1;
                float num = (max - min) * rand() / (RAND_MAX + 1.0) + min;
                wRow.push_back(num);
                // if (layerIndex == 3) {
                //    cout << "init layer: " << layerIndex << " , neuronIndex is " << neuronIndex << " , num is: " << num << endl;
                // }
            }


            W.push_back(vector<float>(wRow)); // column = last layer neuron number
            wRow.clear();
        }

        WList.push_back(W);    // push to W
    }

    // Debug W
//     PrintWeight();

    cout << "-------------------------" << "end Initialize W" << "-------------------------" << endl;

    cout << endl;

    cout << "-------------------------" << "Initialize b" << "-------------------------" << endl;

    // Clear b if trained before
    if (bList.size() != 0) {
        bList.clear();
    }

    // Clear inputSumList if trained before
    if (inputSumList.size() != 0) {
        inputSumList.clear();
    }

    for (int layerIndex = 1; layerIndex < layer; layerIndex++)  // l = 2-L, loop hidden layer
    {
        bList.push_back(vector<float>(neuron[layerIndex], 1));
    }

    // Debug b
    PrintBias();

    cout << "-------------------------" << "end Initialize b" << "-------------------------" << endl;


    // Clear a if trained before
    if (aList.size() != 0) {
        aList.clear();
    }
}

// --------------------------------------------------------------------------- Feed Forward Function ---------------------------------------------------------------------------
void ANN::FeedForward(int l, vector<float> sample, int indexX) {
    vector<float> a = SigmoidActivation(l, sample);
    aList.push_back(a);
}

vector<float> ANN::SigmoidActivation(int l, vector<float> sample) {
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

        for (int previousAIndex = 0;
             previousAIndex < previousA.size(); previousAIndex++)   // every previous layer neuron
        {
//            cout << neuronWeight.at(previousAIndex) << "*" << previousA.at(previousAIndex) << endl;
            summation += neuronWeight.at(previousAIndex) * previousA.at(previousAIndex);

        }

        z.push_back(summation);
    }


    inputSumList.push_back(z);


    // al = sgm(zl)
    // sgm(x) = 1 / (1 + exp(x))
    for (int neuronIndex = 0; neuronIndex < w.size(); neuronIndex++) {

        a.push_back(sigmoid(z.at(neuronIndex)));
//        a.push_back(z.at(neuronIndex));
    }

    return a;
}
// --------------------------------------------------------------------------- Feed Forward Function ---------------------------------------------------------------------------

float ANN::calculate_total_error(int trueYLabel) {
    float sum = 0;
    vector<float> a = aList.at(layer - 2);
    vector<int> labelVector = labelToVector(trueYLabel);
    for (size_t i = 0; i < labelVector.size(); i++) {
        sum += 0.5f * (labelVector.at(i) - a.at((i))) * (labelVector.at(i) - a.at((i)));
    }
    return sum;
}


vector<float> ANN::lastLayerError(int trueY) {
    vector<float> lastLayerErrors;
    vector<float> a = aList.at(layer - 2);

    vector<int> labelVector = labelToVector(trueY);

    for (int i = 0; i < labelVector.size(); i++) {
//        float delta = labelVector.at(i) - a.at(i);

        float delta = a.at(i) - labelVector.at(i);
        float error = delta * sigmoid_derivative(inputSumList.at(layer - 2).at(i));
//        cout << "error of " << trueY << " in last layer " << i << " is " << error <<endl;
        lastLayerErrors.push_back(error);
    }
    return lastLayerErrors;
}


vector<float> ANN::Error(int layer, vector<float> error) {  // l = L-1 to 2
    vector<float> LayerErrors;
    int nextLayerNeuronNumbers = neuron[layer];
    int currentLayerNeronNumbers = neuron[layer - 1];

//    cout << "Curent layer is: " << layer - 1 << " size is " << currentLayerNeronNumbers << endl;
//    cout << "Next layer size: " << nextLayerNeuronNumbers << endl;

    vector<vector<float> > layerW = WList.at(layer - 1);    // W of current layer

//    cout << "layerW  size  is: " << layerW.size() <<  endl;
//
//    cout << "error  size  is: " << error.size() <<  endl;

    // current = 16, next = 12
    // next w = 12 * 16
    // next wT = 16 * 12
    // next loss = 12 * 1

    // a = 16 * 1
    // wTLoss = 16 * 12 and 12 * 1 -> 16 * 1

    // Transpose Vector
    vector<vector<float> > layerWT = layerW;
    transpose(layerWT); // 16 * 12
//
//    cout << "layerW  size  is: " << layerW.size() <<  endl;
//    cout << "layerWT  size  is: " << layerWT.size() <<  endl;

    for (int c = 0; c < currentLayerNeronNumbers; c++) {

        double sum = 0.0;

        for (int n = 0; n < nextLayerNeuronNumbers; n++) {

            vector<float> w = layerWT.at(c);
//            cout << "n " << n << endl;
//            cout << "w " << w.at(n) << endl;
//            cout << "error " << error.at(n) << endl;
            sum += w.at(n) * error.at(n);
        }

//        cout << " sum : " << sum << endl ;
        vector<float> laySumList;
        if (layer - 2 < 0) {
            laySumList = inputSumList.at(0);
        } else {
            laySumList = inputSumList.at(layer - 2);
        }

//        cout << "laySumList  size  is: " << laySumList.size() <<  endl;
        float error = sum * sigmoid_derivative(laySumList.at(c));
        LayerErrors.push_back(error);
    }
    return LayerErrors;

//    }
}

// void ANN::updateWeights(float r, vector<float> sample) {
// sample->layer->value
void ANN::updateWeights(float r, 
                        vector<vector<vector<float> > > miniBatchAList,     // includes input at front
                                                                            // l = 1 to L
                        vector<vector<vector<float> > > miniBatchErrorList) // l = 2 to L
{
    cout << "--------------------------------------------------" << "Update Weight" << "--------------------------------------------------" << endl;

    cout << "-------------------------" << "Update Weight" << "-------------------------" << endl;

    for (int layerIndex = layer - 1; layerIndex >= 1; layerIndex--) // l = L to 2
    {
        cout << "-------------------------" << "Layer " << layerIndex << "-------------------------" << endl;

        // Get Weight and Bias of this layer
        vector<vector<float> > *w = &WList.at(layerIndex - 1);
        vector<float> *b = &bList.at(layerIndex - 1);

        // Summation of w and b update
        vector<vector<float> > wSum;
        vector<float> bSum;

        int row = miniBatchErrorList.at(0).at(layerIndex - 1).size();   // Current layer neuron [m]
        int column = miniBatchAList.at(0).at(layerIndex - 1).size();    // last layer neuron    [n]
                                                                        // Weight               [m*n]
        cout << "row * column = " << row << "*" << column << endl;
        
        for (int rowIndex = 0; rowIndex < row; rowIndex++)  // Initialize w[m*n] and b[m]
        {
            wSum.push_back(vector<float>(column, 0));
            bSum.push_back(0);
        }

        // Add up summation of w and b
        for (int miniBatchIndex = 0; miniBatchIndex < miniBatchAList.size(); miniBatchIndex++)   // loop every sample in mini batch
        {
            // a list and error list in current sample
            vector<vector<float> > aList = miniBatchAList.at(miniBatchIndex);
            vector<vector<float> > errorList = miniBatchErrorList.at(miniBatchIndex);

            // Get current layer error and previous layer a
            vector<float> a = aList.at(layerIndex - 1);
            vector<float> error = errorList.at(layerIndex - 1);

            // Add up sum
            // cout << "W sum" << endl;
            // cout << "Error" << endl;
            for (int rowIndex = 0; rowIndex < row; rowIndex++)
            {
                for (int columnIndex = 0; columnIndex < column; columnIndex++)
                {
                    // w[m, n] = loss[m] * a[n]
                    wSum.at(rowIndex).at(columnIndex) += error.at(rowIndex) * a.at(columnIndex);  // loss * a
                    // cout << wSum.at(rowIndex).at(columnIndex) << " ";
                    // cout << error.at(rowIndex) * a.at(columnIndex) << " ";
                }               
                // cout << endl;

                bSum.at(rowIndex) += error.at(rowIndex);    // loss
            }
            // cout << endl;
            // cout << endl;
        }

        // cout << "wSum" << endl;
        // for (int rowIndex = 0; rowIndex < row; rowIndex++)
        // {
        //     for (int columnIndex = 0; columnIndex < column; columnIndex++)
        //     {
        //         cout << wSum.at(rowIndex).at(columnIndex) << " ";
        //     }
        //     cout << endl;
        // }
        // cout << endl;

        // cout << "bSum" << endl;
        // for (int rowIndex = 0; rowIndex < row; rowIndex++)
        // {
        //     cout << bSum.at(rowIndex) << " ";
        // }
        // cout << endl;

        // cout << "original W and b" << endl;
        // PrintWeight();
        // PrintBias();

        // Update W and b
        for (int rowIndex = 0; rowIndex < row; rowIndex++)
        {
            for (int columnIndex = 0; columnIndex < column; columnIndex++)
            {
                w->at(rowIndex).at(columnIndex) -= r / miniBatchAList.size() * wSum.at(rowIndex).at(columnIndex);  // loss * a
            }     
            
            b->at(rowIndex) -= r / miniBatchAList.size() * bSum.at(rowIndex);
        }

        cout << "Updated W and b" << endl;
        // PrintWeight();
        // PrintBias();

        cout << "-------------------------" << "end Layer " << layerIndex << "-------------------------" << endl;
    }

    cout << "--------------------------------------------------" << "end Update Weight" << "--------------------------------------------------" << endl;
}

void ANN::clearList() {
    // Clear inputSumList if trained before
    if (inputSumList.size() != 0) {
        inputSumList.clear();
    }
    // Clear a if trained before
    if (aList.size() != 0) {
        aList.clear();
    }

    if (errorsList.size() != 0) {
        errorsList.clear();
    }
}

// Inference
void ANN::predict(std::vector<std::vector<float> > X, std::vector<float> Y) {
    for (int indexX = 0; indexX < 3; indexX++) {
        clearList();
        for (int layerIndex = 1; layerIndex < layer; layerIndex++)  // Loop every layer except input layer
        {
            FeedForward(layerIndex, X.at(indexX), indexX);
        }
        PrintActivation();

        cout << "Number " << indexX << " data is " <<  Y.at(indexX) << endl;
    }

    PrintWeight();
    PrintBias();
}


// --------------------------------------------------------------------------- Debugging ---------------------------------------------------------------------------
void ANN::PrintNeuron() {
    cout << "Number o layer: " << layer << endl;

    cout << "Number of neuron in hidden layer: ";
    for (int layerIndex = 0; layerIndex < layer; layerIndex++) {
        cout << neuron[layerIndex] << " ";
    }

    cout << endl;
}

void ANN::PrintWeight() {
    cout << "-------------------------" << "Print Weight" << "-------------------------" << endl;

    cout << "WList size " << WList.size() << endl;
    for (int layerIndex = 1; layerIndex < WList.size() + 1; layerIndex++) {
        cout << "layer " << layerIndex << endl;

        vector<vector<float> > layerW = WList.at(layerIndex - 1);    // W of current layer

        cout << "Layer " << layerIndex << " have " << layerW.size() << " row" << endl;

        for (int neuronIndex = 0; neuronIndex < layerW.size(); neuronIndex++)   // row = neuron number
        {
            vector<float> rowW = layerW.at(neuronIndex);    // W of the neuron

            cout << "Layer " << layerIndex << " have " << layerW.size() << " row" << " each row size : " << rowW.size()
                 << endl;

            for (int row = 0; row < rowW.size(); row++) {
                cout << rowW.at(row) << " ";
            }

            cout << endl;
        }

        cout << endl;
    }

    cout << "-------------------------" << "end Print Weight" << "-------------------------" << endl;
}

void ANN::PrintBias() {
    for (int layerIndex = 1; layerIndex < layer; layerIndex++) {
        vector<float> b = bList.at(layerIndex - 1);    // b of current layer
        cout << "layer " << layerIndex << endl;

        for (int neuronIndex = 0; neuronIndex < b.size(); neuronIndex++)   // length = neuron number
        {
            cout << b.at(neuronIndex) << " ";
        }

        cout << endl;
    }
}

void ANN::PrintActivation() {
    for (int layerIndex = 1; layerIndex < layer; layerIndex++) {
        vector<float> a = aList.at(layerIndex - 1);    // a of current layer
        cout << "layer " << layerIndex << " Activation:" << endl;

        for (int neuronIndex = 0; neuronIndex < a.size(); neuronIndex++)   // length = neuron number
        {
            cout << a.at(neuronIndex) << " ";
        }

        cout << endl;
    }
}

void ANN::PrintinputSumList() {
    for (int layerIndex = 1; layerIndex < layer; layerIndex++) {
        vector<float> sum = inputSumList.at(layerIndex - 1);    // a of current layer
        cout << "layer " << layerIndex << " summation " << endl;

        for (int neuronIndex = 0; neuronIndex < sum.size(); neuronIndex++)   // length = neuron number
        {
            cout << sum.at(neuronIndex) << " ";
        }

        cout << endl;
    }
}

void ANN::PrintErrorList() {
    for (int layerIndex = 1; layerIndex < layer; layerIndex++) {
        vector<float> sum = errorsList.at(layerIndex - 1);    // a of current layer
        cout << "layer " << layerIndex << " error " << endl;

        for (int neuronIndex = 0; neuronIndex < sum.size(); neuronIndex++)   // length = neuron number
        {
            cout << sum.at(neuronIndex) << " ";
        }

        cout << endl;
    }
}