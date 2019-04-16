//
// Created by Jacky Lam on 2019-04-11.
//

#ifndef GROUPPROJECT1_ANN_H
#define GROUPPROJECT1_ANN_H

#include <vector>
#include <algorithm>    // std::random_shuffle


// Train -> FeedForward(0->1->2->...->n) -> Backprogagation(loss->partial derivative->errors) -> sgd
class ANN {
private:
//    int hiddenLayer;    // 2nd - (L-1)th layer
//    int* hiddenLayerNeuron; // array of number of neuron of hidden layer
//    int outputNeuron;   // 0-9
//
//    int inputNeuron;  // column length of W

    int layer;  // number of layer i.e input + hidden + output

    int* neuron;    // array of neuron number in every layer

    std::vector< std::vector < std::vector < float > > > *W;    // Weight layer 2-L, m = last layer neuron number
                                                                // w = (w00, ..., w0m
                                                                //      ...
                                                                //      wn0, ..., wnm)

    std::vector< std::vector< float > > *b;   // bias layer 2-L
                                // b = (b0
                                //      ...
                                //      bn)

    std::vector< std::vector< float > > *a;   // activation sgm(z) layer 2-L
                                // a = (a0
                                //      ...
                                //      an)

private:
    void InitializeW(std::vector< std::vector < float > > X);

    void FeedForward(int layer);
    void Error(int layer);
    void backPropagation(int layer);

    std::vector< float > sigmoidActivation(std::vector < float > z);    // a = (a1
                                                                        //      ...
                                                                        //      an)
    void LossFunction(std::vector < float > trueY, std::vector < float > predictY); // C = (C1
                                                                                    //      ...
                                                                                    //      Cn)

    // Save and load
    void SaveWeight();
    void LoadWeight();

    // Debug
    void PrintNeuron();
    void PrintWeight();
    void PrintBias();

public:
    explicit ANN(int hiddenLayer, ...);    // Set the number of layers
                                        // and the number of neurons in each layer

    void Train(std::vector< std::vector < float > > X,  // start from 2nd digit of dataset
               std::vector < float > Y, // 1st digit of dataset
               float r,
               int miniBatchSize,   // number of mini-batch = total number of dataset / mini-batch size
               int epoch);
};


#endif //GROUPPROJECT1_ANN_H
