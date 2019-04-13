//
// Created by Jacky Lam on 2019-04-11.
//

#ifndef GROUPPROJECT1_ANN_H
#define GROUPPROJECT1_ANN_H

#include <algorithm>    // std::random_shuffle


// Train -> FeedForward(0->1->2->...->n) -> Backprogagation(loss->partial derivative->errors) -> sgd
class ANN {
private:
    int hiddenLayer;    // 2nd - (L-1)th layer
    int* hiddenLayerNeuron;
    int outputNeuron;   // 0-9

    int inputNeuron;  // column length of W
    std::vector< std::vector < float > > *W;    // array of W for each layer
                                                // w = (w00, ..., w0n
                                                //      ...
                                                //      wm0, ..., wmn)

    std::vector< float > *b;   // bias
                                // b = (b0
                                //      ...
                                //      bn)
    std::vector< float > *a;   // activation sgm(z)
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
