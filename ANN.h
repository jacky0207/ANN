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
    int layer;  // number of layer i.e input + hidden + output

    int* neuron;    // array of neuron number in every layer

    std::vector<std::vector<std::vector<float> > > WList;    // Weight layer 2-L, m = last layer neuron number
                                                                // w = (w00, ..., w0m
                                                                //      ...
                                                                //      wn0, ..., wnm)

    std::vector<std::vector<float> > bList; // bias layer 2-L
                                            // b = (b0
                                            //      ...
                                            //      bn)

    std::vector<std::vector<float> > aList; // activation sgm(z) layer 2-L
                                            // a = (a0
                                            //      ...
                                            //      an)
     std::vector<std::vector<float> > inputSumList; // input summation before activation function layer 2-L
    // a = (a0
    //      ...
    //      an)

    std::vector<std::vector<float> > errorsList; // input summation before activation function layer 2-L
    // a = (a0
    //      ...
    //      an)

private:
    void InitializeW(std::vector<std::vector<float> > X);

    void FeedForward(int l, std::vector<float> sample, int indexX);
    std::vector<float> SigmoidActivation(int l, std::vector<float> sample);


    void backPropagation(int l);

    void LossFunction(std::vector<float> trueY, std::vector<float> predictY); // C = (C1
                                                                                    //      ...
    std::vector<float> lastLayerError(int trueYLabel);                                                       //      Cn)
    std::vector<float> Error(int layer, std::vector<float> error);
    // void updateWeights(float r, std::vector<float> sample);
    void updateWeights(float r, int m, std::vector<float> sample);  // m = number of sample
    float calculate_total_error(int trueYLabel);
    void clearList();


    // Save and load
    void SaveWeight();
    void LoadWeight();

    // Debug
    void PrintNeuron();
    void PrintWeight();
    void PrintBias();
    void PrintActivation();
    void PrintinputSumList();
    void PrintErrorList();
public:
    explicit ANN(int hiddenLayer, ...);    // Set the number of layers
                                        // and the number of neurons in each layer

    void Train(std::vector<std::vector<float> > X,  // start from 2nd digit of dataset
               std::vector<float> Y, // 1st digit of dataset
               float r,
               int miniBatchSize,   // number of mini-batch = total number of dataset / mini-batch size
               int epoch);
};


#endif //GROUPPROJECT1_ANN_H