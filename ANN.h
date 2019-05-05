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

    int blockNumber;

private:
    void InitializeW(std::vector<std::vector<float> > X);

    // void FeedForward(int l, std::vector<float> sample, int indexX);
    // std::vector<float> SigmoidActivation(int l, std::vector<float> sample);
    // Changes aList and zList to local
    // Every sample has its own aList and zList
    void FeedForward(int l, std::vector<float> sample, int indexX, std::vector<std::vector<float> > &aList, std::vector<std::vector<float> > &inputSumList);
    std::vector<float> SigmoidActivation(int l, std::vector<float> sample, std::vector<std::vector<float> > &aList, std::vector<std::vector<float> > &inputSumList);


    void backPropagation(int l);

    void LossFunction(std::vector<float> trueY, std::vector<float> predictY); // C = (C1
                                                                                    //      ...
    // std::vector<float> lastLayerError(int trueYLabel);                                                       //      Cn)
    std::vector<float> lastLayerError(int trueYLabel, std::vector<std::vector<float> > aList, std::vector<std::vector<float> > inputSumList);                                                       //      Cn)
    // std::vector<float> Error(int layer, std::vector<float> error);
    std::vector<float> Error(int layer, std::vector<float> error, std::vector<std::vector<float> > inputSumList);
    // void updateWeights(float r, std::vector<float> sample);
    void updateWeights(float r, 
                        std::vector<std::vector<std::vector<float> > > miniBatchAList, 
                        std::vector<std::vector<std::vector<float> > > miniBatchErrorList);
    // float calculate_total_error(int trueYLabel);
    float calculate_total_error(int trueYLabel, std::vector<std::vector<float> > aList);
    void clearList();

    // Save and load sub function
    void SaveConfig();
    void SaveWeight();
    void SaveBias();
    void LoadConfig();
    void LoadWeight();
    void LoadBias();

    // Debug
    void PrintNeuron();
    void PrintWeight();
    void PrintBias();
    void PrintActivation();
    // void PrintinputSumList();
    void PrintinputSumList(std::vector<std::vector<float> > inputSumList);
    void PrintErrorList();
public:
    explicit ANN(int hiddenLayer, ...);    // Set the number of layers
                                        // and the number of neurons in each layer

    void Train(std::vector<std::vector<float> > X,  // start from 2nd digit of dataset
               std::vector<float> Y, // 1st digit of dataset
               float r,
               int miniBatchSize,   // number of mini-batch = total number of dataset / mini-batch size
               int epoch,
               int thread_count,   // number of thread
               int blockNumber);    // number of block for cache blocking

    // Compare result
    void predict(std::vector<std::vector<float> > X,
                 std::vector<float> Y);

    // Save and load
    void Save();  // Save automatically
                        // Each layer each file
    void Load();  // Load if layer and neuron match 
};


#endif //GROUPPROJECT1_ANN_H