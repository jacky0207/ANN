#ifndef _PERCEPTRO_H_
#define _PERCEPTRO_H_

#include <vector>
#include "Vector.h"
#include "Vector.cpp"

class Perceptron
{
private:
    // Data for training
    Vector<float> *W;   // Weight
    int rowLength;  // length of W
    float r;    // learning rate
    int epoch;  // Number of looping dataset X

    bool saveAndLoad;   // Execute LoadWeight()
    
    // Make prediction on an input
    int Predict(Vector<float> x);
    std::vector < int > Predict(std::vector< std::vector < float > > X);    // demo

    // Save/Load function
    void SaveWeight();
    bool LoadWeight();

public:
    explicit Perceptron(float r, int epoch, bool inputWeight = false);

    // Set r and epoch
    void SetLearningRate(float r);
    void SetEpoch(int epoch);
    void SetInputWeight(bool inputWeight);

    // Train bias and weight
    void Train(std::vector< std::vector < float > > X,  // For predict value
                std::vector<float> Y);  // True value
    void Train(std::vector< std::vector < float > > X,  // For predict value
                std::vector<float> Y,   // True value
                float r,
                int epoch);

    // Return all predict Y of all rows in dataset X
    void PrintPredictResult(std::vector< std::vector < float > > X, vector < float > Y);    // demo

    // Training function
    void InitializeW(vector< vector < float > > X);
	Vector<float> GetRowX(std::vector< std::vector <float> >::iterator iteratorX);
};

#endif