//
// Created by Jacky Lam on 2019-04-11.
//

#ifndef GROUPPROJECT1_ANN_H
#define GROUPPROJECT1_ANN_H

#include <algorithm>    // std::random_shuffle


// Train -> FeedForward(0->1->2->...->n) -> Backprogagation(loss->partial derivative->errors) -> sgd
class ANN {
private:
    void FeedForward();
    void BackProgagation();
    void LossFunction(std::vector < float > trueY, std::vector < float > predictY);

public:
    void Train(std::vector< std::vector < float > > X,  // For predict value
               float r,
               int miniBatchSize,
               int epoch);
};


#endif //GROUPPROJECT1_ANN_H
