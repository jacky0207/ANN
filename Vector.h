#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <string>

template<typename T>
class Vector
{
private:
    T *data;
    int size;
    bool vertical;
public:
    Vector(T *data, int size, bool vertical = false);
    int GetSize();
    double GetMean();
    T GetL1Norm();
    double GetEuclideanNorm();
    void Multiplication(T scalar);

    T* GetDataClone();
    bool isColumnVector();
    void Print();   // Debug
    std::string GetData();
};

#endif