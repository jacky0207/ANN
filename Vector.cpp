#ifndef _VECTOR_CPP_
#define _VECTOR_CPP_

#include "Vector.h"
#include "math.h"
#include <iostream>
#include <string>
using namespace std;

template<typename T>
Vector<T>::Vector(T *data, int size, bool vertical)
{
    // Initialize
    // this->data = data;
    this->data = new T[size];
    for (int index = 0; index < size; index++)
    {
        this->data[index] = data[index];
    }
    this->size = size;
    this->vertical = vertical;
}

template<typename T>
int Vector<T>::GetSize()
{
    return size;
}

template<typename T>
double Vector<T>::GetMean()
{
    double sum = 0;

    for (int index = 0; index < size; index++)
    {
        sum += data[index];
    }

    return sum / size;
}

template<typename T>
T Vector<T>::GetL1Norm()
{
    T norm = 0;

    for (int index = 0; index < size; index++)
    {
        norm += abs(data[index]);
    }

    return norm;
}

template<typename T>
double Vector<T>::GetEuclideanNorm()
{
    double normSquare = 0;

    for (int index = 0; index < size; index++)
    {
        normSquare += pow(data[index], 2);
    }

    return sqrt(normSquare);
}

template<typename T>
void Vector<T>::Multiplication(T scalar)
{
    if (*typeid(scalar).name() != 'i' && *typeid(scalar).name() != 'd')
    {
        cout << "Usage: please enter integer or double" << endl;
        return;
    }

    for (int index = 0; index < size; index++)
    {
        data[index] *= scalar;
    }
}

template<typename T>
T* Vector<T>::GetDataClone()
{
    T *data = new T[size];
    for (int index = 0; index < size; index++)
    {
        data[index] = this->data[index];
    }
    return data;
}

template<typename T>
bool Vector<T>::isColumnVector()
{
    return vertical;
}

template<typename T>
void Vector<T>::Print()
{
    for (int index = 0; index < size; index++)
    {
        cout << data[index] << (vertical ? "\n" : "\t");
    }
    cout << endl;
}

// Return data in string form
template<typename T>
string Vector<T>::GetData()
{
    string dataString = "";

    for (int index = 0; index < size; index++)
    {
        if (index != 0)
        {
            dataString += "\n";  // spliter
        }
        dataString += to_string(data[index]);
    }
    
    return dataString;
} 

#endif