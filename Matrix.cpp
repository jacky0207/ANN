#ifndef _MATRIX_CPP_
#define _MATRIX_CPP_

#include "Matrix.h"
#include <iostream>
using namespace std;

template<typename T>
Matrix<T>::Matrix(T *data, int row, int column)
{
    // Initialize
    // this->data = data;
    int size = row * column;
    this->data = new T[size];
    for (int index = 0; index < size; index++)
    {
        this->data[index] = data[index];
    }
    this->row = row;
    this->column = column;
}

template<typename T>
int Matrix<T>::GetRow()
{
    return row;
}

template<typename T>
int Matrix<T>::GetColumn()
{
    return column;
}

template<typename T>
void Matrix<T>::Transpose()
{
    // Rearrange data
    T *dataT = new T[row * column];

    for (int rowIndex = 0; rowIndex < row; rowIndex++)
    {
        for (int columnIndex = 0; columnIndex < column; columnIndex++)
        {
            dataT[columnIndex * row + rowIndex] = data[rowIndex * column + columnIndex];
        }
    }

    data = dataT;

    // Swap row and column
    int temp = row;
    row = column;
    column = temp;
}

template<typename T>
void Matrix<T>::Multiplication(T scalar)
{
    for (int index = 0; index < row * column; index++)
    {
        data[index] *= scalar;
    }
}

template<typename T>
T* Matrix<T>::GetDataClone()
{
    int size = row * column;
    T *data = new T[size];
    for (int index = 0; index < size; index++)
    {
        data[index] = this->data[index];
    }
    return data;
}

template<typename T>
void Matrix<T>::Print()
{
    for (int index = 0; index < row * column; index++)
    {
        if (index != 0 && index % column == 0)
        {
            cout << "\n";
        }
        cout << data[index] << "\t";
    }
    cout << endl;
}

#endif