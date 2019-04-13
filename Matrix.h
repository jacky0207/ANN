#ifndef _MATRIX_H_
#define _MATRIX_H_

template<typename T>
class Matrix
{
private:
    T *data;
    int row;
    int column;
public:
    Matrix(T *data, int row, int column);
    int GetRow();
    int GetColumn();
    void Transpose();
    void Multiplication(T scalar);

    T* GetDataClone();
    void Print();   // Debug
};

#endif