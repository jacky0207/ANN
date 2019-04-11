#include "Vector.h"
#include "Vector.cpp"
#include "Matrix.h"
#include "Matrix.cpp"
#include <iostream>
using namespace std;

template<typename T>
Vector<T>* VectorAddition(Vector<T> a, Vector<T> b, T scalar = 1)
{
    T *dataA = a.GetDataClone();
    T *dataB = b.GetDataClone();

    int sizeA = a.GetSize();
    int sizeB = b.GetSize();

    bool verticalA = a.isColumnVector();
    bool verticalB = b.isColumnVector();

    if (sizeA != sizeB || verticalA != verticalB)
    {
        throw invalid_argument("Size of vector a and b are not the same");
    }

    T *newData = new T[sizeA];
    for (int index = 0; index < sizeA; index++)
    {
        newData[index] = scalar * dataA[index] + dataB[index];
    }

    return new Vector<T>(newData, sizeA);
}

template<typename T>
T VectorDotProduct(Vector<T> a, Vector<T> b)
{
    T *dataA = a.GetDataClone();
    T *dataB = b.GetDataClone();

    int sizeA = a.GetSize();
    int sizeB = b.GetSize();

    if (sizeA != sizeB)
    {
        throw invalid_argument("Size of vector a and b are not the same");
    }

    T result = 0;
    for (int index = 0; index < sizeA; index++)
    {
        result += dataA[index] * dataB[index];
    }

    return result;
}

template<typename T>
Matrix<T>* MatrixAddition(Matrix<T> a, Matrix<T> b)
{
    T *dataA = a.GetDataClone();
    T *dataB = b.GetDataClone();

    int rowA = a.GetRow();
    int rowB = b.GetRow();
    int columnA = a.GetColumn();
    int columnB = b.GetColumn();

    if (rowA != rowB || columnA != columnB)
    {
        throw invalid_argument("Dimension of matrix a and b are not the same");
    }

    int sizeA = rowA * columnA;
    T *newData = new T[sizeA];
    for (int index = 0; index < sizeA; index++)
    {
        newData[index] = dataA[index] + dataB[index];
    }

    return new Matrix<T>(newData, rowA, columnA);
}

template<typename T>
Matrix<T>* VectorMatrixMultiplication(Vector<T> a, Matrix<T> b)
{
    T *dataA = a.GetDataClone();
    T *dataB = b.GetDataClone();

    // Check valid
    int columnA = a.isColumnVector() ? 1 : a.GetSize();
    int rowB = b.GetRow();

    if (columnA != rowB)
    {
        throw invalid_argument("Dimension of matrix a and b are not the same");
    }

    // Size of new data
    int rowA = a.isColumnVector() ? a.GetSize() : 1;
    int columnB = b.GetColumn();

    // New data
    int size = rowA * columnB;
    T *newData = new T[size];
    // every row in a
    for (int rowIndex = 0; rowIndex < rowA; rowIndex++)
    {
        // cross every column of b
        for (int matrixColumnIndex = 0; matrixColumnIndex < columnB; matrixColumnIndex++)
        {
            T value = 0;

            // add every row in that column
            for(int matrixRowIndex = 0; matrixRowIndex < rowB; matrixRowIndex++)
            {
                value += dataA[rowIndex * columnA + matrixRowIndex] * dataB[matrixRowIndex * columnB + matrixColumnIndex];
            }

            newData[rowIndex * columnB + matrixColumnIndex] = value;
        }
    }

    return new Matrix<T>(newData, rowA, columnB);
}

template<typename T>
Matrix<T>* MatrixMatrixMultiplication(Matrix<T> a, Matrix<T> b)
{
    T *dataA = a.GetDataClone();
    T *dataB = b.GetDataClone();

    // Check valid
    int columnA = a.GetColumn();
    int rowB = b.GetRow();

    if (columnA != rowB)
    {
        throw invalid_argument("Dimension of matrix a and b are not the same");
    }

    // Size of new data
    int rowA = a.GetRow();
    int columnB = b.GetColumn();

    // New data
    int size = rowA * columnB;
    T *newData = new T[size];
    // every row in a
    for (int rowIndex = 0; rowIndex < rowA; rowIndex++)
    {
        // cross every column of b
        for (int matrixColumnIndex = 0; matrixColumnIndex < columnB; matrixColumnIndex++)
        {
            T value = 0;

            // add every row in that column
            for(int matrixRowIndex = 0; matrixRowIndex < rowB; matrixRowIndex++)
            {
                value += dataA[rowIndex * columnA + matrixRowIndex] * dataB[matrixRowIndex * columnB + matrixColumnIndex];
            }

            newData[rowIndex * columnB + matrixColumnIndex] = value;
        }
    }

    return new Matrix<T>(newData, rowA, columnB);
}

// int main()
// {
//     int nums1[12] = {-1, 2, 3, 4, 
//                     5, 6, 11, 12, 
//                     13, 14, 15, -16};
//     int nums2[12] = {-21, 22, 23, 24, 
//                     25, 26, 121, 122, 
//                     123, 124, 125, -126};
//     int nums3[11] = {-21, 22, 23};

//     Matrix<int> y(nums2, 4, 3);
//     Matrix<int> z(nums2, 3, 4);
//     Matrix<int> a(nums1, 3, 4);
//     Vector<int> b(nums1, 12);
//     Vector<int> c(nums2, 12);
//     Vector<int> d(nums3, 3);

//     cout << "Data of a: " << endl; a.Print();
//     cout << "Row of a: " << a.GetRow() << endl;
//     cout << "Column of a: " << a.GetColumn() << endl;

//     cout << endl;

//     cout << "Transpose a" << endl; a.Transpose();
//     cout << "Data of a: " << endl; a.Print();
//     cout << "Row of a: " << a.GetRow() << endl;
//     cout << "Column of a: " << a.GetColumn() << endl;

//     cout << endl;

//     cout << "Multply a by 2" << endl; a.Multiplication(2);
//     cout << "Data of a: " << endl; a.Print();
//     cout << "Row of a: " << a.GetRow() << endl;
//     cout << "Column of a: " << a.GetColumn() << endl;

//     cout << endl;

//     cout << "Data of b: " << endl; b.Print();
//     cout << "Size of b: " << b.GetSize() << endl;
//     cout << "Mean of b: " << b.GetMean() << endl;
//     cout << "L1 norm of b: " << b.GetL1Norm() << endl;
//     cout << "L2 norm of b: " << b.GetEuclideanNorm() << endl;

//     cout << endl;

//     cout << "Multply b by 3" << endl; b.Multiplication(3);
//     cout << "Data of b: " << endl; b.Print();
//     cout << "Size of b: " << b.GetSize() << endl;
//     cout << "Mean of b: " << b.GetMean() << endl;
//     cout << "L1 norm of b: " << b.GetL1Norm() << endl;
//     cout << "L2 norm of b: " << b.GetEuclideanNorm() << endl;

//     cout << endl;

//     Vector<int> *bc1 = VectorAddition<int>(b, c);
//     cout << "Data of b: " << endl; b.Print();
//     cout << "Data of c: " << endl; c.Print();
//     cout << "Data of b + c: " << endl; bc1->Print();

//     cout << endl;

//     Vector<int> *bc2 = VectorAddition<int>(b, c, 2);
//     cout << "Data of b: " << endl; b.Print();
//     cout << "Data of c: " << endl; c.Print();
//     cout << "Data of 2b + c: " << endl; bc2->Print();

//     cout << endl;

//     int bDotC = VectorDotProduct<int>(b, c);
//     cout << "Data of b: " << endl; b.Print();
//     cout << "Data of c: " << endl; c.Print();
//     cout << "Dot product of bc: " << bDotC << endl;

//     cout << endl;

//     a.Transpose();
//     Matrix<int> *az = MatrixAddition<int>(a, z);
//     cout << "Data of a: " << endl; a.Print();
//     cout << "Data of z: " << endl; z.Print();
//     cout << "Data of a + z: " << endl; az->Print();

//     cout << endl;

//     // a.Transpose();
//     Matrix<int> *da = VectorMatrixMultiplication<int>(d, a);
//     cout << "Data of d: " << endl; d.Print();
//     cout << "Data of a: " << endl; a.Print();
//     cout << "Data of dCrossA: " << endl; da->Print();

//     cout << endl;

//     // Throw exception of invalid dimension
//     // Matrix<int> *aCrossZ = MatrixMatrixMultiplication<int>(a, z);
//     // cout << "Data of a: " << endl; a.Print();
//     // cout << "Data of z: " << endl; z.Print();
//     // cout << "Data of dCrossA: " << endl; da->Print();

//     // cout << endl;

//     Matrix<int> *aCrossY= MatrixMatrixMultiplication<int>(a, y);
//     cout << "Data of a: " << endl; a.Print();
//     cout << "Data of y: " << endl; y.Print();
//     cout << "Data of aCrossY: " << endl; aCrossY->Print();

//     cout << endl;

//     // Throw exception of size of 2 vector not match
//     // Vector<int> *bd = VectorAddition<int, int>(b, d, 1);
//     // cout << "Data of b: " << endl; b.Print();
//     // cout << "Data of d: " << endl; d.Print();
//     // cout << "Data of bd: " << endl; bd->Print();

//     return 0;
// }