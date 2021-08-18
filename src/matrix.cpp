#include "matrix.h"


Matrix::Matrix(vector<vector<MyDType>>&& data)
{
    // This should allocate memory on the heap and move data there
    _data = make_unique<vector<vector<MyDType>>>(std::move(data));
    _rows = _data->size();
    if(_rows > 0)
    {
        _cols = (*_data)[0].size();
    } else {
        _cols = 0;
    }
    
} 

Matrix::Matrix(Matrix& other)
{
    // Make a deep copy of the data contents
    vector<vector<MyDType>> dataCopy = (*other._data);
    // Call data constructor
    Matrix(move(dataCopy));
}

Matrix::Matrix(int rows, int cols)
{
    // Store shape
    _rows = rows;
    _cols = cols;
    // This should allocate memory on the heap
} 

Matrix& Matrix::operator=(Matrix&&)
{
    // Move assignment operator
    return *this;
}  

float Matrix::operator()(int row, int col)
{
    // Access individual elements
    return (*_data)[row][col];
}  
Matrix Matrix::mathMultiply(Matrix& other)
{
    return *this;
}
Matrix Matrix::operator*(Matrix& other)
{
    return *this;
}
Matrix Matrix::operator+(Matrix& other)
{
    return *this;
}
Matrix Matrix::transpose()
{
    return *this;
}
// Scalar operators
Matrix Matrix::operator*(MyDType&)
{
    return *this;
}


