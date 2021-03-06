#include "matrix.h"


Matrix::Matrix(vector<vector<MyDType>> &data)
{
    // cout << "Matrix data contructor." << endl;
    // cout << "starting address:" << &data << endl;
    // This should allocate memory on the heap and copy the data there
    _data = make_unique<vector<vector<MyDType>>>();
    (*_data) = data;
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
    // cout << "Matrix copy constructor" << endl;
    // Make a deep copy of the data contents
    // vector<vector<MyDType>> dataCopy = (*other._data);
    _data = make_unique<vector<vector<MyDType>>>(*other._data);
    // cout << "data copied to new location " << &(*_data) << endl;
        // Set size of _rows and _cols
    _rows = _data->size();
    if(_rows > 0)
    {
        _cols = (*_data)[0].size();
    } else {
        _cols = 0;
    }
}

Matrix::Matrix(Matrix&& other)
{
    // Move contructor
    // _data = unique_ptr<vector<vector<MyDType>>>(move((*other._data)));
    // Move the _data pointer
    // cout << "Matrix move contructor" << endl;
    _data = move(other._data);
    // cout << "data moved to new object" << endl;
    // Set size of _rows and _cols
    _rows = other._rows;
    _cols = other._cols;
}

Matrix::Matrix(int rows, int cols)
{
    // cout << "args constructor" << endl;
    // Store shape
    _rows = rows;
    _cols = cols;
    // This allocates memory on the heap and initializes the data to zeros
    _data = make_unique<vector<vector<MyDType>>>(_rows);
    for(auto& row: (*_data)) row.resize(_cols);
} 

Matrix& Matrix::operator=(Matrix&& other)
{
    // Move assignment operator
    // This should allocate memory on the heap and move data there
    // _data = make_unique<vector<vector<MyDType>>>(std::move(other._data));
    // cout << "Matrix move assignment operator." << endl;
    // _data = unique_ptr<vector<vector<MyDType>>>(move(other._data));
    _data = move(other._data);
    _rows = other._rows;
    _cols = other._cols;
    return *this;
}

Matrix& Matrix::operator=(Matrix& other)
{
    // Copy assignment operator, creates a deep copy
    _data = make_unique<vector<vector<MyDType>>>(*other._data);
    _rows = other._rows;
    _cols = other._cols;
    return *this;
}

float Matrix::operator()(int row, int col)
{
    // Access individual elements
    // Validate in bounds
    // cout << "_rows:" << _rows << " row:" << row << " _cols:" << _cols << " col:" << col << endl;
    if((row >= _rows) || (col >= _cols)) throw invalid_argument("Attempting to access outside Matrix bounds.");
    return (*_data)[row][col];
}

vector<MyDType>& Matrix::operator[](std::size_t n)
{
    return (*_data)[n];
}

Matrix Matrix::mathMultiply(Matrix& other)
{
    // Validate that both matrix objects are same shape
    if(!((this->rows()==other.rows()) && (this->cols()==other.cols()))) throw invalid_argument(
        "Matrix A must be same shape as Matrix B in Matrix::mathMultiply");
    Matrix targetMatrix(_rows, _cols);
    for(int j=0;j<targetMatrix.rows();j++)
    {
        for(int k=0;k<targetMatrix.cols();k++)
        {
            targetMatrix[j][k] = (*this)[j][k] * other[j][k];
        }
    }
    return targetMatrix;
}

Matrix Matrix::operator*(Matrix& other)
{
    // Validate that the other matrix and this matrix are compatible
    // implies the order this * other
    // this.columns must equal other rows
    if(this->cols() != other.rows()) throw invalid_argument("Matrix A.cols must equal B.rows in Matrix::operator*");
    Matrix targetMatrix(this->rows(), other.cols());
    
    // for(int j=0;j<targetMatrix)
    for(int j=0;j<targetMatrix.rows();j++)
    {
        for(int k=0;k<targetMatrix.cols();k++)
        {
            for(int i=0;i<other.rows();i++)
            {
                targetMatrix[j][k] += (*this)(j, i) * other(i, k);
            }
        }
    }
    return targetMatrix;
}

Matrix Matrix::operator+(Matrix& other)
{
    // Validate that both matrix objects are same shape
    if(!((this->rows()==other.rows()) && (this->cols()==other.cols()))) throw invalid_argument(
        "Matrix A must be same shape as Matrix B in Matrix::operator+");
    Matrix targetMatrix(_rows, _cols);
    for(int j=0;j<targetMatrix.rows();j++)
    {
        for(int k=0;k<targetMatrix.cols();k++)
        {
            targetMatrix[j][k] = (*this)[j][k] + other[j][k];
        }
    }
    return targetMatrix;
}

Matrix Matrix::transpose()
{
    Matrix targetMatrix(_cols, _rows);  // Swap cols and row sizes
    for(int j=0;j<_rows;j++)
    {
        for(int k=0;k<_cols;k++)
        {
            targetMatrix[k][j] = (*this)(j, k);
        }
    }
    return targetMatrix;
}

Matrix Matrix::identity()
{
    // Create identity matrix of size of calling matrix
    Matrix targetMatrix(_rows, _cols);  // Initializes to zeros
    for(int j=0;j<_rows;j++){
        for(int k=0;k<_cols;k++)
        {
            if(j=k){
                targetMatrix[j][k] = 1.0;
            } else {
                targetMatrix[j][k] = 0.0;
            }
        }
    }
    return targetMatrix;
}

MyDType Matrix::max()
{
    MyDType maxValue = -inf;
    for(int j=0;j<_rows;j++)
    {
        for(int k=0;k<_cols;k++)
        {
            maxValue = std::max(maxValue, (*this)(j, k));
        }
    } 
    return maxValue;
}

// Scalar operators
Matrix Matrix::operator*(MyDType&)
{
    return *this;
}


