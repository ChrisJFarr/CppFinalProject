#include "matrix.h"


Matrix::Matrix(vector<vector<MyDType>> &data)
{
    // cout << "Matrix data contructor." << endl;
    // cout << "starting address:" << &data << endl;
    // This should allocate memory on the heap and copy the data there
    _data = make_unique<vector<vector<MyDType>>>();
    (*_data) = data;
    // TODO Test this copy operation works properly...
    // cout << "new address:" << &(*_data) << endl;
    // Manual deep-copy (Maybe this will fix my seg fault bug...)
    // for(vector<MyDType> rows: data)
    // {
    //     (*_data).emplace_back(vector<MyDType>());
    //     for(MyDType dataPoint: rows)
    //     {
    //         (*_data)[(*_data).size()-1].emplace_back(dataPoint);
    //         // cout << (*_data)[(*_data).size()-1].back();
    //     }
    // }
    // cout << endl;
    // cout << "allocated and copied data to new matrix." << endl;

    _rows = _data->size();
    if(_rows > 0)
    {
        _cols = (*_data)[0].size();
    } else {
        _cols = 0;
    }
    // cout << "exiting data constructor" << endl;
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
    _rows = _data->size();
    if(_rows > 0)
    {
        _cols = (*_data)[0].size();
    } else {
        _cols = 0;
    }
}

Matrix::Matrix(int rows, int cols)
{
    // cout << "args constructor" << endl;
    // Store shape
    _rows = rows;
    _cols = cols;
    // This should allocate memory on the heap
} 

Matrix& Matrix::operator=(Matrix&& other)
{
    // Move assignment operator
    // This should allocate memory on the heap and move data there
    // _data = make_unique<vector<vector<MyDType>>>(std::move(other._data));
    // cout << "Matrix move assignment operator." << endl;
    // _data = unique_ptr<vector<vector<MyDType>>>(move(other._data));
    _data = move(other._data);
    _rows = _data->size();
    if(_rows > 0)
    {
        _cols = (*_data)[0].size();
    } else {
        _cols = 0;
    }

    return *this;
}


Matrix& Matrix::operator=(Matrix&)
{
    cout << "Matrix copy assignment operator not implemented yet" << endl;
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


