#ifndef MATRIX_H_
#define MATRIX_H_

#include <memory>
#include <vector>
#include <iostream>

#include "constants.h"

using namespace std;

class Matrix 
{
public:
    Matrix(Matrix&&);  // Move contructor
    Matrix(vector<vector<MyDType>>&); // Allow initialization using an existing 2d-vector, copy by value internally
    Matrix(Matrix&);  // Copy constructor, creates a deep-copy
    Matrix(int rows, int cols);  // This should allocate memory on the heap
    Matrix& operator=(Matrix&&);  // Move assignment operator
    Matrix& operator=(Matrix&);  // Deep-copy assignment operator
    float operator()(int row, int col);  // Access individual elements
    vector<MyDType>& operator[](std::size_t n);  // Trying to use this for assingments
    Matrix mathMultiply(Matrix&);
    Matrix operator*(Matrix&);
    Matrix operator+(Matrix&);
    Matrix transpose();
    // Scalar operators
    Matrix operator*(MyDType&);
    int rows(){return _rows;}
    int cols(){return _cols;}

private:
    Matrix(){}; // No default allowed, must know shape
    unique_ptr<vector<vector<MyDType>>> _data;  // Owns its own data
    int _rows;
    int _cols;
};

#endif