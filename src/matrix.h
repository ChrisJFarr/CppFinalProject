#ifndef MATRIX_H_
#define MATRIX_H_

#include <memory>
#include <vector>

using namespace std;

class Matrix 
{
public:
    Matrix(int rows, int cols);  // This should allocate memory on the heap
    Matrix& operator=(Matrix&&);  // Move assignment operator
    float operator()(int row, int col);  // Access individual elements
    Matrix mathMultiply(Matrix&);
    Matrix operator*(Matrix&);
    Matrix transpose();
    // Scalar operators
    Matrix operator*(float&);

private:
    Matrix(); // No default allowed, must know shape
    Matrix(Matrix&);  // Copy constructor (do I need this?) Blocking in private
    unique_ptr<vector<vector<float>>> _data;  // Owns its own data
};

#endif