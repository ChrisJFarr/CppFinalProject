#include "matrix.h"


// Don't implement //
Matrix::Matrix(){} // No default allowed, must know shape
Matrix::Matrix(Matrix&){} // No copy constructor allowed
/////////////////////

Matrix::Matrix(int rows, int cols)
{
    // This should allocate memory on the heap
} 

Matrix& Matrix::operator=(Matrix&&)
{
    // Move assignment operator
}  

float Matrix::operator()(int row, int col)
{
    // Access individual elements
    return 0.0;
}  
Matrix Matrix::mathMultiply(Matrix&)
{
 
}
Matrix Matrix::operator*(Matrix&)
{

}
Matrix Matrix::transpose()
{

}
// Scalar operators
Matrix Matrix::operator*(float&)
{

}
