#include <iostream>
#include <utility>
#include <vector>
#include <memory>
#include "model.h"


/*
https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
Working with the data

Will need to preprocess the data and store in an easier file type perhaps


    // valgrind --leak-check=yes --track-origins=yes --log-file=valgrind-out.txt ./NeuralNetwork
*/

void functionBlock()
{

    // TODO Test the usage of all underlying data objects
    // 1d-vectors
    std::vector<float> myArray;
    for(float j=0;j<=10;++j)
    {
        std::cout << j << " ";
        myArray.emplace_back((j/(10.)));
    }
    std::cout << std::endl;
    for(int j=0;j<=10;++j){std::cout << myArray[j] << " " << std::endl;}

    // 1d-vector using unique ptr and heap allocation
    std::unique_ptr<std::vector<float>> myArrayPtr = std::make_unique<std::vector<float>>();
    for(float j=0;j<=10;++j)
    {
        std::cout << j;
        (*myArrayPtr).emplace_back((j/(10.)));
    }
    std::cout << std::endl;
    for(int j=0;j<=10;++j){std::cout << (*myArrayPtr)[j] << " " << std::endl;}

    //  2d-vectors vs 2d-arrays

    // Initializing and working with 2d-vector of floats on the stack
    std::vector<std::vector<float>> my2DArray;
    for(int j=0;j<10;++j)
    {
        my2DArray.emplace_back(std::vector<float>());
        for(int k=0;k<10;++k)
        {
            my2DArray[j].emplace_back(j+k*1.0);
        }
    }
    // 2d vector using unique pointers
    for(int j=0;j<10;++j){for(int k=0;k<10;++k)
        {std::cout << " (" << j << "," << k << "): " << my2DArray[j][k] << " ";}std::cout << std::endl;
    }

    //  unique ptr to 2d vector of floats declared on the heap
    std::unique_ptr<std::vector<std::vector<float>>> my2DArrayPtr;
    my2DArrayPtr = std::make_unique<std::vector<std::vector<float>>>();

    for(int j=0;j<10;++j)
    {
        (*my2DArrayPtr).emplace_back(std::vector<float>());
        for(int k=0;k<10;++k)
        {
            (*my2DArrayPtr)[j].emplace_back(j+k*1.0);
        }
    }
    // 2d vector using unique pointers
    for(int j=0;j<10;++j){for(int k=0;k<10;++k)
        {std::cout << " (" << j << "," << k << "): " << (*my2DArrayPtr)[j][k] << " ";}std::cout << std::endl;
    }

    //  unique ptr to 2d vector of floats declared on the heap with user-defined size
    // parameters won't need to be built but once
    int MAX_SIZE = 10;

    std::unique_ptr<std::vector<std::vector<float>>> my2DArrayPtrDefined;
    my2DArrayPtrDefined = std::make_unique<std::vector<std::vector<float>>>(MAX_SIZE);

    for(int j=0;j<10;++j)
    {
        (*my2DArrayPtrDefined)[j] = std::vector<float>(MAX_SIZE);
        for(int k=0;k<10;++k)
        {
            (*my2DArrayPtrDefined)[j][k] = (j+k*1.0);
        }
    }
    // 2d vector using unique pointers
    for(int j=0;j<10;++j){for(int k=0;k<10;++k)
        {std::cout << " (" << j << "," << k << "): " << (*my2DArrayPtr)[j][k] << " ";}std::cout << std::endl;
    }
    //  shared ptr on stack, shared ptr on heap
    //  moving a unique pointer
    std::unique_ptr<std::vector<std::vector<float>>> my2DArrayPtrDefinedMoved;
    my2DArrayPtrDefinedMoved = std::move(my2DArrayPtrDefined);
    // 2d vector using unique pointers
    for(int j=0;j<10;++j){for(int k=0;k<10;++k)
        {std::cout << " (" << j << "," << k << "): " << (*my2DArrayPtrDefinedMoved)[j][k] << " ";}std::cout << std::endl;
    }
    //  copying a shared pointer

}


int main() {

    MyModel model;

    std::cout << "Model declared at:" << &model << "\n";
    MyModel newModel(model);
    std::cout << "Model copy at:" << &newModel << std::endl;

    // Test forward pass
    std::cout << "Testing BaseModel::forward(std::unique_ptr<std::vector<float>> &&inputs);" << std::endl;
    std::unique_ptr<std::vector<float>> inputs = std::make_unique<std::vector<float>>(std::vector<float>({.5, .5, .5}));
    model.forward(std::move(inputs));

    // Test backward pass
    std::cout << "Testing BaseModel::backward(std::unique_ptr<std::vector<float>> &&targets);" << std::endl;
    std::unique_ptr<std::vector<float>> targets = std::make_unique<std::vector<float>>(std::vector<float>({1.0, 0.0, 0.0}));
    model.backward(std::move(targets));

    // Declare outputs
    OutputData outputData;

    // Test BaseModel::getGradients(OutputData&)
    std::cout << "Testing BaseModel::getGradients(OutputData&)" << std::endl;
    model.getGradients(outputData);
    
    // Test BaseModel::getOutputs(OutputData&)
    std::cout << "Testing BaseModel::getGradients(OutputData&)" << std::endl;
    model.getOutputs(outputData);

    return 0;
}