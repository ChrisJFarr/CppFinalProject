#include <iostream>
#include <utility>
#include <vector>
#include <memory>
#include "model.h"
#include "utils.h"

using namespace std;

/*
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
    unique_ptr<vector<vector<MyDType>>> my2DArrayPtrDefinedMoved;
    my2DArrayPtrDefinedMoved = std::move(my2DArrayPtrDefined);
    // 2d vector using unique pointers
    for(int j=0;j<10;++j){for(int k=0;k<10;++k)
        {std::cout << " (" << j << "," << k << "): " << (*my2DArrayPtrDefinedMoved)[j][k] << " ";}std::cout << std::endl;
    }
    //  copying a shared pointer

}

// Build this here, then move to utils.cpp
// This will eventually take a xData Matrix, a yData Matrix, pred Matrix
void printExample(Matrix xData, Matrix yData)
{
    // images are 28 x 28 // TODO START HERE! FInish implementing this 
    for(int j=1;j<xData.cols()+1;j++)
    {
        int imageCols = 28;
        int dataPoint = xData(0, j-1);  // Call Matrix accessor
        // Decide how to represent the data point
        char pixelRep;
        if(dataPoint==0){pixelRep = ' ';}
        else if(dataPoint<50){pixelRep = '\'';}
        else{pixelRep = 'O';}
        cout << pixelRep;  // Print it out
        // If on column 28, end line
        if(j%imageCols == 0){cout << endl;}
    }
    cout << endl << endl << "Target: " << yData(0, 0) << endl << endl;
}


int main() {


    ////////// Testing utils.h ///////////////////
    
    // Initialize DataLoader
    DataLoader dataLoader("/home/chris/Desktop/final-project/src/data/train.csv");

    // Print number of train, test, and validation examples
    // Print single example size (from Matrix class rows() cols())

    // Print top 5 test examples
    int nExamples = 5;
    vector<unique_ptr<vector<Matrix>>> testData;
    dataLoader.getTestDataCopy(0, nExamples, testData);
    unique_ptr<vector<Matrix>> xTest = move(testData[0]);
    unique_ptr<vector<Matrix>> yTest = move(testData[1]);
    for(int i=0;i<nExamples;i++) printExample((*xTest)[i], (*yTest)[i]);
    // Print top 5 validation examples
    vector<unique_ptr<vector<Matrix>>> validData;
    dataLoader.getValidDataCopy(0, nExamples, validData);
    unique_ptr<vector<Matrix>> xValid = move(validData[0]);
    unique_ptr<vector<Matrix>> yValid = move(validData[1]);
    for(int i=0;i<nExamples;i++) printExample((*xValid)[i], (*yValid)[i]);
    // Print top 5 train examples of a single batch


    // Consider taking a batch approach to the test and validation data for memory efficiency (how much memory does it take up?)
    // Print the target distribution of a train batch
    // Print the target distribution of test set
    // Print the target distribution of validation set

    //////// END OF TESTING utils.h //////////////


    /////////// Testing model.h ///////////////////

    // MyModel model;

    // std::cout << "Model declared at:" << &model << "\n";
    // MyModel newModel(model);
    // std::cout << "Model copy at:" << &newModel << std::endl;

    // // Test forward pass
    // std::cout << "Testing BaseModel::forward(std::unique_ptr<Matrix> &&inputs);" << std::endl;
    // // Create a 2d vector
    // vector<vector<MyDType>> myInputs({{.5, .5, .5}});
    // std::unique_ptr<Matrix> inputs = std::make_unique<Matrix>(std::move(myInputs));
    // model.forward(std::move(inputs));

    // // Test backward pass
    // std::cout << "Testing BaseModel::backward(std::unique_ptr<Matrix> &&targets);" << std::endl;
    // vector<vector<MyDType>> myTargets({{1.0, 0.0, 0.0}});
    // std::unique_ptr<Matrix> targets = std::make_unique<Matrix>(std::move(myTargets));
    // model.backward(std::move(targets));

    // // Declare outputs
    // OutputData outputData;

    // // Test BaseModel::getGradients(OutputData&)
    // std::cout << "Testing BaseModel::getGradients(OutputData&)" << std::endl;
    // model.getGradients(outputData);
    
    // // Test BaseModel::getOutputs(OutputData&)
    // std::cout << "Testing BaseModel::getGradients(OutputData&)" << std::endl;
    // model.getOutputs(outputData);

    //////// END OF TESTING model.h //////////////

    return 0;
}
