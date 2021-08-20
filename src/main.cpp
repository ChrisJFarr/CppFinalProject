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

// Build this here, then move to utils.cpp
// This will eventually take a xData Matrix, a yData Matrix, pred Matrix
void printExample(Matrix xData, Matrix yData)
{
    // images are 28 x 28 // TODO START HERE! FInish implementing this 
    for(int j=1;j<xData.cols()+1;j++)
    {
        int imageCols = 28;
        MyDType dataPoint = xData(0, j-1);  // Call Matrix accessor
        // Decide how to represent the data point
        char pixelRep;
        if(dataPoint<.001){pixelRep = ' ';}
        else if(dataPoint<.2){pixelRep = '\'';}
        else{pixelRep = 'O';}
        cout << pixelRep;  // Print it out
        // If on column 28, end line
        if(j%imageCols == 0){cout << endl;}
    }
    cout << endl << endl << "Target: " << yData(0, 0) << endl << endl;
}

void testUtils()
{
        ////////// Testing utils.h ///////////////////
    
    // Initialize DataLoader
    string fileName;
    // prompt user for project directory
    cout << "what's the full path to the data?" << endl;
    cin >> fileName;
    // /home/chris/Desktop/final-project/src/data/train.csv
    // /home/pi/Desktop/CppFinalProject/src/data/train.csv
    DataLoader dataLoader(fileName, true, 0, 0.10, 0.10, 1000);

    // Print number of train, test, and validation examples
    // Print single example size (from Matrix class rows() cols())

    // Print some test examples
    int nExamples = 1;
    vector<unique_ptr<vector<Matrix>>> testData(0);
    dataLoader.getTestDataCopy(0, nExamples, testData);
    unique_ptr<vector<Matrix>> xTest = move(testData[0]), yTest = move(testData[1]);
    cout << "Test Examples:" << endl;
    for(int i=0;i<nExamples;i++) printExample((*xTest)[i], (*yTest)[i]);

    // Print some validation examples
    vector<unique_ptr<vector<Matrix>>> validData(0);
    dataLoader.getValidDataCopy(0, nExamples, validData);
    unique_ptr<vector<Matrix>> xValid = move(validData[0]), yValid = move(validData[1]);
    cout << "Validation Examples:" << endl;
    for(int i=0;i<nExamples;i++) printExample((*xValid)[i], (*yValid)[i]);

    // Print some train examples of a single batch
    vector<unique_ptr<vector<Matrix>>> trainData(0);
    dataLoader.getTrainBatch(nExamples, trainData);
    unique_ptr<vector<Matrix>> xTrain = move(trainData[0]), yTrain = move(trainData[1]);
    cout << "Train Examples:" << endl;
    for(int i=0;i<nExamples;i++) printExample((*xTrain)[i], (*yTrain)[i]);
    
    // Print the target distribution of a train batch
    // Print the target distribution of test set
    // Print the target distribution of validation set

    //////// END OF TESTING utils.h //////////////
}


int main() {


    ////////// Testing utils.h ///////////////////
    testUtils();
    //////// END OF TESTING utils.h //////////////

    ////////// Testing layer.h ///////////////////
    
    // Create test input data that's similar to image inputs (but small)
    vector<vector<MyDType>> data({{0.0, 0.0, 0.5, 0.5, 0.0}});
    Matrix xData(data);

    // Test input layer
    // Initialize input layer using size



    //////// END OF TESTING layer.h //////////////


    // TODO PLan where to go next...
    // I have data and ready to use it
    // Need to build the model now
    // Step 1: It should be able to generate a prediction (with random initialization)

    // Intialize model
    // 




    // Test generating a prediction
    // Test extracting the parameters
    // Test extracting the parameter gradients
    // Test extracting the layer-gradients
    // Implement a print-model-summary

    /////////// Testing model.h ///////////////////

    MyModel model;

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
