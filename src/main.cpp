#include <iostream>
#include <utility>
#include <vector>
#include <memory>
#include "model.h"
#include "utils.h"
#include "layer.h"

using namespace std;

/*
valgrind --leak-check=yes --track-origins=yes --log-file=valgrind-out.txt ./NeuralNetwork

/home/chris/Desktop/final-project/src/data/train.csv
/home/pi/Desktop/CppFinalProject/src/data/train.csv
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

void testLayer()
{
        ////////// Testing layer.h ///////////////////
    
    // Create test input data that's similar to image inputs (but small)
    vector<vector<MyDType>> data({{0.0, 0.0, 0.5, 0.5, 0.0}});
    unique_ptr<Matrix> xData = make_unique<Matrix>(data);

    // Test input layer

    // Initialize input layer using shape of input data
    InputLayer inputLayer(xData->rows(), xData->cols());
    // Test that input shapes == expected shapes
    vector<string> testInputLayerInputSize(2);
    testInputLayerInputSize[0] = (inputLayer.getInputShape()[0] == xData->rows()) ? "success" : "fail";
    testInputLayerInputSize[1] = (inputLayer.getInputShape()[1] == xData->cols()) ? "success" : "fail";
    for(string r: testInputLayerInputSize) {cout << "testing InputLayer::getInputShape... " << r << endl;}
    // Test expected output shape
    vector<string> testInputLayerOutputSize(2);
    testInputLayerOutputSize[0] = (inputLayer.computeOutputShape()[0] == xData->rows()) ? "success" : "fail";;
    testInputLayerOutputSize[1] = (inputLayer.computeOutputShape()[1] == xData->cols()) ? "success" : "fail";;
    for(string r: testInputLayerOutputSize) {cout << "testing InputLayer::computeOutputShape... " << r << endl;}
    // Test InputLayer::setInputs (this invalidates xData, must be reinitialized after moving out)
    cout << "testing InputLayer::setInputs... ";
    inputLayer.setInputs(move(xData));
    cout << "success" << endl;
    // Test InputLayer::foward() prior to and after connecting to child
    cout << "testing InputLayer::forward... ";
    try {inputLayer.forward();} catch(logic_error) {cout << "success" << endl;}
    // Connect to another layer then try forward again

    // Test actual output shape

    // Use try catch and print "success" after testing

    // TODO Consider implementing dropout, seems like it wouldn't be too hard to do
    //  just create a parameter that can be consumed by the child that tells it which
    //  inputs to ignore (this could also serve as a way to implement a mask operation)
    //  it only works when just before a parameterized layer unless a universal masking
    //  approach is taking? Think on it.
    //////// END OF TESTING layer.h //////////////
}

int main() {


    ////////// Testing utils.h ///////////////////
    // testUtils();
    //////// END OF TESTING utils.h //////////////

    ////////// Testing layer.h ///////////////////
    // testLayer();
    //////// END OF TESTING layer.h //////////////


    ////////// Testing matrix.h ///////////////////
    {
    // Test setting values with [] accessor operator
    Matrix myMatrix1(5, 5), myMatrix2(1, 5), myMatrix3(5, 1);
    cout << "visual inspection for Matrix::operator[]... do these match?" << endl;
    for(int j=0;j<5;j++)
    {
        for(int k=0;k<5;k++)
        {
            myMatrix1[j][k] = j+k;
            cout << j+k;
        }
        cout << " ";
        for(int k=0;k<5;k++)
        {
            cout << myMatrix1[j][k];
        }
        cout << endl;
    }
    cout << endl;
    vector<int> check;
    }
    {
        Matrix myMatrix1(5, 5), myMatrix2(1, 5), myMatrix3(5, 1);
        // Test that Matrix(rows,cols) constructor initializes to all zeros
        cout << "testing Matrix(rows,cols) initializing to 0... ";
        vector<bool> testingZeroInitialization;
        for(MyDType d: myMatrix1[0]) testingZeroInitialization.emplace_back(d==0);  // Will this work or do I need a threshold?
        bool zeroInitializationResult = std::all_of(testingZeroInitialization.begin(), testingZeroInitialization.end(), [](bool v) { return v; });
        string zeroInitializationResultStr = (zeroInitializationResult) ? "success" : "fail";
        cout << zeroInitializationResultStr << endl; 
    }
    {
        // Test matrixmul operator*
        Matrix myMatrix1(5, 5), myMatrix2(1, 5), myMatrix3(5, 1);
        // Try 5x5 * 5x1 (expecting 5x1 result)
        Matrix newMatrix = myMatrix1 * myMatrix3;
        cout << "testing Matrix*Matrix output shape...";
        string matrixMultShapeTest = ((newMatrix.rows()==5) && (newMatrix.cols()==1)) ? "success" : "fail";
        cout << matrixMultShapeTest << endl;

        // Test numerical results of 1x2 * 2x3 (expecting 1x3 result)
        Matrix myMatrix4(1, 2); // Similar to an input vector of a dense layer with 2 features
        Matrix myMatrix5(2, 3);  // Similar to a weights matrix of a dense layer with 3 nodes
        // Just set values (i know its wonky)
        for(int j=0;j<2;j++)
        {
            // cout << "looping" << endl;
            myMatrix4[0][j] = .1;
            for(int k=0;k<3;k++)
            {
                myMatrix5[j][k] = .1 * k;
            }
        }
        cout << "testing Matrix*Matrix output values... ";
        Matrix newMatrix2 = myMatrix4 * myMatrix5;
        // For debugging, print the values
        // for(int j=0;j<newMatrix2.rows();j++)
        // {
        //     for(int k=0;k<newMatrix2.cols();k++)
        //     {
        //         cout << newMatrix2(j,k);
        //     }
        //     cout << endl;
        // }
        // Expected results of myMatrix4 * myMatrix5
        /*
            myMatrix4 {{.1, .1}} myMatrix5{{0., .1, .2}, {0., .1, .2}}
            newMatrix2 {{0.0, .02, .04}}
        */
       vector<MyDType> expected({0.0, 0.02, 0.04});
       vector<bool> matrixMultValueTest;
        for(int i=0;i<newMatrix2.cols();i++)
        {
            matrixMultValueTest.emplace_back(abs(newMatrix2(0, i)-expected[i])<.001);
        }
        string matrixMultValueResult = (std::all_of(matrixMultValueTest.begin(), matrixMultValueTest.end(), [](bool v) { return v; }) ? "success" : "fail");
        cout << matrixMultValueResult << endl;

    }
    {
        // Test adding 2 Matrix
    }
    
    // Test multiplying 2 Matrix mathMultiply (element-wise and must be exact same shape)


    //////// END OF TESTING matrix.h //////////////

    

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
