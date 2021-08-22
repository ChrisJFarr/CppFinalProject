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
    {
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
    testInputLayerOutputSize[0] = (inputLayer.getOutputShape()[0] == xData->rows()) ? "success" : "fail";;
    testInputLayerOutputSize[1] = (inputLayer.getOutputShape()[1] == xData->cols()) ? "success" : "fail";;
    for(string r: testInputLayerOutputSize) {cout << "testing InputLayer::getOutputShape... " << r << endl;}
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
    }
    {
    // TODO Develop glorot uniform initializer
    // Draws samples from a uniform distribution within [-limit, limit], 
    // where limit = sqrt(6 / (fan_in + fan_out)) (fan_in is the number of input 
    // units in the weight tensor and fan_out is the number of output units).
    
    // Example: (1x3) inputs (1x5) outputs
    //  weights size: 3x5, bias size: 1x5
    Matrix weights(50, 50);  // Bias should remain 0's
    // MyDType fan_in = static_cast<MyDType>(weights.rows());
    // MyDType fan_out = static_cast<MyDType>(weights.cols());
    // // fan_in=3, fan_out=5
    // // limit = sqrt(6/(fan_in+fan_out)) = sqrt(6/8)
    // MyDType limit = sqrt(6./(fan_in+fan_out));
    // cout << "limit: " << limit;
    // std::random_device rd;  //Will be used to obtain a seed for the random number engine
    // std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    // std::uniform_real_distribution<> dis(-limit, limit);
    glorotUniformInitializer(weights);
    // Test that the weights sum close to zero
    
    MyDType weightsSum = 0.;
    MyDType absMax = 0;
    MyDType fan_in = static_cast<MyDType>(weights.rows());
    MyDType fan_out = static_cast<MyDType>(weights.cols());
    // limit = sqrt(6/(fan_in+fan_out))
    MyDType limit = sqrt(6./(fan_in+fan_out));
    for(int j=0;j<weights.rows();j++)
    {
        for(int k=0;k<weights.cols();k++)
        {
            weightsSum += weights(j, k);
            absMax = max(absMax, abs(weights(j, k)));
        }
    }
    
    // Test to determine how well the random numbers are distributed
    // centered on 0, this test usually fails, not sure if that is an
    // issue with the random number generator or not an issue at all
    // continue to develop and return here if model is unable to learn 
    // or converge.
    string result;
    // cout << "testing glorotUniformInitializer weights sum...";
    // result = (abs(weightsSum) < .01) ? "success" : "fail";
    // cout << result << endl;
    // if(result == "fail")
    // {
    //     cout << "...sum :" << weightsSum << endl;
    // }
    // Test that the weights stay within the limit sqrt(6/(fan_in+fan_out))
    cout << "testing glorotUniformInitializer weights range...";
    result = (absMax <= limit) ? "success" : "fail";
    cout << result << endl;
    }

    // TODO Consider implementing dropout, seems like it wouldn't be too hard to do
    //  just create a parameter that can be consumed by the child that tells it which
    //  inputs to ignore (this could also serve as a way to implement a mask operation)
    //  it only works when just before a parameterized layer unless a universal masking
    //  approach is taking? Think on it.
    //////// END OF TESTING layer.h //////////////
}

void testMatrix()
{
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
        // Test adding 2 Matrix objects
        /*
            * Matrix objects must be the same shape
            myMatrix1({{1, 1, 1}})
            myMatrix2({{2, 2, 2}})
            expected {{3, 3, 3}}
        */
        vector<vector<MyDType>> data1({{1, 1, 1}}), data2({{2, 2, 2}});
        Matrix myMatrix1(data1), myMatrix2(data2);
        Matrix newMatrix = myMatrix1 + myMatrix2;

        cout << "testing Matrix+Matrix output shape...";
        string matrixAddShapeTest = ((newMatrix.rows()==1) && (newMatrix.cols()==3)) ? "success" : "fail";
        cout << matrixAddShapeTest << endl;

        cout << "testing Matrix+Matrix output values...";
        vector<MyDType> expected({3., 3., 3.});
        vector<bool> matrixAddValueTest;
        for(int i=0;i<newMatrix.cols();i++)
        {
            matrixAddValueTest.emplace_back(abs(newMatrix(0, i)-expected[i])<.001);
        }
        string matrixAddValueResult = (std::all_of(matrixAddValueTest.begin(), matrixAddValueTest.end(), [](bool v) { return v; }) ? "success" : "fail");
        cout << matrixAddValueResult << endl;
    }
    
    {
        // Test adding 2 Matrix mathMultiply (element-wise and must be exact same shape)
        /*
            * Matrix objects must be the same shape
            myMatrix1({{3, 3, 3}})
            myMatrix2({{2, 2, 2}})
            expected {{6, 6, 6}}
        */
        vector<vector<MyDType>> data1({{3, 3, 3}}), data2({{2, 2, 2}});
        Matrix myMatrix1(data1), myMatrix2(data2);
        Matrix newMatrix = myMatrix1.mathMultiply(myMatrix2);
        cout << "testing Matrix.mathMultiply(Matrix) output shape...";
        string matrixMathMultShapeTest = ((newMatrix.rows()==1) && (newMatrix.cols()==3)) ? "success" : "fail";
        cout << matrixMathMultShapeTest << endl;

        cout << "testing Matrix.mathMultiply(Matrix) output values...";
        vector<MyDType> expected({6., 6., 6.});
        vector<bool> matrixMathMultValueTest;
        for(int i=0;i<newMatrix.cols();i++)
        {
            matrixMathMultValueTest.emplace_back(abs(newMatrix(0, i)-expected[i])<.001);
        }
        string matrixMathMultValueResult = (std::all_of(matrixMathMultValueTest.begin(), matrixMathMultValueTest.end(), [](bool v) { return v; }) ? "success" : "fail");
        cout << matrixMathMultValueResult << endl;
    }
    {
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
    }
    
    //////// END OF TESTING matrix.h //////////////
}


int main() {

    ////////// Testing utils.h ///////////////////
    // testUtils();
    //////// END OF TESTING utils.h //////////////
    ////////// Testing layer.h ///////////////////
    // testLayer();
    //////// END OF TESTING layer.h //////////////
    ////////// Testing matrix.h ///////////////////
    // testMatrix();
    //////// END OF TESTING matrix.h //////////////

    // TODO PLan where to go next...
    // I have data and ready to use it
    // Need to build the model now
    // Preparing layers and model class for building the model
    // Step 1: It should be able to generate a prediction (with random initialization)

    // Continue building here and dispursing code to layer.h and testLayer()

    // What steps need to be taken to initialize layers, connecting them, 
    // moving them, and copying them.

    // TODO Change course
    //  on current path, by using a modelVector to build graph copies
    //  only single-path-graphs would be able to create. (1 to 1 connections only)
    //  Instead, rely on the connections of the graph for structure
    //  and simply call a function to create a graph, then use
    //  get and set to allow sharing of parameters.
    //  Also consider tensorflow approach to creating a model using
    //  inputs and outputs to verify the graph is connected.
    // compile or something, needs to be a function that...
    //  * runs build on each layer of a graph
    //  * validates that the inputs connect to the outputs without any dead-ends
    //  * dynamically deals with forked connections
    //  


    // Top Considerations:
    // * location and handling of shared parameters (must be shared by all layers and only initialized once)
    // * size-dependencies between connected layers
    // * completeness of the model graph (must start with an input layer and connect through to a loss layer)
    // * layer-specific constraints and interaction between layers

    // To initialize a model...

    // Dependencies
    MyDType regularization = 0.0001;
    int denseLayer1Units = 10;
    string result;

    // Initialize a unique ptr to a modelVector on the heap
    // vector<unique_ptr<BaseLayer>> modelVector;  obsolete
    // Initialize each layer (This goes into MyModel)
    
    // Input layer
    InputLayer inputLayer(1, INPUT_SHAPE);

    // Dense layer
    DenseLayer denseLayer1 = DenseLayer(denseLayer1Units, regularization);
    denseLayer1(inputLayer);  // Connect parent with () operator

    // Relu layer
    ReluLayer reluLayer1;
    reluLayer1(denseLayer1);  // Connect to denselayer
    
    // Softmax output layer
    SoftMaxLayer softMaxLayer;
    softMaxLayer(reluLayer1);

    // Cross entropy loss layer
    CrossEntropyLossLayer crossEntropyLossLayer;
    crossEntropyLossLayer(softMaxLayer);

    // Test that denseLayer is the child of inputLayer(inputLayer._childLayers)
    cout << "Test that denseLayer1 is child of inputLayer...";
    result = (&denseLayer1 == &(*inputLayer._childLayers[0])) ? "success" : "fail";
    cout << result << endl;
    // Test that inputLayer is the parent of denseLayer1(denseLayer1._parentLayers)
    cout << "Test that inputLayer is parent of denseLayer1...";
    result = (&inputLayer == &(*denseLayer1._parentLayers[0])) ? "success" : "fail";
    cout << result << endl;
    // Test that hasParams returns false for inputlayer, relu, softmax and true for dense
    cout << "testing hasParams on each layer type:" << endl;
    cout << "testing InputLayer...";
    result = (!InputLayer(1, INPUT_SHAPE).hasParams()) ? "success" : "fail";  // Expecting false
    cout << result << endl;
    cout << "testing DenseLayer...";
    result = (DenseLayer(denseLayer1Units, regularization).hasParams()) ? "success" : "fail";  // Expecting true
    cout << result << endl;
    cout << "testing ReluLayer...";
    result = (!ReluLayer().hasParams()) ? "success" : "fail";  // Expecting false
    cout << result << endl;
    cout << "testing SoftMaxLayer...";
    result = (!SoftMaxLayer().hasParams()) ? "success" : "fail";  // Expecting false
    cout << result << endl;
    cout << "testing CrossEntropyLoss...";
    result = (!CrossEntropyLossLayer().hasParams()) ? "success" : "fail";  // Expecting false
    cout << result << endl;
    
    // Create a model
    BaseLayer* inputLayerPtr = &inputLayer;
    BaseLayer* crossEntropyLossLayerPtr = &crossEntropyLossLayer;
    BaseModel model(inputLayerPtr, crossEntropyLossLayerPtr);
    // Validate the model with build
    cout << "validating model... " << endl;
    model.build();
    cout << "validating model again..." << endl;
    model.build();  // To validate that the first run doesn't break anything, run it twice
    cout << "success" << endl;

    // Create sample data and perform forward pass

    // Initialize DataLoader
    // string fileName;
    // // prompt user for project directory
    // cout << "what's the full path to the data?" << endl;
    // cin >> fileName;
    // /home/chris/Desktop/final-project/src/data/train.csv
    // /home/pi/Desktop/CppFinalProject/src/data/train.csv
    DataLoader dataLoader("/home/chris/Desktop/final-project/src/data/train.csv", true, 0, 0.10, 0.10, 1000);

    // Print number of train, test, and validation examples
    // Print single example size (from Matrix class rows() cols())

    // Print some test examples
    int nExamples = 1;
    vector<unique_ptr<vector<Matrix>>> testData(0);
    dataLoader.getTestDataCopy(0, nExamples, testData);
    unique_ptr<vector<Matrix>> xTest = move(testData[0]), yTest = move(testData[1]);
    cout << "Test Examples:" << endl;
    for(int i=0;i<nExamples;i++) printExample((*xTest)[i], (*yTest)[i]);
    // Perform forward pass using example
    // TODO implement forward on model
    // Loop over examples and call forward on the model
    cout << "performing forward pass on model..." << endl;
    unique_ptr<Matrix> example;
    example = make_unique<Matrix>(move((*xTest)[0]));
    model.forward(move(example));
    cout << "success" << endl;

    // Test the outputs
    OutputData outputData;
    model.getOutputs(outputData);
    Matrix outputs = *outputData.outputs;
    cout << "viewing model outputs..." << endl;
    for(int j=0;j<outputs.rows();j++)
    {
        for(int k=0;k<outputs.cols();k++)
        {
            cout << outputs(j,k);
        }
        cout << endl;
    }


    //  Connect the child to the parent
    //   connect updates child and parent attributes
    // Move layer to the modelVector
    // Loop through layers and build
    //  build validates compatibilty between layers
    //  once built, the model can perform forward and backward pass

    // To copy a model...
    // Initialize a new modelVector on the heap
    // Loop through layers
    //  copy layer to local variable
    //  Connect the child copy to the parent copy
    //  move the copy to the new modelVector
    


    // Intialize model
    
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
