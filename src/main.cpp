#include <iostream>
#include <utility>
#include <vector>
#include <memory>
#include "model.h"
#include "utils.h"
#include "layer.h"
#include "tests.cpp"

using namespace std;

/*
valgrind --leak-check=yes --track-origins=yes --log-file=valgrind-out.txt ./NeuralNetwork

/home/chris/Desktop/final-project/src/data/train.csv
/home/pi/Desktop/CppFinalProject/src/data/train.csv
/home/chris/Desktop/CppFinalProject/src/data/train.csv
*/

// Build this here, then move to utils.cpp
// This will eventually take a xData Matrix, a yData Matrix, pred Matrix


int main() {

    // Optional, run tests (will need to input address to data location)
    // runAllTests();

    //////////////////////////////////////////////////
    /////// DIY Neural Network Starter in CPP ////////
    //////////////////////////////////////////////////


    /////// Load Data ////////////

    // Initialize DataLoader
    string fileName;
    // prompt user for project directory
    cout << "what's the full path to the data?" << endl;
    cin >> fileName;
    // /home/chris/Desktop/final-project/src/data/train.csv
    // /home/pi/Desktop/CppFinalProject/src/data/train.csv
    // `project directory path`/src/data/train.csv
    // The expected file is a csv with a target in the first 
    //  column and covariates after
    bool header = true;  // The file containers a header row
    // The portion of data to be reserved for validation and test datasets
    float testRatio = 0.10, validRatio = 0.10;  
    // The maximum number of examples to load from the file
    int maxExamples = 1000;
    int targetPosition = 0; // Location of the target column

    DataLoader dataLoader(fileName, header, targetPosition, testRatio, validRatio, maxExamples);

    /*
    What's happening in DataLoader on initialization?
    analyze(); 
        * Open the file to count the rows and compute train, test, and validation sizes 
    split();
        * Decide which indices belong to each dataset in random fashion 
    load();
        * Write train examples to a temporary csv file for random batch reading
        * Load test and validation examples into memory in Matrix objects
    */

    /////// View Sample /////////
    int nExamples = 5;
    vector<unique_ptr<vector<Matrix>>> testData(0);
    dataLoader.getTestDataCopy(0, nExamples, testData);
    unique_ptr<vector<Matrix>> xTest = move(testData[0]), yTest = move(testData[1]);
    cout << "Test Examples:" << endl;
    for(int i=0;i<nExamples;i++) printExample((*xTest)[i], (*yTest)[i]);


    /////// Build a Model ////////

    MyDType regularization = 0.0001;
    int denseLayer1Units = 100;
    int denseLayer2Units = 150;
    int outputShape = 10;
    string result;


    // Declare Two-layer soft-max model
    InputLayer inputLayer(1, INPUT_SHAPE);
    DenseLayer denseLayer1(denseLayer1Units, regularization);
    ReluLayer reluLayer1;
    DenseLayer denseLayer2(denseLayer2Units, regularization);
    ReluLayer reluLayer2;
    DenseLayer denseOutputs(outputShape, regularization);
    SoftMaxLayer softMaxLayer;
    CrossEntropyLossLayer crossEntropyLossLayer;

    // Connect the layers
    denseLayer1(inputLayer);  // Connect parent with () operator
    reluLayer1(denseLayer1);  // Connect to denselayer
    denseLayer2(reluLayer1);  // Connect parent with () operator
    reluLayer2(denseLayer2);  // Connect to denselayer
    denseOutputs(reluLayer2);  // Connect parent with () operator
    softMaxLayer(denseOutputs);
    crossEntropyLossLayer(softMaxLayer);

    // Create a model-graph with the first and last layers
    // Using raw pointers here only to pass the location of the layers, they are not
    // responsible for managing memory of the layers
    BaseLayer* inputLayerPtr = &inputLayer;
    BaseLayer* lossLayerPtr = &crossEntropyLossLayer;
    BaseModel model(inputLayerPtr, lossLayerPtr);
    // Validate the model with build
    model.build();

    ///// Perform forward pass on each example ///
    cout << endl << endl;
    for(int i=0;i<nExamples;i++)
        {
        // Show the example
        printExample((*xTest)[i], (*yTest)[i]);
        cout << "performing forward pass on model..." << endl;
        unique_ptr<Matrix> example;
        example = make_unique<Matrix>(move((*xTest)[i]));
        model.forward(move(example));

        // Test the outputs
        OutputData outputData;
        model.getOutputs(outputData);
        Matrix outputs = *outputData.outputs;
        // Print the random (untrained) predictions
        float maxPred = 0.0;  // 0 is the lowest softmax can output
        int maxPredIndex = -1;
        for(int k=0;k<outputs.cols();k++)
        {
            if(outputs(0, k)>maxPred)
            {
                maxPred = outputs(0, k);
                maxPredIndex = k;
            }
        }
        cout << "I'm not trained yet, but I'm guessing this is a " << maxPredIndex << endl;
    }


    /////////////////////////////
    /////// DIY TODO's //////////
    /////////////////////////////
    // Implement the backward pass of the model
    // Implement the optimizer and training functions
    // Then train the model to learn how to identify handwritten digits!

    return 0;
}
