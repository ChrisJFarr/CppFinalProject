# CPPND: Capstone Hello World Repo

This is a starter repo for the Capstone project in the [Udacity C++ Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213).

The Capstone Project gives you a chance to integrate what you've learned throughout this program. This project will become an important part of your portfolio to share with current and future colleagues and employers.

In this project, you can build your own C++ application starting with this repo, following the principles you have learned throughout this Nanodegree Program. This project will demonstrate that you can independently create applications using a wide range of C++ features.

## Dependencies for Running Locally
* cmake >= 3.7
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./HelloWorld`.

## How to use the library

	// Initialize DataLoader
	string fileName;
	// prompt user for project directory
	cout << "what's the full path to the data?" << endl;
	cin >> fileName;
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



class BaseModel
  Consumes layers from a connected graph
class BaseLayer
class Input: public BaseLayer
class Dense: public BaseLayer
class Relu: public BaseLayer
class Softmax: public BaseLayer
class CrossEntropyLoss

Utilities
  class Optimizer
    * TODO Future implementation of adam optimizer for training a network
  class DataLoader
    * Loads and manages data for training a neural network
  function train (initializes model, trains, and stores final parameters)

class Matrix
  The primary data structure utilized is the Matrix class
