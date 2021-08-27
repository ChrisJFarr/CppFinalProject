# CPPND: Capstone Neural Network Repo

This is the beginnings of a Neural Network Library for building general function approximators using C++

A demonstration of how the library works is included below and in main.cpp.

The following components have been implemented:
* DataLoader class: Reads from a CSV file and manages the train/test/validation data split
	* Splits are random and controlled by the testRatio and validRatio parameters
	* Data is loaded into Matrix objects
* Matrix class: Stores a 2d vector with many matrix functions implemented such as matrix-multiply, addition, transpose, etc. Each Matrix object owns its own data.
* "layer.h/.cpp": Contains a framework for general purpose graph-based neural network layers. So far the InputLayer, ReluLayer, DenseLayer, SoftmaxLayer, and CrossEntropyLossLayer have their forward pass implemented. A BaseLayer class handles the connecting and communication across a connected graph.
* Model class: Consumes an InputLayer and a loss type layer. It validates a graph and uses the BaseLayer interface to move data objects from one layer to the next to generate predictions. Predictions are random until parameters are trained. The backward pass implementation is a future TODO along with the optimization and training algorithms.

So far this library is able to construct a multi-layer neural network, randomly initialize the weights, and perform a forward pass. This was a large undertaking so far and much work is left to be done before being able to train a model using this library.

**Rubric Points**
* The project code compiles and runs without errors
* Control structures are used throughout the libary. See `Matrix::mathMultiply` to see a nested for loop over a 2D vector in `matrix.cpp` line 97.
*  The code is clearly organized into classes and functions. See `layer.h/.cpp` where many kinds of network layers are organized into a standard interface using virtual functions in `BaseLayer`. Many layer types inherit from `BaseLayer` and implement special functions such as uniform-random intialized weight parameters in the `DenseLayer` class.
*  The project can read from any dataset that contains a target in the first column and covariates after. This example is reading handwritten integers from the MNIST dataset found in the `src/data` folder of the repo.
*  The project accepts user input to specify any location of data, the full path to the data is expected and the format required is `.csv`
*  Object oriented programming: as metioned many classes are used as well as inheritance. See `layer.h/.cpp` for an example of inheritance. Also `model.h/.cpp` uses inheritance.
*  All data members are explicitly specified as public, protected, or private.
*  All class member functions are documented with notes throughout the library.
*  Class encapsulation exists. For example see the `Matrix` class and the protected `_data` member in `matrix.h/.cpp`.
*  Class hirearchies are logical. For example see `BaseLayer` and the subsequent classes that inherit many utility functions as well as virtual functions that make up the interface of a graph. This allows for polymorphism when storing pointers to the layers which is required for creating general multi-layer neural networks.
*  Many functions use pass-by-reference in the project code. See `Matrix::mathMultiply` and `Matrix::operator*` among others that accept another Matrix by reference.
* The project uses at least one smart pointer, both `unique_ptr` and `shared_ptr` are used. See `matrix.h` and the private attribute `_data`, which is a `unique_ptr` to a 2d vector. The only place where raw pointers are used is in the layers to allow for polymorphism in the forward pass. According to valgrind this led to no memory issues as the layers are not responsible for managing other layer's data.


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
4. Run it: `./NeuralNetwork`.

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


	/*
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
	*/
