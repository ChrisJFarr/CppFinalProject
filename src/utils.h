#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <random>


#include "constants.h"
#include "matrix.h"

using namespace std;

class AdamOptimizer
{
public:
private:
};
// class Optimizer
//   Use adam https://arxiv.org/pdf/1412.6980.pdf
//   Initialize with default lr and include decay param
//   This class has read/write access to the shared model parameters

//   How does this one work? TODO work on this part
//   regularization occurs already from the layers with params (only Dense)
//   It can get gradients... but does it need anything else? see paper
//   This takes in the gradients for the whole batch
//   Computes the update
//   Requests write access to the parameters
//   updates the parameters
//   releases the lock


class DataLoader
// This is a dynamic loader
// It gets the information about the available data, such as n-examples
// Computes which indices belong to which dataset (indices should be shuffled)
// Reads and stores subsets on demand, such as for batch-size and maybe a buffer
// Perhaps for the validation and test it loads the full sets and stores
// This would, however, require reading from the train file often when training
// It also may not scale super well when large
// Another option is to store the train-only data and randomly read from it
{
public:
    DataLoader(string filePath, bool header, int targetPos, float testRatio, float validRatio, int maxExamples);
    ~DataLoader();  // Delete the temp train file for clean up
    void getTestDataCopy(int startingIndex, int batchSize, vector<unique_ptr<vector<Matrix>>>&);
    void getValidDataCopy(int startingIndex, int batchSize, vector<unique_ptr<vector<Matrix>>>&);
    void getTrainBatch(int batchSize, vector<unique_ptr<vector<Matrix>>>&);

private:
    DataLoader(){};  // No default constructor, must pass fileName
    void analyze();  // Analyze data, sets sizes, selects
    void split(); // select and store indices
    void load();  // (prev parse) loop over files once again and add
    MyDType clean(MyDType); // Project specific preprocessing, a little hacky for this one
    string _filePath;
    string _tempPath = "DataLoader.csv";
    bool _header;
    int _targetPos;
    int _totalSize, _trainSize, _testSize, _validSize, _maxExamples;
    float _testRatio, _validRatio;
    vector<int> _testIndices, _validIndices, _trainIndices;
    // store a pointer to a vector of matrix objects for data and targets
    unique_ptr<vector<Matrix>> _testData;
    unique_ptr<vector<Matrix>> _validData;
    unique_ptr<vector<Matrix>> _testTargets;
    unique_ptr<vector<Matrix>> _validTargets;

    // TODO Add file object for train, this maintains an open connection to the data object
    // Don't forget to close it (open it every time instead)

    // clean()
    // maybe convert to correct dtype
    //  scale the data from 0-1

    // parse()
    // this should be called after analyze() in the constructor
    // using size variables for trainSize, testSize, and validSize
    // parse the file and write data to _test and _valid
    // write the train to a new file in shuffled order
    // open train to allow loadBatch to work

    // loadBatch(int batchSize)
    // read from the open train file the next n==batchSize lines
    // clean
    // read directly into matrix objects...
    // return pointer to 

    // getTestData()
    // getValidData()

};


// This should become a class or part of a class!!!

// function train (initializes model, trains, and stores final parameters)
//   perform setup
//   Initialize MyModel
//   Determine number of threads to spin up, smallest of MAX_THREADS and BATCH_SIZE
//   Use createModelThread to spin up each thread and store in a vector?
//     do they need to be stored? How will they be joined later if not.
//   ask the model for trainable the parameters (these are shared among all threads)
//   analyze the training data to get the total number of training 
//   load the validation data into two vectors, for x-data and y-data
//   loop for n-epochs
//     loop for batch-size / total-examples
//       manage train data buffer
//         2 or 3x's the batch-size should be loaded into a buffer
//         then shuffled... and the batch is selected from this
//         then the next batch-size is loaded next time
//       loop over batch-size
//         package up the example into InputData struct
//         send data to message queue for processing
//       wait until gradient-queue == batch-size
//       wait until parameters are freed up (for extra safety)
//       Loop over the parameter updates to compute the average update
//         divide each update by batch-size then add to parameter
      
//       Outputs should also be collected (not just gradients) and passed to evaluation
//         pass InputData::
//       Average derivatives should be stored for training analysis to analyze gradient flow
//             but not just for parameters, but at every layer
    
//     loop over validation data
//     should validation data persist (yes, speed this up)
//     create vector of pointers to x and y data objects
//     pass the x-data to the message queue
//     pass the outputs and the targets to evaluation

//     print summary
//       train loss, train accuracy
//       validation loss, validation accuracy
//       customized gradients information based on args passed to executable?
//         gradient total per layer l1:float here, l2:... etc for each layer

//     Use validation performance to stop training early

//   TODO function predict (populate folders with predictions for evaluation)
//   TODO function evaluate (should run on the test data)
//   TODO function accuracy, confusion matrix (used while training and for evaluate)