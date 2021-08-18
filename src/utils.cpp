#include "utils.h"

// class Optimizer
//   Use adam
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

DataLoader::DataLoader(string filePath, bool header, int targetPos, float testRatio, float validRatio)
{
    // Set values
    _filePath = filePath;
    _header = header;
    _targetPos = targetPos;
    _testRatio = testRatio;
    _validRatio = validRatio;
    // Reads fileName(train.csv)
    // Accepts %train, %test, %validation floats
    // Accepts bool header (skips first row if true)
    // Internally stores constants for _test, and _valid
    // call analyze() to set _testShape, _validShape
    // call parse()
    // Print summary
    // total example, train, test, valid example counts
}
// This is a dynamic loader
// It gets the information about the available data, such as n-examples
// Computes which indices belong to which dataset (indices should be shuffled)
// Reads and stores subsets on demand, such as for batch-size and maybe a buffer
// Perhaps for the validation and test it loads the full sets and stores
// This would, however, require reading from the train file often when training
// It also may not scale super well when large
// Another option is to store the train-only data and randomly read from it


// How long would it take to read the last 50 rows? (worst case for batch-size of 50)

// analyze()
void DataLoader::analyze()
{
    // this should be called when constructed perhaps
    // open the data source file
    // learn and store the number of examples and the size/shape of each example
    // Set _trainSize based on the number of examples left over after test and valid
    // remember target is at the beginning of the line
    // Set _testSize, _validSize, _trainSize
}

void DataLoader::parse()
{
// this should be called after analyze() in the constructor
// using size variables for trainSize, testSize, and validSize
// parse the file and write data to _test and _valid
// write the train to a new file in shuffled order
// open train to allow loadBatch to work
}

// clean()
//  scale the data from 0-1

// loadTrainBatch(int batchSize)
// this function randomly samples with replacement
// read from the open train file the next n==batchSize lines
// clean
// read directly into matrix objects...
// return pointer to 

// getTestDataCopy()
// getValidDataCopy()



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