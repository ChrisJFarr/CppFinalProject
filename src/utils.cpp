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
    // call analyze() to set _testShape, _validShape
    analyze();
    // call parse()
    // Print summary
    cout << "Total Examples: " << _totalSize;
    // total example, train, test, valid example counts
}




// How long would it take to read the last 50 rows? (worst case for batch-size of 50)

// analyze()
void DataLoader::analyze()
{
    // this should be called when constructed perhaps
    // open the data source file
    ifstream file(_filePath);  //, ios::in
    // learn and store the number of examples and the size/shape of each example
    _totalSize = 0;  // Reset to 0
    bool firstRow = true;  // Intialize this with true, set to false after first line is parsed

    // string line, dataPoint;
    string line;
    // vector<MyDType> row;
    
    if(file.is_open())
    {
        while(file >> line)
        {
            // If header, skip the first line
            if(_header && firstRow){firstRow=false;continue;};
            // Increment _totalSize while looping through file
            _totalSize++;
            // // Clear the row for this iteration
            // row.clear();
            // // Create a stream object
            // stringstream s(line);
            // // Populate the row
            // while (getline(s, dataPoint, ',')) {
            //     row.push_back(stof(dataPoint));
            // }
        }
    }
    // Set _trainSize based on the number of examples left over after test and valid
    // remember target is at the beginning of the line
    // Set _testSize, _validSize, _trainSize
}

void DataLoader::load()
{
    // this should be called after analyze() in the constructor
    // using size variables for trainSize, testSize, and validSize
    // parse the file and write data to _test and _valid
    // write the train to a new file in shuffled order
    // open train to allow loadBatch to work
}

// clean()
//  call this when initializing
//  scale the data from 0-1
//  look at all the data to find the max and min values
//  rescale all the data in-place

// unique_ptr<vector<Matrix>> DataLoader::getTestDataCopy()
// {
//     // Create deep copy of test data
//     // Create a unique_ptr to new vector and return
// }
// unique_ptr<vector<Matrix>> DataLoader::getValidDataCopy()
// {
//     // Create deep copy of validation data
//     // Create a unique_ptr to new vector and return
// }
// unique_ptr<vector<Matrix>> DataLoader::getTrainBatch(int batchSize)
// {
//     // loadTrainBatch(int batchSize)
//     // this function randomly samples with replacement
//     // read from the open train file the next n==batchSize lines
//     // clean
//     // read directly into matrix objects...
//     // return pointer to 
// }




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