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

DataLoader::DataLoader(string filePath, bool header, int targetPos, float testRatio, float validRatio, int maxExamples)
{
    // Set values
    _filePath = filePath;
    _header = header;
    _targetPos = targetPos;
    _testRatio = testRatio;
    _validRatio = validRatio;
    _maxExamples = maxExamples;
    // call analyze() to set _testShape, _validShape
    // Initialize the test and validation pointers
    _testData = make_unique<vector<Matrix>>();
    _testTargets = make_unique<vector<Matrix>>();
    _validData = make_unique<vector<Matrix>>();
    _validTargets = make_unique<vector<Matrix>>();
    analyze();
    split();
    load();
    // Print summary
    cout << "Total Examples: " << _totalSize;
    cout << " Train Examples: " << _trainSize;
    cout << " Test Examples: " << _testSize;
    cout << " Validation Examples: " << _validSize << endl;
}

DataLoader::~DataLoader()
{
    // TODO clean up temporary train file
    const char* removePath = _tempPath.c_str();
    remove(static_cast<const char*>(removePath));
}

void DataLoader::analyze()
{
    // this should be called when constructed perhaps
    // open the data source file
    if(DEBUG) cout << "attempting to read file at " << _filePath << endl;
    ifstream file(_filePath);
    // learn and store the number of examples and the size/shape of each example
    _totalSize = 0;  // Reset to 0
    bool firstRow = true;  // Intialize this with true, set to false after first line is parsed
    string line;

    if(file.is_open())
    {
        while(file >> line)
        {
            // If header, skip the first line
            if(_header && firstRow){firstRow=false;continue;};
            // Increment _totalSize while looping through file
            _totalSize++;
            if(_totalSize>=_maxExamples) break;
        }
    }
    if(DEBUG) cout << "found " << _totalSize << " records" << endl;
    // Set _testSize, _validSize, _trainSize
    _testSize = static_cast<int>(static_cast<float>(_totalSize) * _testRatio);
    _validSize = static_cast<int>(static_cast<float>(_totalSize) * _validRatio);
    // Set _trainSize based on the number of examples left over after test and valid
    _trainSize = _totalSize - (_testSize + _validSize);
    if(DEBUG) cout << "exiting DataLoader::analyze() " << endl;
}


void DataLoader::split()
{
    // this should be called after analyze() in the constructor
    // using size variables for trainSize, testSize, and validSize
    // parse the file and write data to _test and _valid, no need to shuffle

    // Decide which indices belong to which part, this should be random
    // Randomly create testIndices, validIndices, and trainIndices(this one needs to be stored)
    // Iterate through and randomly assign until sizes are met
    srand(1987);  // Set seed once, and always the same for reproducibility
    float myRandomNumber;
    for(int i=0;i<_totalSize;i++) 
    {
        myRandomNumber = static_cast<float>(rand()) / RAND_MAX;
        if((_testIndices.size() < _testSize) && (myRandomNumber < _testRatio))
        {
            _testIndices.emplace_back(i);
        } 
        else if((_validIndices.size() < _validSize) && (myRandomNumber < (_validRatio * 2.0)))
        {
            _validIndices.emplace_back(i);
        } 
        else if((_trainIndices.size() < _trainSize))
        {
            _trainIndices.emplace_back(i);
        } 
        else 
        {
            // If hasn't been assigned, find one that isn't full yet
            if(_testIndices.size() < _testSize){_testIndices.emplace_back(i);}
            else if(_validIndices.size() < _validSize){_validIndices.emplace_back(i);}
            else if(_trainIndices.size() < _trainSize){_trainIndices.emplace_back(i);}
            else {
                cout << "index: " << i << endl;
                throw out_of_range("Found an index that doesn't belong?");
                }
        }
    }

    // Debug, print out indices
    cout << "test: ";
    for(int i=0; i<20;i++){cout << _testIndices[i] << ",";}
    cout << endl;
    cout << "valid: ";
    for(int i=0; i<20;i++){cout << _validIndices[i] << ",";}
    cout << endl;
    cout << "train: ";
    for(int i=0; i<20;i++){cout << _trainIndices[i] << ",";}
    cout << endl;
}



void DataLoader::load()
{
    // iterate through the examples, counting backwards from max index
    // use an if statement to check which portion it belongs to and move it there
    // if(i==testIndices[-1]) {move to test, pop last element from testIndices} 
    // else if(i==validIndices[-1]) {move to valid, pop last element from validIndices}
    // else continue (skip over train)
        // open the data source file
    ifstream file(_filePath);  //, ios::in
    ofstream tempFile(_tempPath);  // write train examples here
    // learn and store the number of examples and the size/shape of each example
    bool firstRow = true;  // Intialize this with true, set to false after first line is parsed
    string line, dataPoint;
    int rowIndex = (_totalSize - 1);  // Start with the max index (since index vectors are sorted ascending)
    
    if(file.is_open())
    {
        while(file >> line)
        {
            // If header, skip the first line
            if(_header && firstRow){firstRow=false;continue;};
            if(rowIndex<0) break;  // THis occurs when _maxExamples > available examples
            // First determine if this is a train example
            if((_trainIndices.size() > 0) && (rowIndex == _trainIndices.back()))
            {
                // Write example to temporary file
                tempFile << line << endl;
                // Remove index from train
                _trainIndices.pop_back();
                rowIndex--;
                continue;
            }
            // Prepare the data structure
            // Matrix can be initialized with a vector<vector<MyDType>>>
            vector<vector<MyDType>> xData;
            vector<MyDType> xRow;
            vector<vector<MyDType>> yData;
            vector<MyDType> yRow;
            // Create a stream object
            stringstream s(line);
            // Populate the row
            // remember target is at the beginning of the line
            getline(s, dataPoint, ',');
            yRow.emplace_back(stof(dataPoint));
            while (getline(s, dataPoint, ',')) {
                xRow.emplace_back(clean(stof(dataPoint)));
            }
            // Insert rows into data objects
            xData.emplace_back(xRow);
            yData.emplace_back(yRow);
            // Decide where it needs to go
            // If test, add to test
            if((_testIndices.size() > 0) && (rowIndex == _testIndices.back()))
            {
                // cout << "adding test data" << endl;
                // Add to test
                _testData->emplace_back(Matrix(xData));
                _testTargets->emplace_back(Matrix(yData));

                // Pop last index on test
                _testIndices.pop_back();
                rowIndex--;
                continue;
            }
            else if((_validIndices.size() > 0) && (rowIndex == _validIndices.back()))
            {
                // cout << "adding validation data" << endl;
                // Add to valid
                _validData->emplace_back(Matrix(xData));
                _validTargets->emplace_back(Matrix(yData));
                // Pop last index on valid
                _validIndices.pop_back();
                rowIndex--;
                continue;
            }
            else {
                throw out_of_range("rowIndex does not equal any available index.");
            }
        }
    }
}

MyDType DataLoader::clean(MyDType dataPoint)
{
    // cout << "Before: " << dataPoint << " After: " << (dataPoint/255.) << endl;
    return dataPoint / 255.;
}
// For a more reusable clean...
//  call this when initializing
//  scale the data from 0-1
//  look at all the data to find the max and min values
//  rescale all the data in-place

void DataLoader::getTestDataCopy(int startingIndex, int batchSize, vector<unique_ptr<vector<Matrix>>>& dataVector)
{
    // Create deep copy of test data
    // Create a unique_ptr to new vector and return
    unique_ptr<vector<Matrix>> xOut = make_unique<vector<Matrix>>();
    unique_ptr<vector<Matrix>> yOut = make_unique<vector<Matrix>>();
    // Copy test data to out data
    for(int i=startingIndex;i<startingIndex+batchSize;i++)
    {
        // Copy the data and targets (running into a binding error 'Matrix&' vs 'const Matrix')
        (*xOut).emplace_back(move((*_testData))[i]);
        (*yOut).emplace_back(move((*_testTargets))[i]);
    }
    dataVector.emplace_back(move(xOut));
    dataVector.emplace_back(move(yOut));
}

void DataLoader::getValidDataCopy(int startingIndex, int batchSize, vector<unique_ptr<vector<Matrix>>>& dataVector)
{
    // Create deep copy of validation data
    // Create a unique_ptr to new vector and return
    unique_ptr<vector<Matrix>> xOut = make_unique<vector<Matrix>>();
    unique_ptr<vector<Matrix>> yOut = make_unique<vector<Matrix>>();
    // Copy test data to out data
    for(int i=startingIndex;i<startingIndex+batchSize;i++)
    {
        // Copy the data and targets (running into a binding error 'Matrix&' vs 'const Matrix')
        (*xOut).emplace_back(move((*_validData))[i]);
        (*yOut).emplace_back(move((*_validTargets))[i]);
    }
    dataVector.emplace_back(move(xOut));
    dataVector.emplace_back(move(yOut));
}

void DataLoader::getTrainBatch(int batchSize, vector<unique_ptr<vector<Matrix>>>& dataVector)
{
    // this function randomly samples with replacement (underlying data is not shuffled)
    // loadTrainBatch(int batchSize)
    // cout << "DataLoader::getTrainBatch" << endl;
    
    // Select n=batchSize random indices from 0-_trainSize
    vector<int> indices;
    srand(time(0));
    for(int i=0;i<batchSize;i++)
    {
        float myRandomNumber = static_cast<float>(rand()) / RAND_MAX;
        int myRandomIndex = static_cast<int>(myRandomNumber * _trainSize);  // Truncate to nearest int
        indices.push_back(myRandomIndex);
    }
    // cout << "sorting indices" << endl;
    // Sort the indices in ascending order
    sort(indices.begin(), indices.end());
    // Prep intermediate vector for storing data as its read
    unique_ptr<vector<Matrix>> xOut = make_unique<vector<Matrix>>();
    unique_ptr<vector<Matrix>> yOut = make_unique<vector<Matrix>>();

    // Open file and get the data
    ifstream file(_tempPath);
    string line, dataPoint;
    int rowIndex = (_trainSize - 1);  // Track the index
    // Search for indices and add data to outputs
    // Loop while indices is not empty and file is open
    // cout << "getting ready to loop" << endl;
    if(file.is_open())
    {
        while((indices.size() > 0) && (file >> line))
        {
            if(rowIndex==indices.back())
            {
                // Pop the indices as they are found
                indices.pop_back();
            } else {
                // Skip to next loop if not a selected index
                rowIndex--;
                continue;
            }
            // cout << "adding an example" << endl;
            // No header exists in the temp train file
            // Prepare the data structure
            vector<vector<MyDType>> xData;
            vector<MyDType> xRow;
            vector<vector<MyDType>> yData;
            vector<MyDType> yRow;
            // Create a stream object
            stringstream s(line);
            // Populate the row
            // remember target is at the beginning of the line
            getline(s, dataPoint, ',');
            yRow.emplace_back(stof(dataPoint));
            while (getline(s, dataPoint, ',')) {
                xRow.emplace_back(clean(stof(dataPoint)));
            }
            // Insert rows into data objects
            xData.emplace_back(xRow);
            yData.emplace_back(yRow);
            // cout << "moving matrix to xOut and yOut" << endl;
            // Insert into provided data object
            xOut->emplace_back(move(Matrix(xData)));
            yOut->emplace_back(move(Matrix(yData)));
        }
    }
    // cout << "Adding to dataVector" << endl;
    dataVector.emplace_back(move(xOut));
    dataVector.emplace_back(move(yOut));
}



// This should become a class or part of a class instead of a function

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