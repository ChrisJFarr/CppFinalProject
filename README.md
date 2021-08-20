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


## Project Design

Purpose: build a lighweight and high performing computational graph to implement 
a relatively simple neural network with a general training and inference operation when given 
input data. With a general implementation the final executable, once trained, is able to
perform prediction on an input file to an output file. Then evaluate can analyze a folder
of output files for performance.

Summary:
  For efficiency during the training process, multi-threading is to be used for computing the loss and
  gradients of examples concurrently. Two message queues are used for passing data to model threads
  and for returning the information for computing the parameter updates. Parameters are shared by all
  threads and model threads only have (and need) read access. Once a batch has finished processing
  the parameter updates can be computed and updated in place once model threads have all released
  their access.

  To evaluate the model performance and training both the outputs and layer gradients are returned 
  from the train data set. A forward pass is only used for the validation data.

https://stackoverflow.com/questions/11935030/representing-a-float-in-a-single-byte
To save memory, use 8-bit precision
float toFloat(uint8_t x) {
    return x / 255.0e7;
}
uint8_t fromFloat(float x) {
    if (x < 0) return 0;
    if (x > 1e-7) return 255;
    return 255.0e7 * x; // this truncates; add 0.5 to round instead
}
Perhaps use 16 bit precision centered around 0 for the 

Modify this to center around 255/2? Perhaps with relu, negatives are never needed anyways
What about the outputs? Sigmoid... are negatives needed?

Completed
* Clean up the design and review

Current Step
* Build the shell with classes and functions

Future
* Pull in mnist digits data
* Build data pipeline objects and loops
* Start building in order as the data flows

Initialize structs with all nullptrs by default
struct InputData
  unique_ptr<vector<floattype>> xData = nullptr;
  unique_ptr<vector<floattype>> targets = nullptr; if exists (send data to queue without targets to just get predictions)
struct OutputData
  unique_ptr<vector<floattype>> outputs = nullptr;
  vector<unique_ptr<vector<floattype>>> parameterGradients;  
  vector<unique_ptr<vector<floattype>>> layerGradients;  (mean-gradloss-wrt-inputs-er-outputs)

class BaseModel
  This class manages multi-threading and generic components
  copy constructor (copies are used in the thread functions)
    loop over and copy the modelVector to the new object
  implement the rule of five

  createModelThread (i think this function is passed to a thread)
    This function copies the modelGraph
    This function creates a thread, copies the modelVector, then loops forever (how can I reduce cost of looping)
    loop over modelVector 
    copy each layer to local modelVector
    reconnect the local graph (loop over vector and call connectParent from the child, passing the parent)
    loop forever 
      wait to get data from the messageQueue
      Parse the InputData:
        get xData
        get targets if exists (send data to queue without targets to just get predictions)
        get evaluateTrain bool
        get evaluateLayerGrads bool
      can the threads claim the parameters while performing forward and backward passes for extra safety?
        such as a shared control object that doesn't lock out the other threads
      perform forward pass, call BaseModel::forward(unique_ptr<paramdatatype[]> inputs)
        This should move the inputs
      Declare and intialize OutputData on the stack
      if targets are present in data object
        perform backward pass, call BaseModel::backward
        call BaseModel::getGradients(OutputData&)
        call BaseModel::getOutputs(OutputData&)
        move OutputData to messageQueue
      else
        call BaseModel::getOutputs(OutputData&)
        package up outputs in output struct
        get the outputs from the final layer (the loss layer hold thems)
        move OutputData to messageQueue
        problem: now model is stuck in forward position
          move pointers from child back to parent to setup for another forward pass
          Maybe add a function for resetting for forward position to the layers?

  TODO Copy the message queue from concurrency project
  2 queues, one for inputs and one for outputs
    MessageQueue<InputData> inputQueue
    MessageQueue<OutputData> outputQueue
  save
    loop over layers
    extract parameters with BaseLayer::getParams()
    write parameters to file in order of layers 
  load
    open file
    loop over layers
    ask layer if it needs params
    if needs then load line and pass to BaseLayer::setParams()
    else go to next layer
  forward(unique_ptr<paramdatatype[]> inputs)
    param: inputs which is the rvalue of a unique pointer to a 1d array
    loop over layers
    call forward on each layer except the last (loss is computed in BaseModel::backward)
    only the input layer requires an argument for forward, or another method could be used to move value to inputLayer
  backward(unique_prt<paramdatatype[]> targets)
    call forward on the final layer, this either requires an argument or set an attribute for target
    call backward on each layer (including the loss layer)
  OutputData getGradients(OutputData&)
    this returns a struct with gradients
    loop over layers
    TODO figure this out: all gradients need to be added from the layers in addition to the parameter gradients
    ask if they have params, hasParams
    call getGradients() on the layer to get the parameter gradients
    copy the shared pointer to vector in output struct
  OutputData getOutputs(OutputData&)
    this returns a struct with prediction/model outputs
    they should be waiting in the inputs of the final layer (loss layer) after forward is called

class MyModel: public BaseModel
  This class contains the models architecture instructions 
  constructor
    call MyModel::modelInstructions to populate the modelVector and paramVector
    call BaseModel::constructor
  Problem-specific implementation
  
  buildModelGraph (this happens once)
    This is implemented specific to model architecture
    Contains instructions for and builds the model graph
    populate modelVector
    Initialize layers in sequential order
    Connect layers

class BaseLayer
  copy constructor (this is called when copying the modelVector should be done for threads only)
    No copying of the inputs/outputs, just need to declare a unique ptr without allocation for the inputs
    Copy the shared pointers of parameters to the new object
    layer parameters are owned by the layer, copies of the parameters and grads are shared pointer
  parentLayer
  childLayer
  connectParent (the child is called and passed the parent)
    other.childLayer = this
    this.parentLayer = other
  std::unique_prt<2d array> inputs
  inputs
    they come from the parent layer
    stored on the heap
    child layers store the pointer from forward pass to use for backward pass
  outputs
    created on the heap
    create a unique pointer on the forward pass
    moved to child layer on forward pass
  parameters
    shared across all layers across threads, layers have read-only access
  hasParams returns boolean defaults to false unless overriden
  getParams()
    throw error if !hasParams
    this needs to coordinate with the training op and save
    return vector of pointers to the params vector<shared_ptr<paramdatatype[]>>
  setParams()
    throw error if !hasParams
    this needs to coordinate with BaseModel::load
  forward(unique-ptr&&)
    accepts a pointer rvalue argument
    moves the unique pointer to outputs to the child laye0r
  backward(unique-ptr&&)
    accepts a pointer rvalue argument
    moves the unique pointer to gradients-wrt-inputs to the parent layer

class Input: public BaseLayer
  int inputShape;
  forward()
    validate inputs shape (to handle incorrect input, should throw exception, also coordinate with the thread call to handle)
    call BaseLayer::forward to move pointer to child

class Dense: public BaseLayer
  constructor (this is done once in the BaseModel build method)
    compute output shape 
      uses input shape from parent and units
    create array of size of output shape on the heap use shared pointer handle
    vector<shared_ptr<paramdatatype[]>>_params; owned by layer
      create array for weights, add pointer to array to _params vector
      create array for bias, add pointer to _params vector
      randomly initialize bias and weights
    vector<shared_ptr<paramdatatype[]>>_grads
    hasParams = true;
  regularization controlled by hyperparam
  parameters, protected with mutex, wait to update weights until backward pass is done
  forward()
    allocate memory for outputs using unique ptr
    compute outputs and set value in memory
    call BaseLayer::forward to move pointer to child
  backward()
    allocate memory for and compute derivative of gradients wrt parameters 
      include regularization
      use a unique ptr, one for grads-wrt-weights and another for grads-wrt-bias
      move both to private _grads in order of [weights, bias]
    allocate memory for gradients wrt inputs on stack using unique ptr
      array is same size as inputs
    compute derivative of gradients wrt inputs
    call BaseLayer::backward to move gradients-wrt-inputs pointer to the parent

class Relu: public BaseLayer
  forward()
    allocate memory for outputs (same size as inputs) on stack with unique_ptr
    compute relu function assign to memory
    call BaseLayer::forward to move pointer to child
  backward()
    allocate memory for gradients-wrt-inputs
    compute relu derivative
      if relu function of input >0 then 1 else 0 * gradients
    multipy by gradients from child to get gradients-wrt-inputs
    call BaseLayer::backward to move pointer to parent

class Softmax: public BaseLayer
  forward()
    safe softmax
  backward()
    ? need work here
    https://github.gatech.edu/cfarr31/DeepLearning7643/blob/master/assignment1/models/softmax_regression.py

class CrossEntropyLoss
  targets are passed directly to this layer
  no child layer
  forward(unique_ptr<paramdatatype[]>&&targets)
    compute loss using inputs from parent and targets
    don't call BaseModel::forward, store the loss here
  backward()
    allocate memory on stack with unique-ptr the same shape as the inputs to store the gradients
    compute the gradients of the loss wtr the inputs
    call BaseLayer::backward to move loss to parent layer

class Optimizer
  Use adam
  Initialize with default lr and include decay param
  This class has read/write access to the shared model parameters

  How does this one work? TODO work on this part
  regularization occurs already from the layers with params (only Dense)
  It can get gradients... but does it need anything else? see paper
  This takes in the gradients for the whole batch
  Computes the update
  Requests write access to the parameters
  updates the parameters
  releases the lock


  class DataLoader
  Reads fileName(train.csv)
  Accepts %train, %test, %validation floats
  Accepts bool header (skips first row if true)
  Internally stores constants for _test, and _valid

  This could instead be a dynamic loader
  It gets the information about the available data, such as n-examples
  Computes which indices belong to which dataset (indices should be shuffled)
  Reads and stores subsets on demand, such as for batch-size and maybe a buffer
  Perhaps for the validation and test it loads the full sets and stores
  This would, however, require reading from the train file often when training
  It also may not scale super well when large
  Another option is to store the train-only data and randomly read from it


  How long would it take to read the last 50 rows? (worst case for batch-size of 50)

  analyze()
  this should be called when constructed perhaps
  open the data source file
  learn and store the number of examples and the size/shape of each example
  remember target is at the beginning of the line

  clean()
    scale the data from 0-1

  parse()
  this should be called after analyze() in the constructor
  using size variables for trainSize, testSize, and validSize
  parse the file and write data to _test and _valid
  write the train to a new file in shuffled order
  open train to allow loadBatch to work

  loadBatch(int batchSize)
  read from the open train file the next n==batchSize lines
  clean
  read directly into matrix objects...
  return pointer to 

  getTestData()
  getValidData()



This should become a class or part of a class!!!
class TBD
function train (initializes model, trains, and stores final parameters)
  perform setup
  Initialize MyModel
  Determine number of threads to spin up, smallest of MAX_THREADS and BATCH_SIZE
  Use createModelThread to spin up each thread and store in a vector?
    do they need to be stored? How will they be joined later if not.
  ask the model for trainable the parameters (these are shared among all threads)
  analyze the training data to get the total number of training 
  load the validation data into two vectors, for x-data and y-data
  loop for n-epochs
    loop for batch-size / total-examples
      manage train data buffer
        2 or 3x's the batch-size should be loaded into a buffer
        then shuffled... and the batch is selected from this
        then the next batch-size is loaded next time
      loop over batch-size
        package up the example into InputData struct
        send data to message queue for processing
      wait until gradient-queue == batch-size
      wait until parameters are freed up (for extra safety)
      Loop over the parameter updates to compute the average update
        divide each update by batch-size then add to parameter
      
      Outputs should also be collected (not just gradients) and passed to evaluation
        pass InputData::
      Average derivatives should be stored for training analysis to analyze gradient flow
            but not just for parameters, but at every layer
    
    loop over validation data
    should validation data persist (yes, speed this up)
    create vector of pointers to x and y data objects
    pass the x-data to the message queue
    pass the outputs and the targets to evaluation

    print summary
      train loss, train accuracy
      validation loss, validation accuracy
      customized gradients information based on args passed to executable?
        gradient total per layer l1:float here, l2:... etc for each layer

    Use validation performance to stop training early

  TODO function predict (populate folders with predictions for evaluation)
  TODO function evaluate (should run on the test data)
  TODO function accuracy, confusion matrix (used while training and for evaluate)

Efficient computation resource: https://gist.github.com/nadavrot/5b35d44e8ba3dd718e595e40184d03f0
For efficiency threads can be used for multiple simultaneous examples to fill
up the batch. Then no need to parallelize the basic matrix function.
OMP or MPI packages for threading the matrix loops


TODO Design a matrix class
  Use this class for all data objects
    load data into a matrix((1,INPUT_SHAPE))
    create a matrix for each parameter
    create a matrix for calculations between layers

  Excellent resource: https://medium.com/@furkanicus/how-to-create-a-matrix-class-using-c-3641f37809c7

class Matrix
  Matrix(shape)
    rows, cols = shape
    create a 2d vector of dtype

  memory will always be shared
  the underlying storage can be a 1d or 2d float vector (or any type)
  default storage is a vector of float vectors
  it should be initialized with a known shape
  shape (rows, cols)
  int[2] shape = (rows, cols);

  private member
  shared_ptr<vector<vector<float>>> _data

  Override accessor
  vector<float>& operator()(int, int);

  mathMultiply(Matrix&)
    use tensorflow's api, this should expect two matrices of exact same shape
    and multiply this[j][k] * other[j][k]
  operator*(Matrix&)
    must be compatible shapes, rows of this must == columns of other
    perform matrix multiplication
    output to a new matrix
  operator+(Matrix&)
  dot(Matrix&)
  Matrix transpose()

  scalar operators
  operator*(float&)



main
  run options
    * train or load
    * predict and write to output file
    * evaluate
