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
of output files for performance?

Components:

Data can be moved from node to node using shared pointers. Each node
creates new data to write to, then passes a shared or unique pointer to 
the child nodes during forward pass and parent nodes during backward pass.
On the backward pass there are dependent connections to the parameters as
well as the layer inputs. On the forward pass, the pointers are kept 
for the backward pass. Only thing is, there is no need to allocate new memory
when the same graph (with same shape and type of objects) is needed time and time again. 

For multithreading, each thread contains the forward and backward pass to compute the 
update. A full batch, populated by one thread at a time, is used for the final update. 
Therefore concurrency seems to rely on batch-size > 1. Each thread is a copy of the graph
and the data is built on the first usage. But how to keep the graph intact between examples?
The threads can be kept in a vector, when one returns... actually we just need a summary
from each thread once the update has been computed for each node. This can be
stored in a single graph and incrementally updated.

Each thread creates a graph and the graph persists throughout training.
Or... it could be risky this route and safer to allocate memory for each
new example.

Concurrency within a single graph....

Simplify, I don't need to use a graph, node, and edge concept.
I can just create different layer classes and then a model class.
However.. does using a graph allow for concurrency that otherwise wouldn't?
Perhaps a model class can still use threads to compute updates for a single example.

Building on concept of implementing layers. Each thread builds a graph, which
is really just a model. But the model class has more information than just a model
including parameter update information. As a forward pass occurs, (decide either allocate memory or overwrite)
perhaps I use the heap (more space?) and keep it static, so it overrides it on each forward and 
backward pass without allocating new memory.

Output shapes of a layer on the foward pass are the same size as the gradients on the backward pass.

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


Steps today
* Clean up the design and review
* Build the shell with classes and functions
Tommorrow or (today if time)
* Pull in mnist digits data
* Build data pipeline objects and loops


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
      can the threads claim the parameters while performing forward and backward passes for extra safety?
        such as a shared control object that doesn't lock out the other threads
      perform forward pass, call BaseModel::forward
      if targets are present in data object
        perform backward pass, call BaseModel::backward
        call BaseModel::getGradients
        send gradients to messageQueue
      else
        call BaseModel::getOutputs
        package up outputs in output struct
        get the outputs from the final layer (the loss layer hold thems)
        send outputs to messageQueue
        problem: now model is stuck in forward position
          move pointers from child back to parent to setup for another forward pass
          Maybe add a function for resetting for forward position to the layers?

  messageQueue (This needs worked out)
  2 queues, one for inputs and one for outputs
  MessageQueue<InputData> inputQueue
    x-data
    y-data

  MessageQueue<OutputData> outputQueue
    model outputs
    parameter gradients
  
  struct InputData{x-data, y-data(optional)}
  struct OutputData{outputs, vector<shared_ptr<floattype>>gradients}
  **(layers implement *getParams()* and *setParams()*)**
  save
    loop over layers
    extract parameters with getParams()
    write parameters to file in order of layers 
  load
    open file
    loop over layers
    ask layer if it needs params
    if needs then load line and pass to setParams()
    else go to next layer
  forward
    param: inputs which is the rvalue of a unique pointer to a 1d array
    loop over layers
    call forward on each layer except the last (loss is computed in BaseModel::backward)
    only the input layer requires an argument for forward, or another method could be used to move value to inputLayer
  backward
    call forward on the final layer, this either requires an argument or set an attribute for target
    call backward on each layer (including the loss layer)
  OutputData getGradients()
    this returns a struct with gradients
    loop over layers
    ask if they have params, hasParams
    call getGradients() on the layer to get the parameter gradients
    copy the shared pointer to vector in output struct
  
  OutputData getOutputs()
    this returns a struct with prediction/model outputs
    they should be waiting in the inputs of the final layer (loss layer)


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



Hyperparameters vs model parameters...
Some layers need hyperparams: n-units, input/output shapes, 

Also need to randomly initialize model weights.

vector<uint8_t> paramVector

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

class Input: public BaseLayer
  int inputShape;
  forward
    validate inputs shape (to handle incorrect input, should throw exception, also coordinate with the thread call to handle)
    returns pointer with move

class Dense: public BaseLayer
  constructor
    compute output shape 
      uses input shape from parent and units
    create array of size of output shape on the heap use shared pointer handle
    vector<shared_ptr<paramdatatype[]>>_params; owned by layer
    vector<shared_ptr<paramdatatype[]>>_grads; owned by layer
    create array for weights, add pointer to array to _params vector
    create array for weights gradients, add pointer to _grads vector
    create array for bias, add pointer to _params vector
    create array for bias gradients, add pointer to _grads vector
    hasParams = true;

  regularization controlled by hyperparam
  parameters, protected with mutex, wait to update weights until backward pass is done
  How does the train op get gradients?
  forward
    compute outputs and move pointer to child layer
  backward
    compute derivative of inputs wrt parameters, compute derivative of gradients wrt inputs

Figure out the outputs data for each layer TODO Working here!!!!!

  On each call to forward
  The layer creates memory for outputs on the stack using unique_ptrs 
  They compute the output then move it to their child layer

  When forward is called with targets

  Delete the below

  This can either be reused or disposed of each time, for now try the stack..?
  THe layer knows ahead of time the input and output shapes
  If initialized with memory for inputs/outputs, then the layer needs to own the data for reuse.

  Lets say then...
  each layer initializes and owns their outputs...
  How then does the final output go?
  Instead, they initialize and move out their outputs
  Then the following layer
  If its an output layer, it returns the outputs with move instead of moving to child
  So... to allow any layer to be an output, it needs to either be void or return?
  Ah, it needs a special output layer to be the child
  The output layer should easily convert to the message structs used


class Relu: public BaseLayer
  on forward: relu function
  on backward: if relu function of input >0 then 1 else 0 * gradients

class Softmax: public BaseLayer
  forward: safe softmax
  backward: ? need work here
  https://github.gatech.edu/cfarr31/DeepLearning7643/blob/master/assignment1/models/softmax_regression.py

class CrossEntropyLoss
  targets are passed directly to this layer
  no child layer
  forward: compute loss
  backward: move loss to parent layer

class Optimizer
  Use adam
  Initialize with default lr and include decay param
  This class has read/write access to the shared model parameters

  How does this one work? TODO work on this part
  regularization occurs already from the layers with params (only Dense)
  It can get gradients... but does it need anything else like the params for regularization?
  This takes in the gradients for the whole batch
  Computes the update
  Requests write access to the parameters
  updates the parameters
  releases the lock

function train (initializes model, trains, and stores final parameters)
  perform setup
  Initialize MyModel
  Determine number of threads to spin up, smallest of MAX_THREADS and BATCH_SIZE
  Use createModelThread to spin up each thread and store in a vector
  ask the model for trainable the parameters (these are shared among all threads)
  loop for n-epochs
    TODO Should I load all train data at once? How else to shuffle?
    loop for batch-size / total-examples
      loop over batch-size
        load example from train data into what object?
        send data to message queue
      wait until gradient-queue == batch-size
      wait until parameters are freed up (for extra safety)
      Loop over the parameter updates to compute the average update
        divide each update by batch-size then add to parameter
    
    loop over validation data
    should validation data persist?
    create vector of pointers to x and y data objects
    
    todo start here!!
    send data to queue without targets to get predictions


How will predictions be generated
Need to predict over a validation dataset
Use validation performance to stop training

only need the forward pass
threading would still be useful
can I reuse the threads from the gradient calc?

Declare 2 structs to use as template types for the queues

From train loop to message queue
struct InputData
  x-data
  y-data

TODO What is x and y data type?
x-data: one example is a vector of floats (or compressed type)
y-data: int target

From model thread to message queue
struct OutputData
  model outputs
  parameter gradients

Efficient computation resource: https://gist.github.com/nadavrot/5b35d44e8ba3dd718e595e40184d03f0
For efficiency threads can be used for multiple simultaneous examples to fill
up the batch. Then no need to parallelize the basic matrix function.
OMP or MPI packages for threading the matrix loops

main
  run options
    * train or load
    * predict and write to output file
    * evaluate
