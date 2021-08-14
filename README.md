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
Modify this to center around 255/2? Perhaps with relu, negatives are never needed anyways
What about the outputs? Sigmoid... are negatives needed?

class BaseModel
  This class manages multi-threading and generic components
  copy constructor (copies are used in the thread functions)
    loop over and copy the modelVector to the new object
  implement the rule of five

  createModelThread
    This function copies the modelGraph
    This function creates a thread, copies the modelVector, then loops forever
    loop over modelVector 
    copy each layer to local modelVector
    reconnect the local graph
    loop forever 
      wait to get data from the messageQueue
      perform forward pass
      return the data through the messageQueue
  messageQueue
    each copy of the base model maintains its own message queue
  forward
    get data somehow
    wait until modelThread is not busy
    send data through the messageQueue
  backward
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
  forward
    this can be called directly, or on a copy
    takes in inputs
    loop over 
  backward

What is the process of spinning up threads during training: TODO Pick up here!!!!!!!!!!!!!!!
Initialize MyModel
Determine number of threads to spin up, smallest of MAX_THREADS and BATCH_SIZE
Use createModelThread to spin up each thread and store in a vector

What is the process of looping through the data while training?



<!-- loop forever 
wait to get data
perform forward pass -->

Loop for n-epochs

model build instructions to allow for threaded training when batch-size>1
Vector of connected layers
vector<BaseLayer> modelVector
Hyperparameters
vector<vector<uint8_t>> hyperParamsVector

vector<vector<>>

Hyperparameters vs model parameters...
Some layers need hyperparams: n-units, input/output shapes, 

Also need to randomly initialize model weights.

vector<uint8_t> paramVector

class BaseLayer
  copy constructor (this is called when copying the modelVector)
    Deep copy the inputs and outputs... where are instructions for input/output shapes?
    Copy the shared pointers of parameters to the new object
  parentLayer
  childLayer
  connectParent
    other.childLayer = this
    this.parentLayer = other
  std::unique_prt<2d array> inputs
  how is data stored and managed within a layer?
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



  vector<uint8_t> paramVector
  forward
    store inputs, move pointer from parent class
  backward

class Input: public BaseLayer
  input shape
  on forward pass, takes inputs pointer and moves it to child layer

class Dense: public BaseLayer
  compute output shape 
    uses input shape from parent
  regularize
  parameters, protected with mutex, wait to update weights until backward pass is done
  weights, bias
  on forward pass, compute output and store the inputs
  on backward pass, compute derivative of inputs wrt parameters, compute derivative of gradients wrt inputs 

class Relu: public BaseLayer
  on forward: relu function
  on backward: if relu function of input >0 then 1 else 0 * gradients
class Sigmoid: public BaseLayer
  forward: sigmoid function in inputs
  backward: sigmoid * (1-sigmoid)
class Dropout: public BaseLayer (nice to have)
  Perhaps this is optional if I get extra time.
  During training, randomly drop connections
  During inference, compute average somehow... TODO I need to figure this out find a paper
class Softmax: public BaseLayer
  forward: safe softmax
  backward: ? need work here
  https://github.gatech.edu/cfarr31/DeepLearning7643/blob/master/assignment1/models/softmax_regression.py

class Optimizer
  Use adam
  Initialize with default lr and include decay param
  This class has read/write access to the shared model parameters

Efficient computation resource: https://gist.github.com/nadavrot/5b35d44e8ba3dd718e595e40184d03f0
For efficiency threads can be used for multiple simultaneous examples to fill
up the batch. Then no need to parallelize the basic matrix function.
OMP or MPI packages for threading the matrix loops

main
  run options
    * train or load
    * predict and write to output file
    * evaluate
