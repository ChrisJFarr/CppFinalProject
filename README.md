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


class BaseModel
  Manages multi-threading and generic components
  forward
  backward
  save
  load
class Model: public BaseModel
  Problem-specific implementation
  forward
  backward

class BaseLayer (needed?)
class Dense: public BaseLayer
  regularize (implement here or optimizer?)
  parameters, protected with mutex, wait to update weights until backward pass is done
class Relu: public BaseLayer
class Sigmoid: public BaseLayer
class Dropout: public BaseLayer
  During training, randomly drop connections
  During inference, compute average somehow... TODO I need to figure this out find a paper

class Optimizer
  Use adam
  Initialize with default lr and include decay param

Efficient computation resource: https://gist.github.com/nadavrot/5b35d44e8ba3dd718e595e40184d03f0
For efficiency threads can be used for multiple simultaneous examples to fill
up the batch. Then no need to parallelize the basic matrix function.
OMP or MPI packages for threading the matrix loops

main
  run options
    * train or load
    * predict and write to output file
    * evaluate
