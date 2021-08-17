#include "layer.h"
#include <iostream>


BaseLayer::BaseLayer()
{
    std::cout << "BaseLayer::constructor" << std::endl;
    _inputs = nullptr;
    // Constructor
}

BaseLayer::BaseLayer(BaseLayer& other)
{
    std::cout << "BaseLayer::copy-constructor" << std::endl;
    // This is called when copying the modelVector and should be done for threads only
    // No copying of the inputs/outputs, just need to declare a unique ptr without allocation for the inputs?
    // Copy the shared pointers of parameters to the new object
    // layer parameters are owned by the layer, copies of the parameters and grads are shared pointer
}

void BaseLayer::connectParent(BaseLayer& parent)
{
    // connectParent (the child is called and passed the parent)
    // parent.childLayer = this
    // this.parentLayer = parent
}

void BaseLayer::moveOutputs(unique_ptr<Matrix>&& outputs)
{
    // called on the forward pass by the parent layer
    // accepts a pointer rvalue argument
    // moves the unique pointer to outputs to the child layer
}

void moveGradients(unique_ptr<Matrix>&& gradients)
{
    // called on the backward pass by the child layer
    // accepts a pointer rvalue argument
    // moves the unique pointer to gradients-wrt-inputs to the parent layer
}


//   parameters
//     shared across all layers across threads, layers have read-only access 
//          (or at least they never alter the params except when loading)
//   hasParams returns boolean defaults to false unless overriden
//   getParams() abstract method
//     throw error if !hasParams
//     this needs to coordinate with the training op and save
//     return vector of pointers to the params vector<shared_ptr<paramdatatype[]>>
//   setParams() abstract method
//     throw error if !hasParams
//     this needs to coordinate with BaseModel::load


// class Input: public BaseLayer
//   int inputShape;
//   forward()
//     validate inputs shape (to handle incorrect input, should throw exception, also coordinate with the thread call to handle)
//     call BaseLayer::moveOutputs to move pointer to child
InputLayer::InputLayer(int inputSize)
{
    // Constructor
}

void InputLayer::setInputs(unique_ptr<Matrix> inputs)
{
    // Input size must be passed along when connecting the graph to allow child layers to create
    // parameters.
}

void InputLayer::forward()
{
    // Validate input shape, pass inputs to child layer
}

// class Dense: public BaseLayer
//   constructor (this is done once in the BaseModel build method)
//     compute output shape 
//       uses input shape from parent and units
//     create array of size of output shape on the heap use shared pointer handle
//     vector<shared_ptr<paramdatatype[]>>_params; owned by layer
//       create array for weights, add pointer to array to _params vector
//       create array for bias, add pointer to _params vector
//       randomly initialize bias and weights
//     vector<shared_ptr<paramdatatype[]>>_grads
//     hasParams = true;
//   regularization controlled by hyperparam
//   parameters, protected with mutex, wait to update weights until backward pass is done
//   forward()
//     allocate memory for outputs using unique ptr
//     compute outputs and set value in memory
//     call BaseLayer::forward to move pointer to child
//   backward()
//     allocate memory for and compute derivative of gradients wrt parameters 
//       include regularization
//       use a unique ptr, one for grads-wrt-weights and another for grads-wrt-bias
//       move both to private _grads in order of [weights, bias]
//     allocate memory for gradients wrt inputs on stack using unique ptr
//       array is same size as inputs
//     compute derivative of gradients wrt inputs
//     call BaseLayer::backward to move gradients-wrt-inputs pointer to the parent

// class Relu: public BaseLayer
//   forward()
//     allocate memory for outputs (same size as inputs) on stack with unique_ptr
//     compute relu function assign to memory
//     call BaseLayer::forward to move pointer to child
//   backward()
//     allocate memory for gradients-wrt-inputs
//     compute relu derivative
//       if relu function of input >0 then 1 else 0 * gradients
//     multipy by gradients from child to get gradients-wrt-inputs
//     call BaseLayer::backward to move pointer to parent

// class Softmax: public BaseLayer
//   forward()
//     safe softmax
//   backward()
//     ? need work here
//     https://github.gatech.edu/cfarr31/DeepLearning7643/blob/master/assignment1/models/softmax_regression.py
