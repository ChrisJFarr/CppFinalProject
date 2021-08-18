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
    // clear relationships after copying, the copies must be reconnected by the model class
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

void BaseLayer::moveGradients(unique_ptr<Matrix>&& gradients)
{
    // called on the backward pass by the child layer
    // accepts a pointer rvalue argument
    // moves the unique pointer to gradients-wrt-inputs to the parent layer
}


InputLayer::InputLayer(int inputSize)
{
    // Constructor
}

void InputLayer::setInputs(unique_ptr<Matrix> inputs)
{
    // Input size must be passed along when connecting the graph to allow child layers to create
    // parameters.
}

vector<int> InputLayer::computeOutputShape()
{
    // Implement this specific to the layer
    // Move and return the resulting vector
    return vector<int>();
}

void InputLayer::forward()
{
    // Validate input shape, pass inputs to child layer
}



// class Dense: public BaseLayer

DenseLayer::DenseLayer(int units, float reg)
{
    // constructor
    // set units
    // Intialize _gradientsAvailable
    // regularization controlled by hyperparam
}

vector<int> DenseLayer::computeOutputShape()
{
    // compute output shape 
    // uses input shape from parent and units
    return vector<int>();
}

void DenseLayer::build()
{
    // create array of size of output shape on the heap use shared pointer handle
    // vector<shared_ptr<paramdatatype[]>>_params; owned by layer
    // create array for weights, add pointer to array to _params vector
    // create array for bias, add pointer to _params vector
    // randomly initialize bias and weights
    built = true;
}

unique_ptr<vector<Matrix>>&& DenseLayer::extractGradients()
{
    // Because hasParams is true, this must be implemented, called once per backward call
    // Return the pointer to _gradients
    if(_gradientsAvailable)
    {
        return move(_parameterGradients);
    } else {
        throw logic_error("Trying to extract gradients when none are available, must first call backward().");
    }
}

void DenseLayer::updateParameters(unique_ptr<vector<Matrix>> updates)
{
    // Validate contents, must have identical shape to _parameterVector
    // Additive, loops over vector and adds updates to parameters
    // This shouldn't be called on the copies, only needs to be called once for all copies
}


void DenseLayer::forward()
{
    // allocate memory for outputs using unique ptr
    // compute outputs and set value in memory
    // call BaseLayer::moveOutputs to move pointer to child

}

void DenseLayer::backward()
{
    // allocate memory for and compute derivative of gradients wrt parameters 
    //     include regularization
    //     use a unique ptr, one for grads-wrt-weights and another for grads-wrt-bias
    //     move both to private _grads in order of [weights, bias]
    // allocate memory for gradients wrt inputs on stack using unique ptr
    //     array is same size as inputs
    // compute derivative of gradients wrt inputs
    // call BaseLayer::backward to move gradients-wrt-inputs pointer to the parent
}


void ReluLayer::forward()
{
    // allocate memory for outputs (same size as inputs) on heap with unique_ptr
    // compute relu function assign to memory
    // call BaseLayer::moveOutputs to move pointer to child
}

void ReluLayer::backward()
{
    // allocate memory for gradients-wrt-inputs
    // compute relu derivative
    //     if relu function of input >0 then 1 else 0 * gradients
    // multipy by gradients from child to get gradients-wrt-inputs
    // call BaseLayer::backward to move pointer to parent
}
vector<int> ReluLayer::computeOutputShape()
{
    return vector<int>();
}

void Softmax::forward()
{
    // safe softmax
}

void Softmax::backward()
{
    // ? need work here
    // https://github.gatech.edu/cfarr31/DeepLearning7643/blob/master/assignment1/models/softmax_regression.py
}
vector<int> Softmax::computeOutputShape()
{
    return vector<int>();
}


CrossEntropyLoss::CrossEntropyLoss()
{

}

CrossEntropyLoss::CrossEntropyLoss(CrossEntropyLoss& other)
{
    
}
void CrossEntropyLoss::forward(unique_ptr<Matrix> targets)
{
    // compute loss using inputs from parent and targets
    // don't call BaseModel::forward, store the loss here
}
void CrossEntropyLoss::backward()
{
    // allocate memory on stack with unique-ptr the same shape as the inputs to store the gradients
    // compute the gradients of the loss wtr the inputs
    // call BaseLayer::backward to move loss to parent layer
}
vector<int> CrossEntropyLoss::computeOutputShape()
{
    return vector<int>();
}
