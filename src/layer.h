#ifndef LAYER_H_
#define LAYER_H_

#include <memory>
#include <vector>
#include "matrix.h"

using namespace std;

// Need to declare a matrix class to handle (and abstract) variable types of data
// TODO After defining the matrix class, return here and replace type where needed

class BaseLayer {

public:
    BaseLayer();  // Default constructor
    BaseLayer(BaseLayer&);  // Copy constructor

    BaseLayer* _parentLayer;  // Pointer is not owned or managed by layers
    BaseLayer* _childLayer;  // Pointer is not owned or managed by layers
    void connectParent(BaseLayer&);  // Must be called in order across full model graph
    bool hasParams(){return false;} // Returns false unless overriden
    // Forward and backward must be called by the child layer
    // void forward();  rename this to moveInputs (or something)
    void moveOutputs(unique_ptr<Matrix>&& outputs);  // Move layer outputs to child inputs in forward pass
    // void backward(); rename this to moveGradients
    void moveGradients(unique_ptr<Matrix>&& gradients);  // Move gradients to parent
    // Must define these functions in the child classes
    virtual vector<shared_ptr<Matrix>>&& getParams() = 0;
    virtual void setParams() = 0;
    virtual void forward() = 0;  // Perform operation, moves outputs to child
    virtual void backward() = 0;  // Compute gradients wrt inputs, moves gradients to parent
    virtual vector<int>&& computeOutputShape() = 0; // Compute output shape for child layer to consume with setShape
    void setInputShape(vector<int>&& shape) {for(auto s: shape){inputShape.emplace_back(s);}};
    vector<int> getInputShape(){return inputShape;};

private:
    unique_ptr<Matrix> _inputs;  // These come from the parent except for InputLayer
    unique_ptr<Matrix> _gradients;  // These come from the child and are set by moveGradients (grad wrt inputs)
    vector<int> inputShape;
    // _inputs and _gradients are always the same size
    //   inputs
    //     they come from the parent layer
    //   outputs
    //     unique-ptr from parent to store as the childs inputs on the forward pass
};

// Each child layer must implement their own inputs depending on the size/shape needed

//   inputs
//     they come from the parent layer
//     stored on the heap
//     child layers store the pointer from forward pass to use for backward pass
//   outputs
//     created on the heap
//     create a unique pointer on the forward pass
//     moved to child layer on forward pass
//     these are never stored by the parent class
//   parameters
//     shared across all layers across threads, layers have read-only access

class InputLayer: public BaseLayer
{
public:
    InputLayer(int);  // Only allows for one dimension
    void setInputs(unique_ptr<Matrix> inputs);  // Input layer has a different interface
    vector<int>&& computeOutputShape();
    void forward();  // Validate input shape, pass inputs to child layer
    
private:
    InputLayer(){};  // No default constructor
    void backward();  // InputLayer has no backward pass
    unique_ptr<Matrix> _inputs;
};

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
//     call BaseLayer::moveOutputs to move pointer to child
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
//     call BaseLayer::moveOutputs to move pointer to child
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


#endif