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

    // TODO Finish implementing rule of 5
    BaseLayer();  // Default constructor
    BaseLayer(BaseLayer&);  // Copy constructor
    ~BaseLayer();

    // Public methods
    void connectParent(BaseLayer&);  // Must be called in order across full model graph
    // Forward and backward must be called by the child layer
    void moveOutputs(unique_ptr<Matrix>&& outputs);  // Move layer outputs to child inputs in forward pass
    void moveGradients(unique_ptr<Matrix>&& gradients);  // Move gradients to parent in backward pass
    // Setters and getters
    void setInputShape(vector<int> shape) {for(auto d: shape){_inputShape.emplace_back(d);}};
    vector<int> getInputShape(){return _inputShape;};
    const Matrix& getInputs(); // {if(_inputs != nullptr) return (*_inputs);};  // Is this correct, pass by reference? Can't change them but can read them
    const Matrix& getGradients(); // {if(_gradients != nullptr) return (*_gradients);};
    // TODO IDEALLY: Would have some op like "secureParameters" on forward pass, released when the op is done...

    // Abstract methods
    virtual vector<shared_ptr<Matrix>> getParams() = 0;
    virtual void setParams() = 0;
    virtual void forward() = 0;  // Perform operation, call to move outputs to child
    virtual void backward() = 0;  // Compute gradients wrt inputs, call to move gradients to parent
    virtual vector<int> computeOutputShape() = 0; // Compute output shape for child layer to consume with setInputShape

    // Attributes
    BaseLayer* _parentLayer;  // Pointer is not owned or managed by layers
    BaseLayer* _childLayer;  // Pointer is not owned or managed by layers
    bool hasParams(){return false;} // Returns false unless overriden

private:
    unique_ptr<Matrix> _inputs;  // These come from the parent except for InputLayer
    unique_ptr<Matrix> _gradients;  // These come from the child and are set by moveGradients (grad wrt inputs)
    vector<int> _inputShape;
    // _inputs and _gradients are always the same size
    //   inputs
    //     they come from the parent layer
    //   outputs
    //     unique-ptr from parent to store as the childs inputs on the forward pass
};


class InputLayer: public BaseLayer
{
public:
    InputLayer(int);  // Only allows for one dimension
    void setInputs(unique_ptr<Matrix> inputs);  // Input layer has a different interface
    vector<int> computeOutputShape();
    void forward();  // Validate input shape, pass inputs to child layer
    
private:
    InputLayer(){};  // No default constructor
    void backward(){};  // InputLayer has no backward pass
    unique_ptr<Matrix> _inputs;
};


class DenseLayer: public BaseLayer
{
public:
    DenseLayer(int units, float reg=0.0);
    DenseLayer(DenseLayer&);
    vector<int> computeOutputShape();
    void build();  // compute output shape, initialize
    bool built = false;
    bool hasParams(){return true;}
    unique_ptr<vector<Matrix>>&& extractGradients();
    // TODO parameters, protected with mutex, wait to update weights until backward pass is done
    void updateParameters(unique_ptr<vector<Matrix>>);  // Additive, loops over vector and adds to parameters
    // getGradients (get an rvalue unique_ptr<vector<unique_ptr<Matrix>>> to gradients declared on heap)
    void forward();
    void backward();
private:
    DenseLayer();  // Must initialize with units
    int _units;
    float _reg;
    // Must implement and manage _params and _paramGrads
    unique_ptr<vector<Matrix>> _parameterGradients;
    bool _gradientsAvailable;  // Track when gradients are available to extract
    // The _params are shared across all copies, this is manged entirely by the class
    shared_ptr<vector<Matrix>> _parameterVector;  // Vector of parameters of the layer
};

class ReluLayer: public BaseLayer
{
public:

    // Contructors
    ReluLayer();
    ReluLayer(ReluLayer&);

    // Methods
    void forward();
    void backward();
    vector<int> computeOutputShape();

    // Attributes
    bool built = false;
};

class Softmax: public BaseLayer
{
public:
    void forward();
    void backward();
    vector<int> computeOutputShape();
};


class CrossEntropyLoss : public BaseLayer
{
//   targets are passed directly to this layer
//   no child layer
public:
    CrossEntropyLoss();
    CrossEntropyLoss(CrossEntropyLoss&);
    void forward(unique_ptr<Matrix>);
    void backward();
    // targets are passed directly to this layer
    // no child layer
private:
    vector<int> computeOutputShape();  // Output shape is not needed since no child layer
};

#endif