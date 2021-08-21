#ifndef LAYER_H_
#define LAYER_H_

#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include "math.h"
#include "matrix.h"

using namespace std;

// Initializer for parameter layers
void glorotUniformInitializer(Matrix& matrix);


class BaseLayer {

public:

    // TODO Finish implementing rule of 5
    BaseLayer();  // Default constructor
    BaseLayer(BaseLayer&);  // Copy constructor: Copies parameters and reinitializes all other build vars
    BaseLayer(BaseLayer&&); // Move constructor: Moves to new location and nullifies original
    BaseLayer& operator=(BaseLayer&);  // Copy assignment operator: Copies parameters and reinitializes all other build vars
    BaseLayer& operator=(BaseLayer&&); // Move assignment operator: Moves to new location and nullifies original
    ~BaseLayer();

    // Public methods to be used by inheriting layers
    void connectParent(BaseLayer&);  // Must be called in order across full model graph
    // Forward and backward must be called by the child layer
    void _sendOutputs(shared_ptr<Matrix>);  // Move layer outputs to child inputs in forward pass
    void _sendGradients(shared_ptr<Matrix>);  // Move gradients to parent in backward pass
    // Setters and getters
    void _setInputShape(vector<int> shape) {for(auto d: shape){_inputShape.emplace_back(d);}};
    vector<int> getInputShape(){return _inputShape;};
    const Matrix& getInputs(); // {if(_inputs != nullptr) return (*_inputs);};  // Is this correct, pass by reference? Can't change them but can read them
    const Matrix& getGradients(); // {if(_gradients != nullptr) return (*_gradients);};
    // TODO IDEALLY: Would have some op like "secureParameters" on forward pass, released when the op is done...

    // Abstract methods
    virtual void getParams() = 0;
    virtual void setParams() = 0;
    virtual void forward() = 0;  // Perform operation, call to move outputs to child
    virtual void backward() = 0;  // Compute gradients wrt inputs, call to move gradients to parent
    virtual vector<int> computeOutputShape() = 0; // Compute output shape for child layer to consume with setInputShape
    virtual void build() = 0;  // To be ran after a new model or model copy is created for validating the model

    // Attributes
    vector<BaseLayer*> _parentLayers;  // Pointer is not owned or managed by layers
    vector<BaseLayer*> _childLayers;  // Pointer is not owned or managed by layers
    bool hasParams(){return false;} // Returns false unless overriden
    // These are shared to allow for multi-to-multi-connections
    shared_ptr<Matrix> _inputs;  // These come from the parent except for InputLayer
    shared_ptr<Matrix> _gradients;  // These come from the child and are set by moveGradients (grad wrt inputs)

private:
    // These must be shared pointers to allow 1-to-many connections between child and parent
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
    InputLayer(int, int);
    InputLayer(InputLayer&);  // Copy constructor: Copies parameters and reinitializes all other build vars
    InputLayer(InputLayer&&); // Move constructor: Moves to new location and nullifies original
    InputLayer& operator=(InputLayer&);  // Copy assignment operator: Copies parameters and reinitializes all other build vars
    InputLayer& operator=(InputLayer&&); // Move assignment operator: Moves to new location and nullifies original
    ~InputLayer();

    void setInputs(unique_ptr<Matrix>&& inputs);  // Input layer has a different interface
    vector<int> computeOutputShape();
    void forward();  // Validate input shape, pass inputs to child layer
    void build();  // Validate setup
private:
    // Uneeded abstract methods
    InputLayer(){};  // No default constructor
    void backward(){};  // InputLayer has no backward pass
    // This layer has no params
    void setParams(){};  
    void getParams(){};
};


class DenseLayer: public BaseLayer
{
public:
    DenseLayer(int units, float reg=0.00001);
    DenseLayer(DenseLayer&);
    DenseLayer& operator=(DenseLayer&);  // Copy assignment operator
    void operator=(DenseLayer&&) = delete;  // No moving allowed... maybe, or just use same logic as copy assignnment
    vector<int> computeOutputShape();
    void build();  // Validate setup, initialize parameters if not already present
    bool hasParams(){return true;}
    unique_ptr<vector<Matrix>>&& extractGradients();
    // TODO parameters, protected with mutex, wait to update weights until backward pass is done
    void updateParameters(unique_ptr<vector<Matrix>>);  // Additive, loops over vector and adds to parameters
    // getGradients (get an rvalue unique_ptr<vector<unique_ptr<Matrix>>> to gradients declared on heap)
    void forward();
    void backward();
    // TODO Figure these out
    void getParams();  // TODO Consider sending a reference? The datatype can't be right...
    void setParams();  // Pass a reference to the params
    // Public attributes
    int _units;
    float _reg;
    bool _built;  // Set to true after parameters are initialized (or loaded)
    bool _gradientsAvailable;  // Track when gradients are available to extract
private:
    DenseLayer();  // Must initialize with units, hiding default initializer
    // Must implement and manage _params and _paramGrads
    unique_ptr<vector<Matrix>> _parameterGradients;
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
    void build();  // Validate setup

    // Attributes
    bool built = false;
};


class Softmax: public BaseLayer
{
public:
    void forward();
    void backward();
    vector<int> computeOutputShape();
    void build();  // Validate setup
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
    void build();  // Validate setup
    // targets are passed directly to this layer
    // no child layer
private:
    vector<int> computeOutputShape();  // Output shape is not needed since no child layer
};

#endif