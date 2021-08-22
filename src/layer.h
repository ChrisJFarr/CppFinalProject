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

    BaseLayer();  // Default constructor
    BaseLayer(BaseLayer&);  // Copy constructor: Copies parameters and reinitializes all other build vars
    BaseLayer(BaseLayer&&); // Move constructor: Moves to new location and nullifies original
    BaseLayer& operator=(BaseLayer&);  // Copy assignment operator: Copies parameters and reinitializes all other build vars
    BaseLayer& operator=(BaseLayer&&); // Move assignment operator: Moves to new location and nullifies original
    ~BaseLayer();

    // Public methods to be used by inheriting layers
    void connectParent(BaseLayer&);  // Must be called in order across full model graph
    void operator()(BaseLayer& other){return BaseLayer::connectParent(other);};  // Proxy for connectParent
    // These must be called by the child layer during forward and backward as needed
    void _sendOutputs(shared_ptr<Matrix>);  // Move layer outputs to child inputs in forward pass
    void _sendGradients(shared_ptr<Matrix>);  // Move gradients to parent in backward pass
    // Setters and getters
    void _setInputShape(vector<int> shape) {for(auto d: shape){_inputShape.emplace_back(d);}};
    vector<int> getInputShape(){return _inputShape;};
    // TODO IDEALLY: Would have some op like "secureParameters" on forward pass, released when the op is done...

    // Virtual methods
    virtual void build() = 0;  // To be ran after a new model or model copy is created for validating the model
    virtual void setInputs(){};  // Only used for the input layer
    virtual void forward()=0;  // Perform operation, call to move outputs to child
    virtual void backward()=0;  // Compute gradients wrt inputs, call to move gradients to parent
    virtual vector<int> getOutputShape(){return vector<int>();}; // Compute output shape for child layer to consume with setInputShape
    virtual bool hasParams(){return false;} // Returns false unless overriden
    virtual void getParams(){};
    virtual void setParams(){};
    virtual void extractGradients(unique_ptr<vector<Matrix>>&){};  // Pass a unique_ptr to a Matrix vector by reference and set there

    // Attributes
    vector<BaseLayer*> _parentLayers;  // Pointer is not owned or managed by layers
    vector<BaseLayer*> _childLayers;  // Pointer is not owned or managed by layers
    
    // These are shared to allow for multi-to-multi-connections
    // These must be shared pointers to allow 1-to-many connections between child and parent
    shared_ptr<Matrix> _inputs;  // These come from the parent except for InputLayer
    shared_ptr<Matrix> _gradients;  // These come from the child and are set by moveGradients (grad wrt inputs)

    // Store attributes
    vector<int> _inputShape;
    bool _built;
private:
};


class InputLayer: public BaseLayer
{
public:
    InputLayer() = delete;  // No default constructor
    InputLayer(int, int);
    InputLayer(InputLayer&);  // Copy constructor: Copies parameters and reinitializes all other build vars
    InputLayer(InputLayer&&); // Move constructor: Moves to new location and nullifies original
    InputLayer& operator=(InputLayer&);  // Copy assignment operator: Copies parameters and reinitializes all other build vars
    InputLayer& operator=(InputLayer&&); // Move assignment operator: Moves to new location and nullifies original
    ~InputLayer();

    void build();  // To be ran after a new model or model copy is created for validating the model
    void setInputs(unique_ptr<Matrix>&& inputs);  // Input layer has a different interface
    void forward();  // Perform operation, call to move outputs to child
    void backward();  // Compute gradients wrt inputs, call to move gradients to parent
    vector<int> getOutputShape(); // Compute output shape for child layer to consume with setInputShape
private:
};


class DenseLayer: public BaseLayer
{
public:
    DenseLayer(int units, float reg=0.00001);
    DenseLayer(DenseLayer&);
    DenseLayer(DenseLayer&&); // Move constructor: Moves to new location and nullifies original
    DenseLayer& operator=(DenseLayer&);  // Copy assignment operator
    DenseLayer& operator=(DenseLayer&&);  // No moving allowed... maybe, or just use same logic as copy assignnment?
    ~DenseLayer();
    
    vector<int> getOutputShape();
    void build();  // Validate setup, initialize parameters if not already present
    bool hasParams(){return true;}
    void extractGradients(unique_ptr<vector<Matrix>>& gradients);
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
    ReluLayer();  // Default constructor
    ReluLayer(ReluLayer&);  // Copy constructor: Copies parameters and reinitializes all other build vars
    ReluLayer(ReluLayer&&); // Move constructor: Moves to new location and nullifies original
    ReluLayer& operator=(ReluLayer&);  // Copy assignment operator: Copies parameters and reinitializes all other build vars
    ReluLayer& operator=(ReluLayer&&); // Move assignment operator: Moves to new location and nullifies original
    ~ReluLayer();

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