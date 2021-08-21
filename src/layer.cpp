#include "layer.h"

// Initializer for parameter layers
void glorotUniformInitializer(Matrix& matrix)
{
    // http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    MyDType fan_in = static_cast<MyDType>(matrix.rows());
    MyDType fan_out = static_cast<MyDType>(matrix.cols());
    // limit = sqrt(6/(fan_in+fan_out))
    MyDType limit = sqrt(6./(fan_in+fan_out));
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-limit, limit);
    for(int j=0;j<matrix.rows();j++)
    {
        for(int k=0;k<matrix.cols();k++)
        {
            matrix[j][k] = dis(gen);
        }
    }
    return;
}


BaseLayer::BaseLayer()
{
    if(DEBUG) std::cout << "BaseLayer::constructor" << std::endl;
    // _inputs = nullptr; Not needed since these are smart pointers
    // _gradients = nullptr;
    // Constructor
    // Initialize the child and parent pointers (these go unused if not needed by inheriting layer)
    _parentLayers = vector<BaseLayer*>();
    _childLayers = vector<BaseLayer*>();
    _inputs = nullptr;  // Already nullptr but being explicit
    _gradients = nullptr; 
}

BaseLayer::BaseLayer(BaseLayer& other)
{
    if(DEBUG) std::cout << "BaseLayer::copy-constructor" << std::endl;
    // Copy constructor: Copies parameters and reinitializes all other build vars
    // TODO Implement copy constructor
    // Copy layer shape
    _inputShape = other._inputShape;
    // Reinitialize these
    _parentLayers = vector<BaseLayer*>();
    _childLayers = vector<BaseLayer*>();
    _inputs = nullptr;  // Already nullptr but being explicit
    _gradients = nullptr; 

}

BaseLayer::BaseLayer(BaseLayer&& other)
{
    if(DEBUG) std::cout << "BaseLayer::move-constructor" << std::endl;
    // Move constructor: Moves to new location and nullifies original
    // Move all vars connection and build variables
    _inputShape = move(other._inputShape);
    _parentLayers = move(other._parentLayers);
    _childLayers = move(other._childLayers);
    // Clear the forward/backward results vars
    _inputs = nullptr;  // Already nullptr but being explicit
    _gradients = nullptr;
    // Nullify the originals
    other._inputs = nullptr;
    other._gradients = nullptr;
}

BaseLayer& BaseLayer::operator=(BaseLayer&)
{
    if(DEBUG) std::cout << "BaseLayer::copy-assignment-operator" << std::endl;
    // Copy assignment operator: Copies parameters and reinitializes all other build vars
    // TODO Implement copy assignment operator 
    return *this;
}

BaseLayer& BaseLayer::operator=(BaseLayer&&)
{
    if(DEBUG) std::cout << "BaseLayer::move-assignment-operator" << std::endl;
    // Move assignment operator: Moves to new location and nullifies original
    // TODO Implement move assignment operator
    return *this;
}

BaseLayer::~BaseLayer()
{
    if(DEBUG) std::cout << "BaseLayer::destructor" << std::endl;
    // TODO?
    // Drop the connections
    _parentLayers.clear();
    _childLayers.clear();
}

void BaseLayer::connectParent(BaseLayer& parent)
{
    // connectParent (the child is called and passed to the parent)
    // parent.childLayer = this
    // Make a pointer and move it to childLayer with emplace back, repeat for parent
    BaseLayer* childLayer = this;  // Could this create that pointer lock loop thing on destruction?
    parent._childLayers.emplace_back(childLayer);
    // BaseLayer* parentLayer = &parent;
    // _parentLayers.emplace_back(parentLayer);
}

void BaseLayer::_sendOutputs(shared_ptr<Matrix> outputs)
{
    // called on the forward pass by the parent layer
    // accepts a pointer rvalue argument
    // moves the unique pointer to outputs to the child layer
    for(int i=0;i<_childLayers.size();i++)
    {
        // Copy the shared pointer to each child
        _childLayers[i]->_inputs = outputs;
    }
}

void BaseLayer::_sendGradients(shared_ptr<Matrix> gradients)
{
    // called on the backward pass by the child layer
    // accepts a pointer rvalue argument
    // moves the unique pointer to gradients-wrt-inputs to the parent layer
    for(int i=0;i<_parentLayers.size();i++)
    {
        // Copy the shared pointer to each child
        _parentLayers[i]->_gradients = gradients;
    }
}


InputLayer::InputLayer(int rows, int cols)
{
    if(DEBUG) std::cout << "InputLayer::parameterized-constructor" << std::endl;
    // Constructor
    vector<int> inputShape({rows, cols});
    BaseLayer::_setInputShape(inputShape);
}

InputLayer::InputLayer(InputLayer& other)
{
    if(DEBUG) std::cout << "InputLayer::copy-constructor" << std::endl;
    // Copy constructor: Copies parameters and reinitializes all other build vars
    // TODO Implement copy constructor

}

InputLayer::InputLayer(InputLayer&& other)
{
    if(DEBUG) std::cout << "InputLayer::move-constructor" << std::endl;
    // Move constructor: Moves to new location and nullifies original
    // TODO Implement move constructor
}

InputLayer& InputLayer::operator=(InputLayer& other)
{
    if(DEBUG) std::cout << "InputLayer::copy-assignment-operator" << std::endl;
    // Copy assignment operator: Copies parameters and reinitializes all other build vars
    // TODO Implement copy assignment operator
    return *this;
}

InputLayer& InputLayer::operator=(InputLayer&& other)
{
    if(DEBUG) std::cout << "InputLayer::move-assignment-operator" << std::endl;
    // Move assignment operator: Moves to new location and nullifies original
    // TODO Implement move assignment operator
    return *this;
}

InputLayer::~InputLayer()
{
    // TODO Implement destructor
}

void InputLayer::setInputs(unique_ptr<Matrix>&& inputs)
{
    // Input size must be passed along when connecting the graph to allow child layers to create
    // parameters.
    // Validate inputs shape (if passing all local tests, the only test needed should be here for new inputs.)
    bool dim1Test, dim2Test;
    dim1Test = (*inputs).rows() == getInputShape()[0];
    dim2Test = (*inputs).cols() == getInputShape()[1];
    if(!(dim1Test && dim2Test)) throw logic_error(
        "Passing wrong shape inputs to InputLayer.");
    _inputs = move(inputs);  // Move from a unique ptr to a shared one in the input layer only
}

vector<int> InputLayer::computeOutputShape()
{
    // Implement this specific to the layer
    // Move and return the resulting vector
    // This layer has the same input and output shape
    vector<int> outputs = getInputShape();
    // Do I need to move this out?
    return outputs;
}

void InputLayer::forward()
{
    // First validate that the layer is connected to a child layer
    if(_childLayers.size()==0) throw logic_error(
        "Calling forward on a disconnected graph. \n Error in InputLayer::forward()");
    // Pass inputs to child layers as outputs
    BaseLayer::_sendOutputs(_inputs);
}

void InputLayer::build()
{
    // TODO Validate connnections exist
}


DenseLayer::DenseLayer(int units, float reg)
{
    // constructor
    // set attributes
    _units = units;
    _reg = reg;  // regularization controlled by hyperparam
    _gradientsAvailable = false;
    _built = false;
}

DenseLayer::DenseLayer(DenseLayer& other)
{
    // Implement copy constructor
    _units = other._units;
    _reg = other._reg;
    _built = false;  // Must be rebuilt after copying for validation
    _gradientsAvailable = false;  // New copy doesn't start with gradients
    // Copy the shared pointer to the parameter vector as is
    _parameterVector = other._parameterVector;
    // parameter gradients remains uninitialized here
}

vector<int> DenseLayer::computeOutputShape()
{
    // compute output shape 
    // uses input shape from parent and units
    // the operations are... matrix-multiply xw + b
    // (1x3)*(3*10) for size 10
    // expected output shape (1, 10)
    int inputRows = getInputShape()[0];
    int unitsOut = _units;
    return vector<int>({inputRows, unitsOut});
}

void DenseLayer::build()
{
    // TODO HOw can I make sure this is called each time a thread is created
    // This must be ran once per initialization or copy (parameters are only created once)
    // Validate that only one parent connection exists
    if(_parentLayers.size() != 1) throw invalid_argument("DenseLayer must have exactly 1 parent layer");
    // Validate that input shape rows == 1, can only deal with 1xn-features shape (one example and flattened)
    if(getInputShape()[0] != 1) throw invalid_argument("DenseLayer inputs must be a flattened single example");
    // Check if layer has previously been initialized, only create parameters upon first initialization
    if(_parameterVector->size()==0)
    {
        // Initialize the _parameterVector on the heap
        _parameterVector = make_shared<vector<Matrix>>();
        // Weights must be position 0, and bias position 1
        // Build the weights matrix and initialize weights
        // weights-shape (input-shape[1], _units)
        int inputFeatures = getInputShape()[1];
        Matrix weights(inputFeatures, _units);
        // Initialize with glorot uniform
        glorotUniformInitializer(weights);
        // Move weights to the shared parameter vector
        _parameterVector->emplace_back(move(weights));
        // Build the bias and explicity set to 0's
        // bias-shape (1, _units)
        Matrix bias(1, _units);
        for(MyDType& b: bias[0]) b = 0.0;
        // Move bias to the shared parameter vector
        _parameterVector->emplace_back(move(bias));        
    }
    _built = true;  // After complete, set _built to true
}

void DenseLayer::forward()
{
    // allocate memory for outputs using unique ptr, get outputshape
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

void DenseLayer::getParams()
{
    // TODO Figure this out...
}

void DenseLayer::setParams()
{
    // TODO Figure this out
}


void ReluLayer::build()
{

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


void Softmax::build()
{
    // TODO
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

void CrossEntropyLoss::build()
{
    // TODO
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
