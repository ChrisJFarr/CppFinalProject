#include "model.h"



BaseModel::BaseModel(BaseLayer* inputLayer, BaseLayer* lossLayer)
{
    if(DEBUG) cout << "BaseModel::constructor" << endl;
    // Move the pointers to the base model
    _inputLayer = inputLayer;
    _lossLayer = lossLayer;
};

BaseModel::BaseModel(BaseModel& other)
{
    if(DEBUG) cout << "BaseModel::copy-constructor" << endl;
    // copy constructor (copies are used in the thread functions)
    // loop over and copy the modelVector to the new object
};

BaseModel::~BaseModel()
{
    if(DEBUG) cout << "BaseModel::destructor" << endl;
    // Loop through and join the threads, need to stop them somehow...?
    // set up thread barrier before this object is destroyed
    // std::for_each(threads.begin(), threads.end(), [](std::thread &t) {
    //     t.join();
    // });
};
// TODO implement the rule of five

void BaseModel::build()
{
    // Start at the inputThread (validate that isn't a nullptr)
    if(_inputLayer==nullptr) throw logic_error("Attempting to build without an InputLayer");
    // Validate that every path leads to a loss layer
    // DFS for loss layers
    vector<BaseLayer*> queue;
    vector<BaseLayer*> explored;
    BaseLayer* startingNode = &(*_inputLayer);
    queue.emplace_back(startingNode);  // Start at the input layer
    while(true)
    {
        // If queue empty, break
        if(queue.size()==0) break;
        // Check if last node is a loss layer, continue if true
        if(queue.back()->isLossLayer())
        {
            queue.pop_back();
            continue;
        }
        if(DEBUG) cout << "checking " << queue.back()->getLayerType() << endl;
        // Check if last node has child
        //  if false, throw logic_error, last node doesn't have a child and is not a loss layer
        if(queue.back()->_childLayers.size()==0) throw logic_error("Disconnected graph, end layer is not a loss layer");
        //  if true, add all children to back of queue
        // Move last node from queue to explored
        BaseLayer* currentNode = queue.back();
        // Build the layer
        currentNode->build();
        // Proceed with search
        explored.emplace_back(queue.back());
        queue.pop_back();
        // Add new layers to queue
        for(BaseLayer* l: currentNode->_childLayers) 
        {
            // Skip over layers that have already been explored
            for(BaseLayer* e: explored) if(&(*e)==&(*l)) continue;
            BaseLayer* newPointer = &(*l);
            queue.emplace_back(newPointer);
        }
    }
    built = true;
}

void BaseModel::createModelThread()
{
// TODO Add all new threads to the _threads vector
//   createModelThread (i think this function is passed to a thread)
//     This function copies the modelGraph to a local copy
//     This function creates a thread, copies the modelVector, then loops forever (how can I reduce cost of looping)
//     loop over modelVector 
//     copy each layer to local modelVector
//     reconnect the local graph (loop over vector and call connectParent from the child, passing the parent)
//     add a method in BaseLayer to be called here that verifies everything is setup properly (.build()?)
//     loop forever 
//       wait to get data from the messageQueue
//       Parse the InputData:
//         get xData
//         get targets if exists (send data to queue without targets to just get predictions)
//         get evaluateTrain bool
//         get evaluateLayerGrads bool
//       can the threads claim the parameters while performing forward and backward passes for extra safety?
//         such as a shared control object that doesn't lock out the other threads
//       perform forward pass, call BaseModel::forward(unique_ptr<paramdatatype[]> inputs)
//         This should move the inputs
//       Declare and intialize OutputData on the stack
//       if targets are present in data object
//         perform backward pass, call BaseModel::backward
//         call BaseModel::getGradients(OutputData&)
//         call BaseModel::getOutputs(OutputData&)
//         move OutputData to messageQueue
//       else
//         call BaseModel::getOutputs(OutputData&)
//         package up outputs in output struct
//         get the outputs from the final layer (the loss layer hold thems)
//         move OutputData to messageQueue
//         problem: now model is stuck in forward position
//           move pointers from child back to parent to setup for another forward pass
//           Maybe add a function for resetting for forward position to the layers?
};


void BaseModel::save()
{
    // loop over layers
    // extract parameters with BaseLayer::getParams()
    // write parameters to file in order of layers 
    // TODO Need a file to save to
}

void BaseModel::load()
{
    // load
    // open file
    // loop over layers
    // ask layer if it needs params
    // if needs then load line and pass to BaseLayer::setParams()
    // else go to next layer
}

void BaseModel::forward(unique_ptr<Matrix> &&inputs)
{
    if(!built) throw logic_error("Attempting forward pass on model that is not built");
    // Start with inputLayer
    // Perform DFS and call forward on each layer in the graph
    // TODO Add a flag after inputs are ready from parent..
    // If all parents have performed forward pass, then call forward()
    //  If unable to perform pass, check the layer prior in the list
    //  If all are stuck, throw an error
    vector<BaseLayer*> queue;
    vector<BaseLayer*> explored;
    BaseLayer* startingNode = &(*_inputLayer);
    // Set the inputs into the input layer
    _inputLayer->setInputs(move(inputs));
    if(DEBUG) cout << "input layer has inputs..." << startingNode->_hasInputs << endl;
    // Start search at the input layer
    queue.emplace_back(startingNode);
    // Traverse graph and perform forward pass
    if(DEBUG) cout << "traversing graph calling forward()" << endl;
    while(true)
    {
        // If queue empty, break
        if(queue.size()==0) break;
        // Check if last node is a loss layer, continue if true
        if(queue.back()->isLossLayer())
        {
            // forward pass is complete
            // TODO Still need to call on loss layer
            break;
        }
        // Select from the back the first node ready to perform a forward pass
        int i;
        for(i=queue.size()-1;i>=0;i--) if(queue[i]->_hasInputs) break;
        if(i<0) throw logic_error("No layer has available inputs for forward pass");
        BaseLayer* currentNode = queue[i];
        if(DEBUG) cout << "index:" << i << endl;
        if(DEBUG) cout << "shape:" << queue.size() << endl;
        // Call forward on the layer
        if(DEBUG) cout << "performing forward pass on " << currentNode->getLayerType() << endl;
        currentNode->forward();
        // Proceed with search
        explored.emplace_back(currentNode);
        // Erase from queue by index
        queue.erase(queue.begin() + i);  
        // Add new layers to queue
        for(BaseLayer* l: currentNode->_childLayers) 
        {
            // Skip over layers that have already been explored
            bool alreadyExplored = false;
            for(BaseLayer* e: explored) if(&(*e)==&(*l)) alreadyExplored = true;
            if(alreadyExplored) continue;
            BaseLayer* newPointer = &(*l);
            queue.emplace_back(newPointer);
        }
    }
    // TODO Perform forward pass on the final layer somewhere
}

void BaseModel::backward(unique_ptr<Matrix> &&targets)
{
    // Debugging
    cout << "targets:";
    for(int r=0;r<targets->rows();r++)
    {
        for(int c=0;c<targets->cols();c++)
        {
            cout << (*targets)(r, c) << " ";
        }
    }
    cout << endl;
    // call forward on the final layer, this either requires an argument or set an attribute for target
    // call backward on each layer (including the loss layer)
}

OutputData& BaseModel::getGradients(OutputData& outputData)
{
    // this returns a struct with gradients added
    // loop over layers
    // TODO figure this out: all gradients need to be added from the layers in addition to the parameter gradients
    // ask if they have params, hasParams
    // call getGradients() on the layer to get the parameter gradients
    // copy the shared pointer to vector in output struct
    return outputData;
}

OutputData& BaseModel::getOutputs(OutputData& outputData)
{
    // this returns a struct with prediction/model outputs added
    // they should be waiting in the inputs of the final layer (loss layer) after forward is called
    outputData.outputs = make_unique<Matrix>(*_lossLayer->_inputs);
    return outputData;
}

// MyModel::MyModel(): BaseModel()
// {
//     if(DEBUG) cout << "MyModel::constructor" << endl;
//     // call MyModel::modelInstructions to populate the modelVector and paramVector
//     // call BaseModel::constructor
// }
  
// void MyModel::buildModelGraph(){
//     // (this happens once)
//     // TODO I should think about adding a skip connection to the model
//     // This is implemented specific to model architecture
//     // Contains instructions for and builds the model graph
//     // This function needs to connect the graph (connect parent(s) to child layer(s))
//     // This function needs to initialize parameters, 
//     //  to do that, each child layer needs to know the output shape of parent layer
//     // populate modelVector
//     // Initialize layers in sequential order
//     // Connect layers
// }
