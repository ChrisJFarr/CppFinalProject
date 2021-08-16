#include <iostream>
#include "model.h"



BaseModel::BaseModel()
{
    std::cout << "BaseModel::constructor" << std::endl;
};
BaseModel::BaseModel(BaseModel&)
{
    std::cout << "BaseModel::copy-constructor" << std::endl;
    // copy constructor (copies are used in the thread functions)
    // loop over and copy the modelVector to the new object
};
BaseModel::~BaseModel()
{
    std::cout << "BaseModel::destructor" << std::endl;
    // Loop through and join the threads, need to stop them somehow...?
    // set up thread barrier before this object is destroyed
    // std::for_each(threads.begin(), threads.end(), [](std::thread &t) {
    //     t.join();
    // });
};
// TODO implement the rule of five


void BaseModel::createModelThread()
{
// TODO Add all new threads to the _threads vector
//   createModelThread (i think this function is passed to a thread)
//     This function copies the modelGraph to a local copy
//     This function creates a thread, copies the modelVector, then loops forever (how can I reduce cost of looping)
//     loop over modelVector 
//     copy each layer to local modelVector
//     reconnect the local graph (loop over vector and call connectParent from the child, passing the parent)
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



// //   TODO Copy the message queue from concurrency project
// //   2 queues, one for inputs and one for outputs
// //     MessageQueue<InputData> inputQueue
// //     MessageQueue<OutputData> outputQueue
// //   save
// //     loop over layers
// //     extract parameters with BaseLayer::getParams()
// //     write parameters to file in order of layers 
// //   load
// //     open file
// //     loop over layers
// //     ask layer if it needs params
// //     if needs then load line and pass to BaseLayer::setParams()
// //     else go to next layer
// //   forward(unique_ptr<paramdatatype[]> inputs)
// //     param: inputs which is the rvalue of a unique pointer to a 1d array
// //     loop over layers
// //     call forward on each layer except the last (loss is computed in BaseModel::backward)
// //     only the input layer requires an argument for forward, or another method could be used to move value to inputLayer
// //   backward(unique_prt<paramdatatype[]> targets)
// //     call forward on the final layer, this either requires an argument or set an attribute for target
// //     call backward on each layer (including the loss layer)
// //   OutputData getGradients(OutputData&)
// //     this returns a struct with gradients
// //     loop over layers
// //     TODO figure this out: all gradients need to be added from the layers in addition to the parameter gradients
// //     ask if they have params, hasParams
// //     call getGradients() on the layer to get the parameter gradients
// //     copy the shared pointer to vector in output struct
// //   OutputData getOutputs(OutputData&)
// //     this returns a struct with prediction/model outputs
// //     they should be waiting in the inputs of the final layer (loss layer) after forward is called
// };

// class MyModel: public BaseModel
// {
// //   This class contains the models architecture instructions 
// //   constructor
// //     call MyModel::modelInstructions to populate the modelVector and paramVector
// //     call BaseModel::constructor
// //   Problem-specific implementation
  
// //   buildModelGraph (this happens once)
// //     This is implemented specific to model architecture
// //     Contains instructions for and builds the model graph
// //     populate modelVector
// //     Initialize layers in sequential order
// //     Connect layers
// };
