#ifndef MODEL_H_
#define MODEL_H_

#include <vector>
#include <memory>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <thread>



// Data objects for communicating with message queue
struct InputData 
{
    // std::unique_ptr<std::vector<float>> xData = nullptr;
    // // if exists (send data to queue without targets to just get predictions)
    // std::unique_ptr<std::vector<float>> targets = nullptr;
};

struct OutputData 
{
    // std::unique_ptr<std::vector<float>> outputs = nullptr;
    // std::vector<std::unique_ptr<std::vector<float>>> parameterGradients;
    // // (mean-gradloss-wrt-inputs-er-outputs) 
    // std::vector<std::unique_ptr<std::vector<float>>> layerGradients;
};


// Use message queue to send data to model threads
template <class T>
class InputQueue
{
public:
    T receive();
    void send(T &&msg);
private:
    std::deque<InputData> _queue;
    std::condition_variable _condition;
    std::mutex mtx;
};

// Use message queue to get outputs from model threads
template <class T>
class OutputQueue
{
public:
    T receive();
    void send(T &&msg);
private:
    std::deque<OutputData> _queue;
    std::condition_variable _condition;
    std::mutex mtx;
};


class BaseModel
{
// This class manages multi-threading and generic components
public:
    // TODO implement the rule of five
    BaseModel();
    BaseModel(BaseModel&);
    ~BaseModel();

    void createModelThread();

//   save
//     loop over layers
//     extract parameters with BaseLayer::getParams()
//     write parameters to file in order of layers 
//   load
//     open file
//     loop over layers
//     ask layer if it needs params
//     if needs then load line and pass to BaseLayer::setParams()
//     else go to next layer
//   forward(unique_ptr<paramdatatype[]> inputs)
//     param: inputs which is the rvalue of a unique pointer to a 1d array
//     loop over layers
//     call forward on each layer except the last (loss is computed in BaseModel::backward)
//     only the input layer requires an argument for forward, or another method could be used to move value to inputLayer
//   backward(unique_prt<paramdatatype[]> targets)
//     call forward on the final layer, this either requires an argument or set an attribute for target
//     call backward on each layer (including the loss layer)
//   OutputData getGradients(OutputData&)
//     this returns a struct with gradients
//     loop over layers
//     TODO figure this out: all gradients need to be added from the layers in addition to the parameter gradients
//     ask if they have params, hasParams
//     call getGradients() on the layer to get the parameter gradients
//     copy the shared pointer to vector in output struct
//   OutputData getOutputs(OutputData&)
//     this returns a struct with prediction/model outputs
//     they should be waiting in the inputs of the final layer (loss layer) after forward is called

private:

    std::vector<std::thread> threads; // holds all threads that have been launched within this object
    static std::mutex _mtx;           // mutex shared by all traffic objects for protecting cout 

    // 2 queues, one for inputs and one for outputs
    // TODO Probably need to add a public method like "sendData" for external usage
    InputQueue<InputData> _inputQueue;
    OutputQueue<OutputData> _outputQueue;

};

class MyModel: public BaseModel
{
//   This class contains the models architecture instructions 
//   constructor
//     call MyModel::modelInstructions to populate the modelVector and paramVector
//     call BaseModel::constructor
//   Problem-specific implementation
  
//   buildModelGraph (this happens once)
//     This is implemented specific to model architecture
//     Contains instructions for and builds the model graph
//     populate modelVector
//     Initialize layers in sequential order
//     Connect layers
};


#endif