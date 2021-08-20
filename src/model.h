#ifndef MODEL_H_
#define MODEL_H_

#include <vector>
#include <memory>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <thread>
#include <iostream>

#include "constants.h"
#include "matrix.h"


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

    void save();
    void load();
    void forward(std::unique_ptr<Matrix> &&inputs);
    void backward(std::unique_ptr<Matrix> &&targets);
    OutputData& getGradients(OutputData&);
    OutputData& getOutputs(OutputData&);

private:

    std::vector<std::thread> threads; // holds all threads that have been launched within this object
    static std::mutex _mtx;           // mutex shared by all traffic objects for protecting cout 

    // 2 queues, one for inputs and one for outputs
    // TODO Probably need to add a public method like "sendData" for external usage
    InputQueue<InputData> _inputQueue;
    OutputQueue<OutputData> _outputQueue;

    // ModelVector stores the model architecture information
    // std::vector<BaseLayer> modelVector;  // TODO Implement after declaring BaseLayer class

};

class MyModel: public BaseModel
{
public:
    // This class contains the models architecture instructions and is problem-specific
    MyModel();
    void buildModelGraph();
};


#endif