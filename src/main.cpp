#include <iostream>
#include <utility>
#include "model.h"

int main() {

    // valgrind --leak-check=yes --track-origins=yes --log-file=valgrind-out.txt ./membot

    MyModel model;

    std::cout << "Model declared at:" << &model << "\n";
    MyModel newModel(model);
    std::cout << "Model copy at:" << &newModel << std::endl;

    // Test forward pass
    std::cout << "Testing BaseModel::forward(std::unique_ptr<std::vector<float>> &&inputs);" << std::endl;
    std::unique_ptr<std::vector<float>> inputs = std::make_unique<std::vector<float>>(std::vector<float>({.5, .5, .5}));
    model.forward(std::move(inputs));

    // Test backward pass
    std::cout << "Testing BaseModel::backward(std::unique_ptr<std::vector<float>> &&targets);" << std::endl;
    std::unique_ptr<std::vector<float>> targets = std::make_unique<std::vector<float>>(std::vector<float>({1.0, 0.0, 0.0}));
    model.backward(std::move(targets));

    // Declare outputs
    OutputData outputData;

    // Test BaseModel::getGradients(OutputData&)
    std::cout << "Testing BaseModel::getGradients(OutputData&)" << std::endl;
    model.getGradients(outputData);
    
    // Test BaseModel::getOutputs(OutputData&)
    std::cout << "Testing BaseModel::getGradients(OutputData&)" << std::endl;
    model.getOutputs(outputData);

    // Test MyModel::MyModel



    return 0;
}