#include <iostream>
#include "model.h"

int main() {

    MyModel model;

    std::cout << "Model declared at:" << &model << "\n";
    MyModel newModel(model);
    std::cout << "Model copy at:" << &newModel << std::endl;
    return 0;
}