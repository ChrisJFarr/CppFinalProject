#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <limits>

const static int INPUT_SHAPE = 784;
const static bool DEBUG = false;
typedef float MyDType;
const MyDType static inf = std::numeric_limits<MyDType>::infinity();

#endif