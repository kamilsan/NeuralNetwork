#include "reluLayer.h"

NNDataType ReLULayer::activationFunction(NNDataType value) const
{
    if(value < 0.0f) return 0.0f;
    else return value;
}

NNDataType ReLULayer::activationDerivative(NNDataType value) const
{
    if(value < 0.0f) return 0.0f;
    else return 1.0f;
}