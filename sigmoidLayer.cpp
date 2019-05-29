#include "sigmoidLayer.h"

NNDataType SigmoidLayer::activationFunction(NNDataType value) const
{
    return 1.0/(1.0 + exp(-value));
}

NNDataType SigmoidLayer::activationDerivative(NNDataType value) const
{
    NNDataType sigmoid = 1.0/(1.0 + exp(-value));
    return sigmoid*(1.0-sigmoid);
}