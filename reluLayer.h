#pragma once

#include "layer.h"

class ReLULayer : public Layer
{
public:
    ReLULayer(unsigned int nodes, unsigned int prevNodes) : Layer(nodes, prevNodes) {};

    virtual NNDataType activationFunction(NNDataType value) const;
    virtual NNDataType activationDerivative(NNDataType value) const;
};