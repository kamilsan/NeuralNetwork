#pragma once

#include "layer.hpp"

class ReLULayer : public Layer
{
friend class NeuralNetwork;
public:
    ReLULayer(unsigned int nodes, unsigned int prevNodes) : Layer(nodes, prevNodes) {};

    virtual NNDataType activationFunction(NNDataType value) const;
    virtual NNDataType activationDerivative(NNDataType value) const;

    virtual void serialize(std::ofstream& ofile) const;
};