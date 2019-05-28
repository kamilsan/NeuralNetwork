#pragma once

#include "layer.h"

class SigmoidLayer : public Layer
{
public:
    SigmoidLayer(unsigned int nodes, unsigned int prevNodes);

    virtual NNMatrixType feedforward(const NNMatrixType& input, 
                                     NNMatrixType& weightedInput);
    virtual NNMatrixType feedforward(const NNMatrixType& input) const;
    virtual NNMatrixType backpropagate(const NNMatrixType& error, 
                                       const NNMatrixType& weightedInput,
                                       const NNMatrixType& prevOutput);
private:
    //Sigmoid activation function
    static float sigmoid(float x)
    {
        return 1.0/(1.0 + exp(-x));
    }
    
    //Derivative of sigmoid, used in backpropagation
    static float dsigmoid(float x)
    {
        return sigmoid(x)*(1.0-sigmoid(x));
    }
};