#pragma once

#include "layer.h"

class ReLULayer : public Layer
{
public:
    ReLULayer(unsigned int nodes, unsigned int prevNodes);

    virtual NNMatrixType feedforward(const NNMatrixType& input, 
                                     NNMatrixType& weightedInput);
    virtual NNMatrixType feedforward(const NNMatrixType& input) const;
    virtual NNMatrixType backpropagate(const NNMatrixType& error, 
                                       const NNMatrixType& weightedInput,
                                       const NNMatrixType& prevOutput);
private:
    //Rectified linear unit activation function
    static float relu(float value)
    {
        if(value < 0.0f) return 0.0f;
        else return value;
    }

    //Derivative of ReLU, used in backpropagation
    static float drelu(float value)
    {
        if(value < 0.0f) return 0.0f;
        else return 1.0f;
    }
};