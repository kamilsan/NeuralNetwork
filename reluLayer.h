#pragma once

#include "layer.h"

class ReLULayer : public Layer
{
public:
    ReLULayer(int nodes, int nodesPrevious);

    virtual NNMatrixType feedforward(const NNMatrixType& input) const;
    virtual NNMatrixType backpropagate(const NNMatrixType& input);
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