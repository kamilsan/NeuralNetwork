#pragma once

#include "neuralnetwork.h"

class Layer
{
public:
    //Getter for nodes_
    virtual unsigned int getNodesCount() const;

    //Calculates f(weights * input + bias), where f is an activation function
    virtual NNMatrixType feedforward(const NNMatrixType& input) const = 0;
    //Performs step of backpropagation, return gradient and adjusts weights & bias
    virtual NNMatrixType backpropagate(const NNMatrixType& input) = 0;
protected:
    //Return weights * input + bias. This value needs to be calculated in all layer types so this function is shared
    virtual NNMatrixType calculateWeightedInput(const NNMatrixType& input) const;
private:
    Layer();

    unsigned int nodes_;
    NNMatrixType weights_;
    NNMatrixType bias_;
};