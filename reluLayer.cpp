#include "reluLayer.h"

NNMatrixType ReLULayer::feedforward(const NNMatrixType& input) const
{
    NNMatrixType weightedInput = calculateWeightedInput(input);
    return weightedInput.map(relu);
}

NNMatrixType ReLULayer::backpropagate(const NNMatrixType& input)
{
    //TODO
    return NNMatrixType();
}