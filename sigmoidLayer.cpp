#include "sigmoidLayer.h"

SigmoidLayer::SigmoidLayer(unsigned int nodes, unsigned int prevNodes)
{
    weights_ = NNMatrixType(nodes, prevNodes);
    bias_ = NNMatrixType(nodes, 1);

    float r = 4.0*std::sqrt(6.0/(nodes + prevNodes));
    weights_.randomize(-r, r);

    bias_.zero();

    nablaW_ = NNMatrixType(nodes, prevNodes);
    nablaB_ = NNMatrixType(nodes, 1);

    nablaW_.zero();
    nablaB_.zero();
}

NNMatrixType SigmoidLayer::feedforward(const NNMatrixType& input) const
{
    NNMatrixType weightedInput = calculateWeightedInput(input);
    return weightedInput.map(sigmoid);
}

NNMatrixType SigmoidLayer::feedforward(const NNMatrixType& input, 
                                    NNMatrixType& weightedInput)
{
    weightedInput = calculateWeightedInput(input);
    return weightedInput.map(sigmoid);
}

NNMatrixType SigmoidLayer::backpropagate(const NNMatrixType& error,
                                       const NNMatrixType& weightedInput,
                                       const NNMatrixType& prevOutput)
{
    NNMatrixType delta = error.hadamard(weightedInput.map(dsigmoid));
    NNMatrixType nablaW = delta * NNMatrixType::transpose(prevOutput);

    NNMatrixType output = NNMatrixType::transpose(weights_) * delta;

    nablaW_ += nablaW;
    nablaB_ += delta;

    return output;
}