#include "layer.h"

#include <functional>

Layer::Layer(unsigned int nodes, unsigned int prevNodes)
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

unsigned int Layer::getNodesCount() const
{
    return nodes_;
}

NNMatrixType Layer::feedforward(const NNMatrixType& input, NNMatrixType& weightedInput)
{
    weightedInput = calculateWeightedInput(input);
    return weightedInput.map(std::bind(&Layer::activationFunction, this, std::placeholders::_1));
}

NNMatrixType Layer::feedforward(const NNMatrixType& input) const
{
    NNMatrixType weightedInput = calculateWeightedInput(input);
    return weightedInput.map(std::bind(&Layer::activationFunction, this, std::placeholders::_1));
}

NNMatrixType Layer::backpropagate(const NNMatrixType& error,
                                    const NNMatrixType& weightedInput,
                                    const NNMatrixType& prevOutput)
{
    NNMatrixType delta = error.hadamard(weightedInput.map(std::bind(&Layer::activationDerivative, this, std::placeholders::_1)));
    NNMatrixType nablaW = delta * NNMatrixType::transpose(prevOutput);

    NNMatrixType output = NNMatrixType::transpose(weights_) * delta;

    nablaW_ += nablaW;
    nablaB_ += delta;

    return output;
}

NNMatrixType Layer::calculateWeightedInput(const NNMatrixType& input) const
{
    return weights_*input + bias_;
}

void Layer::performSDGStep(float learningRate)
{
    weights_ -= nablaW_ * learningRate;
    bias_ -= nablaB_ * learningRate;

    nablaW_.zero();
    nablaB_.zero();
}