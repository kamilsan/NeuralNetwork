#include "layer.h"

#include <functional>

Layer::Layer(unsigned int nodes, unsigned int prevNodes)
{
    weights_ = NNMatrixType(nodes, prevNodes);
    bias_ = NNMatrixType(nodes, 1);

    //initializes weights is this way so as to the keep values reasonably small
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
    //Calculates dC/dz = dC/da * da/dz, where da/dz is the derivative of the activation function
    NNMatrixType delta = error.hadamard(weightedInput.map(std::bind(&Layer::activationDerivative, this, std::placeholders::_1)));
    //dC/dw = dC/dz * dz/dw
    NNMatrixType nablaW = delta * NNMatrixType::transpose(prevOutput);

    //dC/da for the next layer
    NNMatrixType output = NNMatrixType::transpose(weights_) * delta;

    //Accumulate the gradients
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

    //Reset gradient values (for next batch)
    nablaW_.zero();
    nablaB_.zero();
}