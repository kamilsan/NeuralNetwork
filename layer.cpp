#include "layer.h"

unsigned int Layer::getNodesCount() const
{
    return nodes_;
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