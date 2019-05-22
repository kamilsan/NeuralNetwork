#include "layer.h"

unsigned int Layer::getNodesCount() const
{
    return nodes_;
}

NNMatrixType Layer::calculateWeightedInput(const NNMatrixType& input) const
{
    return weights_*input + bias_;
}