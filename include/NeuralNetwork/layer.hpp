#pragma once

#include "neuralnetwork.hpp"

#include <fstream>

class Layer
{
friend class NeuralNetwork;
public:
    Layer(unsigned int nodes, unsigned int prevNodes);

    // Getter for nodes_
    virtual unsigned int getNodesCount() const;

    // Layer's abstract activation function
    virtual NNDataType activationFunction(NNDataType value) const = 0;

    // Derivative of layer's abstract activation function
    virtual NNDataType activationDerivative(NNDataType value) const = 0;

    // Calculates f(weights * input + bias), where f is an activation function and sets weightedInput to calculated value
    virtual NNMatrixType feedforward(const NNMatrixType& input, NNMatrixType& weightedInput);

    // Calculates f(weights * input + bias), where f is an activation function
    virtual NNMatrixType feedforward(const NNMatrixType& input) const;

    // Caclulates cost derivatives with respect to weights and biases and returns error (derivative of cost w.r.t this layer nodes) to be used in next layer
    virtual NNMatrixType backpropagate(const NNMatrixType& error,
                                       const NNMatrixType& weightedInput,
                                       const NNMatrixType& prevOutput);

    // Nudges weights and biases in direction of steepest descent
    virtual void performSDGStep(float learingRate);

    virtual void serialize(std::ofstream& ofile) const = 0;
protected:
    // Return weights * input + bias. This value needs to be calculated in all layer types so this function is shared
    virtual NNMatrixType calculateWeightedInput(const NNMatrixType& input) const;

    virtual void serializeMatricies(std::ofstream& ofile) const;

    unsigned int nodes_;
    
    NNMatrixType weights_;
    NNMatrixType bias_;

    NNMatrixType nablaW_;   // Accumulated cost derivative w.r.t weights
    NNMatrixType nablaB_;   // Accumulated cost derivative w.r.t biases
};