#pragma once

#include <cmath>
#include <vector>
#include <memory>
#include <random>

#include "matrix.h"

class Layer;

typedef float NNDataType;
typedef Matrix<NNDataType> NNMatrixType;

class NeuralNetwork
{
public:
    NeuralNetwork(unsigned int inputNodes, float learningRate);

    unsigned int getLayersCount() const;

    template<typename T>
    void addLayer(unsigned int nodes);

    NNMatrixType feedforward(const NNMatrixType& input) const;

    void train(unsigned int epochs, 
                unsigned int batchSize, 
                const std::vector<std::shared_ptr<NNMatrixType>>& inputs, 
                const std::vector<std::shared_ptr<NNMatrixType>>& targets);

    float test(const std::vector<std::shared_ptr<NNMatrixType>>& inputs, 
               const std::vector<std::shared_ptr<NNMatrixType>>& targets) const;

    void save(const char* filename) const;
    static NeuralNetwork* load(const char* filename);
private:
    unsigned int inputNodes_;
    unsigned int outputNodes_;
    float learningRate_;
    std::vector<std::shared_ptr<Layer>> layers_;
};

template<typename T>
void NeuralNetwork::addLayer(unsigned int nodes)
{
    layers_.push_back(std::make_shared<T>(nodes, outputNodes_));
    outputNodes_ = nodes;
}