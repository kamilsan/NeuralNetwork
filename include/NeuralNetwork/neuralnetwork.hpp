#pragma once

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "matrix.hpp"

class Layer;
class CostFunctionStrategy;

typedef float NNDataType;
typedef Matrix<NNDataType> NNMatrixType;

class NeuralNetwork
{
public:
    NeuralNetwork(unsigned int inputNodes, float learningRate, std::unique_ptr<CostFunctionStrategy> costFunction);

    unsigned int getLayersCount() const;
    unsigned int getOutputNodesCount() const;

    template<typename T>
    void addLayer(unsigned int nodes);

    // Get output from neural net
    NNMatrixType feedforward(const NNMatrixType& input) const;

    // The name of the game
    void train(unsigned int epochs, 
                unsigned int batchSize, 
                const std::vector<NNMatrixType>& inputs, 
                const std::vector<NNMatrixType>& targets);

    // Testing nn performance
    float test(const std::vector<NNMatrixType>& inputs, 
               const std::vector<NNMatrixType>& targets) const;

    // Serialization and deserialization
    void save(const char* filename) const;
    static NeuralNetwork load(const char* filename);
private:
    void singleInputTrain(const NNMatrixType& input, const NNMatrixType& target); // used in train
    void addLayer(std::shared_ptr<Layer> layer); // used in serialization

    unsigned int inputNodes_;
    unsigned int outputNodes_;
    float learningRate_;
    std::unique_ptr<CostFunctionStrategy> costFunction_;
    std::vector<std::shared_ptr<Layer>> layers_;
};

template<typename T>
void NeuralNetwork::addLayer(unsigned int nodes)
{
    layers_.push_back(std::make_shared<T>(nodes, outputNodes_));
    outputNodes_ = nodes;
}
