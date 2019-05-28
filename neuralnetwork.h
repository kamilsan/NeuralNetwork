#pragma once

#include <cmath>
#include <vector>
#include <memory>
#include <random>

#include "matrix.h"

class Layer;

typedef Matrix<float> NNMatrixType;

class NeuralNetwork
{
public:
    NeuralNetwork(int inputNodes, float learningRate);

    void addLayer(const std::shared_ptr<Layer>& layer);

    void feedforward(const NNMatrixType& input, NNMatrixType &result) const;
    void train(int epochs, 
                int batchSize, 
                const std::vector<std::shared_ptr<NNMatrixType>>& inputs, 
                const std::vector<std::shared_ptr<NNMatrixType>>& targets);
    float test(const std::vector<std::shared_ptr<NNMatrixType>>& inputs, 
               const std::vector<std::shared_ptr<NNMatrixType>>& targets) const;
    void save(const char* filename) const;
    static NeuralNetwork* load(const char* filename);
private:
    int inputNodes_;
    float learningRate_;
    std::vector<std::shared_ptr<Layer>> layers_;
};