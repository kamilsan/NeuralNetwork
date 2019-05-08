#pragma once

#include <cmath>
#include <vector>
#include <memory>
#include <random>

#include "matrix.h"

typedef Matrix<float> NNMatrixType;

class NeuralNetwork
{
public:
    NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes, float learningRate);

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
    int input_nodes_, hidden_nodes_, output_nodes_;
    float learningRate;
    NNMatrixType weights_ih, weights_ho;
    NNMatrixType bias_h, bias_o;

    static float sigmoid(float x)
    {
        return 1.0/(1.0 + exp(-x));
    }
    static float dsigmoid(float x)
    {
        return x*(1.0-x);
    }

    static float relu(float x)
    {
        if(x < 0.0f) return 0;
        else return x;
    }
    static float drelu(float x)
    {
        if(x < 0.0f) return 0;
        else return 1.0;
    }
};