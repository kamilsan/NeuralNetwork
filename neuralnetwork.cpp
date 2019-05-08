#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <chrono>

#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes, float learningRate): input_nodes_(input_nodes), hidden_nodes_(hidden_nodes), output_nodes_(output_nodes), learningRate(learningRate)
{
    weights_ih = NNMatrixType(hidden_nodes, input_nodes);
    weights_ho = NNMatrixType(output_nodes, hidden_nodes);
    bias_h = NNMatrixType(hidden_nodes, 1);
    bias_o = NNMatrixType(output_nodes, 1);

    float r = 4.0*std::sqrt(6.0/(input_nodes + hidden_nodes));
    weights_ih.randomize(-r, r);

    r = 4.0*std::sqrt(6.0/(hidden_nodes + output_nodes));
    weights_ho.randomize(-r, r);

    bias_h.zero();
    bias_o.zero();
}

void NeuralNetwork::feedforward(const NNMatrixType& input, NNMatrixType &result) const
{
    if(input.getRows() != input_nodes_ || input.getColumns() != 1)
    {
        std::cout << "ERROR: passed input matrix has wrong dimensions!\n";
        return;
    }

    result = weights_ih*input + bias_h;
    result = result.map(relu);
    result = weights_ho*result + bias_o;
    //Softmax
    result = result.map(std::exp);
    float den = 1.0f/result.sum();
    result *= den;
}

void NeuralNetwork::train(int epochs, 
                          int batchSize, 
                          const std::vector<std::shared_ptr<NNMatrixType>>& inputs, 
                          const std::vector<std::shared_ptr<NNMatrixType>>& targets)
{
    int trainingSize = inputs.size();
    std::vector<int> permutaionTable(trainingSize);
    for(int i = 0; i < trainingSize; ++i)
    {
        permutaionTable[i] = i;
    }

    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);

    int numBatches = std::ceil((float)trainingSize / batchSize);

    NNMatrixType input;
    NNMatrixType target;

    NNMatrixType bh_grad(hidden_nodes_, 1);
    NNMatrixType bo_grad(output_nodes_, 1);
    NNMatrixType wih_grad(hidden_nodes_, input_nodes_);
    NNMatrixType who_grad(output_nodes_, hidden_nodes_);

    for(int epoch = 0; epoch < epochs; ++epoch)
    {
        std::cout << "Epoch " << epoch + 1 << " out of " << epochs << "\n";
        std::shuffle(permutaionTable.begin(), permutaionTable.end(), generator);
        for(int n = 0; n < numBatches; ++n)
        {
        bh_grad.zero();
        bo_grad.zero();
        wih_grad.zero();
        who_grad.zero();

        int startIdx = n*batchSize;

        for(int i = 0; i < batchSize; ++i)
        {
            int idx = startIdx + i;
            if(idx >= trainingSize) break;

            input = *inputs[permutaionTable[idx]];
            target = *targets[permutaionTable[idx]];

            if(input.getRows() != input_nodes_ || input.getColumns() != 1)
            {
                std::cout << "ERROR: passed input matrix has wrong dimensions!\n";
                return;
            }

            NNMatrixType hidden_unactive = weights_ih*input + bias_h;
            NNMatrixType hidden = hidden_unactive.map(relu);
            NNMatrixType output_unactive = weights_ho*hidden + bias_o;

            //Softmax
            NNMatrixType output = output_unactive.map(std::exp);
            float den = 1.0/output.sum();
            output *= den;

            NNMatrixType difference = target - output;
            //softmax + negative log-likelihood
            NNMatrixType deltaOutput = difference;

            bo_grad += deltaOutput;
            who_grad += deltaOutput * NNMatrixType::transpose(hidden);

            NNMatrixType error_hidden = NNMatrixType::transpose(weights_ho) * deltaOutput;
            NNMatrixType gradActivation_hidden = hidden_unactive.map(drelu);

            NNMatrixType deltaHidden = error_hidden.hadamard(gradActivation_hidden);
            
            bh_grad += deltaHidden;
            wih_grad += deltaHidden * NNMatrixType::transpose(input);
        }

        bias_o += learningRate * bo_grad;
        weights_ho += learningRate * who_grad;
        bias_h += learningRate * bh_grad;
        weights_ih += learningRate * wih_grad;
        }
    }
}

float NeuralNetwork::test(const std::vector<std::shared_ptr<NNMatrixType>>& inputs, 
                          const std::vector<std::shared_ptr<NNMatrixType>>& targets) const
{
    NNMatrixType result;
    unsigned correctPredictions = 0;
    unsigned predictions = inputs.size();
    for(size_t n = 0; n < predictions; ++n)
    {
        feedforward(*inputs[n], result);

        int predictedLabel = 0;
        float currentValue = result.get(0, 0);
        float maxValue = currentValue;
        for(int i = 0; i < result.getRows(); ++i)
        {
            currentValue = result.get(i, 0);
            if(currentValue > maxValue)
            {
                predictedLabel = i;
                maxValue = currentValue;
            }
        }

        int expectedLabel = 0;
        currentValue = targets[n]->get(0, 0);
        maxValue = currentValue;
        for(int i = 0; i < result.getRows(); ++i)
        {
            currentValue = targets[n]->get(i, 0);
            if(currentValue > maxValue)
            {
                expectedLabel = i;
                maxValue = currentValue;
            }
        }

        if(expectedLabel == predictedLabel) correctPredictions++;
    }

    return 100.0f*correctPredictions/predictions;
}

void NeuralNetwork::save(const char* filename) const
{
    std::cout << "Not Implemented!\n";
}

NeuralNetwork* NeuralNetwork::load(const char* filename)
{
    std::cout << "Not Implemented!\n";
    return nullptr;
}
