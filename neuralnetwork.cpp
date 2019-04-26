#include <iostream>
#include <memory>
#include <algorithm>
#include <chrono>

#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes, float learningRate): input_nodes_(input_nodes), hidden_nodes_(hidden_nodes), output_nodes_(output_nodes), learningRate(learningRate)
{
    weights_ih = MatrixType(hidden_nodes, input_nodes);
    weights_ho = MatrixType(output_nodes, hidden_nodes);
    bias_h = MatrixType(hidden_nodes, 1);
    bias_o = MatrixType(output_nodes, 1);

    float r = 4.0*std::sqrt(6.0/(input_nodes + hidden_nodes));
    //weights_ih.randomize();
    //weights_ih.map(random);
    weights_ih *= r;

    //weights_ho.randomize();
    r = 4.0*std::sqrt(6.0/(hidden_nodes + output_nodes));
    weights_ho *= r;

    bias_h.zero();
    bias_o.zero();
}

void NeuralNetwork::feedforward(const MatrixType& input, MatrixType &result) const
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
                          const std::vector<std::shared_ptr<MatrixType>>& inputs, 
                          const std::vector<std::shared_ptr<MatrixType>>& targets)
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

    MatrixType input;
    MatrixType target;

    MatrixType bh_grad(hidden_nodes_, 1);
    MatrixType bo_grad(output_nodes_, 1);
    MatrixType wih_grad(hidden_nodes_, input_nodes_);
    MatrixType who_grad(output_nodes_, hidden_nodes_);

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

            MatrixType hidden_unactive = weights_ih*input + bias_h;
            MatrixType hidden = hidden_unactive.map(relu);
            MatrixType output_unactive = weights_ho*hidden + bias_o;

            //Softmax
            MatrixType output = output_unactive.map(std::exp);
            float den = 1.0/output.sum();
            output *= den;

            MatrixType difference = target - output;
            //softmax + negative log-likelihood
            MatrixType deltaOutput = difference;

            bo_grad += deltaOutput;
            who_grad += deltaOutput * MatrixType::transpose(hidden);

            MatrixType error_hidden = MatrixType::transpose(weights_ho) * deltaOutput;
            MatrixType gradActivation_hidden = hidden_unactive.map(drelu);

            MatrixType deltaHidden = error_hidden.hadamard(gradActivation_hidden);
            
            bh_grad += deltaHidden;
            wih_grad += deltaHidden * MatrixType::transpose(input);
        }

        bias_o += learningRate * bo_grad;
        weights_ho += learningRate * who_grad;
        bias_h += learningRate * bh_grad;
        weights_ih += learningRate * wih_grad;
        }
    }
}