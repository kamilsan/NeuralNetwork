#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <chrono>

#include "neuralnetwork.h"
#include "data_load_failure.h"

NeuralNetwork::NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes, float learningRate): input_nodes_(input_nodes), hidden_nodes_(hidden_nodes), output_nodes_(output_nodes), learningRate_(learningRate)
{
    weights_ih_ = NNMatrixType(hidden_nodes, input_nodes);
    weights_ho_ = NNMatrixType(output_nodes, hidden_nodes);
    bias_h_ = NNMatrixType(hidden_nodes, 1);
    bias_o_ = NNMatrixType(output_nodes, 1);

    float r = 4.0*std::sqrt(6.0/(input_nodes + hidden_nodes));
    weights_ih_.randomize(-r, r);

    r = 4.0*std::sqrt(6.0/(hidden_nodes + output_nodes));
    weights_ho_.randomize(-r, r);

    bias_h_.zero();
    bias_o_.zero();
}

NeuralNetwork::NeuralNetwork(const NNMatrixType& weights_ih, const NNMatrixType& bias_h, const NNMatrixType& weights_ho, const NNMatrixType& bias_o, float learningRate): weights_ih_(weights_ih), bias_h_(bias_h), weights_ho_(weights_ho), bias_o_(bias_o), learningRate_(learningRate)
{
    input_nodes_ = weights_ih_.getColumns();
    hidden_nodes_ = weights_ho_.getColumns();
    output_nodes_ = weights_ho_.getRows();
}

void NeuralNetwork::feedforward(const NNMatrixType& input, NNMatrixType &result) const
{
    if(input.getRows() != input_nodes_ || input.getColumns() != 1)
    {
        std::cout << "ERROR: passed input matrix has wrong dimensions!\n";
        return;
    }

    result = weights_ih_*input + bias_h_;
    result = result.map(relu);
    result = weights_ho_*result + bias_o_;
    //Softmax
    result = result.map(std::exp);
    float den = 1.0f/result.sum();
    if(std::isinf(den))
    {
        result.zero();
    }
    else
    {
        result *= den;
    }
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

                NNMatrixType hidden_unactive = weights_ih_*input + bias_h_;
                NNMatrixType hidden = hidden_unactive.map(relu);
                NNMatrixType output_unactive = weights_ho_*hidden + bias_o_;

                //Softmax
                NNMatrixType output = output_unactive.map(std::exp);
                float den = 1.0/output.sum();
                output *= den;

                NNMatrixType difference = target - output;
                //softmax + negative log-likelihood
                NNMatrixType deltaOutput = difference;

                bo_grad += deltaOutput;
                who_grad += deltaOutput * NNMatrixType::transpose(hidden);

                NNMatrixType error_hidden = NNMatrixType::transpose(weights_ho_) * deltaOutput;
                NNMatrixType gradActivation_hidden = hidden_unactive.map(drelu);

                NNMatrixType deltaHidden = error_hidden.hadamard(gradActivation_hidden);
                
                bh_grad += deltaHidden;
                wih_grad += deltaHidden * NNMatrixType::transpose(input);
            }

            bias_o_ += learningRate_ * bo_grad;
            weights_ho_ += learningRate_ * who_grad;
            bias_h_ += learningRate_ * bh_grad;
            weights_ih_ += learningRate_ * wih_grad;
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
    std::ofstream ofile(filename, std::ios::binary);

    ofile.write((char*)&learningRate_, sizeof(learningRate_));
    int data = weights_ih_.getRows();
    ofile.write((char*)&data, sizeof(data));
    data = weights_ih_.getColumns();
    ofile.write((char*)&data, sizeof(data));
    data = weights_ho_.getRows();
    ofile.write((char*)&data, sizeof(data));
    data = weights_ho_.getColumns();
    ofile.write((char*)&data, sizeof(data));

    int len = weights_ih_.getRows()*weights_ih_.getColumns();
    ofile.write((char*)weights_ih_.getData(), len*sizeof(float));

    len = bias_h_.getRows()*bias_h_.getColumns();
    ofile.write((char*)bias_h_.getData(), len*sizeof(float));

    len = weights_ho_.getRows()*weights_ho_.getColumns();
    ofile.write((char*)weights_ho_.getData(), len*sizeof(float));

    len = bias_o_.getRows()*bias_o_.getColumns();
    ofile.write((char*)bias_o_.getData(), len*sizeof(float));

    ofile.close();
}

NeuralNetwork* NeuralNetwork::load(const char* filename)
{
    std::ifstream ifile(filename, std::ios::binary);

    if(!ifile.is_open())
    {
        throw data_load_failure(filename);
    }

    float learingRate;
    int weights_ih_rows, weights_ih_columns, weights_ho_rows, weights_ho_columns;

    ifile.read((char*)&learingRate, sizeof(float));
    ifile.read((char*)&weights_ih_rows, sizeof(int));
    ifile.read((char*)&weights_ih_columns, sizeof(int));
    ifile.read((char*)&weights_ho_rows, sizeof(int));
    ifile.read((char*)&weights_ho_columns, sizeof(int));

    int len = weights_ih_rows*weights_ih_columns;
    float* weights_ih_buffer = new float[len];
    ifile.read((char*)weights_ih_buffer, len*sizeof(float));

    len = weights_ih_rows;
    float* bias_h_buffer = new float[len];
    ifile.read((char*)bias_h_buffer, len*sizeof(float));

    len = weights_ho_rows*weights_ho_columns;
    float* weights_ho_buffer = new float[len];
    ifile.read((char*)weights_ho_buffer, len*sizeof(float));

    len = weights_ho_rows;
    float* bias_o_buffer = new float[len];
    ifile.read((char*)bias_o_buffer, len*sizeof(float));
    
    ifile.close();

    NNMatrixType weights_ih = NNMatrixType(weights_ih_buffer, weights_ih_rows, weights_ih_columns);
    NNMatrixType bias_h = NNMatrixType(bias_h_buffer, weights_ih_rows, 1);
    NNMatrixType weights_ho = NNMatrixType(weights_ho_buffer, weights_ho_rows, weights_ho_columns);
    NNMatrixType bias_o = NNMatrixType(bias_o_buffer, weights_ho_rows, 1);
    
    delete[] weights_ih_buffer;
    delete[] bias_h_buffer;
    delete[] weights_ho_buffer;
    delete[] bias_o_buffer;

    return new NeuralNetwork(weights_ih, bias_h, weights_ho, bias_o, learingRate);
}
