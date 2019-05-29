#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <chrono>

#include "neuralnetwork.h"
#include "data_load_failure.h"
#include "layer.h"

NeuralNetwork::NeuralNetwork(int inputNodes, float learningRate): 
    inputNodes_(inputNodes),
    outputNodes_(inputNodes),
    learningRate_(learningRate)
{}

NNMatrixType NeuralNetwork::feedforward(const NNMatrixType& input) const
{
    if(input.getRows() != inputNodes_ || input.getColumns() != 1)
    {
        std::cout << "ERROR: passed input matrix has wrong dimensions!\n";
        return NNMatrixType();
    }

    NNMatrixType result = input;
    for(auto it = layers_.begin(); it < layers_.end(); ++it)
    {
        result = (*it)->feedforward(result);
    }

    return result;
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

    for(int epoch = 0; epoch < epochs; ++epoch)
    {
        std::cout << "Epoch " << epoch + 1 << " out of " << epochs << "\n";
        std::shuffle(permutaionTable.begin(), permutaionTable.end(), generator);
        for(int n = 0; n < numBatches; ++n)
        {
            int startIdx = n*batchSize;

            for(int i = 0; i < batchSize; ++i)
            {
                int idx = startIdx + i;
                if(idx >= trainingSize) break;

                input = *inputs[permutaionTable[idx]];
                target = *targets[permutaionTable[idx]];

                if(input.getRows() != inputNodes_ || input.getColumns() != 1)
                {
                    std::cout << "ERROR: passed input matrix has wrong dimensions!\n";
                    return;
                }

                //forward pass
                NNMatrixType output = input;
                //Vectors storing results of layers' calculations
                //used in backpropagation
                std::vector<NNMatrixType> weightedInputs;
                std::vector<NNMatrixType> outputs;
                weightedInputs.reserve(layers_.size());
                outputs.reserve(layers_.size());

                for(auto it = layers_.begin(); it < layers_.end(); ++it)
                {
                    NNMatrixType weightedInput;
                    output = (*it)->feedforward(output, weightedInput);
                    weightedInputs.emplace_back(weightedInput);
                    outputs.emplace_back(output);
                }

                NNMatrixType error = output - target;

                int backpropIdx = layers_.size() - 1;
                for(auto it = layers_.rbegin(); it < layers_.rend() - 1; ++it)
                {
                    error = (*it)->backpropagate(error, weightedInputs[backpropIdx], outputs[backpropIdx - 1]);
                    backpropIdx--;
                }

                //Handle first layer differently - pass input instead of last layer's output
                error = layers_[0]->backpropagate(error, weightedInputs[0], input);
            }
            
            //Adjust weights and biases
            for(auto it = layers_.begin(); it < layers_.end(); ++it)
            {
                (*it)->performSDGStep(learningRate_);
            }
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
        result = feedforward(*inputs[n]);

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
    //TODO
    std::cout << "NOT IMPLEMENTED!\n";
    /*
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
    */
}

NeuralNetwork* NeuralNetwork::load(const char* filename)
{
    //TODO
    std::cout << "NOT IMPLEMENTED!\n";
    return nullptr;
    
    /*
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
    */
}
