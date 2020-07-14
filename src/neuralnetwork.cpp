#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

#include "costFunctionStrategy.h"
#include "crossEntropyCost.h"
#include "data_load_failure.h"
#include "meanSquereErrorCost.h"
#include "neuralnetwork.h"
#include "reluLayer.h"
#include "sigmoidLayer.h"

NeuralNetwork::NeuralNetwork(unsigned int inputNodes, float learningRate, std::unique_ptr<CostFunctionStrategy> costFunction): 
    inputNodes_(inputNodes),
    outputNodes_(inputNodes),
    learningRate_(learningRate),
    costFunction_(std::move(costFunction))
{}

unsigned int NeuralNetwork::getLayersCount() const
{
    return layers_.size();
}

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

void NeuralNetwork::train(unsigned int epochs, 
                          unsigned int batchSize, 
                          const std::vector<std::shared_ptr<NNMatrixType>>& inputs, 
                          const std::vector<std::shared_ptr<NNMatrixType>>& targets)
{
    // Prepare permutation table for training data shuffle
    size_t trainingSize = inputs.size();
    std::vector<unsigned int> permutaionTable(trainingSize);
    for(size_t i = 0; i < trainingSize; ++i)
    {
        permutaionTable[i] = i;
    }

    // Initialize PRNG
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);

    unsigned int numBatches = std::ceil((float)trainingSize / batchSize);

    NNMatrixType input;
    NNMatrixType target;

    for(unsigned int epoch = 0; epoch < epochs; ++epoch)
    {
        std::cout << "Epoch " << epoch + 1 << " out of " << epochs << "\n";
        std::shuffle(permutaionTable.begin(), permutaionTable.end(), generator);
        for(unsigned int n = 0; n < numBatches; ++n)
        {
            // Train on single batch
            unsigned int startIdx = n*batchSize;

            for(unsigned int i = 0; i < batchSize; ++i)
            {
                unsigned int idx = startIdx + i;
                if(idx >= trainingSize) break;

                input = *inputs[permutaionTable[idx]];
                target = *targets[permutaionTable[idx]];

                if(input.getRows() != inputNodes_ || input.getColumns() != 1)
                {
                    std::cout << "ERROR: passed input matrix has wrong dimensions!\n";
                    return;
                }

                singleInputTrain(input, target);
            }
            
            // Adjust weights and biases after finishing batch
            for(auto it = layers_.begin(); it < layers_.end(); ++it)
            {
                (*it)->performSDGStep(learningRate_);
            }
        }
    }
}

void NeuralNetwork::singleInputTrain(const NNMatrixType& input, const NNMatrixType& target)
{
    // forward pass
    NNMatrixType output = input;
    // Vectors storing results of layers' calculations
    // used in backpropagation
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

    // dC/da
    NNMatrixType costDerivative = costFunction_->calculateCostDerivative(output, target);

    unsigned int backpropIdx = layers_.size() - 1;
    for(auto it = layers_.rbegin(); it < layers_.rend() - 1; ++it)
    {
        costDerivative = (*it)->backpropagate(costDerivative, weightedInputs[backpropIdx], outputs[backpropIdx - 1]);
        backpropIdx--;
    }

    // Handle first layer differently - pass input instead of last layer's output
    costDerivative = layers_[0]->backpropagate(costDerivative, weightedInputs[0], input);
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

        unsigned int predictedLabel = 0;
        float currentValue = result.get(0, 0);
        float maxValue = currentValue;
        for(unsigned int i = 0; i < result.getRows(); ++i)
        {
            currentValue = result.get(i, 0);
            if(currentValue > maxValue)
            {
                predictedLabel = i;
                maxValue = currentValue;
            }
        }

        unsigned int expectedLabel = 0;
        currentValue = targets[n]->get(0, 0);
        maxValue = currentValue;
        for(unsigned int i = 0; i < result.getRows(); ++i)
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

void NeuralNetwork::addLayer(Layer* layer)
{
    layers_.emplace_back(std::unique_ptr<Layer>(layer));
    outputNodes_ = layer->getNodesCount();
}

void NeuralNetwork::save(const char* filename) const
{
    std::ofstream ofile(filename, std::ios::binary);

    ofile.write((char*)&learningRate_, sizeof(learningRate_));
    ofile.write((char*)&inputNodes_, sizeof(inputNodes_));
    ofile.write((char*)&outputNodes_, sizeof(outputNodes_));
    
    costFunction_->serialize(ofile);

    unsigned int layersCount = getLayersCount();
    ofile.write((char*)&layersCount, sizeof(layersCount));

    for(auto it = layers_.begin(); it < layers_.end(); ++it)
    {
        (*it)->serialize(ofile);
    }

    ofile.close();
}

NeuralNetwork* NeuralNetwork::load(const char* filename)
{
    std::ifstream ifile(filename, std::ios::binary);

    if(!ifile.is_open())
    {
        throw data_load_failure(filename);
    }

    // Loading basic info
    float learingRate;
    unsigned int inputNodes, outputNodes;
    
    ifile.read((char*)&learingRate, sizeof(learingRate));
    ifile.read((char*)&inputNodes, sizeof(inputNodes));
    ifile.read((char*)&outputNodes, sizeof(outputNodes));

    // Cost function
    unsigned int idLen;
    ifile.read((char*)&idLen, sizeof(idLen));

    char* id = new char[idLen];
    ifile.read(id, idLen*sizeof(char));

    // do not delete this!
    CostFunctionStrategy* costFunction = nullptr;

    if(id[0] == 'M' && id[1] == 'S' && id[2] == 'E')
    {
        costFunction = new MeanSquereErrorCost();
    }
    else if(id[0] == 'C' && id[1] == 'E' && id[2] == 'X')
    {
        costFunction = new CrossEntropyCost();
    }
    else
    {
        delete[] id;
        throw data_load_failure(filename);
    }
    
    NeuralNetwork* nn = new NeuralNetwork(inputNodes, learingRate, std::unique_ptr<CostFunctionStrategy>(costFunction));

    // Layers
    unsigned int layersCount;
    ifile.read((char*)&layersCount, sizeof(layersCount));

    for(unsigned int i = 0; i < layersCount; ++i)
    {
        ifile.read((char*)&idLen, sizeof(idLen));
        if(id) delete[] id;
        id = new char[idLen];
        ifile.read(id, idLen*sizeof(char));

        unsigned int rows, columns;
        ifile.read((char*)&rows, sizeof(rows));
        ifile.read((char*)&columns, sizeof(columns));

        // this resource cannot be deleted!
        Layer* layer = nullptr;

        if(id[0] == 'S' && id[1] == 'I' && id[2] == 'G')
        {
            layer = new SigmoidLayer(rows, columns);
        }
        else if(id[0] == 'R' && id[1] == 'E' && id[2] == 'L')
        {
            layer = new ReLULayer(rows, columns);
        }

        unsigned int len = rows*columns;
        NNDataType* bufferWeights = new NNDataType[len];
        ifile.read((char*)bufferWeights, len*sizeof(NNDataType));

        NNDataType* bufferBias = new NNDataType[rows];
        ifile.read((char*)bufferBias, rows*sizeof(NNDataType));

        NNMatrixType weights = NNMatrixType(bufferWeights, rows, columns);
        NNMatrixType bias = NNMatrixType(bufferBias, rows, 1);

        delete[] bufferWeights;
        delete[] bufferBias;

        layer->weights_ = weights;
        layer->bias_ = bias;

        nn->addLayer(layer);
    }

    if(id) delete[] id;

    ifile.close();

    return nn;
}
