#include "matrix.h"
#include "mnistDataLoader.h"
#include "neuralnetwork.h"

int main()
{
    MNISTData data;

    MNISTDataLoader::loadData(data, "data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 
                                    "data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");


    NeuralNetwork nn = NeuralNetwork(784, 100, 10, 0.0019);
    
    std::cout << "Training...\n";
    nn.train(1, 8, data.getTrainingData(), data.getTrainingLabels());
    
    std::cout << "Testing...\n";
    float acc = nn.test(data.getTestingData(), data.getTestingLabels());

    std::cout << "Accuracity: " << acc << "%\n";

    return 0;
}