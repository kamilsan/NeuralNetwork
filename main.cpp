#include "userInterface.h"
#include "mnistDataLoader.h"
#include "reluLayer.h"
#include "sigmoidLayer.h"

#include <memory>

int main()
{
    //UserInterface::handleInteraction();

    MNISTData* data = MNISTDataLoader::loadData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 
                                             "data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");

    NeuralNetwork nn = NeuralNetwork(784, 0.03);
    nn.addLayer(std::make_shared<ReLULayer>(300, 784));
    nn.addLayer(std::make_shared<SigmoidLayer>(10, 300));


    nn.train(1, 32, data->getTrainingData(), data->getTrainingLabels());
    
    float acc = nn.test(data->getTestingData(), data->getTestingLabels());
    std::cout << "Acc: " << acc << "\n";

    delete data;

    return 0;
}