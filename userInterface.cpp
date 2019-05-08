#include <iostream>
#include <limits>
#include <exception>

#include "userInterface.h"
#include "mnistDataLoader.h"
#include "data_load_failure.h"
#include "image.h"

void UserInterface::clearInputBuffer()
{
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

void UserInterface::handleInteraction()
{    
    NeuralNetwork* nn = nullptr;
    MNISTData* mnistData = nullptr;

    State state = State::ModelNotLoaded;

    while(state != State::Exit)
    {
        handleCurrentState(mnistData, nn, state);
    }

    delete mnistData;
    delete nn;
}

void UserInterface::handleCurrentState(MNISTData* &data, NeuralNetwork* &nn, State &state)
{
    switch(state)
    {
        case State::ModelNotLoaded:
            handleStateModelNotLoaded(data, nn, state);
            break;
        case State::ModelLoaded:
            handleStateModelLoaded(nn, state);
            break;
        default:
            break;
    }
}

void UserInterface::handleStateModelNotLoaded(MNISTData* &data, NeuralNetwork* &nn, State &state)
{
    int choice;
    do
    {
        std::cout << "No model is currently loaded. Please choose one of the listed options:\n";
        std::cout << "1. load model from file\n";
        std::cout << "2. train new model\n";
        std::cout << "3. exit\n";
        std::cout << ">>>";

        std::cin >> choice;
        if(!std::cin) clearInputBuffer();
    } while(choice < 1 || choice > 3);

    switch(choice)
    {
        case 1:
            handleModelLoading(nn);
            state = State::ModelLoaded;
            break;
        case 2:
            handleModelCreation(data, nn);
            state = State::ModelLoaded;
            break;
        case 3:
            state = State::Exit;
            break;
    }
}

void UserInterface::handleStateModelLoaded(NeuralNetwork* &nn, State &state)
{
    int choice;
    do
    {
        std::cout << "Choose one of the listed actions:\n";
        std::cout << "1. recognize digit\n";
        std::cout << "2. new model\n";
        std::cout << "3. save model\n";
        std::cout << "4. exit\n";
        std::cout << ">>>";

        std::cin >> choice;
        if(!std::cin) clearInputBuffer();
    } while(choice < 1 || choice > 4);

    switch(choice)
    {
        case 1:
            handleDigitRecognition(nn);
            break;
        case 2:
            state = State::ModelNotLoaded;
            break;
        case 3:
            handleModelSave(nn);
            break;
        case 4:
            state = State::Exit;
            break;
    }
}

void UserInterface::handleModelLoading(NeuralNetwork* &nn)
{
    std::cout << "Enter model filename: ";
    std::string filename;
    std::cin >> filename;

    nn = NeuralNetwork::load(filename.c_str());
}

void UserInterface::handleModelCreation(MNISTData* &data, NeuralNetwork* &nn)
{
    if(!data)
    {
        data = MNISTDataLoader::loadData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 
                                  "data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
    }

    if(nn != nullptr) delete nn;

    unsigned nHiddenLayerNodes;
    float learingRate;
    unsigned nEpochs;
    unsigned batchSize; 

    std::cout << "Enter number of neurons in hidden layer: ";
    std::cin >> nHiddenLayerNodes;
    std::cout << "Learning rate: ";
    std::cin >> learingRate;
    std::cout << "Epochs: ";
    std::cin >> nEpochs;
    std::cout << "Batch size: ";
    std::cin >> batchSize;

    nn = new NeuralNetwork(784, nHiddenLayerNodes, 10, learingRate);
    
    std::cout << "Training...\n";
    nn->train(nEpochs, batchSize, data->getTrainingData(), data->getTrainingLabels());
    
    std::cout << "Testing...\n";
    float acc = nn->test(data->getTestingData(), data->getTestingLabels());
    std::cout << "Model created! Accuracity: " << acc << "%\n";
}

void UserInterface::handleDigitRecognition(NeuralNetwork* &nn)
{
    std::cout << "Enter image filename: ";
    std::string filename;
    std::cin >> filename;
    
    std::unique_ptr<Image> img;
    try
    {
        img = std::make_unique<Image>(filename.c_str());
    }
    catch(const data_load_failure& ex)
    {
        std::cout << ex.what() << "\n";
        return;
    }
    
    if(img->getWidth() != 28 || img->getHeight() != 28)
    {
        std::cout << "Improper image size! Make sure that the image has dimentions 28x28.\n";
        return;
    }

    const int IMAGE_PIXELS = 784;
    float matrixData[IMAGE_PIXELS];
    const char* imagePixels = img->getPixels();

    for(int i = 0; i < IMAGE_PIXELS; ++i)
    {
        matrixData[i] = (unsigned char)(imagePixels[i])/255.0f;
    }

    NNMatrixType inputMatrix = NNMatrixType(matrixData, IMAGE_PIXELS, 1);
    NNMatrixType resultMatrix;
    nn->feedforward(inputMatrix, resultMatrix);

    int predictedLabel = 0;
    float max = resultMatrix.get(0, 0);
    for(int i = 0; i < resultMatrix.getRows(); ++i)
    {
        if(resultMatrix.get(i, 0) > max)
        {
            predictedLabel = i;
            max = resultMatrix.get(i, 0);
        }
    }

    std::cout << "Result matrix:\n" << resultMatrix << "\n";
    std::cout << "Predicted Label:\n" << predictedLabel << "\n";
}

void UserInterface::handleModelSave(NeuralNetwork* &nn)
{
    std::cout << "Enter model filename: ";
    std::string filename;
    std::cin >> filename;

    nn->save(filename.c_str());
}