#include <iostream>
#include <limits>
#include <exception>
#include <chrono>
#include <memory>

#include "userInterface.h"
#include "mnistDataLoader.h"
#include "data_load_failure.h"
#include "image.h"
#include "reluLayer.h"
#include "sigmoidLayer.h"
#include "meanSquereErrorCost.h"
#include "crossEntropyCost.h"

void UserInterface::clearInputBuffer()
{
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

void UserInterface::printAsciiImage(const char* image)
{
    const unsigned int IMG_SIZE = 28;
    std::cout << "\n";
    for(unsigned int y = 0, k = 0; y < IMG_SIZE; ++y)
    {
        for(unsigned int x = 0; x < IMG_SIZE; ++x, k+=3)
        {
            //if pixel is bright enough
            if((unsigned char)image[k] > 127)
            {
                std::cout << "X";
            }
            else std::cout << " ";
        }
        std::cout << "\n";
    }
}

void UserInterface::handleInteraction()
{    
    NeuralNetwork* nn = nullptr;
    MNISTData* mnistData = nullptr;

    //handles states in loop
    //it's a finite-state automaton
    State state = State::ModelNotLoaded;
    unsigned int outputNodes = 0;

    while(state != State::Exit)
    {
        handleCurrentState(mnistData, nn, state, outputNodes);
    }

    if(mnistData != nullptr) delete mnistData;
    if(nn != nullptr) delete nn;
}

void UserInterface::handleCurrentState(MNISTData* &data, NeuralNetwork* &nn, State &state, unsigned int& outputNodes)
{
    switch(state)
    {
        case State::ModelNotLoaded:
            handleStateModelNotLoaded(data, nn, state, outputNodes);
            break;
        case State::LayersAddition:
            handleStateLayersAddition(data, nn, state, outputNodes);
            break;
        case State::ModelLoaded:
            handleStateModelLoaded(nn, state);
            break;
        default:
            break;
    }
}

void UserInterface::handleStateModelNotLoaded(MNISTData* &data, NeuralNetwork* &nn, State &state, unsigned int& outputNodes)
{
    outputNodes = 0;

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
            std::cout << "\n";
            handleModelLoading(nn, state);
            break;
        case 2:
            std::cout << "\n";
            handleModelCreation(data, nn, state);
            break;
        case 3:
            state = State::Exit;
            break;
    }
}

void UserInterface::handleStateLayersAddition(const MNISTData* data, NeuralNetwork* nn, State &state, unsigned int& outputNodes)
{
    int choice;
    do
    {
        std::cout << "What type of layer do you want to add?\n";
        std::cout << "1. relu\n";
        std::cout << "2. sigmoid\n";
        std::cout << "3. none\n";
        std::cout << ">>>";

        std::cin >> choice;
        if(!std::cin) clearInputBuffer();
    } while(choice < 1 || choice > 3);

    std::cout << "\n";

    switch(choice)
    {
        case 1:
            addLayer<ReLULayer>(nn, outputNodes);
            break;
        case 2:
            addLayer<SigmoidLayer>(nn, outputNodes);
            break;
        case 3:
            if(nn->getLayersCount() == 0)
            {
                std::cout << "Please add at least one layer.\n\n";
            }
            else if(outputNodes != 10)
            {
                std::cout << "Last layer has to have 10 nodes!\n";
                state = State::ModelNotLoaded;
            }
            else trainModel(data, nn, state);
            break;
    }
}

template <typename T>
void UserInterface::addLayer(NeuralNetwork* nn, unsigned int& outputNodes)
{
    unsigned int nodes = 0;
    do
    {
        std::cout << "How many nodes should this layer have?\n>>>";
        std::cin >> nodes;
        if(!std::cin)
        {
            clearInputBuffer();
            nodes = 0;
        }
    } while(nodes < 1);

    nn->addLayer<T>(nodes);
    outputNodes = nodes;
    
    std::cout << "\n";
}

void UserInterface::trainModel(const MNISTData* data, NeuralNetwork* nn, State &state)
{
    auto readIntFromUserInputAndVerify = [](const char* prompt, unsigned int &value, unsigned int condition) 
    { 
        do
        {
            std::cout << prompt;
            std::cin >> value;
            if(!std::cin)
            {
                clearInputBuffer();
                value = 0;
            }
        } while(value < condition);
    };

    unsigned int nEpochs;
    unsigned int batchSize; 

    readIntFromUserInputAndVerify("Epochs: ", nEpochs, 1);
    readIntFromUserInputAndVerify("Batch size: ", batchSize, 1);

    //train and measure time
    std::cout << "\nTraining...\n";
    auto timeStart = std::chrono::high_resolution_clock::now();
    nn->train(nEpochs, batchSize, data->getTrainingData(), data->getTrainingLabels());
    auto timeEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timeElapsed = timeEnd - timeStart;
    std::cout << "Training took " << timeElapsed.count() << "s\n";

    std::cout << "\nTesting...\n";
    float acc = nn->test(data->getTestingData(), data->getTestingLabels());
    std::cout << "Model created! Accuracity: " << acc << "%\n\n";

    state = State::ModelLoaded;
}

void UserInterface::handleStateModelLoaded(NeuralNetwork* nn, State &state)
{
    unsigned int choice;
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
            std::cout << "\n";
            handleDigitRecognition(nn);
            break;
        case 2:
            state = State::ModelNotLoaded;
            std::cout << "\n";
            break;
        case 3:
            std::cout << "\n";
            handleModelSave(nn);
            break;
        case 4:
            state = State::Exit;
            break;
    }
}

void UserInterface::handleModelLoading(NeuralNetwork* &nn, State &state)
{
    std::cout << "Enter model filename: ";
    std::string filename;
    std::cin >> filename;

    if(nn != nullptr) delete nn;
    try
    {
        nn = NeuralNetwork::load(filename.c_str());
    }
    catch(const data_load_failure& ex)
    {
        std::cout << ex.what() << "\n";
        state = State::ModelNotLoaded;
        return;
    }

    std::cout << "Model loaded!\n\n";
    state = State::ModelLoaded;
}

void UserInterface::handleModelCreation(MNISTData* &data, NeuralNetwork* &nn, State &state)
{
    if(!data)
    {
        try
        {
            data = MNISTDataLoader::loadData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 
                                             "data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
        }
        catch(const data_load_failure& ex)
        {
            std::cout << "Loading model data failed! Application will be closed now.\n";
            state = State::Exit;
            return;
        }
    }

    if(nn != nullptr) delete nn;

    float learingRate;
    do
    {
        std::cout << "Learning rate: ";
        std::cin >> learingRate;
        if(!std::cin)
        {
            clearInputBuffer();
            learingRate = -1;
        }
    } while(learingRate <= 0);

    std::unique_ptr<CostFunctionStrategy> costFunction = nullptr;

    unsigned int choice;
    do
    {
        std::cout << "Choose one of the listed cost functions:\n";
        std::cout << "1. mean squere error\n";
        std::cout << "2. cross-entropy\n";
        std::cout << ">>>";

        std::cin >> choice;
        if(!std::cin) clearInputBuffer();
    } while(choice < 1 || choice > 2);

    switch(choice)
    {
        case 1:
            costFunction = std::make_unique<MeanSquereErrorCost>();
            break;
        case 2:
            costFunction = std::make_unique<CrossEntropyCost>();
    }

    nn = new NeuralNetwork(784, learingRate, std::move(costFunction));

    state = State::LayersAddition;

    std::cout << "\n";
}

void UserInterface::handleDigitRecognition(const NeuralNetwork* nn)
{
    std::cout << "Enter image filename: ";
    std::string filename;
    std::cin >> filename;
    
    std::unique_ptr<Image> img = nullptr;
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

    const unsigned int IMAGE_PIXELS = 784;
    float matrixData[IMAGE_PIXELS];
    const char* imagePixels = img->getPixels();

    for(unsigned int i = 0; i < IMAGE_PIXELS; ++i)
    {
        matrixData[i] = ((unsigned char)imagePixels[3*i])/255.0f;
    }

    NNMatrixType inputMatrix = NNMatrixType(matrixData, IMAGE_PIXELS, 1);
    NNMatrixType resultMatrix = nn->feedforward(inputMatrix);

    unsigned int predictedLabel = 0;
    float max = resultMatrix.get(0, 0);
    for(unsigned int i = 0; i < resultMatrix.getRows(); ++i)
    {
        if(resultMatrix.get(i, 0) > max)
        {
            predictedLabel = i;
            max = resultMatrix.get(i, 0);
        }
    }

    printAsciiImage(imagePixels);

    std::cout << "Result matrix:\n" << resultMatrix << "\n\n";
    std::cout << "Predicted Label:\n" << predictedLabel << "\n\n";
}

void UserInterface::handleModelSave(const NeuralNetwork* nn)
{
    std::cout << "Enter model filename: ";
    std::string filename;
    std::cin >> filename;

    nn->save(filename.c_str());
    std::cout << "Model saved!\n\n";
}