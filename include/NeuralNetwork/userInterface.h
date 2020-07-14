#pragma once

#include <memory>

#include "mnistData.h"

class UserInterface
{
public:
    static void handleInteraction();
private:
    UserInterface();

    enum class State { ModelNotLoaded, LayersAddition, ModelLoaded, Exit };

    static void clearInputBuffer();
    static void printAsciiImage(const char* image);

    static void handleCurrentState(MNISTData* &data, NeuralNetwork* &nn, State &state, unsigned int& outputNodes);
    static void handleStateModelNotLoaded(MNISTData* &data, NeuralNetwork* &nn, State &state, unsigned int& outputNodes);
    
    static void handleStateLayersAddition(const MNISTData* data, NeuralNetwork* nn, State &state, unsigned int& outputNodes);
    template <typename T>
    
    static void addLayer(NeuralNetwork* nn, unsigned int& outputNodes);
    
    static void trainModel(const MNISTData* data, NeuralNetwork* nn, State &state);
    
    static void handleStateModelLoaded(NeuralNetwork* nn, State &state);
    static void handleModelLoading(NeuralNetwork* &nn, State &state);
    static void handleModelCreation(MNISTData* &data, NeuralNetwork* &nn, State &state);
    static void handleDigitRecognition(const NeuralNetwork* nn);
    static void handleModelSave(const NeuralNetwork* nn);
};
