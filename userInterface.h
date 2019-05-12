#pragma once

#include <memory>
#include "mnistData.h"

class UserInterface
{
public:
    static void handleInteraction();
private:
    UserInterface();

    enum class State { ModelNotLoaded, ModelLoaded, Exit };

    static void clearInputBuffer();
    static void printAsciiImage(const char* image);
    static void handleCurrentState(MNISTData* &data, NeuralNetwork* &nn, State &state);
    static void handleStateModelNotLoaded(MNISTData* &data, NeuralNetwork* &nn, State &state);
    static void handleStateModelLoaded(NeuralNetwork* &nn, State &state);
    static void handleModelLoading(NeuralNetwork* &nn, State &state);
    static void handleModelCreation(MNISTData* &data, NeuralNetwork* &nn, State &state);
    static void handleDigitRecognition(NeuralNetwork* &nn);
    static void handleModelSave(NeuralNetwork* &nn);
};