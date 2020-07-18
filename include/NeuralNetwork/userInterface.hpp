#pragma once

#include <memory>
#include <optional>

#include "mnistData.hpp"

class Image;

class UserInterface
{
public:
    static void handleInteraction();
private:
    UserInterface();

    enum class State { ModelNotLoaded, LayersAddition, ModelLoaded, Exit };

    static void clearInputBuffer();
    static void printAsciiImage(const Image& image);

    static void handleCurrentState(std::optional<MNISTData>& data, std::optional<NeuralNetwork>& nn, State& state);
    static void handleStateModelNotLoaded(std::optional<MNISTData>& data, std::optional<NeuralNetwork>& nn, State& state);
    
    static void handleStateLayersAddition(std::optional<MNISTData>& data, std::optional<NeuralNetwork>& nn, State& state);
    
    template <typename T>
    static void addLayer(std::optional<NeuralNetwork>& nn);
    
    static void trainModel(std::optional<MNISTData>& data, std::optional<NeuralNetwork>& nn, State& state);
    
    static void handleStateModelLoaded(std::optional<NeuralNetwork>& nn, State& state);
    static void handleModelLoading(std::optional<NeuralNetwork>& nn, State& state);
    static void handleModelCreation(std::optional<MNISTData>& data, std::optional<NeuralNetwork>& nn, State& state);
    static void handleDigitRecognition(std::optional<NeuralNetwork>& nn);
    static void handleModelSave(std::optional<NeuralNetwork>& nn);
};
