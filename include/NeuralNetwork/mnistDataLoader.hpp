#pragma once

#include <memory>

#include "mnistData.hpp"

class MNISTDataLoader
{
typedef std::vector<std::shared_ptr<NNMatrixType>> MatrixVec;
public:
    static MNISTData loadData(const char* trainingImagesFilename,
                              const char* trainingLabelsFilename,
                              const char* testingImagesFilename,
                              const char* testingLabelsFilename);
private:
    static int reverseInt(int number);

    static void loadImages(const char* imagesFilename, std::vector<std::unique_ptr<char[]>>& images, int& nLoadedImages, int& imagePixels);
    static std::unique_ptr<char[]> loadLabels(const char* labelsFilename, int& nLoadedLabels);

    static void createMatriciesFromRawData(const std::vector<std::unique_ptr<char[]>>& images, std::unique_ptr<char[]>, int nData, int imagePixels, MatrixVec& imagesMatricies, MatrixVec& lablesMatricies);

    MNISTDataLoader();
};