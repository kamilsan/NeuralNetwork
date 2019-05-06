#pragma once

#include <exception>

#include "mnistData.h"

class data_load_failure : public std::exception
{
public:
    data_load_failure(const char* filename): filename_(filename) {}

    const char* what() const throw()
    {
    	return (std::string("Could not load data from file named: ") + filename_).c_str();
    }
private:
    std::string filename_;
};

class MNISTDataLoader
{
typedef std::vector<std::shared_ptr<NNMatrixType>> MatrixVec;
public:
    static void loadData(MNISTData& data, const char* trainingImagesFilename,
                       const char* trainingLabelsFilename,
                       const char* testingImagesFilename,
                       const char* testingLabelsFilename);
private:
    static int reverseInt(int number);

    static void loadImages(const char* imagesFilename, std::vector<char*> &images, int &nLoadedImages, int &imagePixels);
    static char* loadLabels(const char* labelsFilename, int &nLoadedLabels);

    static void createMatriciesFromRawData(const std::vector<char*> &images, const char* labels, int nData, int imagePixels, MatrixVec& imagesMatricies, MatrixVec& lablesMatricies);

    MNISTDataLoader();
};