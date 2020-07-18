#include <fstream>

#include "mnistDataLoader.hpp"

#include "data_load_failure.hpp"

typedef std::vector<std::shared_ptr<NNMatrixType>> MatrixVec;

int MNISTDataLoader::reverseInt(int number)
{
    // divides int into 4 chunks and concats them in reversed order
    unsigned char c1, c2, c3, c4;
    c1 = number & 255;
    c2 = (number >> 8) & 255;
    c3 = (number >> 16) & 255;
    c4 = (number >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

MNISTData MNISTDataLoader::loadData(const char* trainingImagesFilename,
                                    const char* trainingLabelsFilename,
                                    const char* testingImagesFilename,
                                    const char* testingLabelsFilename)
{
    std::vector<std::unique_ptr<char[]>> trainingImages;
    int nLoadedTrainingImages;
    int imagePixels;
    loadImages(trainingImagesFilename, trainingImages, nLoadedTrainingImages, imagePixels);

    int nLoadedTrainingLabels;
    auto trainingLabels = loadLabels(trainingLabelsFilename, nLoadedTrainingLabels);

    std::vector<std::unique_ptr<char[]>> testingImages;
    int nLoadedTestingImages;
    loadImages(testingImagesFilename, testingImages, nLoadedTestingImages, imagePixels);

    int nLoadedTestingLabels;
    auto testingLabels = loadLabels(testingLabelsFilename, nLoadedTestingLabels);

    MatrixVec trainingDataMatrix, trainingLabelsMatrix, testingDataMatrix, testingLabelsMatrix;
    createMatriciesFromRawData(trainingImages, std::move(trainingLabels), nLoadedTrainingImages, 
                                imagePixels, trainingDataMatrix, trainingLabelsMatrix);
    createMatriciesFromRawData(testingImages, std::move(testingLabels), nLoadedTestingImages, 
                                imagePixels, testingDataMatrix, testingLabelsMatrix);
    
    return MNISTData(trainingDataMatrix, trainingLabelsMatrix, testingDataMatrix, testingLabelsMatrix);
}

void MNISTDataLoader::loadImages(const char* imagesFilename, 
                             std::vector<std::unique_ptr<char[]>>& images, 
                             int& nLoadedImages,
                             int& imagePixels)
{
    std::ifstream file;
    file.open(imagesFilename, std::ios::binary);
    if(!file.is_open())
    {
        throw data_load_failure(imagesFilename);
    }

    int magicNum, nItems, w, h;
    file.read((char *)&magicNum, 4);
    magicNum = reverseInt(magicNum);
    file.read((char *)&nItems, 4);
    nItems = reverseInt(nItems);
    file.read((char *)&w, 4);
    w = reverseInt(w);
    file.read((char *)&h, 4);
    h = reverseInt(h);

    images.clear();
    images.reserve(nItems);
    int len = w*h;
    for(int i = 0; i < nItems; ++i)
    {
        images.push_back(std::make_unique<char[]>(len));
        file.read(images[i].get(), len);
    }

    file.close();

    nLoadedImages = nItems;
    imagePixels = w*h;
}

std::unique_ptr<char[]> MNISTDataLoader::loadLabels(const char* labelsFilename, int& nLoadedLabels)
{
    std::ifstream file;
    file.open(labelsFilename, std::ios::binary);
    if(!file.is_open())
    {
        throw data_load_failure(labelsFilename);
    }

    int magicNum, nItems;
    file.read((char *)&magicNum, sizeof(int));
    magicNum = reverseInt(magicNum);
    file.read((char *)&nItems, sizeof(int));
    nItems = reverseInt(nItems);

    std::unique_ptr<char[]> labels = std::make_unique<char[]>(nItems);
    file.read(labels.get(), nItems);
    file.close();

    nLoadedLabels = nItems;
    return labels;
}

void MNISTDataLoader::createMatriciesFromRawData(const std::vector<std::unique_ptr<char[]>>& images, 
                                             std::unique_ptr<char[]> labels, 
                                             int nData, 
                                             int imagePixels, 
                                             MatrixVec& imagesMatricies, 
                                             MatrixVec& lablesMatricies)
{
    const int POSSIBLE_LABELS = 10;

    std::unique_ptr<float[]> imageMatrixData = std::make_unique<float[]>(imagePixels);
    float labelMatrixData[POSSIBLE_LABELS];

    imagesMatricies.clear();
    lablesMatricies.clear();
    imagesMatricies.reserve(nData);
    lablesMatricies.reserve(nData);

    for(int i = 0; i < nData; ++i)
    {
        for(int j = 0; j < imagePixels; ++j)
        {
            imageMatrixData[j] = (unsigned char)(images[i][j])/255.0f;
        }
        imagesMatricies.push_back(std::make_shared<NNMatrixType>(imageMatrixData.get(), imagePixels, 1));
        
        // convert from label to matrix by setting matrix entry to 1 in specific place
        int label = +labels[i];
        for(int n = 0; n < POSSIBLE_LABELS; ++n)
        {
            if(n != label) labelMatrixData[n] = 0.0f;
            else labelMatrixData[n] = 1.0f;
        }
        lablesMatricies.push_back(std::make_shared<NNMatrixType>(labelMatrixData, POSSIBLE_LABELS, 1));
    }
}