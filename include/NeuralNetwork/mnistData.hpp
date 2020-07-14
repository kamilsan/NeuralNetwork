#pragma once

#include "matrix.hpp"
#include "neuralnetwork.hpp"

class MNISTData
{
typedef std::vector<std::shared_ptr<NNMatrixType>> MatrixVec;
public:
    MNISTData() = default;

    MNISTData(const MatrixVec& trainingData, const MatrixVec& trainingLabels,
              const MatrixVec& testingData, const MatrixVec& testingLabels):
              trainingData_(trainingData), trainingLabels_(trainingLabels),
              testingData_(testingData), testingLabels_(testingLabels) 
    {}

    const MatrixVec& getTrainingData() const
    {
        return trainingData_;
    }
    void setTrainingData(const MatrixVec& trainingData)
    {
        trainingData_ = trainingData;
    }

    const MatrixVec& getTrainingLabels() const
    {
        return trainingLabels_;
    }
    void setTrainingLabels(const MatrixVec& trainingLabels)
    {
        trainingLabels_ = trainingLabels;
    }

    const MatrixVec& getTestingData() const
    {
        return testingData_;
    }
    void setTestingData(const MatrixVec& testingData)
    {
        testingData_ = testingData;
    }

    const MatrixVec& getTestingLabels() const
    {
        return testingLabels_;
    }
    void setTestingLabels(const MatrixVec& testingLabels)
    {
        testingLabels_ = testingLabels;
    }
private:
    MatrixVec trainingData_;
    MatrixVec trainingLabels_;
    MatrixVec testingData_;
    MatrixVec testingLabels_;
};
