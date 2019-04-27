#pragma once

#include "matrix.h"
#include "neuralnetwork.h"

class MNISTData
{
typedef std::vector<std::shared_ptr<NNMatrixType>> MatrixVec;
public:
    MNISTData(const MatrixVec& trainingData, const MatrixVec& trainingLabels,
              const MatrixVec& testingData, const MatrixVec& testingLabels):
              trainingData_(trainingData), trainingLabels_(trainingLabels),
              testingData_(testingData), testingLabels_(testingLabels) {}

    MatrixVec const& getTrainingData() const
    {
        return trainingData_;
    }
    void setTrainingData(const MatrixVec& trainingData)
    {
        trainingData_ = trainingData;
    }

    MatrixVec const& getTrainingLabels() const
    {
        return trainingLabels_;
    }
    void setTrainingLabels(const MatrixVec& trainingLabels)
    {
        trainingLabels_ = trainingLabels;
    }

    MatrixVec const& getTestingData() const
    {
        return testingData_;
    }
    void setTestingData(const MatrixVec& testingData)
    {
        testingData_ = testingData;
    }

    MatrixVec const& getTestingLabels() const
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